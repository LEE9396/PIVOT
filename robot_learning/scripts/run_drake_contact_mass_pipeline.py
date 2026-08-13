#!/usr/bin/env python3
"""RB5 contact → 3-camera angle → AFT200 → 2-link mass posterior."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
from itertools import combinations
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from pydrake.all import StartMeshcat


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

import run_custom_cad_configuration_information_experiment as study  # noqa: E402
import run_drake_static_mass_tracking as static  # noqa: E402
import simulate_drake_rb5_contact_ft_custom_object as contact  # noqa: E402


DEFAULT_OUTPUT = ROOT / "results/contact_mass_pipeline_2link.json"
WRIST_PITCH_CANDIDATES_DEG = (40.0, 80.0, 100.0, 120.0)
WRIST_PITCH_CANDIDATES_3LINK_DEG = (-10.0, 10.0, -10.0, 15.0)
WRIST_PITCH_FREE_3LINK_DEG = (
    -20.0,
    -15.0,
    -10.0,
    -5.0,
    5.0,
    10.0,
    15.0,
    20.0,
)
WRIST_ROLL_FREE_3LINK_DEG = (
    -20.0,
    20.0,
    -20.0,
    20.0,
    -20.0,
    20.0,
    -20.0,
    20.0,
)
MIN_UPPER_JOINT_EXCITATION_DEG = 10.0
MIN_JOINT_RESPONSE_DEG = 1.0
MIN_INFORMATIVE_JOINT_TORQUE_NM = 2.0 * contact.AFT200_TORQUE_RESOLUTION_NM
MAX_UNRESPONSIVE_INFORMATIVE_ACTIONS = 2
MAX_CAMERA_ANGLE_SPAN_DEG = 2.0
FOUNDATIONPOSE_CONSENSUS_FRAMES = 3
PGC_CALIBRATED_KP = 3000.0
MAX_NORMALIZED_INNOVATION_RMS = 2.0
DYNAMIC_LIFT_WEIGHT = 1.0
ROBUST_COM_OFFSET_BOUND_M = 0.005
ROBUST_ANGLE_BIAS_BOUND_DEG = 1.0
ROBUST_FORCE_BIAS_BOUND_N = 0.07
ROBUST_TORQUE_BIAS_BOUND_NM = 0.007
ROBUST_VOLUME_RELATIVE_BOUND = 0.03
TRUE_PART_MASS_KG = np.array([0.8, 0.4])
FOUNDATIONPOSE_ROTATION_STD_DEG = {
    "cam_d435i_left": 0.8,
    "cam_d435i_right": 0.8,
    "cam_d456_front": 0.5,
}
PARENT_HINGE_XYZ_M = np.array((0.083, 0.0, 0.010))
CHILD_HINGE_XYZ_M = np.array((0.106, 0.0, 0.0))


def _rotation_y(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array(((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c)))


def _project_rotation(matrices: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(np.mean(matrices, axis=0))
    correction = np.diag((1.0, 1.0, np.linalg.det(u @ vt)))
    return u @ correction @ vt


def _filter_foundationpose_track(
    translations: np.ndarray,
    rotations: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    translations = translations[valid]
    rotations = rotations[valid]
    center = np.median(translations, axis=0)
    distance = np.linalg.norm(translations - center, axis=1)
    distance_mad = max(float(np.median(np.abs(distance - np.median(distance)))), 1e-6)
    keep = distance <= np.median(distance) + 3.5 * 1.4826 * distance_mad
    rotations = rotations[keep]
    translations = translations[keep]
    mean_rotation = _project_rotation(rotations)
    angular_error = np.array(
        [
            math.acos(np.clip((np.trace(mean_rotation.T @ r) - 1.0) / 2.0, -1.0, 1.0))
            for r in rotations
        ]
    )
    keep = angular_error <= math.radians(5.0)
    return (
        np.median(translations[keep], axis=0),
        _project_rotation(rotations[keep]),
        int(np.count_nonzero(keep)),
    )


def _opening_from_relative(
    relative: np.ndarray,
    joint_min_deg: float,
    joint_max_deg: float,
) -> float:
    angle = (
        180.0
        + math.degrees(math.atan2(relative[0, 2], relative[0, 0]))
    ) % 360.0
    if joint_max_deg <= 180.0:
        angle = min(angle, 360.0 - angle)
    return float(np.clip(angle, joint_min_deg, joint_max_deg))


def foundationpose_joint_angle(
    actual_opening_deg: float,
    rng: np.random.Generator,
    *,
    joint_min_deg: float,
    joint_max_deg: float,
    joint_axis: tuple[float, float, float] | None = None,
) -> dict[str, object]:
    axis = np.asarray(joint_axis or (0.0, 1.0, 0.0), dtype=float)
    bend = math.radians(
        actual_opening_deg if joint_axis is not None else actual_opening_deg - 180.0
    )
    nominal_relative = _axis_rotation(axis, bend)

    def measured_angle(relative: np.ndarray) -> float:
        if joint_axis is None:
            return _opening_from_relative(
                relative,
                joint_min_deg,
                joint_max_deg,
            )
        sine = 0.5 * np.dot(
            axis / np.linalg.norm(axis),
            np.array(
                (
                    relative[2, 1] - relative[1, 2],
                    relative[0, 2] - relative[2, 0],
                    relative[1, 0] - relative[0, 1],
                )
            ),
        )
        cosine = (np.trace(relative) - 1.0) / 2.0
        return float(
            np.clip(
                math.degrees(math.atan2(sine, cosine)),
                joint_min_deg,
                joint_max_deg,
            )
        )

    per_camera = {}
    parent_rotations = []
    child_rotations = []
    for name, std_deg in FOUNDATIONPOSE_ROTATION_STD_DEG.items():
        parent_samples = []
        child_samples = []
        parent_xyz = []
        child_xyz = []
        valid = []
        for _ in range(30):
            parent_noise = _axis_rotation(
                rng.normal(size=3),
                math.radians(rng.normal(0.0, std_deg)),
            )
            child_noise = _axis_rotation(
                rng.normal(size=3),
                math.radians(rng.normal(0.0, std_deg)),
            )
            is_valid = bool(rng.random() >= 0.08)
            if not is_valid:
                child_noise = _rotation_y(math.pi) @ child_noise
            parent_samples.append(parent_noise)
            child_samples.append(child_noise @ nominal_relative)
            parent_xyz.append(rng.normal(0.0, 0.002, 3))
            child_xyz.append(np.array((0.1, 0.0, 0.0)) + rng.normal(0.0, 0.002, 3))
            valid.append(is_valid)
        _, parent_rotation, parent_inliers = _filter_foundationpose_track(
            np.asarray(parent_xyz), np.asarray(parent_samples), np.asarray(valid)
        )
        _, child_rotation, child_inliers = _filter_foundationpose_track(
            np.asarray(child_xyz), np.asarray(child_samples), np.asarray(valid)
        )
        relative = parent_rotation.T @ child_rotation
        angle = measured_angle(relative)
        per_camera[name] = {
            "opening_deg": angle,
            "parent_inliers": parent_inliers,
            "child_inliers": child_inliers,
        }
        parent_rotations.append(parent_rotation)
        child_rotations.append(child_rotation)
    relative = (
        _project_rotation(np.asarray(parent_rotations)).T
        @ _project_rotation(np.asarray(child_rotations))
    )
    fused = measured_angle(relative)
    return {
        "source": "filtered_parent_child_6d_foundationpose",
        "per_camera": per_camera,
        "fused_opening_deg": fused,
    }


def _foundationpose_tracking_measurement(path: Path) -> dict[str, object]:
    result = json.loads(path.read_text())
    summary = result["summary"]
    return {
        "source": "official_foundationpose_track_one",
        "fused_opening_deg": summary["filtered_static_opening_deg"],
        "stationary": summary["stationary"],
        "static_window_span_deg": summary["static_window_span_deg"],
        "tracking_fps": summary[
            "tracking_fps_three_cameras_two_links"
        ],
    }


def _load_foundationpose_results(
    paths: tuple[Path, ...],
) -> dict[str, object]:
    results = [json.loads(path.read_text()) for path in paths]
    if len(results) == 1:
        return results[0]
    merged = deepcopy(results[0])
    merged["frames"] = []
    merged["summary"]["static_holds"] = []
    next_index = 0
    for result in results:
        index_map = {}
        for frame in result["frames"]:
            copied = deepcopy(frame)
            index_map[frame["index"]] = next_index
            copied["index"] = next_index
            merged["frames"].append(copied)
            next_index += 1
        for hold in result["summary"]["static_holds"]:
            copied = deepcopy(hold)
            copied["frame_indices"] = [
                index_map[index] for index in hold["frame_indices"]
            ]
            merged["summary"]["static_holds"].append(copied)
    merged["summary"]["stationary"] = all(
        hold["stationary"]
        for hold in merged["summary"]["static_holds"]
    )
    merged["source_results"] = [str(path) for path in paths]
    return merged


def _axis_rotation(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    cross = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return (
        np.eye(3) * math.cos(angle_rad)
        + (1.0 - math.cos(angle_rad)) * np.outer(axis, axis)
        + math.sin(angle_rad) * cross
    )


def measured_mass_regressor(
    result: contact.ContactFtResult,
    fused_opening_deg: float,
) -> np.ndarray:
    gravity = np.asarray(result.gravity_sensor_m_s2)
    positions = np.asarray(result.part_com_sensor_m).copy()
    origin = np.asarray(result.joint_origin_sensor_m)
    axis = np.asarray(result.joint_axis_sensor)
    delta = math.radians(
        fused_opening_deg - result.actual_opening_angles_deg[0]
    )
    positions[1] = (
        origin
        + _axis_rotation(axis, delta) @ (positions[1] - origin)
    )
    # AFT200 joint reaction is the support wrench, opposite gravity load.
    return np.column_stack(
        [
            np.concatenate((-gravity, -np.cross(position, gravity)))
            for position in positions
        ]
    )


def _hold_mass_regressor(
    hold: contact.StaticHoldMeasurement,
    measured_angles_deg: list[float] | tuple[float, ...],
    part_com_sensor_m: np.ndarray | None = None,
) -> np.ndarray:
    gravity = np.asarray(hold.gravity_sensor_m_s2)
    positions = np.asarray(
        hold.part_com_sensor_m
        if part_com_sensor_m is None
        else part_com_sensor_m
    ).copy()
    origins = np.asarray(hold.joint_origins_sensor_m).copy()
    axes = np.asarray(hold.joint_axes_sensor).copy()
    for joint_index, measured_angle in enumerate(measured_angles_deg):
        delta = math.radians(
            measured_angle - hold.actual_opening_angles_deg[joint_index]
        )
        rotation = _axis_rotation(axes[joint_index], delta)
        origin = origins[joint_index]
        positions[joint_index + 1 :] = (
            origin
            + (rotation @ (positions[joint_index + 1 :] - origin).T).T
        )
        if joint_index + 1 < len(origins):
            origins[joint_index + 1 :] = (
                origin
                + (
                    rotation
                    @ (origins[joint_index + 1 :] - origin).T
                ).T
            )
            axes[joint_index + 1 :] = (
                rotation @ axes[joint_index + 1 :].T
            ).T
    return np.column_stack(
        [
            np.concatenate((-gravity, -np.cross(position, gravity)))
            for position in positions
        ]
    )


def _systematic_error_regressor(
    hold: contact.StaticHoldMeasurement,
    measured_angles_deg: list[float] | tuple[float, ...],
    com_offsets_body_m: np.ndarray,
    angle_bias_deg: np.ndarray,
) -> np.ndarray:
    positions = np.asarray(hold.part_com_sensor_m) + np.einsum(
        "nij,nj->ni",
        np.asarray(hold.part_rotations_sensor),
        com_offsets_body_m,
    )
    return _hold_mass_regressor(
        hold,
        np.asarray(measured_angles_deg) + angle_bias_deg,
        positions,
    )


def _bounded_mass_fit(
    design: np.ndarray,
    observed: np.ndarray,
    minimum_mass_kg: float = contact.MINIMUM_PART_MASS_KG,
) -> np.ndarray:
    part_count = design.shape[1]
    best = None
    for fixed_bits in range(1 << part_count):
        fixed = np.array(
            [bool(fixed_bits & (1 << index)) for index in range(part_count)]
        )
        candidate = np.full(part_count, minimum_mass_kg)
        free = ~fixed
        if np.any(free):
            target = observed - design[:, fixed] @ candidate[fixed]
            candidate[free] = np.linalg.lstsq(
                design[:, free],
                target,
                rcond=None,
            )[0]
        if np.any(candidate < minimum_mass_kg):
            continue
        residual = float(np.linalg.norm(design @ candidate - observed))
        if best is None or residual < best[0]:
            best = (residual, candidate)
    if best is None:
        raise RuntimeError("no feasible positive mass estimate")
    return best[1]


def _bootstrap_uncertainty_summary(
    mass_samples: list[np.ndarray],
    density_samples: list[np.ndarray],
    mean: np.ndarray,
    volume: np.ndarray,
) -> dict[str, object]:
    mass_interval = np.percentile(
        np.asarray(mass_samples),
        (2.5, 97.5),
        axis=0,
    )
    density_interval = np.percentile(
        np.asarray(density_samples),
        (2.5, 97.5),
        axis=0,
    )
    mass_width = (mass_interval[1] - mass_interval[0]) / (2.0 * mean)
    density_width = (
        density_interval[1] - density_interval[0]
    ) / (2.0 * mean / volume)
    return {
        "part_mass_95_interval_kg": mass_interval.T.tolist(),
        "relative_95_half_width": mass_width.tolist(),
        "part_density_95_interval_kg_m3": density_interval.T.tolist(),
        "density_relative_95_half_width": density_width.tolist(),
        "maximum_relative_95_half_width": float(
            max(np.max(mass_width), np.max(density_width))
        ),
    }


def _adaptive_hold_estimate(
    holds: tuple[contact.StaticHoldMeasurement, ...],
    *,
    part_count: int,
    object_profile: str,
    volume: np.ndarray,
    true_mass: np.ndarray,
    target_relative_95_half_width: float,
    minimum_static_holds: int,
    previous_mean: np.ndarray | None,
    seed: int,
    robust_error_scale: float,
) -> dict[str, object]:
    profile = contact.HOUSEHOLD_PROFILES[object_profile]
    angle_bounds = profile.get(
        "joint_angle_bounds_deg",
        tuple((0.0, limit) for limit in profile["joint_limits_deg"]),
    )
    joint_axes = profile.get(
        "joint_axes",
        (None,) * (part_count - 1),
    )
    regressors = []
    wrenches = []
    blocks = []
    accepted_holds = []
    accepted_angles = []
    camera_results = []
    accepted = False
    physical_safe = False
    innovation_rms = None
    measurement_consistent = True
    wrench_weight = np.array(
        [1.0 / contact.AFT200_FORCE_RESOLUTION_N] * 3
        + [1.0 / contact.AFT200_TORQUE_RESOLUTION_NM] * 3
    )
    reference = np.asarray(holds[0].grasp_relative_pose)
    translation_drift = rotation_drift = 0.0
    for index, hold in enumerate(holds):
        camera_joint_frames = [
            [
                foundationpose_joint_angle(
                    angle,
                    np.random.default_rng(
                        seed
                        + 1009 * index
                        + 97 * frame_index
                        + joint_index
                    ),
                    joint_min_deg=lower,
                    joint_max_deg=upper,
                    joint_axis=axis,
                )
                for joint_index, (
                    angle,
                    (lower, upper),
                    axis,
                ) in enumerate(
                    zip(
                        hold.actual_opening_angles_deg,
                        angle_bounds,
                        joint_axes,
                        strict=True,
                    )
                )
            ]
            for frame_index in range(FOUNDATIONPOSE_CONSENSUS_FRAMES)
        ]
        fused_angles = [
            float(
                np.mean(
                    [
                        frame[joint_index]["fused_opening_deg"]
                        for frame in camera_joint_frames
                    ]
                )
            )
            for joint_index in range(part_count - 1)
        ]
        camera_results.append(
            {
                "source": "filtered_foundationpose_simulation_with_gt_first_mask",
                "per_camera": camera_joint_frames[0][0]["per_camera"],
                "consensus_frame_count": (
                    FOUNDATIONPOSE_CONSENSUS_FRAMES
                ),
                "fused_opening_deg": fused_angles[0],
                "fused_opening_angles_deg": fused_angles,
                "stationary": True,
                "all_links_visible": True,
            }
        )
        regressor = _hold_mass_regressor(hold, fused_angles)
        known_tool = np.asarray(hold.known_tool_wrench)
        wrench = (
            np.concatenate(
                (hold.ft_reaction_force, hold.ft_reaction_torque)
            )
            - known_tool
        )
        relative = np.linalg.inv(reference) @ np.asarray(
            hold.grasp_relative_pose
        )
        translation_drift = float(np.linalg.norm(relative[:3, 3]))
        rotation_drift = math.degrees(
            math.acos(
                np.clip(
                    (np.trace(relative[:3, :3]) - 1.0) / 2.0,
                    -1.0,
                    1.0,
                )
            )
        )
        physical_safe = bool(
            hold.ft_stationary
            and hold.joints_stationary
            and hold.contact_free
            and translation_drift
            <= contact.MAX_GRASP_TRANSLATION_DRIFT_M
            and rotation_drift
            <= contact.MAX_FREE_GRASP_ROTATION_DRIFT_DEG
        )
        innovation_rms = None
        measurement_consistent = True
        if physical_safe and regressors:
            previous_design = np.vstack(
                [
                    previous_regressor * wrench_weight[:, None]
                    for previous_regressor in regressors
                ]
            )
            if (
                np.linalg.matrix_rank(previous_design) == part_count
                and np.linalg.cond(previous_design) < 50.0
            ):
                previous_observed = np.concatenate(
                    [
                        previous_wrench * wrench_weight
                        for previous_wrench in wrenches
                    ]
                )
                previous_fit = _bounded_mass_fit(
                    previous_design,
                    previous_observed,
                )
                innovation_rms = float(
                    np.linalg.norm(
                        (regressor @ previous_fit - wrench)
                        * wrench_weight
                    )
                    / math.sqrt(6.0)
                )
                measurement_consistent = (
                    innovation_rms <= MAX_NORMALIZED_INNOVATION_RMS
                )
        accepted = physical_safe and measurement_consistent
        if accepted:
            regressors.append(regressor)
            wrenches.append(wrench)
            blocks.append(np.asarray(hold.ft_block_means) - known_tool)
            accepted_holds.append(hold)
            accepted_angles.append(fused_angles)
    if not regressors:
        raise RuntimeError("no safe static hold available for estimation")
    design = np.vstack(
        [regressor * wrench_weight[:, None] for regressor in regressors]
    )
    observed = np.concatenate(
        [wrench * wrench_weight for wrench in wrenches]
    )
    mean = _bounded_mass_fit(design, observed)
    residual_pool = np.asarray(
        [
            wrench - regressor @ mean
            for regressor, wrench in zip(
                regressors,
                wrenches,
                strict=True,
            )
        ]
    )
    residual_pool -= residual_pool.mean(axis=0)
    rng = np.random.default_rng(seed + 7919 * len(holds))
    sources = (
        "nominal_repeatability",
        "aft200_bias",
        "joint_angle_bias",
        "body_frame_com_offset",
        "volume_scale",
        "combined",
    )
    mass_bootstrap = {source: [] for source in sources}
    density_bootstrap = {source: [] for source in sources}
    for _ in range(400):
        com_offsets = rng.uniform(
            -ROBUST_COM_OFFSET_BOUND_M,
            ROBUST_COM_OFFSET_BOUND_M,
            (part_count, 3),
        ) * robust_error_scale
        angle_bias = rng.uniform(
            -ROBUST_ANGLE_BIAS_BOUND_DEG,
            ROBUST_ANGLE_BIAS_BOUND_DEG,
            part_count - 1,
        ) * robust_error_scale
        angle_design = np.vstack(
            [
                _systematic_error_regressor(
                    hold,
                    angles,
                    np.zeros_like(com_offsets),
                    angle_bias,
                )
                * wrench_weight[:, None]
                for hold, angles in zip(
                    accepted_holds,
                    accepted_angles,
                    strict=True,
                )
            ]
        )
        com_design = np.vstack(
            [
                _systematic_error_regressor(
                    hold,
                    angles,
                    com_offsets,
                    np.zeros_like(angle_bias),
                )
                * wrench_weight[:, None]
                for hold, angles in zip(
                    accepted_holds,
                    accepted_angles,
                    strict=True,
                )
            ]
        )
        combined_design = np.vstack(
            [
                _systematic_error_regressor(
                    hold,
                    angles,
                    com_offsets,
                    angle_bias,
                )
                * wrench_weight[:, None]
                for hold, angles in zip(
                    accepted_holds,
                    accepted_angles,
                    strict=True,
                )
            ]
        )
        residual_bias = np.concatenate(
            (
                rng.uniform(
                    -ROBUST_FORCE_BIAS_BOUND_N,
                    ROBUST_FORCE_BIAS_BOUND_N,
                    3,
                ),
                rng.uniform(
                    -ROBUST_TORQUE_BIAS_BOUND_NM,
                    ROBUST_TORQUE_BIAS_BOUND_NM,
                    3,
                ),
            )
        ) * robust_error_scale
        sampled_wrenches = [
            block[rng.integers(0, len(block), len(block))].mean(axis=0)
            + residual_pool[rng.integers(0, len(residual_pool))]
            for block in blocks
        ]
        sampled_observed = np.concatenate(
            [wrench * wrench_weight for wrench in sampled_wrenches]
        )
        biased_observed = np.concatenate(
            [
                (wrench - residual_bias) * wrench_weight
                for wrench in sampled_wrenches
            ]
        )
        sampled_masses = {
            "nominal_repeatability": _bounded_mass_fit(
                design,
                sampled_observed,
            ),
            "aft200_bias": _bounded_mass_fit(design, biased_observed),
            "joint_angle_bias": _bounded_mass_fit(
                angle_design,
                sampled_observed,
            ),
            "body_frame_com_offset": _bounded_mass_fit(
                com_design,
                sampled_observed,
            ),
            "combined": _bounded_mass_fit(
                combined_design,
                biased_observed,
            ),
        }
        sampled_masses["volume_scale"] = sampled_masses[
            "nominal_repeatability"
        ]
        volume_scale = 1.0 + rng.uniform(
            -ROBUST_VOLUME_RELATIVE_BOUND,
            ROBUST_VOLUME_RELATIVE_BOUND,
            part_count,
        ) * robust_error_scale
        for source, sampled_mass in sampled_masses.items():
            mass_bootstrap[source].append(sampled_mass)
            density_bootstrap[source].append(
                sampled_mass
                / (
                    volume * volume_scale
                    if source in ("volume_scale", "combined")
                    else volume
                )
            )
    uncertainty_by_error_source = {
        source: _bootstrap_uncertainty_summary(
            mass_bootstrap[source],
            density_bootstrap[source],
            mean,
            volume,
        )
        for source in sources
    }
    nominal_maximum = uncertainty_by_error_source[
        "nominal_repeatability"
    ]["maximum_relative_95_half_width"]
    for summary in uncertainty_by_error_source.values():
        summary["increase_over_nominal"] = max(
            0.0,
            summary["maximum_relative_95_half_width"] - nominal_maximum,
        )
    combined = uncertainty_by_error_source["combined"]
    lower, upper = np.asarray(combined["part_mass_95_interval_kg"]).T
    density_lower, density_upper = np.asarray(
        combined["part_density_95_interval_kg_m3"]
    ).T
    relative_95_half_width = np.asarray(
        combined["relative_95_half_width"]
    )
    nominal_density = mean / volume
    density_relative_95_half_width = np.asarray(
        combined["density_relative_95_half_width"]
    )
    maximum_robust_relative_95_half_width = combined[
        "maximum_relative_95_half_width"
    ]
    data_rank = int(np.linalg.matrix_rank(design))
    condition_number = (
        float(np.linalg.cond(design))
        if data_rank == part_count
        else None
    )
    convergence = (
        None
        if previous_mean is None
        else float(np.max(np.abs(mean - previous_mean) / mean))
    )
    mass_boundary_active = bool(
        np.any(mean <= contact.MINIMUM_PART_MASS_KG + 1e-6)
    )
    total_mass_feasible = bool(
        contact.MINIMUM_OBJECT_MASS_KG
        <= float(mean.sum())
        <= contact.PGC_MAX_PAYLOAD_KG
    )
    joint_excitation_range = np.ptp(
        np.asarray(accepted_angles),
        axis=0,
    )
    internal_joints_excited = bool(
        robust_error_scale == 0.0
        or np.all(
            joint_excitation_range >= MIN_UPPER_JOINT_EXCITATION_DEG
        )
    )
    target_met = bool(
        accepted
        and len(regressors) >= minimum_static_holds
        and data_rank == part_count
        and condition_number is not None
        and condition_number <= 50.0
        and maximum_robust_relative_95_half_width
        <= target_relative_95_half_width
        and convergence is not None
        and convergence <= target_relative_95_half_width / 2.0
        and not mass_boundary_active
        and total_mass_feasible
        and internal_joints_excited
    )
    latest_fit_residual = wrench - regressor @ mean
    latest = holds[-1]
    return {
        "step": len(holds),
        "selected_wrist_pitch_deg": latest.wrist_pitch_deg,
        "three_camera_foundationpose_filter": camera_results[-1],
        "aft200_filtered_loaded_wrench": np.concatenate(
            (latest.ft_reaction_force, latest.ft_reaction_torque)
        ).tolist(),
        "aft200_empty_tool_tare_wrench": list(latest.known_tool_wrench),
        "aft200_contact_object_wrench": wrench.tolist(),
        "accepted": accepted,
        "contact_gate": {
            "grasp_translation_drift_m": translation_drift,
            "grasp_rotation_drift_deg": rotation_drift,
            "grasp_stable": bool(
                translation_drift
                <= contact.MAX_GRASP_TRANSLATION_DRIFT_M
                and rotation_drift
                <= contact.MAX_FREE_GRASP_ROTATION_DRIFT_DEG
            ),
            "physical_safe": physical_safe,
            "normalized_innovation_rms": innovation_rms,
            "measurement_consistent": measurement_consistent,
            "fit_residual_force_norm_n": float(
                np.linalg.norm(latest_fit_residual[:3])
            ),
            "fit_residual_torque_norm_nm": float(
                np.linalg.norm(latest_fit_residual[3:])
            ),
            "opening_angle_span_deg": list(latest.opening_angle_span_deg),
            "actual_opening_angles_deg": list(
                latest.actual_opening_angles_deg
            ),
            "maximum_abs_joint_velocity_deg_s": list(
                latest.maximum_abs_joint_velocity_deg_s
            ),
            "maximum_abs_joint_acceleration_deg_s2": list(
                latest.maximum_abs_joint_acceleration_deg_s2
            ),
            "joints_stationary": latest.joints_stationary,
        },
        "estimate": {
            "part_mass_kg": mean.tolist(),
            "part_mass_95_interval_kg": np.column_stack(
                (lower, upper)
            ).tolist(),
            "part_density_kg_m3": nominal_density.tolist(),
            "part_density_95_interval_kg_m3": np.column_stack(
                (density_lower, density_upper)
            ).tolist(),
        },
        "uncertainty_stop": {
            "data_rank": data_rank,
            "design_condition_number": condition_number,
            "mass_change_from_previous": convergence,
            "relative_95_half_width": relative_95_half_width.tolist(),
            "maximum_relative_95_half_width": float(
                np.max(relative_95_half_width)
            ),
            "density_relative_95_half_width": (
                density_relative_95_half_width.tolist()
            ),
            "maximum_robust_relative_95_half_width": (
                maximum_robust_relative_95_half_width
            ),
            "target_relative_95_half_width": (
                target_relative_95_half_width
            ),
            "target_met": target_met,
            "minimum_mass_boundary_active": mass_boundary_active,
            "total_mass_feasible": total_mass_feasible,
            "joint_excitation_range_deg": joint_excitation_range.tolist(),
            "minimum_joint_excitation_deg": (
                MIN_UPPER_JOINT_EXCITATION_DEG
                if robust_error_scale > 0.0
                else None
            ),
            "internal_joints_excited": internal_joints_excited,
            "uncertainty_by_error_source": uncertainty_by_error_source,
        },
        "evaluator_only": {
            "true_part_mass_kg": true_mass.tolist(),
            "mass_relative_error": (
                np.abs(mean - true_mass) / true_mass
            ).tolist(),
        },
        "_mean": mean,
        "_regressors": regressors,
    }


def _geometry_centroids_body_m(
    object_profile: str,
) -> tuple[tuple[float, float, float], ...]:
    profile = contact.HOUSEHOLD_PROFILES[object_profile]
    if "centroids_body_m" in profile:
        return profile["centroids_body_m"]
    if object_profile == "desk_lamp":
        arm, head, _ = profile["sizes_m"]
        return (
            (arm[0] / 2.0, 0.0, 0.0),
            (-head[0] / 2.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        )
    base, lid = profile["sizes_m"]
    return (
        (0.0, 0.0, 0.0),
        (0.0, -base[1] / 2.0, 0.003 + lid[2] / 2.0),
    )


def _foundationpose_static_mass_regressor(
    hold: contact.StaticHoldMeasurement,
    tracked_result: dict[str, object],
    tracked_hold: dict[str, object],
    com_offsets_body_m: tuple[tuple[float, float, float], ...],
) -> np.ndarray:
    frames = {
        frame["index"]: frame for frame in tracked_result["frames"]
    }
    selected = [
        frames[index]
        for index in tracked_hold["frame_indices"]
        if index in frames
    ]
    if not selected:
        raise ValueError("FoundationPose hold has no fused pose frames")
    links = tuple(
        tracked_result.get(
            "mass_order_links",
            ("parent",)
            + tuple(
                f"child_{index}"
                for index in range(1, len(com_offsets_body_m))
            ),
        )
    )
    positions_world = []
    for link, offset in zip(links, com_offsets_body_m, strict=True):
        samples = []
        for frame in selected:
            pose = np.asarray(frame["fused_world_poses"][link])
            samples.append(
                pose[:3, :3] @ np.asarray(offset) + pose[:3, 3]
            )
        positions_world.append(np.median(samples, axis=0))
    x_sensor_world = np.linalg.inv(np.asarray(hold.sensor_world_pose))
    positions_sensor = [
        (x_sensor_world @ np.append(position, 1.0))[:3]
        for position in positions_world
    ]
    gravity = np.asarray(hold.gravity_sensor_m_s2)
    return np.column_stack(
        [
            np.concatenate((-gravity, -np.cross(position, gravity)))
            for position in positions_sensor
        ]
    )


def _foundationpose_com_trajectory(
    frames: list[dict[str, object]],
    com_offsets_body_m: tuple[tuple[float, float, float], ...],
) -> tuple[np.ndarray, np.ndarray]:
    tracked = [
        frame for frame in frames if "fused_world_poses" in frame
    ]
    if len(tracked) < 7:
        return np.empty(0), np.empty((0, 0, 3))
    times = np.asarray([frame["time_s"] for frame in tracked])
    links = ("parent",) + tuple(
        f"child_{index}" for index in range(1, len(com_offsets_body_m))
    )
    com_world = np.empty((len(tracked), len(links), 3))
    for frame_index, frame in enumerate(tracked):
        poses = []
        if "encoder_parent_world_pose" in frame:
            poses.append(np.asarray(frame["encoder_parent_world_pose"]))
            for joint_index, opening_deg in enumerate(
                frame["foundationpose_opening_angles_deg"]
            ):
                parent = poses[-1]
                pose = np.eye(4)
                pose[:3, :3] = (
                    parent[:3, :3]
                    @ _axis_rotation(
                        np.array([0.0, 1.0, 0.0]),
                        math.radians(opening_deg - 180.0),
                    )
                )
                hinge = (
                    PARENT_HINGE_XYZ_M
                    if joint_index == 0
                    else CHILD_HINGE_XYZ_M
                )
                pose[:3, 3] = (
                    parent[:3, :3] @ hinge + parent[:3, 3]
                )
                poses.append(pose)
        else:
            poses = [
                np.asarray(frame["fused_world_poses"][link])
                for link in links
            ]
        for link_index, (pose, offset) in enumerate(
            zip(poses, com_offsets_body_m, strict=True)
        ):
            com_world[frame_index, link_index] = (
                pose[:3, :3] @ np.asarray(offset) + pose[:3, 3]
            )
    return times, com_world


def _foundationpose_dynamic_force_regressors(
    frames: list[dict[str, object]],
    measurements: tuple[contact.DynamicLiftMeasurement, ...],
    com_offsets_body_m: tuple[tuple[float, float, float], ...],
) -> dict[float, np.ndarray]:
    times, com_world = _foundationpose_com_trajectory(
        frames,
        com_offsets_body_m,
    )
    if not len(times):
        return {}
    part_count = com_world.shape[1]
    regressors = {}
    for measurement in measurements:
        time_s = float(measurement.time_s)
        nearest = np.argsort(np.abs(times - time_s))[:7]
        local_time = times[nearest] - time_s
        acceleration_world = np.empty((part_count, 3))
        for link_index in range(part_count):
            for axis in range(3):
                coefficients = np.polynomial.polynomial.polyfit(
                    local_time,
                    com_world[nearest, link_index, axis],
                    3,
                )
                acceleration_world[link_index, axis] = (
                    2.0 * coefficients[2]
                )
        sensor_rotation = np.asarray(
            measurement.sensor_rotation_world_to_sensor
        )
        force_columns = (
            sensor_rotation
            @ (
                acceleration_world
                - np.array([0.0, 0.0, -9.81])
            ).T
        )
        regressor = np.zeros((6, part_count))
        regressor[:3] = force_columns
        regressors[time_s] = regressor
    return regressors


def _foundationpose_impulse_regressors(
    frames: list[dict[str, object]],
    com_offsets_body_m: tuple[tuple[float, float, float], ...],
    boundaries_s: np.ndarray,
) -> list[np.ndarray]:
    times, com_world = _foundationpose_com_trajectory(
        frames,
        com_offsets_body_m,
    )
    if not len(times):
        return []
    velocities = []
    for boundary in boundaries_s:
        nearest = np.argsort(np.abs(times - boundary))[:15]
        local_time = times[nearest] - boundary
        velocity = np.empty((com_world.shape[1], 3))
        for part_index in range(com_world.shape[1]):
            for axis in range(3):
                coefficients = np.polynomial.polynomial.polyfit(
                    local_time,
                    com_world[nearest, part_index, axis],
                    3,
                )
                velocity[part_index, axis] = coefficients[1]
        velocities.append(velocity)
    regressors = []
    gravity = np.array([0.0, 0.0, -9.81])
    for index, duration in enumerate(np.diff(boundaries_s)):
        average_specific_force = (
            (velocities[index + 1] - velocities[index]) / duration
            - gravity
        )
        regressor = np.zeros((6, com_world.shape[1]))
        regressor[:3] = average_specific_force.T
        regressors.append(regressor)
    return regressors


def _contact_wrench(
    loaded: contact.ContactFtResult,
    empty: contact.ContactFtResult,
) -> np.ndarray:
    return np.concatenate(
        (
            np.asarray(loaded.ft_reaction_force)
            - np.asarray(empty.ft_reaction_force),
            np.asarray(loaded.ft_reaction_torque)
            - np.asarray(empty.ft_reaction_torque),
        )
    )


def _contact_wrench_variance(
    loaded: contact.ContactFtResult,
    empty: contact.ContactFtResult,
) -> np.ndarray:
    return (
        np.asarray(loaded.ft_mean_standard_error) ** 2
        + np.asarray(empty.ft_mean_standard_error) ** 2
    )


def _fit(
    regressors: list[np.ndarray],
    wrenches: list[np.ndarray],
    wrench_variances: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    design = np.vstack(regressors)
    observed = np.concatenate(wrenches)
    mean = np.maximum(np.linalg.lstsq(design, observed, rcond=None)[0], 1e-6)
    inverse = np.linalg.pinv(design)
    covariance = (
        inverse * np.concatenate(wrench_variances)[None, :]
    ) @ inverse.T
    return mean, covariance


def select_geometry_action_set(
    candidate_pitches_deg: tuple[float, ...],
    candidate_rolls_deg: tuple[float, ...],
    regressor_sets: list[list[np.ndarray]],
    valid_sets: list[set[int]],
    planned_steps: int,
) -> dict[str, object]:
    if (
        len(candidate_pitches_deg) != len(candidate_rolls_deg)
        or not 1 <= planned_steps <= len(candidate_pitches_deg)
        or len(regressor_sets) != len(valid_sets)
    ):
        raise ValueError("invalid geometry action candidates")
    wrench_weight = np.array(
        [1.0 / contact.AFT200_FORCE_RESOLUTION_N] * 3
        + [1.0 / contact.AFT200_TORQUE_RESOLUTION_NM] * 3
    )
    evaluations = []
    for selected in combinations(range(len(candidate_pitches_deg)), planned_steps):
        if any(index not in valid for valid in valid_sets for index in selected):
            continue
        conditions = []
        information = []
        for regressors in regressor_sets:
            design = np.vstack(
                [
                    np.asarray(regressors[index]) * wrench_weight[:, None]
                    for index in selected
                ]
            )
            if design.shape[1] != 3 or np.linalg.matrix_rank(design) < 3:
                conditions.append(math.inf)
                information.append(-math.inf)
                continue
            conditions.append(float(np.linalg.cond(design)))
            information.append(float(np.linalg.slogdet(design.T @ design)[1]))
        evaluations.append(
            {
                "indices": list(selected),
                "wrist_pitch_sequence_deg": [
                    candidate_pitches_deg[index] for index in selected
                ],
                "wrist_roll_sequence_deg": [
                    candidate_rolls_deg[index] for index in selected
                ],
                "worst_design_condition_number": max(conditions),
                "minimum_logdet_information": min(information),
            }
        )
    finite = [
        item
        for item in evaluations
        if math.isfinite(item["worst_design_condition_number"])
    ]
    if not finite:
        raise RuntimeError("no identifiable safe and camera-visible action set remains")
    return {
        "selected": min(
            finite,
            key=lambda item: (
                item["worst_design_condition_number"],
                -item["minimum_logdet_information"],
            ),
        ),
        "evaluated_action_sets": evaluations,
    }


def select_next_geometry_action(
    current_regressors: list[np.ndarray],
    candidate_regressors: list[np.ndarray],
    valid: set[int],
    movement_costs: list[float] | None = None,
) -> dict[str, object]:
    wrench_weight = np.array(
        [1.0 / contact.AFT200_FORCE_RESOLUTION_N] * 3
        + [1.0 / contact.AFT200_TORQUE_RESOLUTION_NM] * 3
    )
    current = [
        np.asarray(regressor) * wrench_weight[:, None]
        for regressor in current_regressors
    ]
    use_difficulty = movement_costs is not None
    movement_costs = movement_costs or [0.0] * len(candidate_regressors)
    if len(movement_costs) != len(candidate_regressors):
        raise ValueError("one movement cost is required per candidate")
    evaluations = []
    for index in sorted(valid):
        design = np.vstack(
            current
            + [
                np.asarray(candidate_regressors[index])
                * wrench_weight[:, None]
            ]
        )
        rank = int(np.linalg.matrix_rank(design))
        gram = design.T @ design
        regularized_information = float(
            np.linalg.slogdet(gram + np.eye(gram.shape[0]) * 1e-9)[1]
        )
        evaluations.append(
            {
                "index": index,
                "data_rank": rank,
                "design_condition_number": (
                    float(np.linalg.cond(design))
                    if rank == design.shape[1]
                    else None
                ),
                "logdet_information": regularized_information,
                "movement_cost": float(movement_costs[index]),
            }
        )
    if not evaluations:
        raise RuntimeError("no identifiable safe and camera-visible next action")
    quality = lambda item: (
        item["data_rank"],
        item["design_condition_number"] is not None
        and item["design_condition_number"] <= 50.0,
    )
    best_quality = max(map(quality, evaluations))
    comparable = [item for item in evaluations if quality(item) == best_quality]
    if use_difficulty:
        selected = max(
            comparable,
            key=lambda item: (
                item["logdet_information"] - item["movement_cost"],
                -item["movement_cost"],
                -(
                    item["design_condition_number"]
                    if item["design_condition_number"] is not None
                    else math.inf
                ),
            ),
        )
    else:
        selected = max(
            comparable,
            key=lambda item: (
                item["logdet_information"],
                -(
                    item["design_condition_number"]
                    if item["design_condition_number"] is not None
                    else math.inf
                ),
            ),
        )
    return {"selected": selected, "evaluated_actions": evaluations}


def _joint_response_counts(
    previous_angles_deg: np.ndarray,
    measured_angles_deg: np.ndarray,
    predicted_torque_nm: np.ndarray,
    counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    informative = predicted_torque_nm >= MIN_INFORMATIVE_JOINT_TORQUE_NM
    responsive = (
        np.abs(measured_angles_deg - previous_angles_deg)
        >= MIN_JOINT_RESPONSE_DEG
    )
    return (
        np.where(informative, np.where(responsive, 0, counts + 1), counts),
        informative,
        responsive,
    )


def plan_actions_from_geometry(path: Path, planned_steps: int) -> dict[str, object]:
    source = json.loads(path.read_text())
    candidates = source["candidates"]
    pitches = tuple(float(item["wrist_pitch_deg"]) for item in candidates)
    rolls = tuple(float(item["wrist_roll_deg"]) for item in candidates)
    regressors = []
    for item in candidates:
        if "mass_regressor" in item:
            regressors.append(np.asarray(item["mass_regressor"], dtype=float))
            continue
        if not {"gravity_sensor_m_s2", "part_com_sensor_m"} <= item.keys():
            raise ValueError(
                "candidate needs mass_regressor or gravity and part COMs"
            )
        gravity = np.asarray(item["gravity_sensor_m_s2"], dtype=float)
        positions = np.asarray(item["part_com_sensor_m"], dtype=float)
        regressors.append(
            np.column_stack(
                [
                    np.concatenate((-gravity, -np.cross(position, gravity)))
                    for position in positions
                ]
            )
        )
    if not candidates or any(
        regressor.shape != (6, 3) or not np.isfinite(regressor).all()
        for regressor in regressors
    ):
        raise ValueError("every geometry mass_regressor must be finite 6x3")
    valid = {
        index
        for index, item in enumerate(candidates)
        if item.get("grasp_safe", True)
        and item.get("all_links_visible", True)
        and item.get("stationary", True)
        and max(item.get("camera_angle_span_deg", [0.0]))
        <= MAX_CAMERA_ANGLE_SPAN_DEG
    }
    current_regressors = [
        np.asarray(regressor, dtype=float)
        for regressor in source.get("current_regressors", [])
    ]
    if current_regressors:
        selection = select_next_geometry_action(
            current_regressors,
            regressors,
            valid,
        )
        index = selection["selected"]["index"]
        selection["selected"].update(
            {
                "indices": [index],
                "wrist_pitch_sequence_deg": [pitches[index]],
                "wrist_roll_sequence_deg": [rolls[index]],
            }
        )
    else:
        selection = select_geometry_action_set(
            pitches,
            rolls,
            [regressors],
            [valid],
            planned_steps,
        )
    return {
        "schema": "rb5-3link-geometry-action-plan-v2",
        "selection": (
            "current-object geometry regressor condition number and "
            "information under grasp and camera gates"
        ),
        "ground_truth_usage": "none",
        "geometry_action_input": str(path),
        **selection,
    }


def plan_wrist_actions(
    *,
    candidate_pitches_deg: tuple[float, ...] = (
        -20.0,
        -15.0,
        -10.0,
        -5.0,
        5.0,
        10.0,
        15.0,
        20.0,
        -20.0,
        0.0,
        20.0,
        0.0,
    ),
    candidate_rolls_deg: tuple[float, ...] = (
        -20.0,
        20.0,
        -20.0,
        20.0,
        -20.0,
        20.0,
        -20.0,
        20.0,
        0.0,
        -20.0,
        0.0,
        20.0,
    ),
    planned_steps: int = 8,
    planning_mass_hypotheses_kg: tuple[tuple[float, float, float], ...] = (
        (0.8, 0.4, 0.25),
        (0.55, 0.7, 0.35),
        (1.1, 0.3, 0.6),
        (1.0, 1.0, 1.0),
    ),
    camera_calibration_result: Path | None = None,
    seed: int = 20260728,
) -> dict[str, object]:
    if len(candidate_pitches_deg) != len(candidate_rolls_deg):
        raise ValueError("one roll is required per candidate pitch")
    if not 1 <= planned_steps <= len(candidate_pitches_deg):
        raise ValueError("planned steps exceed candidate wrist poses")
    camera_holds = (
        json.loads(camera_calibration_result.read_text())["summary"].get(
            "static_holds",
            [],
        )
        if camera_calibration_result is not None
        else []
    )
    camera_valid = {
        index
        for index in range(len(candidate_pitches_deg))
        if not camera_holds
        or index >= len(camera_holds)
        or (
            camera_holds[index]["stationary"]
            and camera_holds[index].get("all_links_visible", True)
        )
    }
    trials = []
    for trial_index, masses in enumerate(planning_mass_hypotheses_kg):
        loaded = contact.simulate(
            3,
            initial_opening_angle_deg=180.0,
            opening_angle_deg=180.0,
            pgc_controller_kp=PGC_CALIBRATED_KP,
            grasp_offset_m=contact.PARENT_END_GRASP_OFFSET_M,
            part_masses_kg=masses,
            wrist_pitch_sequence_deg=candidate_pitches_deg,
            wrist_roll_sequence_deg=candidate_rolls_deg,
            free_hinges=True,
        )
        if not (
            loaded.passed_lift
            and loaded.passed_grasp_translation
            and loaded.grasp_relative_rotation_drift_deg
            <= contact.MAX_FREE_GRASP_ROTATION_DRIFT_DEG
        ):
            raise RuntimeError("single grasp failed safety gate")
        rng = np.random.default_rng(seed + trial_index)
        regressors = []
        valid = set(camera_valid)
        grasp_reference = np.asarray(
            loaded.holds[0].grasp_relative_pose
        )
        for index, loaded_hold in enumerate(loaded.holds):
            measured_angles = (
                camera_holds[index]["filtered_opening_angles_deg"]
                if (
                    camera_holds
                    and index < len(camera_holds)
                    and trial_index == 0
                )
                else [
                    foundationpose_joint_angle(
                        angle,
                        rng,
                        joint_min_deg=0.0,
                        joint_max_deg=180.0,
                    )["fused_opening_deg"]
                    for angle in loaded_hold.actual_opening_angles_deg
                ]
            )
            regressors.append(
                _hold_mass_regressor(loaded_hold, measured_angles)
            )
            grasp_delta = (
                np.linalg.inv(grasp_reference)
                @ np.asarray(loaded_hold.grasp_relative_pose)
            )
            grasp_rotation_drift = math.degrees(
                math.acos(
                    np.clip(
                        (np.trace(grasp_delta[:3, :3]) - 1.0) / 2.0,
                        -1.0,
                        1.0,
                    )
                )
            )
            grasp_translation_drift = float(
                np.linalg.norm(grasp_delta[:3, 3])
            )
            if not (
                loaded_hold.ft_stationary
                and loaded_hold.joints_stationary
                and loaded_hold.contact_free
                and grasp_translation_drift
                <= contact.MAX_GRASP_TRANSLATION_DRIFT_M
                and grasp_rotation_drift
                <= contact.MAX_FREE_GRASP_ROTATION_DRIFT_DEG
            ):
                valid.discard(index)
        trials.append((regressors, valid))

    selection = select_geometry_action_set(
        candidate_pitches_deg,
        candidate_rolls_deg,
        [trial[0] for trial in trials],
        [trial[1] for trial in trials],
        planned_steps,
    )
    return {
        "schema": "rb5-3link-geometry-action-plan-v2",
        "selection": (
            "minimum worst-case geometry-regressor condition number, then "
            "maximum information, under grasp and three-camera gates"
        ),
        "ground_truth_usage": "none",
        "candidate_wrist_pitch_deg": list(candidate_pitches_deg),
        "candidate_wrist_roll_deg": list(candidate_rolls_deg),
        "planning_mass_hypotheses_kg": [
            list(masses) for masses in planning_mass_hypotheses_kg
        ],
        "camera_calibration_result": (
            str(camera_calibration_result)
            if camera_calibration_result is not None
            else None
        ),
        **selection,
    }


def _run_adaptive(
    *,
    opening_angle_deg: float,
    initial_opening_angle_deg: float,
    initial_opening_angles_deg: tuple[float, ...] | None,
    part_count: int,
    steps: int,
    seed: int,
    target_relative_95_half_width: float,
    minimum_static_holds: int,
    meshcat,
    part_masses_kg: tuple[float, ...] | None,
    object_profile: str,
    collision_clearance_m: float,
    robust_error_scale: float,
) -> dict[str, object]:
    profile = contact.HOUSEHOLD_PROFILES[object_profile]
    volume = np.asarray(
        profile.get(
            "volumes_m3",
            np.prod(np.asarray(profile["sizes_m"]), axis=1),
        )
    )
    true_mass = np.asarray(
        part_masses_kg or profile["default_masses_kg"],
        dtype=float,
    )
    prior_mass = volume * study.VLM_PRIOR_MEAN_KG_M3[:part_count]
    prior_std = volume * study.VLM_PRIOR_STD_KG_M3[:part_count]
    trace = []
    previous_mean = None
    current_roll_deg = 0.0
    excitation_axis = None
    bias_axis_set = False
    sweep_direction = None
    previous_measured_angles = None
    pending_joint_excitation = None
    unresponsive_action_counts = np.zeros(part_count - 1, dtype=int)
    stop_reason = "maximum_pose_count_reached"

    def select_action(holds, candidates):
        nonlocal previous_mean, current_roll_deg
        nonlocal excitation_axis, bias_axis_set, sweep_direction, stop_reason
        nonlocal previous_measured_angles, pending_joint_excitation
        nonlocal unresponsive_action_counts
        snapshot = _adaptive_hold_estimate(
            holds,
            part_count=part_count,
            object_profile=object_profile,
            volume=volume,
            true_mass=true_mass,
            target_relative_95_half_width=(
                target_relative_95_half_width
            ),
            minimum_static_holds=minimum_static_holds,
            previous_mean=previous_mean,
            seed=seed,
            robust_error_scale=robust_error_scale,
        )
        mean = snapshot.pop("_mean")
        regressors = snapshot.pop("_regressors")
        snapshot["selected_wrist_roll_deg"] = current_roll_deg
        trace.append(snapshot)
        previous_mean = mean
        if not snapshot["contact_gate"]["physical_safe"]:
            stop_reason = "safe_hold_failed"
            return None
        measured_angles = np.asarray(
            snapshot["three_camera_foundationpose_filter"][
                "fused_opening_angles_deg"
            ]
        )
        if (
            previous_measured_angles is not None
            and pending_joint_excitation is not None
        ):
            (
                unresponsive_action_counts,
                informative,
                responsive,
            ) = _joint_response_counts(
                previous_measured_angles,
                measured_angles,
                pending_joint_excitation,
                unresponsive_action_counts,
            )
            target_joints = (
                np.asarray(
                    snapshot["uncertainty_stop"][
                        "joint_excitation_range_deg"
                    ]
                )
                < MIN_UPPER_JOINT_EXCITATION_DEG
            )
            snapshot["joint_response_check"] = {
                "predicted_joint_excitation_torque_nm": (
                    pending_joint_excitation.tolist()
                ),
                "measured_joint_angle_change_deg": np.abs(
                    measured_angles - previous_measured_angles
                ).tolist(),
                "informative_joint": informative.tolist(),
                "responsive_joint": responsive.tolist(),
                "consecutive_unresponsive_informative_actions": (
                    unresponsive_action_counts.tolist()
                ),
                "minimum_informative_joint_torque_nm": (
                    MIN_INFORMATIVE_JOINT_TORQUE_NM
                ),
                "minimum_joint_response_deg": MIN_JOINT_RESPONSE_DEG,
            }
            if (
                robust_error_scale > 0.0
                and np.any(
                    target_joints
                    & (
                        unresponsive_action_counts
                        >= MAX_UNRESPONSIVE_INFORMATIVE_ACTIONS
                    )
                )
            ):
                stop_reason = "joint_not_responsive_under_single_grasp"
                return None
        else:
            snapshot["joint_response_check"] = {
                "status": "awaiting_first_informative_action",
                "minimum_informative_joint_torque_nm": (
                    MIN_INFORMATIVE_JOINT_TORQUE_NM
                ),
                "minimum_joint_response_deg": MIN_JOINT_RESPONSE_DEG,
            }
        previous_measured_angles = measured_angles
        pending_joint_excitation = None
        if snapshot["uncertainty_stop"]["target_met"]:
            stop_reason = "repeatability_interval_and_mass_converged"
            return None
        if len(holds) >= steps:
            stop_reason = "maximum_pose_count_reached"
            return None
        rotation_drift = snapshot["contact_gate"][
            "grasp_rotation_drift_deg"
        ]
        last_rotation_increase = (
            max(
                0.0,
                rotation_drift
                - trace[-2]["contact_gate"][
                    "grasp_rotation_drift_deg"
                ],
            )
            if len(trace) > 1
            else 0.0
        )
        uncertainty = snapshot["uncertainty_stop"]
        identifiable = bool(
            uncertainty["data_rank"] == part_count
            and uncertainty["design_condition_number"] is not None
            and uncertainty["design_condition_number"] <= 50.0
        )
        repeat_for_confirmation = bool(
            identifiable
            and uncertainty["maximum_robust_relative_95_half_width"]
            <= target_relative_95_half_width
        )
        opening_span = np.ptp(
            np.asarray(
                [
                    row["three_camera_foundationpose_filter"][
                        "fused_opening_angles_deg"
                    ]
                    for row in trace
                    if row["accepted"]
                ]
            ),
            axis=0,
        )
        needs_joint_excitation = bool(
            robust_error_scale > 0.0
            and np.any(
                opening_span < MIN_UPPER_JOINT_EXCITATION_DEG
            )
        )
        target_joints = opening_span < MIN_UPPER_JOINT_EXCITATION_DEG
        safe_candidates = [
            candidate for candidate in candidates if candidate["safe"]
        ]
        positions = np.asarray(holds[-1].part_com_sensor_m)
        origins = np.asarray(holds[-1].joint_origins_sensor_m)
        axes = np.asarray(holds[-1].joint_axes_sensor)
        for candidate in safe_candidates:
            regressor = np.asarray(candidate["mass_regressor"])
            gravity = -regressor[:3, 0]
            candidate["predicted_gravity_torque_nm"] = float(
                np.linalg.norm(regressor[3:] @ mean)
            )
            joint_torques = np.asarray(
                [
                    abs(
                        sum(
                            axes[joint_index]
                            @ np.cross(
                                positions[part_index]
                                - origins[joint_index],
                                -mean[part_index] * gravity,
                            )
                            for part_index in range(
                                joint_index + 1,
                                part_count,
                            )
                        )
                    )
                    for joint_index in range(part_count - 1)
                ]
            )
            candidate["predicted_joint_excitation_torque_nm_by_joint"] = (
                joint_torques.tolist()
            )
            candidate["predicted_joint_excitation_torque_nm"] = float(
                joint_torques.sum()
            )
            candidate["joint_excitation_per_grasp_torque"] = (
                (
                    float(np.min(joint_torques[target_joints]))
                    if np.any(target_joints)
                    else float(joint_torques.sum())
                )
                / max(
                    candidate["predicted_gravity_torque_nm"],
                    contact.AFT200_TORQUE_RESOLUTION_NM,
                )
            )
        maximum_grasp_torque = max(
            (
                candidate["predicted_gravity_torque_nm"]
                for candidate in safe_candidates
            ),
            default=contact.AFT200_TORQUE_RESOLUTION_NM,
        )
        for candidate in safe_candidates:
            local_step = math.hypot(
                candidate["pitch_step_deg"],
                candidate["roll_step_deg"],
            )
            predicted_rotation_drift = (
                rotation_drift
                + max(last_rotation_increase, 1.0)
                * local_step
                / 15.0
            )
            rotation_limit = contact.MAX_FREE_GRASP_ROTATION_DRIFT_DEG
            candidate["predicted_grasp_rotation_drift_deg"] = (
                predicted_rotation_drift
            )
            candidate["pose_difficulty"] = {
                "absolute_wrist": (
                    abs(candidate["wrist_pitch_deg"])
                    + abs(candidate["wrist_roll_deg"])
                )
                / 15.0,
                "transition_from_preferred_15_deg": (
                    0.0
                    if local_step == 0.0
                    else abs(local_step - 15.0) / 15.0
                ),
                "predicted_grasp_torque": (
                    candidate["predicted_gravity_torque_nm"]
                    / max(
                        maximum_grasp_torque,
                        contact.AFT200_TORQUE_RESOLUTION_NM,
                    )
                ),
                "grasp_rotation_limit_proximity": (
                    predicted_rotation_drift
                    / max(rotation_limit - predicted_rotation_drift, 1.0)
                ),
            }
            candidate["pose_difficulty"]["total"] = float(
                sum(candidate["pose_difficulty"].values())
            )
        maximum_excitation_efficiency = max(
            (
                candidate["joint_excitation_per_grasp_torque"]
                for candidate in safe_candidates
                if not (
                    candidate["pitch_step_deg"]
                    and candidate["roll_step_deg"]
                )
            ),
            default=0.0,
        )
        local_candidates = []
        for index, candidate in enumerate(candidates):
            if not candidate["safe"]:
                continue
            local_step = math.hypot(
                candidate["pitch_step_deg"],
                candidate["roll_step_deg"],
            )
            candidate["adaptive_gate"] = None
            if (
                candidate["predicted_grasp_rotation_drift_deg"]
                > contact.MAX_FREE_GRASP_ROTATION_DRIFT_DEG
            ):
                candidate["adaptive_gate"] = "predicted_grasp_rotation_limit"
            elif repeat_for_confirmation and local_step > 0.0:
                candidate["adaptive_gate"] = "confirmation_hold_only"
            elif not repeat_for_confirmation and local_step == 0.0:
                candidate["adaptive_gate"] = "new_pose_required"
            elif (
                not needs_joint_excitation
                and
                candidate["pitch_step_deg"] != 0.0
                and candidate["roll_step_deg"] != 0.0
            ):
                candidate["adaptive_gate"] = "multi_axis_transition"
            elif excitation_axis is not None and local_step > 0.0:
                candidate_axis = (
                    "pitch"
                    if candidate["pitch_step_deg"] != 0.0
                    else "roll"
                )
                candidate_step = (
                    candidate["pitch_step_deg"]
                    if candidate_axis == "pitch"
                    else candidate["roll_step_deg"]
                )
                if np.any(unresponsive_action_counts > 0):
                    if candidate_axis != excitation_axis:
                        candidate["adaptive_gate"] = "hold_excitation_axis"
                    elif candidate_step * sweep_direction <= 0.0:
                        candidate["adaptive_gate"] = "sweep_direction"
                elif (
                    not bias_axis_set
                    and candidate_axis == excitation_axis
                ):
                    candidate["adaptive_gate"] = (
                        "orthogonal_bias_required"
                    )
                elif bias_axis_set and candidate_axis != excitation_axis:
                    candidate["adaptive_gate"] = "hold_excitation_axis"
                elif (
                    bias_axis_set
                    and candidate_step * sweep_direction <= 0.0
                ):
                    candidate["adaptive_gate"] = "sweep_direction"
            elif (
                needs_joint_excitation
                and excitation_axis is None
                and candidate["joint_excitation_per_grasp_torque"]
                < 0.95 * maximum_excitation_efficiency
            ):
                candidate["adaptive_gate"] = (
                    "lower_joint_excitation_per_grasp_torque"
                )
            if candidate["adaptive_gate"] is None:
                local_candidates.append((index, candidate))
        if not local_candidates:
            stop_reason = (
                "insufficient_internal_joint_excitation"
                if needs_joint_excitation
                else "no_safe_geometry_action"
            )
            snapshot["action_selection"] = {
                "ground_truth_usage": "none",
                "selected_candidate_index": None,
                "rejected_candidates": [
                    {
                        "wrist_pitch_deg": candidate["wrist_pitch_deg"],
                        "wrist_roll_deg": candidate["wrist_roll_deg"],
                        "safe_path": candidate["safe"],
                        "predicted_gravity_torque_nm": candidate.get(
                            "predicted_gravity_torque_nm"
                        ),
                        "predicted_joint_excitation_torque_nm": (
                            candidate.get(
                                "predicted_joint_excitation_torque_nm"
                            )
                        ),
                        "movement_cost": candidate.get("movement_cost"),
                        "reason": (
                            candidate.get("adaptive_gate")
                            or candidate["rejection_reason"]
                        ),
                    }
                    for candidate in candidates
                ],
            }
            return None
        selection = select_next_geometry_action(
            regressors,
            [
                np.asarray(candidate["mass_regressor"], dtype=float)
                for _, candidate in local_candidates
            ],
            set(range(len(local_candidates))),
            [
                float(candidate["pose_difficulty"]["total"])
                for _, candidate in local_candidates
            ],
        )
        safe_index = selection["selected"]["index"]
        selected_index, selected = local_candidates[safe_index]
        snapshot["action_selection"] = {
            "ground_truth_usage": "none",
            "selected_candidate_index": selected_index,
            "selected_wrist_pitch_deg": selected["wrist_pitch_deg"],
            "selected_wrist_roll_deg": selected["wrist_roll_deg"],
            "selected_minimum_clearance_m": selected[
                "minimum_clearance_m"
            ],
            "selection_metrics": selection["selected"],
            "predicted_gravity_torque_nm": selected[
                "predicted_gravity_torque_nm"
            ],
            "predicted_joint_excitation_torque_nm": selected[
                "predicted_joint_excitation_torque_nm"
            ],
            "predicted_joint_excitation_torque_nm_by_joint": selected[
                "predicted_joint_excitation_torque_nm_by_joint"
            ],
            "joint_excitation_per_grasp_torque": selected[
                "joint_excitation_per_grasp_torque"
            ],
            "pose_difficulty": selected["pose_difficulty"],
            "rejected_candidates": [
                {
                    "wrist_pitch_deg": candidate["wrist_pitch_deg"],
                    "wrist_roll_deg": candidate["wrist_roll_deg"],
                    "reason": (
                        candidate.get("adaptive_gate")
                        or candidate["rejection_reason"]
                    ),
                }
                for candidate in candidates
                if (
                    not candidate["safe"]
                    or candidate.get("adaptive_gate") is not None
                )
            ],
        }
        current_roll_deg = float(selected["wrist_roll_deg"])
        pending_joint_excitation = np.asarray(
            selected["predicted_joint_excitation_torque_nm_by_joint"]
        )
        if excitation_axis is None and (
            selected["pitch_step_deg"] != 0.0
            or selected["roll_step_deg"] != 0.0
        ):
            selected_step = (
                selected["pitch_step_deg"]
                if selected["pitch_step_deg"] != 0.0
                else selected["roll_step_deg"]
            )
            excitation_axis = (
                "pitch"
                if selected["pitch_step_deg"] != 0.0
                else "roll"
            )
            sweep_direction = -math.copysign(1.0, selected_step)
        elif not bias_axis_set and (
            selected["pitch_step_deg"] != 0.0
            or selected["roll_step_deg"] != 0.0
        ):
            bias_axis_set = True
        return selected_index

    loaded = contact.simulate(
        part_count,
        meshcat=meshcat,
        initial_opening_angle_deg=initial_opening_angle_deg,
        initial_opening_angles_deg=initial_opening_angles_deg,
        opening_angle_deg=opening_angle_deg,
        pgc_controller_kp=profile.get(
            "pgc_controller_kp",
            PGC_CALIBRATED_KP,
        ),
        grasp_offset_m=contact.PARENT_END_GRASP_OFFSET_M,
        part_masses_kg=tuple(true_mass),
        free_hinges=True,
        object_profile=object_profile,
        adaptive_action_selector=select_action,
        adaptive_max_holds=steps,
        adaptive_collision_clearance_m=collision_clearance_m,
    )
    for row in trace:
        row["contact_gate"].update(
            {
                "parent_lift_m": loaded.parent_lift_m,
                "final_contact_count": loaded.final_contact_count,
            }
        )
    loaded_safe = bool(
        loaded.passed_lift
        and loaded.passed_grasp_translation
        and loaded.grasp_relative_rotation_drift_deg
        <= contact.MAX_FREE_GRASP_ROTATION_DRIFT_DEG
    )
    target_met = bool(
        trace[-1]["uncertainty_stop"]["target_met"] and loaded_safe
    )
    if not loaded_safe and stop_reason != "safe_hold_failed":
        stop_reason = "single_grasp_failed_safety_gate"
    selected_actions = [
        action["selected"] for action in loaded.adaptive_actions
    ]
    return {
        "schema": "rb5-static-gravity-mass-pipeline-v6",
        "scope": (
            "Online safe adaptive wrist actions, filtered FoundationPose "
            "joint angles, and known-tool-compensated AFT200 gravity wrenches"
        ),
        "estimator": {
            "measurement_equation": (
                "b_i = w_loaded_i - w_known_tool_i = "
                "A_i(q_i, g_i, r_ij) m"
            ),
            "solution": (
                "argmin ||W(A m-b)||^2 subject to m_j>=0.1 kg; "
                "accept only when 1<=sum(m)<=3 kg"
            ),
            "pose_source": "simulated_foundationpose_angle_sensor",
            "foundationpose_consensus_frames": (
                FOUNDATIONPOSE_CONSENSUS_FRAMES
            ),
            "ground_truth_usage": "evaluator_and_simulated_pose_sensor",
            "systematic_error_model": {
                "scale": robust_error_scale,
                "body_frame_com_offset_bound_m": (
                    ROBUST_COM_OFFSET_BOUND_M * robust_error_scale
                ),
                "foundationpose_joint_bias_bound_deg": (
                    ROBUST_ANGLE_BIAS_BOUND_DEG * robust_error_scale
                ),
                "aft200_force_bias_bound_n": (
                    ROBUST_FORCE_BIAS_BOUND_N * robust_error_scale
                ),
                "aft200_torque_bias_bound_nm": (
                    ROBUST_TORQUE_BIAS_BOUND_NM * robust_error_scale
                ),
                "volume_relative_bound": (
                    ROBUST_VOLUME_RELATIVE_BOUND * robust_error_scale
                ),
                "grasp_slip_included": False,
            },
        },
        "action_policy": {
            "mode": "online_replan_after_every_static_hold",
            "ground_truth_usage": "none",
            "candidate_pitch_roll_step_deg": list(
                contact.ADAPTIVE_WRIST_STEP_DEG
            ),
            "absolute_wrist_limit_deg": (
                contact.ADAPTIVE_WRIST_LIMIT_DEG
            ),
            "collision_clearance_m": collision_clearance_m,
            "selection_order": (
                "safe path, rank, information minus pose difficulty"
            ),
            "experiment_design": (
                "maximize passive-joint excitation per predicted grasp "
                "torque, prefer 15 deg increments and low wrist/grasp "
                "difficulty, then stop after two informative actions when "
                "a target joint does not respond"
            ),
            "minimum_joint_response_deg": MIN_JOINT_RESPONSE_DEG,
            "minimum_informative_joint_torque_nm": (
                MIN_INFORMATIVE_JOINT_TORQUE_NM
            ),
            "maximum_unresponsive_informative_actions": (
                MAX_UNRESPONSIVE_INFORMATIVE_ACTIONS
            ),
            "fixed_household_pose_sequence_used": False,
        },
        "safety_constraints": {
            "required_path_clearance_m": collision_clearance_m,
            "verified_selected_path_clearance_lower_bound_m": min(
                (
                    float(action["minimum_clearance_m"])
                    for action in selected_actions
                ),
                default=collision_clearance_m,
            ),
            "maximum_total_payload_kg": contact.PGC_MAX_PAYLOAD_KG,
            "maximum_grasp_translation_drift_m": (
                contact.MAX_GRASP_TRANSLATION_DRIFT_M
            ),
            "maximum_grasp_rotation_drift_deg": (
                contact.MAX_FREE_GRASP_ROTATION_DRIFT_DEG
            ),
            "accepted_static_holds": sum(
                bool(row["accepted"]) for row in trace
            ),
            "minimum_static_holds": minimum_static_holds,
            "maximum_candidate_holds": steps,
            "regrasp_performed": False,
        },
        "initial_values": {
            "part_count": part_count,
            "object_profile": object_profile,
            "part_volumes_m3": volume.tolist(),
            "joint_opening_limits_deg": list(
                profile["joint_limits_deg"]
            ),
            "initial_opening_angle_deg": initial_opening_angle_deg,
            "initial_opening_angles_deg": list(
                initial_opening_angles_deg
                or (initial_opening_angle_deg,) * (part_count - 1)
            ),
            "prior_mass_kg": prior_mass.tolist(),
            "prior_mass_std_kg": prior_std.tolist(),
            "target_relative_95_half_width": (
                target_relative_95_half_width
            ),
            "continuous_wrist_pitch_sequence_deg": [
                0.0,
                *[
                    float(action["wrist_pitch_deg"])
                    for action in selected_actions
                ],
            ],
            "continuous_wrist_roll_sequence_deg": [
                0.0,
                *[
                    float(action["wrist_roll_deg"])
                    for action in selected_actions
                ],
            ],
            "grasped_part": (
                profile["part_names"][1]
                if object_profile == "jewelry_box"
                else profile["part_names"][0]
            ),
        },
        "trace": trace,
        "final": trace[-1]["estimate"],
        "termination": {
            "reason": stop_reason,
            "next_action": (
                "accept_mass_estimate"
                if target_met
                else "safe_stop_without_regrasp"
            ),
            "evaluated_pose_count": len(trace),
            "controller_stop_after_pose": len(trace),
            "maximum_candidate_pose_count": steps,
            "physical_trajectory": "single_grasp_without_reset",
            **trace[-1]["uncertainty_stop"],
        },
        "validation": {
            "passed": target_met,
            "criterion": (
                "single grasp, safe paths, full rank, condition <= 50, "
                f"<= {100 * target_relative_95_half_width:.1f}% relative "
                "95% mass/density half-width under declared systematic "
                "errors, every internal joint excited by at least "
                f"{MIN_UPPER_JOINT_EXCITATION_DEG:.0f} deg, mass "
                "convergence, and feasible masses"
            ),
        },
        "uncertainty_estimator": {
            "method": (
                "400 AFT200 block/residual refits with run-constant F/T "
                "bias, joint-angle bias, body-frame COM offsets, and "
                "volume-scale perturbations"
            ),
            "source_decomposition": (
                "nominal repeatability, AFT200 bias, joint-angle bias, "
                "body-frame COM offset, volume scale, and combined"
            ),
            "relative_95_half_width_equation": (
                "u_j=(q97.5(m_j)-q2.5(m_j))/(2*m_hat_j)"
            ),
        },
        "evaluator_only_ground_truth": {
            "part_mass_kg": true_mass.tolist(),
            "part_density_kg_m3": (true_mass / volume).tolist(),
            "mass_relative_error": trace[-1]["evaluator_only"][
                "mass_relative_error"
            ],
        },
    }


def run(
    *,
    opening_angle_deg: float,
    initial_opening_angle_deg: float | None = None,
    steps: int,
    seed: int,
    part_count: int = 2,
    vary_internal_angles: bool = False,
    target_relative_95_half_width: float = 0.06,
    meshcat=None,
    foundationpose_result: Path | tuple[Path, ...] | None = None,
    part_masses_kg: tuple[float, ...] | None = None,
    initial_opening_angles_deg: tuple[float, ...] | None = None,
    wrist_pitch_sequence_deg: tuple[float, ...] | None = None,
    wrist_roll_sequence_deg: tuple[float, ...] | None = None,
    include_dynamic_lift: bool = False,
    minimum_static_holds: int = 3,
    reset_between_poses: bool = False,
    object_profile: str | None = None,
    adaptive_actions: bool | None = None,
    collision_clearance_m: float = (
        contact.ADAPTIVE_COLLISION_CLEARANCE_M
    ),
    robust_error_scale: float = 1.0,
) -> dict[str, object]:
    if not 0.0 < opening_angle_deg <= 360.0:
        raise ValueError("opening_angle_deg must be within (0, 360]")
    if (
        initial_opening_angles_deg is None
        and initial_opening_angle_deg is not None
        and not 0.0 <= initial_opening_angle_deg <= opening_angle_deg
    ):
        raise ValueError("initial opening angle must be within joint limits")
    if part_count not in (2, 3):
        raise ValueError("part_count must be 2 or 3")
    profile = None
    if object_profile is not None:
        profile = contact.HOUSEHOLD_PROFILES[object_profile]
        if part_count != profile["part_count"]:
            raise ValueError(
                f"{object_profile} requires {profile['part_count']} parts"
            )
        initial_angles = (
            initial_opening_angles_deg
            if initial_opening_angles_deg is not None
            else profile["default_initial_angles_deg"]
            if initial_opening_angle_deg is None
            else (initial_opening_angle_deg,) * (part_count - 1)
        )
        initial_opening_angles_deg = tuple(initial_angles)
        bounds = profile.get(
            "joint_angle_bounds_deg",
            tuple(
                (0.0, limit)
                for limit in profile["joint_limits_deg"]
            ),
        )
        if any(
            not lower <= angle <= upper
            for angle, (lower, upper) in zip(
                initial_angles,
                bounds,
                strict=True,
            )
        ):
            raise ValueError("initial opening angles exceed profile limits")
    if initial_opening_angle_deg is None:
        initial_opening_angle_deg = 0.0
    use_adaptive_actions = (
        bool(object_profile)
        and wrist_pitch_sequence_deg is None
        and wrist_roll_sequence_deg is None
        and foundationpose_result is None
        and not include_dynamic_lift
        and not reset_between_poses
        if adaptive_actions is None
        else adaptive_actions
    )
    if use_adaptive_actions and object_profile is None:
        raise ValueError("adaptive actions require an object profile")
    if use_adaptive_actions and (
        wrist_pitch_sequence_deg is not None
        or wrist_roll_sequence_deg is not None
        or foundationpose_result is not None
        or include_dynamic_lift
        or reset_between_poses
    ):
        raise ValueError(
            "adaptive actions cannot use fixed poses, offline camera data, "
            "dynamic lift, or regrasp"
        )
    wrist_candidates = (
        WRIST_PITCH_CANDIDATES_DEG
        if part_count == 2
        else (
            WRIST_PITCH_FREE_3LINK_DEG
            if vary_internal_angles
            else WRIST_PITCH_CANDIDATES_3LINK_DEG
        )
    )
    maximum_steps = (
        steps
        if use_adaptive_actions
        else
        len(wrist_pitch_sequence_deg)
        if wrist_pitch_sequence_deg is not None
        else len(wrist_candidates)
    )
    if not 1 <= steps <= maximum_steps:
        raise ValueError("steps exceeds available wrist poses")
    if not 0.0 < target_relative_95_half_width < 1.0:
        raise ValueError("uncertainty target must be within (0, 1)")
    if robust_error_scale < 0.0:
        raise ValueError("robust error scale must be nonnegative")
    mass_convergence_target = target_relative_95_half_width / 2.0
    if not 1 <= minimum_static_holds <= steps or (
        part_count == 3 and minimum_static_holds < 3
    ):
        raise ValueError("minimum static holds must be between 3 and steps")
    if use_adaptive_actions:
        return _run_adaptive(
            opening_angle_deg=opening_angle_deg,
            initial_opening_angle_deg=initial_opening_angle_deg,
            initial_opening_angles_deg=initial_opening_angles_deg,
            part_count=part_count,
            steps=steps,
            seed=seed,
            target_relative_95_half_width=(
                target_relative_95_half_width
            ),
            minimum_static_holds=minimum_static_holds,
            meshcat=meshcat,
            part_masses_kg=part_masses_kg,
            object_profile=object_profile,
            collision_clearance_m=collision_clearance_m,
            robust_error_scale=robust_error_scale,
        )

    volume = (
        np.asarray(
            profile.get(
                "volumes_m3",
                np.prod(np.asarray(profile["sizes_m"]), axis=1),
            )
        )
        if object_profile is not None
        else study.nominal_envelope_volumes(part_count)
    )
    true_mass = np.asarray(
        part_masses_kg
        or (
            profile["default_masses_kg"]
            if object_profile is not None
            else (0.8, 0.4, 0.25)[:part_count]
        ),
        dtype=float,
    )
    if true_mass.shape != (part_count,):
        raise ValueError("one mass is required per part")
    prior_mass = volume * study.VLM_PRIOR_MEAN_KG_M3[:part_count]
    prior_std = volume * study.VLM_PRIOR_STD_KG_M3[:part_count]
    rng = np.random.default_rng(seed)
    wrist_sequence = wrist_pitch_sequence_deg or wrist_candidates[:steps]
    if len(wrist_sequence) != steps:
        raise ValueError("steps must match the wrist pitch sequence length")
    wrist_roll_sequence = wrist_roll_sequence_deg or (
        WRIST_ROLL_FREE_3LINK_DEG[:steps]
        if (
            part_count == 3
            and vary_internal_angles
            and wrist_pitch_sequence_deg is None
        )
        else (0.0,) * steps
    )
    if len(wrist_roll_sequence) != steps:
        raise ValueError("wrist pitch and roll sequences must have equal length")
    free_hinges = True
    initial_wrist_pitch_deg = 0.0
    selected_pgc_kp = (
        profile.get("pgc_controller_kp", PGC_CALIBRATED_KP)
        if profile is not None
        else PGC_CALIBRATED_KP
    )

    def simulate_sessions(*, grasp_object: bool, kp: float):
        common = dict(
            initial_opening_angle_deg=initial_opening_angle_deg,
            initial_opening_angles_deg=initial_opening_angles_deg,
            initial_wrist_pitch_deg=initial_wrist_pitch_deg,
            opening_angle_deg=opening_angle_deg,
            pgc_controller_kp=kp,
            grasp_offset_m=contact.PARENT_END_GRASP_OFFSET_M,
            part_masses_kg=tuple(true_mass),
            free_hinges=free_hinges,
            object_profile=object_profile,
            grasp_object=grasp_object,
        )
        if not reset_between_poses:
            result = contact.simulate(
                part_count,
                meshcat=meshcat,
                wrist_pitch_sequence_deg=wrist_sequence,
                wrist_roll_sequence_deg=wrist_roll_sequence,
                **common,
            )
            return result, None
        sessions = [
            contact.simulate(
                part_count,
                meshcat=meshcat if index == 0 else None,
                wrist_pitch_sequence_deg=(pitch,),
                wrist_roll_sequence_deg=(roll,),
                **common,
            )
            for index, (pitch, roll) in enumerate(
                zip(wrist_sequence, wrist_roll_sequence, strict=True)
            )
        ]
        return (
            replace(
                sessions[-1],
                simulation_end_s=sum(
                    item.simulation_end_s for item in sessions
                ),
                parent_lift_m=min(item.parent_lift_m for item in sessions),
                dynamic_lift_measurements=tuple(
                    sample
                    for item in sessions
                    for sample in item.dynamic_lift_measurements
                ),
                holds=tuple(item.holds[0] for item in sessions),
                passed_lift=all(item.passed_lift for item in sessions),
                passed_grasp_translation=all(
                    item.passed_grasp_translation for item in sessions
                ),
                passed_grasp_rotation=all(
                    item.passed_grasp_rotation for item in sessions
                ),
                passed_joint_stability=all(
                    item.passed_joint_stability for item in sessions
                ),
            ),
            tuple(
                (
                    item.grasp_relative_translation_drift_m,
                    item.grasp_relative_rotation_drift_deg,
                )
                for item in sessions
            ),
        )

    loaded, loaded_pose_drifts = simulate_sessions(
        grasp_object=True,
        kp=selected_pgc_kp,
    )
    if not (
        loaded.passed_lift
        and loaded.passed_grasp_translation
        and loaded.grasp_relative_rotation_drift_deg
        <= contact.MAX_FREE_GRASP_ROTATION_DRIFT_DEG
    ):
        raise RuntimeError("single grasp failed safety gate")
    regrasp_performed = False
    empty, _ = simulate_sessions(
        grasp_object=False,
        kp=selected_pgc_kp,
    )
    trace = []
    accepted_regressors = []
    accepted_regressor_samples = []
    accepted_wrenches = []
    accepted_blocks = []
    accepted_angle_history = []
    accepted_static_hold_count = 0
    previous_mean = None
    termination_reason = "maximum_pose_count_reached"
    foundationpose_paths = (
        ()
        if foundationpose_result is None
        else (foundationpose_result,)
        if isinstance(foundationpose_result, Path)
        else foundationpose_result
    )
    tracked_result = (
        _load_foundationpose_results(foundationpose_paths)
        if foundationpose_paths
        else None
    )
    if tracked_result is not None and len(foundationpose_paths) == 1:
        metadata_path = foundationpose_paths[0].parent / "metadata.json"
        if metadata_path.exists():
            metadata_frames = {
                frame["index"]: frame
                for frame in json.loads(metadata_path.read_text())["frames"]
            }
            for frame in tracked_result.get("frames", []):
                metadata_frame = metadata_frames.get(frame["index"])
                if (
                    metadata_frame is not None
                    and "encoder_parent_world_pose" in metadata_frame
                ):
                    frame["encoder_parent_world_pose"] = metadata_frame[
                        "encoder_parent_world_pose"
                    ]
    tracked_summary = (
        tracked_result["summary"] if tracked_result is not None else None
    )
    camera_com_offsets = (
        _geometry_centroids_body_m(object_profile)
        if tracked_result is not None and object_profile is not None
        else None
    )
    camera_dynamic_regressors = (
        _foundationpose_dynamic_force_regressors(
            tracked_result.get("frames", []),
            loaded.dynamic_lift_measurements,
            loaded.part_com_offsets_body_m,
        )
        if include_dynamic_lift and tracked_result is not None
        else {}
    )
    impulse_boundaries_s = np.linspace(
        contact.ORIENT_START_S,
        contact.ORIENT_END_S,
        5,
    )
    camera_impulse_regressors = (
        _foundationpose_impulse_regressors(
            tracked_result.get("frames", []),
            loaded.part_com_offsets_body_m,
            impulse_boundaries_s,
        )
        if include_dynamic_lift and tracked_result is not None
        else []
    )
    force_norms = np.asarray(
        [
            np.linalg.norm(
                np.asarray(loaded_hold.ft_reaction_force)
                - np.asarray(loaded_hold.known_tool_wrench[:3])
            )
            for loaded_hold in loaded.holds
        ]
    )
    median_force_norm = float(np.median(force_norms))
    wrench_weight = np.array(
        [1.0 / contact.AFT200_FORCE_RESOLUTION_N] * 3
        + [1.0 / contact.AFT200_TORQUE_RESOLUTION_NM] * 3
    )
    max_grasp_rotation_drift = (
        contact.MAX_FREE_GRASP_ROTATION_DRIFT_DEG
        if free_hinges
        else contact.MAX_GRASP_ROTATION_DRIFT_DEG
    )
    dynamic_lift_sample_count = 0
    if include_dynamic_lift and camera_impulse_regressors:
        dynamic_times = np.asarray(
            [
                measurement.time_s
                for measurement in loaded.dynamic_lift_measurements
            ]
        )
        dynamic_force_world = np.asarray(
            [
                np.asarray(
                    measurement.sensor_rotation_world_to_sensor
                ).T
                @ (
                    np.asarray(measurement.ft_reaction_force)
                    - np.asarray(measurement.known_tool_wrench[:3])
                )
                for measurement in loaded.dynamic_lift_measurements
            ]
        )
        for index, regressor in enumerate(camera_impulse_regressors):
            start = impulse_boundaries_s[index]
            end = impulse_boundaries_s[index + 1]
            keep = (dynamic_times >= start) & (dynamic_times <= end)
            duration = end - start
            average_force = np.trapezoid(
                dynamic_force_world[keep],
                dynamic_times[keep],
                axis=0,
            ) / duration
            wrench = np.zeros(6)
            wrench[:3] = average_force
            accepted_regressors.append(regressor)
            accepted_wrenches.append(wrench)
            accepted_blocks.append(np.repeat(wrench[None, :], 10, axis=0))
            dynamic_lift_sample_count += 1
    for loaded_lift, empty_lift in zip(
        loaded.dynamic_lift_measurements,
        empty.dynamic_lift_measurements,
        strict=True,
    ):
        if (
            include_dynamic_lift
            and not camera_impulse_regressors
            and
            loaded_lift.contact_free
            and loaded_lift.time_s >= contact.LIFT_START_S + 3.5
        ):
            measured_regressor = camera_dynamic_regressors.get(
                float(loaded_lift.time_s)
            )
            regressor = DYNAMIC_LIFT_WEIGHT * (
                measured_regressor
                if measured_regressor is not None
                else np.asarray(loaded_lift.mass_regressor)
            )
            wrench = DYNAMIC_LIFT_WEIGHT * (
                np.concatenate(
                    (
                        loaded_lift.ft_reaction_force,
                        loaded_lift.ft_reaction_torque,
                    )
                )
                - np.asarray(loaded_lift.known_tool_wrench)
                )
            accepted_regressors.append(regressor)
            accepted_wrenches.append(wrench)
            accepted_blocks.append(np.repeat(wrench[None, :], 10, axis=0))
            dynamic_lift_sample_count += 1

    for step, (loaded_hold, empty_hold) in enumerate(
        zip(loaded.holds, empty.holds, strict=True),
        start=1,
    ):
        tracked_hold = None
        if tracked_summary is None:
            angle_bounds = (
                profile.get(
                    "joint_angle_bounds_deg",
                    tuple(
                        (0.0, limit)
                        for limit in profile["joint_limits_deg"]
                    ),
                )
                if object_profile is not None
                else ((0.0, opening_angle_deg),) * (part_count - 1)
            )
            joint_axes = (
                profile.get("joint_axes", (None,) * (part_count - 1))
                if object_profile is not None
                else (None,) * (part_count - 1)
            )
            camera_joints = [
                foundationpose_joint_angle(
                    angle,
                    rng,
                    joint_min_deg=lower,
                    joint_max_deg=upper,
                    joint_axis=axis,
                )
                for angle, (lower, upper), axis in zip(
                    loaded_hold.actual_opening_angles_deg,
                    angle_bounds,
                    joint_axes,
                    strict=True,
                )
            ]
            fused_angles = [
                item["fused_opening_deg"] for item in camera_joints
            ]
            per_camera = camera_joints[0]["per_camera"]
            camera_source = "filtered_foundationpose_simulation_with_gt_first_mask"
            camera_stationary = True
        else:
            static_holds = tracked_summary.get("static_holds", [])
            tracked_hold = (
                static_holds[step - 1]
                if step <= len(static_holds)
                else None
            )
            fused_angles = (
                tracked_hold["filtered_opening_angles_deg"]
                if tracked_hold is not None
                else tracked_summary.get(
                    "filtered_static_opening_angles_deg",
                    [tracked_summary["filtered_static_opening_deg"]],
                )
            )
            per_camera = {}
            camera_source = "official_foundationpose_track_one"
            camera_stationary = (
                tracked_hold["stationary"]
                if tracked_hold is not None
                else tracked_summary["stationary"]
            )
            camera_visible = (
                tracked_hold.get("all_links_visible", True)
                if tracked_hold is not None
                else True
            )
        cameras = {
            "source": camera_source,
            "per_camera": per_camera,
            "fused_opening_deg": fused_angles[0],
            "fused_opening_angles_deg": fused_angles,
            "stationary": camera_stationary,
            "all_links_visible": (
                camera_visible if tracked_summary is not None else True
            ),
        }
        regressor = (
            _foundationpose_static_mass_regressor(
                loaded_hold,
                tracked_result,
                tracked_hold,
                camera_com_offsets,
            )
            if tracked_hold is not None
            and camera_com_offsets is not None
            else _hold_mass_regressor(loaded_hold, fused_angles)
        )
        regressor_samples = (
            tuple(
                _foundationpose_static_mass_regressor(
                    loaded_hold,
                    tracked_result,
                    {**tracked_hold, "frame_indices": [frame_index]},
                    camera_com_offsets,
                )
                for frame_index in tracked_hold["frame_indices"]
            )
            if tracked_hold is not None
            and camera_com_offsets is not None
            else (regressor,)
        )
        loaded_wrench = np.concatenate(
            (loaded_hold.ft_reaction_force, loaded_hold.ft_reaction_torque)
        )
        empty_wrench = np.asarray(loaded_hold.known_tool_wrench)
        wrench = loaded_wrench - empty_wrench
        blocks = (
            np.asarray(loaded_hold.ft_block_means)
            - empty_wrench
        )
        if loaded_pose_drifts is not None:
            (
                grasp_translation_drift,
                grasp_rotation_drift,
            ) = loaded_pose_drifts[step - 1]
        else:
            grasp_delta = (
                np.linalg.inv(
                    np.asarray(loaded.holds[0].grasp_relative_pose)
                )
                @ np.asarray(loaded_hold.grasp_relative_pose)
            )
            grasp_translation_drift = float(
                np.linalg.norm(grasp_delta[:3, 3])
            )
            grasp_rotation_drift = math.degrees(
                math.acos(
                    np.clip(
                        (np.trace(grasp_delta[:3, :3]) - 1.0) / 2.0,
                        -1.0,
                        1.0,
                    )
                )
            )
        innovation_rms = None
        candidate_mass_change = None
        measurement_consistent = True
        if previous_mean is not None and len(accepted_regressors) >= 3:
            previous_design = np.vstack(
                [
                    accepted_regressor * wrench_weight[:, None]
                    for accepted_regressor in accepted_regressors
                ]
            )
            if (
                np.linalg.matrix_rank(previous_design) == part_count
                and np.linalg.cond(previous_design) < 50.0
            ):
                innovation_rms = float(
                    np.linalg.norm(
                        (regressor @ previous_mean - wrench)
                        * wrench_weight
                    )
                    / math.sqrt(6.0)
                )
                measurement_consistent = (
                    innovation_rms <= MAX_NORMALIZED_INNOVATION_RMS
                )
                candidate_design = np.vstack(
                    (previous_design, regressor * wrench_weight[:, None])
                )
                candidate_observed = np.concatenate(
                    (
                        np.concatenate(
                            [
                                accepted_wrench * wrench_weight
                                for accepted_wrench in accepted_wrenches
                            ]
                        ),
                        wrench * wrench_weight,
                    )
                )
                candidate_mean = _bounded_mass_fit(
                    candidate_design,
                    candidate_observed,
                )
                candidate_mass_change = float(
                    np.max(
                        np.abs(candidate_mean - previous_mean)
                        / previous_mean
                    )
                )
        accepted = bool(
            loaded.passed_lift
            and loaded.final_contact_count >= 2
            and grasp_translation_drift
            <= contact.MAX_GRASP_TRANSLATION_DRIFT_M
            and grasp_rotation_drift
            <= max_grasp_rotation_drift
            and loaded_hold.ft_stationary
            and empty_hold.ft_stationary
            and loaded_hold.contact_free
            and empty_hold.contact_free
            and loaded_hold.joints_stationary
            and cameras["stationary"]
            and cameras["all_links_visible"]
            and abs(force_norms[step - 1] - median_force_norm)
            <= 0.2 * median_force_norm
        )
        if accepted:
            accepted_regressors.append(regressor)
            accepted_regressor_samples.append(regressor_samples)
            accepted_wrenches.append(wrench)
            accepted_blocks.append(blocks)
            accepted_angle_history.append(fused_angles)
            accepted_static_hold_count += 1
        if accepted_regressors:
            design = np.vstack(
                [
                    regressor * wrench_weight[:, None]
                    for regressor in accepted_regressors
                ]
            )
            observed = np.concatenate(
                [wrench * wrench_weight for wrench in accepted_wrenches]
            )
            mean = _bounded_mass_fit(design, observed)
            residual_pool = np.asarray(
                [
                    wrench - regressor @ mean
                    for regressor, wrench in zip(
                        accepted_regressors,
                        accepted_wrenches,
                        strict=True,
                    )
                ]
            )
            residual_pool -= residual_pool.mean(axis=0)
            bootstrap = []
            for _ in range(400):
                sampled_design = np.vstack(
                    [
                        samples[rng.integers(0, len(samples))]
                        * wrench_weight[:, None]
                        for samples in accepted_regressor_samples
                    ]
                )
                sampled = [
                    block[
                        rng.integers(0, len(block), len(block))
                    ].mean(axis=0)
                    + residual_pool[
                        rng.integers(0, len(residual_pool))
                    ]
                    for block in accepted_blocks
                ]
                bootstrap.append(
                    _bounded_mass_fit(
                        sampled_design,
                        np.concatenate(
                            [
                                wrench * wrench_weight
                                for wrench in sampled
                            ]
                        ),
                    )
                )
            lower, upper = np.percentile(
                np.asarray(bootstrap),
                (2.5, 97.5),
                axis=0,
            )
        else:
            design = np.empty((0, part_count))
            mean = prior_mass.copy()
            lower = mean - 1.96 * prior_std
            upper = mean + 1.96 * prior_std
        relative_95_half_width = (upper - lower) / (2.0 * mean)
        data_rank = (
            int(np.linalg.matrix_rank(design))
            if accepted_regressors
            else 0
        )
        condition_number = (
            float(np.linalg.cond(design)) if data_rank == part_count else None
        )
        convergence = (
            None
            if previous_mean is None
            else float(np.max(np.abs(mean - previous_mean) / mean))
        )
        joint_excitation_range = (
            np.ptp(np.asarray(accepted_angle_history), axis=0)
            if accepted_angle_history
            else np.zeros(part_count - 1)
        )
        required_joint_excitation = (
            joint_excitation_range
            if part_count == 3
            else np.array(())
        )
        upper_joint_excited = bool(
            np.all(
                required_joint_excitation
                >= MIN_UPPER_JOINT_EXCITATION_DEG
            )
        )
        mass_boundary_active = bool(
            np.any(mean <= contact.MINIMUM_PART_MASS_KG + 1e-6)
        )
        uncertainty_target_met = (
            accepted
            and data_rank == part_count
            and condition_number is not None
            and condition_number <= 50.0
            and float(np.max(relative_95_half_width))
            <= target_relative_95_half_width
            and convergence is not None
            and convergence <= mass_convergence_target
            and accepted_static_hold_count >= minimum_static_holds
            and upper_joint_excited
            and not mass_boundary_active
        )
        trace.append(
            {
                "step": step,
                "selected_wrist_pitch_deg": loaded_hold.wrist_pitch_deg,
                "selected_wrist_roll_deg": wrist_roll_sequence[step - 1],
                "three_camera_foundationpose_filter": cameras,
                "aft200_filtered_loaded_wrench": loaded_wrench.tolist(),
                "aft200_empty_tool_tare_wrench": empty_wrench.tolist(),
                "aft200_contact_object_wrench": wrench.tolist(),
                "aft200_filter": {
                    "samples_per_hold": contact.FT_HOLD_SAMPLE_COUNT,
                    "loaded_inliers": loaded_hold.ft_inlier_count,
                    "loaded_stationary": loaded_hold.ft_stationary,
                    "empty_inliers": empty_hold.ft_inlier_count,
                    "empty_stationary": empty_hold.ft_stationary,
                    "aft200_quantization": {
                        "force_n": contact.AFT200_FORCE_RESOLUTION_N,
                        "torque_nm": contact.AFT200_TORQUE_RESOLUTION_NM,
                    },
                },
                "accepted": accepted,
                "contact_gate": {
                    "parent_lift_m": loaded.parent_lift_m,
                    "grasp_translation_drift_m": grasp_translation_drift,
                    "grasp_rotation_drift_deg": grasp_rotation_drift,
                    "grasp_stable": bool(
                        grasp_translation_drift
                        <= contact.MAX_GRASP_TRANSLATION_DRIFT_M
                        and grasp_rotation_drift
                        <= max_grasp_rotation_drift
                    ),
                    "joint_drift_deg": (
                        loaded.maximum_internal_joint_drift_deg
                    ),
                    "opening_angle_span_deg": list(
                        loaded_hold.opening_angle_span_deg
                    ),
                    "maximum_abs_joint_velocity_deg_s": list(
                        loaded_hold.maximum_abs_joint_velocity_deg_s
                    ),
                    "maximum_abs_joint_acceleration_deg_s2": list(
                        loaded_hold.maximum_abs_joint_acceleration_deg_s2
                    ),
                    "joints_stationary": loaded_hold.joints_stationary,
                    "final_contact_count": loaded.final_contact_count,
                    "object_force_norm_n": force_norms[step - 1],
                    "trajectory_median_force_norm_n": median_force_norm,
                    "force_norm_consistent": bool(
                        abs(force_norms[step - 1] - median_force_norm)
                        <= 0.2 * median_force_norm
                    ),
                    "normalized_innovation_rms": innovation_rms,
                    "candidate_mass_change": candidate_mass_change,
                    "measurement_consistent": measurement_consistent,
                },
                "estimate": {
                    "part_mass_kg": mean.tolist(),
                    "part_mass_95_interval_kg": np.column_stack(
                        (lower, upper)
                    ).tolist(),
                    "part_density_kg_m3": (mean / volume).tolist(),
                },
                "uncertainty_stop": {
                    "data_rank": data_rank,
                    "design_condition_number": condition_number,
                    "mass_change_from_previous": convergence,
                    "relative_95_half_width": (
                        relative_95_half_width.tolist()
                    ),
                    "maximum_relative_95_half_width": float(
                        np.max(relative_95_half_width)
                    ),
                    "target_relative_95_half_width": (
                        target_relative_95_half_width
                    ),
                    "target_met": uncertainty_target_met,
                    "joint_excitation_range_deg": (
                        joint_excitation_range.tolist()
                    ),
                    "upper_joint_excitation_target_deg": (
                        MIN_UPPER_JOINT_EXCITATION_DEG
                        if part_count == 3
                        else None
                    ),
                    "upper_joint_excitation_target_met": upper_joint_excited,
                    "minimum_mass_boundary_active": mass_boundary_active,
                },
                "evaluator_only": {
                    "true_part_mass_kg": true_mass.tolist(),
                    "mass_relative_error": (
                        np.abs(mean - true_mass) / true_mass
                    ).tolist(),
                    "ideal_support_wrench": (
                        regressor @ true_mass
                    ).tolist(),
                    "static_wrench_residual": (
                        wrench - regressor @ true_mass
                    ).tolist(),
                },
            }
        )
        if uncertainty_target_met:
            termination_reason = "repeatability_interval_and_mass_converged"
            break
        previous_mean = mean.copy()

    if part_count == 3 and not upper_joint_excited:
        termination_reason = "insufficient_internal_joint_excitation"

    return {
        "schema": "rb5-static-gravity-mass-pipeline-v4",
        "scope": (
            "Adaptive static wrist poses, filtered FoundationPose joint angles, "
            "and AFT200 empty-tool-tared gravity wrenches"
        ),
        "estimator": {
            "measurement_equation": (
                "b_i = w_loaded_i - w_empty_i = A_i(q_i, g_i, r_ij) m"
            ),
            "solution": (
                "m_hat = max(0, argmin_m sum_i "
                "||W(A_i m - b_i)||_2^2)"
            ),
            "pose_source": (
                "three_camera_foundationpose_fused_6d_link_poses"
                if tracked_result is not None
                else "simulated_foundationpose_angle_sensor"
            ),
            "ground_truth_usage": (
                "evaluator_only"
                if tracked_result is not None
                else "evaluator_and_simulated_pose_sensor"
            ),
        },
        "safety_constraints": {
            "maximum_total_payload_kg": contact.PGC_MAX_PAYLOAD_KG,
            "maximum_grasp_translation_drift_m": (
                contact.MAX_GRASP_TRANSLATION_DRIFT_M
            ),
            "maximum_grasp_rotation_drift_deg": (
                max_grasp_rotation_drift
            ),
            "maximum_normalized_innovation_rms": (
                MAX_NORMALIZED_INNOVATION_RMS
            ),
            "minimum_upper_joint_excitation_deg": (
                MIN_UPPER_JOINT_EXCITATION_DEG
                if part_count == 3
                else None
            ),
            "dynamic_lift_weight": DYNAMIC_LIFT_WEIGHT,
            "accepted_dynamic_lift_samples": dynamic_lift_sample_count,
            "accepted_static_holds": accepted_static_hold_count,
            "minimum_static_holds": minimum_static_holds,
            "maximum_candidate_holds": steps,
            "regrasp_performed": regrasp_performed,
            "dynamic_tool_tare": (
                "loaded_state_cad_newton_euler"
                if include_dynamic_lift
                else "not_used"
            ),
            "dynamic_pose_source": (
                "disabled_static_gravity_only"
                if not include_dynamic_lift
                else
                "official_foundationpose_impulse_momentum"
                if camera_impulse_regressors
                else "official_foundationpose_finite_difference"
                if camera_dynamic_regressors
                else "simulated_pose_sensor"
            ),
        },
        "initial_values": {
            "part_count": part_count,
            "object_profile": object_profile or "generic",
            "part_volumes_m3": volume.tolist(),
            "opening_limit_deg": opening_angle_deg,
            "joint_opening_limits_deg": list(
                profile["joint_limits_deg"]
                if object_profile is not None
                else [opening_angle_deg] * (part_count - 1)
            ),
            "initial_opening_angle_deg": initial_opening_angle_deg,
            "initial_opening_angles_deg": (
                list(initial_opening_angles_deg)
                if initial_opening_angles_deg is not None
                else [initial_opening_angle_deg] * (part_count - 1)
            ),
            "prior_mass_kg": prior_mass.tolist(),
            "prior_mass_std_kg": prior_std.tolist(),
            "foundationpose_rotation_std_deg": (
                FOUNDATIONPOSE_ROTATION_STD_DEG
            ),
            "pgc_controller_kp": selected_pgc_kp,
            "aft200_hold_samples": contact.FT_HOLD_SAMPLE_COUNT,
            "target_relative_95_half_width": (
                target_relative_95_half_width
            ),
            "mass_convergence_target": mass_convergence_target,
            "continuous_wrist_pitch_sequence_deg": list(wrist_sequence),
            "continuous_wrist_roll_sequence_deg": list(
                wrist_roll_sequence
            ),
            "vary_internal_angles": vary_internal_angles,
            "grasp_width_m": (
                float(profile["sizes_m"][0][1])
                if object_profile is not None
                else study.BOX_DEPTH_M
            ),
            "grasped_part": (
                (
                    profile["part_names"][1]
                    if object_profile == "jewelry_box"
                    else profile["part_names"][0]
                )
                if object_profile is not None
                else "parent"
            ),
        },
        "trace": trace,
        "final": trace[-1]["estimate"],
        "termination": {
            "reason": termination_reason,
            "next_action": (
                "accept_mass_estimate"
                if trace[-1]["uncertainty_stop"]["target_met"]
                else "add_safe_pose_or_terminate"
            ),
            "evaluated_pose_count": len(trace),
            "controller_stop_after_pose": len(trace),
            "maximum_candidate_pose_count": steps,
            "physical_trajectory": (
                "regrasp_between_static_poses"
                if reset_between_poses
                else "single_grasp_without_reset"
            ),
            **trace[-1]["uncertainty_stop"],
        },
        "validation": {
            "passed": bool(trace[-1]["uncertainty_stop"]["target_met"]),
            "criterion": (
                "all required static holds accepted, full rank, condition "
                f"<= 50, <= {100 * target_relative_95_half_width:.1f}% "
                "relative 95% half-width, and estimate change <= half "
                "the uncertainty target; required articulated-joint excitation "
                f">= {MIN_UPPER_JOINT_EXCITATION_DEG:.0f} deg"
            ),
        },
        "uncertainty_estimator": {
            "method": (
                "400 refits with quantized AFT200 block resampling, "
                "FoundationPose frame resampling, and centered hold-residual "
                "resampling"
            ),
            "relative_95_half_width_equation": (
                "u_j = (q97.5(m_j) - q2.5(m_j)) / (2 * m_hat_j)"
            ),
            "stop_equation": (
                "rank(A)=part_count and cond(A)<=50 and max_j(u_j)<=target "
                "and max_j(|m_t-m_prev|/m_t)<=target/2; desk lamp also "
                "requires upper-joint angle range >= 10 deg"
            ),
        },
        "evaluator_only_ground_truth": {
            "part_mass_kg": true_mass.tolist(),
            "part_density_kg_m3": (true_mass / volume).tolist(),
            "mass_relative_error": trace[-1]["evaluator_only"][
                "mass_relative_error"
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--opening-angle-deg", type=float)
    parser.add_argument("--initial-opening-angle-deg", type=float)
    parser.add_argument("--part-count", type=int, choices=(2, 3))
    parser.add_argument(
        "--object-profile",
        choices=tuple(contact.HOUSEHOLD_PROFILES),
    )
    parser.add_argument("--vary-internal-angles", action="store_true")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--part-masses-kg", type=float, nargs="+")
    parser.add_argument("--initial-opening-angles-deg", type=float, nargs="+")
    parser.add_argument("--wrist-pitch-sequence-deg", type=float, nargs="+")
    parser.add_argument("--wrist-roll-sequence-deg", type=float, nargs="+")
    parser.add_argument("--auto-plan", action="store_true")
    parser.add_argument("--planned-steps", type=int, default=8)
    parser.add_argument("--candidate-wrist-pitches-deg", type=float, nargs="+")
    parser.add_argument("--candidate-wrist-rolls-deg", type=float, nargs="+")
    parser.add_argument("--geometry-action-input", type=Path)
    parser.add_argument("--camera-calibration-result", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--target-relative-95-half-width",
        type=float,
        default=0.06,
    )
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--foundationpose-result", type=Path, nargs="+")
    parser.add_argument("--include-dynamic-lift", action="store_true")
    parser.add_argument("--reset-between-poses", action="store_true")
    parser.add_argument("--minimum-static-holds", type=int, default=3)
    parser.add_argument("--fixed-actions", action="store_true")
    parser.add_argument(
        "--collision-clearance-m",
        type=float,
        default=contact.ADAPTIVE_COLLISION_CLEARANCE_M,
    )
    parser.add_argument("--robust-error-scale", type=float, default=1.0)
    args = parser.parse_args()
    profile = (
        contact.HOUSEHOLD_PROFILES[args.object_profile]
        if args.object_profile
        else None
    )
    if args.part_count is None:
        args.part_count = profile["part_count"] if profile else 2
    if args.opening_angle_deg is None:
        args.opening_angle_deg = (
            max(profile["joint_limits_deg"]) if profile else 180.0
        )
    if (
        profile
        and args.initial_opening_angles_deg is None
        and args.initial_opening_angle_deg is None
    ):
        args.initial_opening_angles_deg = list(
            profile["default_initial_angles_deg"]
        )
    if args.initial_opening_angle_deg is None:
        args.initial_opening_angle_deg = 0.0
    if args.steps is None:
        args.steps = 12 if args.object_profile else 3
    meshcat = StartMeshcat() if args.live else None
    if meshcat is not None:
        print(f"Meshcat: {meshcat.web_url()}", flush=True)
    planner_candidates = {}
    if args.candidate_wrist_pitches_deg:
        planner_candidates["candidate_pitches_deg"] = tuple(
            args.candidate_wrist_pitches_deg
        )
    if args.candidate_wrist_rolls_deg:
        planner_candidates["candidate_rolls_deg"] = tuple(
            args.candidate_wrist_rolls_deg
        )
    action_plan = (
        (
            plan_actions_from_geometry(
                args.geometry_action_input,
                args.planned_steps,
            )
            if args.geometry_action_input is not None
            else plan_wrist_actions(
                **planner_candidates,
                planned_steps=args.planned_steps,
                camera_calibration_result=args.camera_calibration_result,
                seed=args.seed,
            )
        )
        if args.auto_plan
        else None
    )
    if action_plan is not None:
        args.wrist_pitch_sequence_deg = action_plan["selected"][
            "wrist_pitch_sequence_deg"
        ]
        args.wrist_roll_sequence_deg = action_plan["selected"][
            "wrist_roll_sequence_deg"
        ]
        args.steps = len(args.wrist_pitch_sequence_deg)
        print(
            "planned wrist pitch/roll: "
            f"{list(zip(args.wrist_pitch_sequence_deg, args.wrist_roll_sequence_deg))}",
            flush=True,
        )
    result = run(
        opening_angle_deg=args.opening_angle_deg,
        initial_opening_angle_deg=args.initial_opening_angle_deg,
        part_count=args.part_count,
        vary_internal_angles=args.vary_internal_angles,
        steps=args.steps,
        seed=args.seed,
        target_relative_95_half_width=(
            args.target_relative_95_half_width
        ),
        meshcat=meshcat,
        foundationpose_result=(
            tuple(args.foundationpose_result)
            if args.foundationpose_result
            else None
        ),
        part_masses_kg=(
            tuple(args.part_masses_kg) if args.part_masses_kg else None
        ),
        initial_opening_angles_deg=(
            tuple(args.initial_opening_angles_deg)
            if args.initial_opening_angles_deg
            else None
        ),
        wrist_pitch_sequence_deg=(
            tuple(args.wrist_pitch_sequence_deg)
            if args.wrist_pitch_sequence_deg
            else None
        ),
        wrist_roll_sequence_deg=(
            tuple(args.wrist_roll_sequence_deg)
            if args.wrist_roll_sequence_deg
            else None
        ),
        include_dynamic_lift=args.include_dynamic_lift,
        minimum_static_holds=args.minimum_static_holds,
        reset_between_poses=args.reset_between_poses,
        object_profile=args.object_profile,
        adaptive_actions=False if args.fixed_actions else None,
        collision_clearance_m=args.collision_clearance_m,
        robust_error_scale=args.robust_error_scale,
    )
    if action_plan is not None:
        result["action_plan"] = action_plan
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    for row in result["trace"]:
        print(
            f"step {row['step']}: "
            f"wrist={row['selected_wrist_pitch_deg']:.0f}°, "
            "q="
            f"{np.round(row['three_camera_foundationpose_filter']['fused_opening_angles_deg'], 2)}, "
            f"accepted={row['accepted']}, "
            f"mass={np.round(row['estimate']['part_mass_kg'], 4)}, "
            f"error={np.round(row['evaluator_only']['mass_relative_error'], 4)}"
        )
    if meshcat is not None:
        stop = "종료"
        meshcat.AddButton(stop)
        while meshcat.GetButtonClicks(stop) < 1:
            time.sleep(0.2)


if __name__ == "__main__":
    main()
