#!/usr/bin/env python3
"""Realistic hidden-GT configuration-information study for custom CAD boxes.

The 2/3/4-link family is derived from ``custom_object_cad``:

* link 1 is the 150 x 100 x 80 mm Parent box;
* links 2..4 are 120 x 100 x 80 mm modular Child/Middle boxes;
* adjacent boxes use two torque hinges at the top edge with an 8 mm gap.

Only the evaluator sees manufacturing deviations, insert masses, true grasp
offsets, true joint angles, hinge capacity, and F/T bias.  The robot sees
estimated exterior-envelope volumes, a VLM prior, camera angles, tare, and
six-axis wrist F/T observations.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import itertools
import json
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

import run_hidden_gt_quasistatic_experiment as core  # noqa: E402


DEFAULT_OUTPUT = (
    ROOT
    / "progress/artifacts/2026-07-28/"
    "custom_cad_configuration_information"
)
PARENT_LENGTH_M = 0.160
CHILD_LENGTH_M = 0.100
# Purpose-built articulated plate: the stock 50 mm PGC stroke closes
# directly on the 40 mm width, with no custom fingertip.
BOX_DEPTH_M = 0.040
BOX_HEIGHT_M = 0.020
GRASP_NECK_LENGTH_M = 0.035
GRASP_FLANGE_LENGTH_M = 0.005
GRASP_NECK_DEPTH_M = 0.035
HINGE_GAP_M = 0.006
TCP_OFFSET_Z_M = 0.125
OPENING_ANGLE_GRID_DEG = (75, 105, 135, 165)

# STL-audited printed material volumes, including the removable lids.
PRINTED_SOLID_VOLUME_M3 = np.array(
    [300.57e-6, 259.35e-6, 259.35e-6, 259.35e-6]
)
PLA_DENSITY_KG_M3 = 1240.0
# Centered hidden inserts make distinct effective densities while preserving
# the one-uniform-effective-density-per-observed-envelope model.
HIDDEN_INSERT_MASS_KG = np.array([0.10, 0.30, 0.05, 0.40])

VLM_PRIOR_MEAN_KG_M3 = np.array([390.0, 410.0, 410.0, 410.0])
VLM_PRIOR_STD_KG_M3 = np.array([150.0, 260.0, 230.0, 300.0])

VOLUME_ESTIMATE_STD_FRACTION = 0.025
MANUFACTURING_SCALE_STD_FRACTION = 0.003
COM_MODEL_STD_M = 0.0015
JOINT_AXIS_MODEL_STD_M = 0.0010
GRASP_MODEL_STD_M = 0.0020
HINGE_PAIR_CAPACITY_MEAN_NM = 2.40
HINGE_PAIR_CAPACITY_STD_NM = 0.25
MINIMUM_HOLD_PROBABILITY = 0.95
HOLD_CAPACITY_USAGE_LIMIT = 0.80
MODEL_COVARIANCE_SAMPLES = 80
# Finite-sample covariance is inflated to cover unmodelled correlations among
# mesh scale, grasp, joint-axis, and COM errors.  This factor is reported and
# must be replaced by calibration against real residuals.
MODEL_COVARIANCE_INFLATION = 2.25
SAFETY_SAMPLES = 500
GRAVITY = np.array([0.0, 0.0, -9.81])


@dataclass(frozen=True)
class RobotObjectModel:
    part_count: int
    lengths_m: tuple[float, ...]
    envelope_volumes_m3: tuple[float, ...]
    prior_density_mean_kg_m3: tuple[float, ...]
    prior_density_std_kg_m3: tuple[float, ...]
    grasp_offset_m: tuple[float, float, float]
    hinge_gap_m: float


@dataclass(frozen=True)
class HiddenObjectWorld:
    true_lengths_m: tuple[float, ...]
    true_envelope_volumes_m3: tuple[float, ...]
    true_density_kg_m3: tuple[float, ...]
    true_part_mass_kg: tuple[float, ...]
    com_offsets_m: tuple[tuple[float, float, float], ...]
    joint_axis_offsets_m: tuple[tuple[float, float, float], ...]
    grasp_offset_m: tuple[float, float, float]
    hinge_capacity_nm: tuple[float, ...]
    sensor_bias: tuple[float, ...]


def nominal_lengths(part_count: int) -> np.ndarray:
    if part_count not in (2, 3, 4):
        raise ValueError("part_count must be 2, 3, or 4")
    return np.array(
        [PARENT_LENGTH_M]
        + [CHILD_LENGTH_M] * (part_count - 1),
        dtype=float,
    )


def nominal_envelope_volumes(part_count: int) -> np.ndarray:
    volumes = nominal_lengths(part_count) * BOX_DEPTH_M * BOX_HEIGHT_M
    volumes[0] -= (
        GRASP_NECK_LENGTH_M
        * (
            BOX_DEPTH_M * BOX_HEIGHT_M
            - GRASP_NECK_DEPTH_M * BOX_HEIGHT_M
        )
    )
    return volumes


def make_robot_model(
    part_count: int,
    rng: np.random.Generator,
) -> RobotObjectModel:
    nominal_volumes = nominal_envelope_volumes(part_count)
    estimated_volumes = nominal_volumes * (
        1.0
        + rng.normal(
            0.0,
            VOLUME_ESTIMATE_STD_FRACTION,
            part_count,
        )
    )
    return RobotObjectModel(
        part_count=part_count,
        lengths_m=tuple(
            float(value) for value in nominal_lengths(part_count)
        ),
        envelope_volumes_m3=tuple(
            float(value) for value in estimated_volumes
        ),
        prior_density_mean_kg_m3=tuple(
            float(value) for value in VLM_PRIOR_MEAN_KG_M3[:part_count]
        ),
        prior_density_std_kg_m3=tuple(
            float(value) for value in VLM_PRIOR_STD_KG_M3[:part_count]
        ),
        grasp_offset_m=(0.0, 0.0, TCP_OFFSET_Z_M),
        hinge_gap_m=HINGE_GAP_M,
    )


def make_hidden_world(
    part_count: int,
    rng: np.random.Generator,
) -> HiddenObjectWorld:
    lengths = nominal_lengths(part_count) * (
        1.0
        + rng.normal(
            0.0,
            MANUFACTURING_SCALE_STD_FRACTION,
            part_count,
        )
    )
    volumes = nominal_envelope_volumes(part_count) * (
        lengths / nominal_lengths(part_count)
    )
    printed_mass = (
        PRINTED_SOLID_VOLUME_M3[:part_count] * PLA_DENSITY_KG_M3
    )
    masses = printed_mass + HIDDEN_INSERT_MASS_KG[:part_count]
    densities = masses / volumes
    return HiddenObjectWorld(
        true_lengths_m=tuple(float(value) for value in lengths),
        true_envelope_volumes_m3=tuple(
            float(value) for value in volumes
        ),
        true_density_kg_m3=tuple(float(value) for value in densities),
        true_part_mass_kg=tuple(float(value) for value in masses),
        com_offsets_m=tuple(
            tuple(float(value) for value in row)
            for row in rng.normal(
                0.0,
                COM_MODEL_STD_M,
                (part_count, 3),
            )
        ),
        joint_axis_offsets_m=tuple(
            tuple(float(value) for value in row)
            for row in rng.normal(
                0.0,
                JOINT_AXIS_MODEL_STD_M,
                (part_count - 1, 3),
            )
        ),
        grasp_offset_m=tuple(
            float(value)
            for value in (
                np.array([0.0, 0.0, TCP_OFFSET_Z_M])
                + rng.normal(0.0, GRASP_MODEL_STD_M, 3)
            )
        ),
        hinge_capacity_nm=tuple(
            float(max(value, 0.5))
            for value in rng.normal(
                HINGE_PAIR_CAPACITY_MEAN_NM,
                HINGE_PAIR_CAPACITY_STD_NM,
                part_count - 1,
            )
        ),
        sensor_bias=tuple(
            float(value)
            for value in rng.normal(0.0, core.BIAS_RUN_STD)
        ),
    )


def _rotation_y(angle_rad: float) -> np.ndarray:
    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)
    return np.array(
        [
            [cosine, 0.0, sine],
            [0.0, 1.0, 0.0],
            [-sine, 0.0, cosine],
        ]
    )


def part_com_positions(
    lengths_m: np.ndarray,
    opening_angles_deg,
    *,
    grasp_offset_m: np.ndarray,
    com_offsets_m: np.ndarray | None = None,
    joint_axis_offsets_m: np.ndarray | None = None,
    hinge_gap_m: float = HINGE_GAP_M,
) -> np.ndarray:
    """Return box COM positions in the wrist F/T frame.

    An opening angle of 180 deg is the flat assembly preview.  Smaller opening
    angles bend the next box upward about the +y hinge axis.
    """
    part_count = lengths_m.size
    opening = np.asarray(opening_angles_deg, dtype=float)
    if opening.shape != (part_count - 1,):
        raise ValueError("one opening angle is required per hinge")
    com_offsets = (
        np.zeros((part_count, 3))
        if com_offsets_m is None
        else np.asarray(com_offsets_m, dtype=float)
    )
    joint_offsets = (
        np.zeros((part_count - 1, 3))
        if joint_axis_offsets_m is None
        else np.asarray(joint_axis_offsets_m, dtype=float)
    )
    positions = [np.asarray(grasp_offset_m) + com_offsets[0]]
    root_center = np.asarray(grasp_offset_m)
    joint_position = root_center + np.array(
        [lengths_m[0] / 2.0 + hinge_gap_m / 2.0, 0.0, BOX_HEIGHT_M / 2.0]
    )
    cumulative_rotation = np.eye(3)
    for child_index in range(1, part_count):
        local_bend = -math.radians(
            180.0 - opening[child_index - 1]
        )
        cumulative_rotation = (
            cumulative_rotation @ _rotation_y(local_bend)
        )
        joint_position = (
            joint_position + joint_offsets[child_index - 1]
        )
        child_from_joint = np.array(
            [
                hinge_gap_m / 2.0 + lengths_m[child_index] / 2.0,
                0.0,
                -BOX_HEIGHT_M / 2.0,
            ]
        )
        positions.append(
            joint_position
            + cumulative_rotation @ child_from_joint
            + com_offsets[child_index]
        )
        distal_from_joint = np.array(
            [hinge_gap_m + lengths_m[child_index], 0.0, 0.0]
        )
        joint_position = (
            joint_position
            + cumulative_rotation @ distal_from_joint
        )
    return np.asarray(positions)


def wrench_regressor(
    model: RobotObjectModel,
    opening_angles_deg,
) -> np.ndarray:
    positions = part_com_positions(
        np.asarray(model.lengths_m),
        opening_angles_deg,
        grasp_offset_m=np.asarray(model.grasp_offset_m),
        hinge_gap_m=model.hinge_gap_m,
    )
    volumes = np.asarray(model.envelope_volumes_m3)
    regressor = np.zeros((6, model.part_count))
    for part in range(model.part_count):
        force = volumes[part] * GRAVITY
        regressor[:3, part] = force
        regressor[3:, part] = np.cross(positions[part], force)
    return regressor


def hidden_wrench_regressor(
    hidden: HiddenObjectWorld,
    opening_angles_deg,
) -> np.ndarray:
    lengths = np.asarray(hidden.true_lengths_m)
    positions = part_com_positions(
        lengths,
        opening_angles_deg,
        grasp_offset_m=np.asarray(hidden.grasp_offset_m),
        com_offsets_m=np.asarray(hidden.com_offsets_m),
        joint_axis_offsets_m=np.asarray(hidden.joint_axis_offsets_m),
    )
    volumes = np.asarray(hidden.true_envelope_volumes_m3)
    regressor = np.zeros((6, lengths.size))
    for part in range(lengths.size):
        force = volumes[part] * GRAVITY
        regressor[:3, part] = force
        regressor[3:, part] = np.cross(positions[part], force)
    return regressor


def joint_gravity_torques(
    lengths_m: np.ndarray,
    masses_kg: np.ndarray,
    opening_angles_deg,
    *,
    grasp_offset_m: np.ndarray,
) -> np.ndarray:
    positions = part_com_positions(
        lengths_m,
        opening_angles_deg,
        grasp_offset_m=grasp_offset_m,
    )
    part_count = lengths_m.size
    opening = np.asarray(opening_angles_deg)
    root_center = grasp_offset_m
    joint_positions = []
    joint_position = root_center + np.array(
        [lengths_m[0] / 2.0 + HINGE_GAP_M / 2.0, 0.0, BOX_HEIGHT_M / 2.0]
    )
    cumulative_rotation = np.eye(3)
    for joint in range(part_count - 1):
        joint_positions.append(joint_position.copy())
        bend = -math.radians(180.0 - opening[joint])
        cumulative_rotation = cumulative_rotation @ _rotation_y(bend)
        child_index = joint + 1
        joint_position = joint_position + cumulative_rotation @ np.array(
            [HINGE_GAP_M + lengths_m[child_index], 0.0, 0.0]
        )
    torques = []
    for joint, position in enumerate(joint_positions):
        torque = 0.0
        for part in range(joint + 1, part_count):
            force = masses_kg[part] * GRAVITY
            torque += np.cross(positions[part] - position, force)[1]
        torques.append(abs(float(torque)))
    return np.asarray(torques)


def candidate_configurations(part_count: int):
    return tuple(
        itertools.product(
            OPENING_ANGLE_GRID_DEG,
            repeat=part_count - 1,
        )
    )


def safety_probability(
    model: RobotObjectModel,
    configuration,
    *,
    seed: int,
    samples: int = SAFETY_SAMPLES,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    lengths = np.asarray(model.lengths_m)
    volumes = np.asarray(model.envelope_volumes_m3)
    prior_mean = np.asarray(model.prior_density_mean_kg_m3)
    prior_std = np.asarray(model.prior_density_std_kg_m3)
    safe = 0
    maximum_usage = []
    for _ in range(samples):
        density = np.maximum(
            rng.normal(prior_mean, prior_std),
            50.0,
        )
        masses = volumes * density
        capacity = np.maximum(
            rng.normal(
                HINGE_PAIR_CAPACITY_MEAN_NM,
                HINGE_PAIR_CAPACITY_STD_NM,
                model.part_count - 1,
            ),
            0.5,
        )
        torque = joint_gravity_torques(
            lengths,
            masses,
            configuration,
            grasp_offset_m=np.asarray(model.grasp_offset_m),
        )
        usage = torque / capacity
        maximum_usage.append(float(np.max(usage)))
        safe += int(np.all(usage <= HOLD_CAPACITY_USAGE_LIMIT))
    return safe / samples, float(np.quantile(maximum_usage, 0.95))


def _angle_covariance(model: RobotObjectModel, configuration) -> np.ndarray:
    precision = sum(
        1.0 / math.radians(std) ** 2
        for std in core.CAMERA_STD_DEG.values()
    )
    return np.eye(model.part_count - 1) / precision


def effective_measurement_covariance(
    model: RobotObjectModel,
    configuration,
    density_linearization: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    base_covariance = np.diag(
        core.FT_SAMPLE_STD**2 / core.HOLD_SAMPLE_COUNT
    )
    q = np.asarray(configuration, dtype=float)
    angle_covariance = _angle_covariance(model, configuration)
    angle_jacobian = np.zeros((6, q.size))
    step_deg = 0.05
    for joint in range(q.size):
        plus = q.copy()
        minus = q.copy()
        plus[joint] += step_deg
        minus[joint] -= step_deg
        angle_jacobian[:, joint] = (
            wrench_regressor(model, plus) @ density_linearization
            - wrench_regressor(model, minus) @ density_linearization
        ) / math.radians(2.0 * step_deg)
    camera_covariance = (
        angle_jacobian @ angle_covariance @ angle_jacobian.T
    )

    # Approximate geometry/grasp model discrepancy as a measurement covariance.
    rng = np.random.default_rng(seed)
    nominal_wrench = wrench_regressor(model, q) @ density_linearization
    deviations = []
    lengths = np.asarray(model.lengths_m)
    volumes = np.asarray(model.envelope_volumes_m3)
    for _ in range(MODEL_COVARIANCE_SAMPLES):
        perturbed_lengths = lengths * (
            1.0
            + rng.normal(
                0.0,
                MANUFACTURING_SCALE_STD_FRACTION,
                model.part_count,
            )
        )
        perturbed_volumes = volumes * (
            1.0
            + rng.normal(
                0.0,
                VOLUME_ESTIMATE_STD_FRACTION,
                model.part_count,
            )
        )
        positions = part_com_positions(
            perturbed_lengths,
            q,
            grasp_offset_m=(
                np.asarray(model.grasp_offset_m)
                + rng.normal(0.0, GRASP_MODEL_STD_M, 3)
            ),
            com_offsets_m=rng.normal(
                0.0,
                COM_MODEL_STD_M,
                (model.part_count, 3),
            ),
            joint_axis_offsets_m=rng.normal(
                0.0,
                JOINT_AXIS_MODEL_STD_M,
                (model.part_count - 1, 3),
            ),
        )
        regressor = np.zeros((6, model.part_count))
        for part in range(model.part_count):
            force = perturbed_volumes[part] * GRAVITY
            regressor[:3, part] = force
            regressor[3:, part] = np.cross(positions[part], force)
        deviations.append(
            regressor @ density_linearization - nominal_wrench
        )
    model_covariance = np.cov(np.asarray(deviations), rowvar=False)
    return (
        base_covariance
        + camera_covariance
        + MODEL_COVARIANCE_INFLATION * model_covariance
        + np.eye(6) * 1e-12
    )


def matrix_rank(regressors: list[np.ndarray], part_count: int) -> int:
    if not regressors:
        return 0
    stacked = np.vstack(regressors)
    singular = np.linalg.svd(stacked, compute_uv=False)
    tolerance = (
        max(stacked.shape)
        * np.finfo(float).eps
        * max(float(singular[0]), 1.0)
        * 100.0
    )
    return int(np.sum(singular > tolerance))


def select_configurations(
    model: RobotObjectModel,
) -> tuple[tuple[int, ...], ...]:
    prior_covariance = np.diag(
        np.asarray(model.prior_density_std_kg_m3) ** 2
    )
    covariance = prior_covariance.copy()
    selected = []
    stacked = []
    for selection_index in range(model.part_count - 1):
        best = None
        best_objective = None
        for candidate_index, candidate in enumerate(
            candidate_configurations(model.part_count)
        ):
            if candidate in selected:
                continue
            safe_probability, p95_usage = safety_probability(
                model,
                candidate,
                seed=10000 * model.part_count + candidate_index,
            )
            if safe_probability < MINIMUM_HOLD_PROBABILITY:
                continue
            regressor = wrench_regressor(model, candidate)
            noise = effective_measurement_covariance(
                model,
                candidate,
                np.asarray(model.prior_density_mean_kg_m3),
                seed=(
                    100000 * model.part_count
                    + 1000 * selection_index
                    + candidate_index
                ),
            )
            fisher = regressor.T @ np.linalg.inv(noise) @ regressor
            sign, logdet = np.linalg.slogdet(
                np.eye(model.part_count) + covariance @ fisher
            )
            if sign <= 0:
                continue
            rank = matrix_rank([*stacked, regressor], model.part_count)
            information_gain = 0.5 * logdet
            objective = (
                rank,
                information_gain,
                -p95_usage,
            )
            if best_objective is None or objective > best_objective:
                best = (candidate, regressor, fisher)
                best_objective = objective
        if best is None:
            raise RuntimeError(
                f"No safe configuration for {model.part_count}-link step "
                f"{selection_index + 1}"
            )
        candidate, regressor, fisher = best
        selected.append(candidate)
        stacked.append(regressor)
        covariance = np.linalg.inv(
            np.linalg.inv(covariance) + fisher
        )
    return tuple(selected)


def initialize_state(model: RobotObjectModel) -> core.EstimatorState:
    p = model.part_count
    mean = np.concatenate(
        (
            np.asarray(model.prior_density_mean_kg_m3),
            np.zeros(6),
        )
    )
    covariance = np.zeros((p + 6, p + 6))
    covariance[:p, :p] = np.diag(
        np.asarray(model.prior_density_std_kg_m3) ** 2
    )
    covariance[p:, p:] = np.diag(core.BIAS_RUN_STD**2)
    return core.EstimatorState(mean, covariance)


def apply_tare(
    model: RobotObjectModel,
    state: core.EstimatorState,
    tare: np.ndarray,
) -> core.EstimatorState:
    p = model.part_count
    matrix = np.hstack((np.zeros((6, p)), np.eye(6)))
    covariance = np.diag(
        core.FT_SAMPLE_STD**2 / core.TARE_SAMPLE_COUNT
    )
    return core._linear_update(state, tare, matrix, covariance)


def fuse_camera(
    actual_angles_deg: np.ndarray,
    rng: np.random.Generator,
):
    return core._fuse_camera_observation(actual_angles_deg, rng)


def run_trial(
    part_count: int,
    *,
    seed: int,
    configurations: tuple[tuple[int, ...], ...] | None = None,
    include_debug_hidden: bool = True,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    model = make_robot_model(part_count, rng)
    hidden = make_hidden_world(part_count, rng)
    configurations = (
        select_configurations(model)
        if configurations is None
        else configurations
    )
    state = initialize_state(model)
    tare_covariance = np.diag(
        core.FT_SAMPLE_STD**2 / core.TARE_SAMPLE_COUNT
    )
    tare = np.asarray(hidden.sensor_bias) + rng.multivariate_normal(
        np.zeros(6),
        tare_covariance,
    )
    state = apply_tare(model, state, tare)
    trace = []
    stacked_accepted = []
    for index, command in enumerate(configurations, start=1):
        actual_angles = np.asarray(command, dtype=float) + rng.normal(
            0.0,
            core.JOINT_SETTING_STD_DEG,
            part_count - 1,
        )
        camera = fuse_camera(actual_angles, rng)
        drift = float(abs(rng.normal(0.0, core.HOLD_DRIFT_STD_DEG)))
        true_torque = joint_gravity_torques(
            np.asarray(hidden.true_lengths_m),
            np.asarray(hidden.true_part_mass_kg),
            actual_angles,
            grasp_offset_m=np.asarray(hidden.grasp_offset_m),
        )
        capacity = np.asarray(hidden.hinge_capacity_nm)
        reason = None
        if camera is None:
            reason = "fewer_than_two_cameras"
        elif drift > core.HARD_DRIFT_GATE_DEG:
            reason = "joint_drift_exceeded_hard_gate"
        elif np.any(true_torque > capacity):
            reason = "hidden_hinge_capacity_exceeded"
        hold_covariance = np.diag(
            core.FT_SAMPLE_STD**2 / core.HOLD_SAMPLE_COUNT
        )
        raw_wrench = (
            hidden_wrench_regressor(hidden, actual_angles)
            @ np.asarray(hidden.true_density_kg_m3)
            + np.asarray(hidden.sensor_bias)
            + rng.multivariate_normal(np.zeros(6), hold_covariance)
        )
        prior_covariance = state.covariance[:part_count, :part_count].copy()
        prior_logdet = np.linalg.slogdet(prior_covariance)[1]
        if reason is None:
            measured_angles = np.asarray(
                camera.fused_joint_angles_deg
            )
            regressor = wrench_regressor(model, measured_angles)
            matrix = np.hstack((regressor, np.eye(6)))
            effective_covariance = effective_measurement_covariance(
                model,
                measured_angles,
                state.mean[:part_count],
                seed=seed * 100 + index,
            )
            state = core._linear_update(
                state,
                raw_wrench,
                matrix,
                effective_covariance,
            )
            stacked_accepted.append(regressor)
        posterior_covariance = state.covariance[:part_count, :part_count]
        posterior_logdet = np.linalg.slogdet(posterior_covariance)[1]
        information_gain = 0.5 * (
            prior_logdet - posterior_logdet
        )
        estimate = state.mean[:part_count]
        std = np.sqrt(np.diag(posterior_covariance))
        gt_density = np.asarray(hidden.true_density_kg_m3)
        estimated_mass = (
            np.asarray(model.envelope_volumes_m3) * estimate
        )
        gt_mass = np.asarray(hidden.true_part_mass_kg)
        trace.append(
            {
                "state_index": index,
                "commanded_opening_angles_deg": list(command),
                "camera_opening_angles_deg": (
                    []
                    if camera is None
                    else list(camera.fused_joint_angles_deg)
                ),
                "visible_cameras": (
                    [] if camera is None else list(camera.visible_cameras)
                ),
                "raw_wrench_ft": raw_wrench.tolist(),
                "accepted": reason is None,
                "rejection_reason": reason,
                "maximum_joint_drift_deg": drift,
                "robot_output": {
                    "density_mean_kg_m3": estimate.tolist(),
                    "density_std_kg_m3": std.tolist(),
                    "part_mass_mean_kg": estimated_mass.tolist(),
                    "incremental_information_gain_nats": information_gain,
                    "data_rank": matrix_rank(
                        stacked_accepted,
                        part_count,
                    ),
                    "nullity": (
                        part_count
                        - matrix_rank(stacked_accepted, part_count)
                    ),
                },
                "evaluator_only": {
                    "density_relative_error": (
                        np.abs(estimate - gt_density) / gt_density
                    ).tolist(),
                    "mass_relative_error": (
                        np.abs(estimated_mass - gt_mass) / gt_mass
                    ).tolist(),
                    "gt_inside_95pct_interval": (
                        np.abs(estimate - gt_density) <= 1.96 * std
                    ).tolist(),
                    "true_joint_torque_nm": true_torque.tolist(),
                    "true_hinge_capacity_nm": capacity.tolist(),
                },
            }
        )
    result = {
        "schema": "custom-cad-hidden-gt-configuration-experiment-v1",
        "claim_scope": (
            "Robust synthetic validation derived from manufacturing CAD; "
            "replace assumed sensor/hinge distributions with measured values "
            "before hardware-performance claims."
        ),
        "robot_visible_model": asdict(model),
        "selected_configurations_deg": [
            list(value) for value in configurations
        ],
        "tare_observation": tare.tolist(),
        "trace": trace,
    }
    if include_debug_hidden:
        result["evaluator_only_hidden_world"] = asdict(hidden)
    return result


def run_monte_carlo(
    part_count: int,
    *,
    trials: int,
    seed: int,
    configurations: tuple[tuple[int, ...], ...],
) -> dict[str, object]:
    errors = []
    mass_errors = []
    coverage = []
    accepted = []
    for trial in range(trials):
        result = run_trial(
            part_count,
            seed=seed + trial,
            configurations=configurations,
        )
        final = result["trace"][-1]
        errors.append(final["evaluator_only"]["density_relative_error"])
        mass_errors.append(final["evaluator_only"]["mass_relative_error"])
        coverage.append(
            final["evaluator_only"]["gt_inside_95pct_interval"]
        )
        accepted.append(
            sum(row["accepted"] for row in result["trace"])
        )
    errors_array = np.asarray(errors)
    mass_array = np.asarray(mass_errors)
    return {
        "trials": trials,
        "density_relative_rmse": np.sqrt(
            np.mean(errors_array**2, axis=0)
        ).tolist(),
        "density_relative_p95": np.quantile(
            errors_array,
            0.95,
            axis=0,
        ).tolist(),
        "mass_relative_rmse": np.sqrt(
            np.mean(mass_array**2, axis=0)
        ).tolist(),
        "empirical_95pct_coverage": np.mean(
            np.asarray(coverage, dtype=float),
            axis=0,
        ).tolist(),
        "all_states_accepted_rate": float(
            np.mean(
                np.asarray(accepted)
                == part_count - 1
            )
        ),
    }


def plot_trace(result: dict[str, object], path: Path) -> None:
    part_count = result["robot_visible_model"]["part_count"]
    gt = np.asarray(
        result["evaluator_only_hidden_world"]["true_density_kg_m3"]
    )
    prior = np.asarray(
        result["robot_visible_model"]["prior_density_mean_kg_m3"]
    )
    prior_std = np.asarray(
        result["robot_visible_model"]["prior_density_std_kg_m3"]
    )
    means = [prior]
    stds = [prior_std]
    info = [0.0]
    ranks = [0]
    for row in result["trace"]:
        means.append(np.asarray(row["robot_output"]["density_mean_kg_m3"]))
        stds.append(np.asarray(row["robot_output"]["density_std_kg_m3"]))
        info.append(row["robot_output"]["incremental_information_gain_nats"])
        ranks.append(row["robot_output"]["data_rank"])
    means = np.asarray(means)
    stds = np.asarray(stds)
    x = np.arange(len(means))
    figure, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    for part in range(part_count):
        axes[0].errorbar(
            x,
            means[:, part],
            yerr=1.96 * stds[:, part],
            marker="o",
            capsize=3,
            label=f"part {part + 1}",
        )
        axes[0].axhline(gt[part], linestyle="--", alpha=0.55)
        axes[1].plot(
            x,
            stds[:, part] / prior_std[part],
            marker="o",
            label=f"part {part + 1}",
        )
    axes[0].set_title("Robot estimate ±95% / dashed hidden GT")
    axes[0].set_ylabel("effective density [kg/m³]")
    axes[0].set_xlabel("configuration count")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].set_title("Uncertainty remaining")
    axes[1].set_ylabel("posterior std / prior std")
    axes[1].set_xlabel("configuration count")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    axes[2].bar(x, info, label="incremental information [nat]")
    axes[2].plot(x, ranks, "o-", color="black", label="data rank")
    axes[2].set_title("Information acquired at each state")
    axes[2].set_xlabel("configuration count")
    axes[2].grid(alpha=0.25)
    axes[2].legend()
    figure.suptitle(f"Custom CAD-derived {part_count}-link hidden-GT validation")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run_suite(
    output_dir: Path,
    *,
    trials: int,
    seed: int,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for part_count in (2, 3, 4):
        selection_model = make_robot_model(
            part_count,
            np.random.default_rng(seed + 100 * part_count),
        )
        configurations = select_configurations(selection_model)
        nominal = run_trial(
            part_count,
            seed=seed + part_count,
            configurations=configurations,
        )
        monte_carlo = run_monte_carlo(
            part_count,
            trials=trials,
            seed=seed + 1000 * part_count,
            configurations=configurations,
        )
        plot_trace(
            nominal,
            output_dir / f"custom_cad_{part_count}_link_trace.png",
        )
        results.append(
            {
                "part_count": part_count,
                "nominal_trace": nominal,
                "monte_carlo": monte_carlo,
            }
        )
    report = {
        "schema": "custom-cad-configuration-information-suite-v1",
        "cad_source": str(ROOT / "custom_object_cad"),
        "assumption_status": {
            "geometry": "derived from audited Parent/Child CAD dimensions",
            "printed_mass": "derived from audited STL solid volume and PLA density",
            "inserts": "controlled centered hidden-mass design values",
            "ft_camera_noise": "engineering assumptions pending calibration",
            "hinge_capacity": "recommended 2-hinge range, pending measurement",
            "3_4_link_cad": (
                "modular repetition of Child envelope; middle/end hinge "
                "features require CAD export before fabrication"
            ),
        },
        "results": results,
    }
    (output_dir / "custom_cad_results.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260728)
    args = parser.parse_args()
    report = run_suite(
        args.output_dir,
        trials=args.trials,
        seed=args.seed,
    )
    for row in report["results"]:
        final = row["nominal_trace"]["trace"][-1]
        print(
            f"{row['part_count']}-link "
            f"configs={row['nominal_trace']['selected_configurations_deg']} "
            f"rank={final['robot_output']['data_rank']} "
            f"error={np.round(final['evaluator_only']['density_relative_error'], 3)} "
            f"MC-RMSE={np.round(row['monte_carlo']['density_relative_rmse'], 3)} "
            f"coverage={np.round(row['monte_carlo']['empirical_95pct_coverage'], 3)}"
        )


if __name__ == "__main__":
    main()
