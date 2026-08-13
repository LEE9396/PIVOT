#!/usr/bin/env python3
"""GT-calibrated Drake proof of static 2-link mass tracking."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from pydrake.all import (
    Box,
    FixedOffsetFrame,
    MultibodyForces,
    MultibodyPlant,
    PrismaticJoint,
    RevoluteJoint,
    RigidTransform,
    Rgba,
    SpatialInertia,
    StartMeshcat,
    UnitInertia,
    WeldJoint,
)


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

import run_custom_cad_configuration_information_experiment as study  # noqa: E402
import run_hidden_gt_quasistatic_experiment as noise  # noqa: E402


DEFAULT_OUTPUT = ROOT / "results/static_mass_tracking_2link.json"
GRAVITY = np.array([0.0, 0.0, -9.81])
WRIST_PITCH_GRID_DEG = (-60.0, -30.0, 0.0, 30.0, 60.0)
FOUNDATIONPOSE_STD_DEG = 0.25
KNOWN_SUPPORT_MASS_KG = 0.002


@dataclass(frozen=True)
class DrakeObject:
    plant: MultibodyPlant
    root_translation: PrismaticJoint
    wrist_pitch: RevoluteJoint
    object_joint: RevoluteJoint
    bodies: tuple[object, object]
    volumes_m3: np.ndarray


def _inertia(mass: float, length: float, com: np.ndarray) -> SpatialInertia:
    return SpatialInertia.MakeFromCentralInertia(
        mass,
        com,
        mass
        * UnitInertia.SolidBox(
            length,
            study.BOX_DEPTH_M,
            study.BOX_HEIGHT_M,
        ),
    )


def build_object(masses_kg: np.ndarray) -> DrakeObject:
    if masses_kg.shape != (2,) or np.any(masses_kg <= 0.0):
        raise ValueError("masses_kg must contain two positive values")
    plant = MultibodyPlant(time_step=0.0)
    carrier = plant.AddRigidBody(
        "vertical_carrier",
        _inertia(0.001, 0.001, np.zeros(3)),
    )
    root_translation = plant.AddJoint(
        PrismaticJoint(
            "wrist_vertical_translation",
            plant.world_frame(),
            carrier.body_frame(),
            [0.0, 0.0, 1.0],
        )
    )
    wrist = plant.AddRigidBody(
        "wrist_pitch_frame",
        _inertia(0.001, 0.001, np.zeros(3)),
    )
    wrist_pitch = plant.AddJoint(
        RevoluteJoint(
            "wrist_pitch",
            carrier.body_frame(),
            wrist.body_frame(),
            [0.0, 1.0, 0.0],
        )
    )
    parent_com = np.array([0.0, 0.0, study.TCP_OFFSET_Z_M])
    parent = plant.AddRigidBody(
        "parent",
        _inertia(masses_kg[0], study.PARENT_LENGTH_M, parent_com),
    )
    plant.AddJoint(
        WeldJoint(
            "wrist_to_parent",
            wrist.body_frame(),
            parent.body_frame(),
            RigidTransform(),
        )
    )
    hinge = plant.AddFrame(
        FixedOffsetFrame(
            "parent_hinge",
            parent.body_frame(),
            RigidTransform(
                parent_com
                + np.array(
                    [
                        study.PARENT_LENGTH_M / 2.0
                        + study.HINGE_GAP_M / 2.0,
                        0.0,
                        study.BOX_HEIGHT_M / 2.0,
                    ]
                )
            ),
        )
    )
    child_com = np.array(
        [
            study.HINGE_GAP_M / 2.0 + study.CHILD_LENGTH_M / 2.0,
            0.0,
            -study.BOX_HEIGHT_M / 2.0,
        ]
    )
    child = plant.AddRigidBody(
        "child",
        _inertia(masses_kg[1], study.CHILD_LENGTH_M, child_com),
    )
    object_joint = plant.AddJoint(
        RevoluteJoint(
            "object_hinge",
            hinge,
            child.body_frame(),
            [0.0, 1.0, 0.0],
        )
    )
    plant.Finalize()
    volumes = study.nominal_envelope_volumes(2)
    return DrakeObject(
        plant,
        root_translation,
        wrist_pitch,
        object_joint,
        (parent, child),
        volumes,
    )


def _context(
    model: DrakeObject,
    wrist_pitch_deg: float,
    opening_angle_deg: float,
):
    context = model.plant.CreateDefaultContext()
    model.root_translation.set_translation(context, 0.0)
    model.wrist_pitch.set_angle(context, math.radians(wrist_pitch_deg))
    model.object_joint.set_angle(
        context,
        -math.radians(180.0 - opening_angle_deg),
    )
    model.plant.SetVelocities(context, np.zeros(model.plant.num_velocities()))
    return context


def mass_regressor(
    model: DrakeObject,
    wrist_pitch_deg: float,
    opening_angle_deg: float,
) -> np.ndarray:
    context = _context(model, wrist_pitch_deg, opening_angle_deg)
    regressor = np.zeros((6, 2))
    for index, body in enumerate(model.bodies):
        pose = model.plant.EvalBodyPoseInWorld(context, body)
        position = pose.multiply(body.default_spatial_inertia().get_com())
        regressor[:3, index] = GRAVITY
        regressor[3:, index] = np.cross(position, GRAVITY)
    return regressor


def drake_inverse_dynamics_error(
    model: DrakeObject,
    wrist_pitch_deg: float,
    opening_angle_deg: float,
    wrench: np.ndarray,
) -> float:
    context = _context(model, wrist_pitch_deg, opening_angle_deg)
    forces = MultibodyForces(model.plant)
    model.plant.CalcForceElementsContribution(context, forces)
    support = model.plant.CalcInverseDynamics(
        context,
        np.zeros(model.plant.num_velocities()),
        forces,
    )
    expected_support = np.array(
        [
            -wrench[2] + KNOWN_SUPPORT_MASS_KG * 9.81,
            -wrench[4],
        ]
    )
    actual_support = np.array(
        [
            support[model.root_translation.velocity_start()],
            support[model.wrist_pitch.velocity_start()],
        ]
    )
    return float(np.max(np.abs(actual_support - expected_support)))


def _measurement_covariance() -> np.ndarray:
    return np.diag(noise.FT_SAMPLE_STD**2 / noise.HOLD_SAMPLE_COUNT)


def _effective_covariance(
    model: DrakeObject,
    pose: tuple[float, float],
    mass_mean: np.ndarray,
) -> np.ndarray:
    wrist_pitch_deg, opening_angle_deg = pose
    step = 0.05
    plus = mass_regressor(
        model,
        wrist_pitch_deg,
        opening_angle_deg + step,
    ) @ mass_mean
    minus = mass_regressor(
        model,
        wrist_pitch_deg,
        opening_angle_deg - step,
    ) @ mass_mean
    jacobian = (plus - minus) / math.radians(2.0 * step)
    angle_variance = math.radians(FOUNDATIONPOSE_STD_DEG) ** 2
    return (
        _measurement_covariance()
        + np.outer(jacobian, jacobian) * angle_variance
        + np.eye(6) * 1e-12
    )


def _posterior_covariance(
    covariance: np.ndarray,
    regressor: np.ndarray,
    measurement_covariance: np.ndarray,
) -> np.ndarray:
    return np.linalg.inv(
        np.linalg.inv(covariance)
        + regressor.T
        @ np.linalg.inv(measurement_covariance)
        @ regressor
    )


def _select_pose(
    model: DrakeObject,
    candidates: list[tuple[float, float]],
    used: set[tuple[float, float]],
    mass_mean: np.ndarray,
    covariance: np.ndarray,
) -> tuple[tuple[float, float], float]:
    prior_logdet = np.linalg.slogdet(covariance)[1]
    best = None
    for pose in candidates:
        if pose in used:
            continue
        regressor = mass_regressor(model, *pose)
        measurement_covariance = _effective_covariance(
            model,
            pose,
            mass_mean,
        )
        posterior = _posterior_covariance(
            covariance,
            regressor,
            measurement_covariance,
        )
        information_gain = 0.5 * (
            prior_logdet - np.linalg.slogdet(posterior)[1]
        )
        if best is None or information_gain > best[1]:
            best = (pose, float(information_gain))
    if best is None:
        raise RuntimeError("no unused static pose remains")
    return best


def run(
    *,
    joint_min_deg: float,
    joint_max_deg: float,
    steps: int,
    seed: int,
) -> dict[str, object]:
    if not 0.0 <= joint_min_deg < joint_max_deg:
        raise ValueError("joint limits must satisfy 0 <= min < max")
    opening_grid = np.linspace(joint_min_deg, joint_max_deg, 7)
    candidates = [
        (pitch, float(opening))
        for pitch in WRIST_PITCH_GRID_DEG
        for opening in opening_grid
    ]
    if not 1 <= steps <= len(candidates):
        raise ValueError("steps exceeds available candidate poses")

    true_mass = (
        study.PRINTED_SOLID_VOLUME_M3[:2] * study.PLA_DENSITY_KG_M3
        + study.HIDDEN_INSERT_MASS_KG[:2]
    )
    model = build_object(true_mass)
    volume = model.volumes_m3
    prior_mass = volume * study.VLM_PRIOR_MEAN_KG_M3[:2]
    prior_std = volume * study.VLM_PRIOR_STD_KG_M3[:2]
    mean = prior_mass.copy()
    covariance = np.diag(prior_std**2)
    rng = np.random.default_rng(seed)
    true_bias = rng.normal(0.0, noise.BIAS_RUN_STD)
    tare_covariance = np.diag(
        noise.FT_SAMPLE_STD**2 / noise.TARE_SAMPLE_COUNT
    )
    tare = true_bias + rng.multivariate_normal(np.zeros(6), tare_covariance)
    used: set[tuple[float, float]] = set()
    trace = []

    for index in range(1, steps + 1):
        pose, expected_information_gain = _select_pose(
            model,
            candidates,
            used,
            mean,
            covariance,
        )
        used.add(pose)
        wrist_pitch_deg, true_opening_deg = pose
        foundationpose_raw_deg = true_opening_deg + rng.normal(
            0.0,
            FOUNDATIONPOSE_STD_DEG,
        )
        measured_opening_deg = float(
            np.clip(
                foundationpose_raw_deg,
                joint_min_deg,
                joint_max_deg,
            )
        )
        true_regressor = mass_regressor(model, *pose)
        true_wrench = true_regressor @ true_mass
        raw_wrench = (
            true_wrench
            + true_bias
            + rng.multivariate_normal(
                np.zeros(6),
                _measurement_covariance(),
            )
        )
        measured_wrench = raw_wrench - tare
        measured_regressor = mass_regressor(
            model,
            wrist_pitch_deg,
            measured_opening_deg,
        )
        effective_covariance = _effective_covariance(
            model,
            (wrist_pitch_deg, measured_opening_deg),
            mean,
        )
        innovation_covariance = (
            measured_regressor
            @ covariance
            @ measured_regressor.T
            + effective_covariance
        )
        gain = (
            covariance
            @ measured_regressor.T
            @ np.linalg.inv(innovation_covariance)
        )
        mean = mean + gain @ (
            measured_wrench - measured_regressor @ mean
        )
        residual = np.eye(2) - gain @ measured_regressor
        covariance = (
            residual @ covariance @ residual.T
            + gain @ effective_covariance @ gain.T
        )
        trace.append(
            {
                "step": index,
                "selected_static_pose": {
                    "wrist_pitch_deg": wrist_pitch_deg,
                    "true_opening_angle_deg": true_opening_deg,
                    "foundationpose_raw_opening_angle_deg": (
                        foundationpose_raw_deg
                    ),
                    "foundationpose_opening_angle_deg": (
                        measured_opening_deg
                    ),
                },
                "expected_information_gain_nats": (
                    expected_information_gain
                ),
                "raw_wrench": raw_wrench.tolist(),
                "tare_corrected_wrench": measured_wrench.tolist(),
                "mass_mean_kg": mean.tolist(),
                "mass_std_kg": np.sqrt(np.diag(covariance)).tolist(),
                "density_mean_kg_m3": (mean / volume).tolist(),
                "mass_relative_error_evaluator_only": (
                    np.abs(mean - true_mass) / true_mass
                ).tolist(),
                "drake_inverse_dynamics_max_error": (
                    drake_inverse_dynamics_error(
                        model,
                        wrist_pitch_deg,
                        true_opening_deg,
                        true_wrench,
                    )
                ),
            }
        )

    return {
        "schema": "drake-static-2link-mass-tracking-v1",
        "scope": "static pose selection and part-mass tracking",
        "formulation": {
            "measurement": "z_k = Y(q_k, R_k) m + epsilon_k",
            "density": "rho_i = m_i / V_i",
            "selection": (
                "argmax_a 0.5*(logdet(P_k)-logdet(P_{k+1}(a)))"
            ),
        },
        "robot_visible_initial_values": {
            "volumes_m3": volume.tolist(),
            "prior_mass_mean_kg": prior_mass.tolist(),
            "prior_mass_std_kg": prior_std.tolist(),
            "joint_limits_deg": [joint_min_deg, joint_max_deg],
            "foundationpose_std_deg": FOUNDATIONPOSE_STD_DEG,
            "ft_hold_covariance": _measurement_covariance().tolist(),
            "tare_observation": tare.tolist(),
        },
        "evaluator_only_ground_truth": {
            "part_mass_kg": true_mass.tolist(),
            "density_kg_m3": (true_mass / volume).tolist(),
            "sensor_bias": true_bias.tolist(),
        },
        "trace": trace,
        "final": {
            "part_mass_kg": mean.tolist(),
            "part_mass_std_kg": np.sqrt(np.diag(covariance)).tolist(),
            "density_kg_m3": (mean / volume).tolist(),
            "mass_relative_error": (
                np.abs(mean - true_mass) / true_mass
            ).tolist(),
        },
    }


def run_live(result: dict[str, object], cycle_seconds: float) -> None:
    meshcat = StartMeshcat()
    true_mass = np.asarray(
        result["evaluator_only_ground_truth"]["part_mass_kg"]
    )
    model = build_object(true_mass)
    meshcat.SetObject(
        "/object/parent",
        Box(
            study.PARENT_LENGTH_M,
            study.BOX_DEPTH_M,
            study.BOX_HEIGHT_M,
        ),
        Rgba(0.08, 0.24, 0.62, 1.0),
    )
    meshcat.SetObject(
        "/object/child",
        Box(
            study.CHILD_LENGTH_M,
            study.BOX_DEPTH_M,
            study.BOX_HEIGHT_M,
        ),
        Rgba(0.08, 0.58, 0.46, 1.0),
    )
    meshcat.SetObject(
        "/wrist",
        Box(0.03, 0.14, 0.03),
        Rgba(0.35, 0.35, 0.38, 1.0),
    )
    meshcat.SetProperty("/Background", "top_color", [0.92, 0.95, 1.0])
    meshcat.SetProperty("/Background", "bottom_color", [0.72, 0.78, 0.86])
    stop = "종료"
    meshcat.AddButton(stop)
    current_label = None
    print(f"Meshcat: {meshcat.web_url()}", flush=True)
    while meshcat.GetButtonClicks(stop) < 1:
        for row in result["trace"]:
            if meshcat.GetButtonClicks(stop) >= 1:
                break
            pose = row["selected_static_pose"]
            context = _context(
                model,
                pose["wrist_pitch_deg"],
                pose["foundationpose_opening_angle_deg"],
            )
            parent_pose = model.plant.EvalBodyPoseInWorld(
                context,
                model.bodies[0],
            )
            child_pose = model.plant.EvalBodyPoseInWorld(
                context,
                model.bodies[1],
            )
            meshcat.SetTransform(
                "/object/parent",
                parent_pose
                @ RigidTransform([0.0, 0.0, study.TCP_OFFSET_Z_M]),
            )
            meshcat.SetTransform(
                "/object/child",
                child_pose
                @ RigidTransform(
                    [
                        study.HINGE_GAP_M / 2.0
                        + study.CHILD_LENGTH_M / 2.0,
                        0.0,
                        -study.BOX_HEIGHT_M / 2.0,
                    ]
                ),
            )
            meshcat.SetTransform("/wrist", parent_pose)
            if current_label is not None:
                meshcat.DeleteButton(current_label)
            masses = row["mass_mean_kg"]
            current_label = (
                f"STEP {row['step']} | wrist={pose['wrist_pitch_deg']:.0f}° "
                f"| q={pose['foundationpose_opening_angle_deg']:.1f}° "
                f"| m=({masses[0]:.3f}, {masses[1]:.3f}) kg"
            )
            meshcat.AddButton(current_label)
            time.sleep(cycle_seconds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--joint-min-deg", type=float, default=20.0)
    parser.add_argument("--joint-max-deg", type=float, default=120.0)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--cycle-seconds", type=float, default=3.0)
    args = parser.parse_args()
    result = run(
        joint_min_deg=args.joint_min_deg,
        joint_max_deg=args.joint_max_deg,
        steps=args.steps,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    for row in result["trace"]:
        pose = row["selected_static_pose"]
        print(
            f"step {row['step']}: wrist={pose['wrist_pitch_deg']:.0f}°, "
            f"q={pose['foundationpose_opening_angle_deg']:.2f}°, "
            f"mass={np.round(row['mass_mean_kg'], 4)}, "
            f"std={np.round(row['mass_std_kg'], 4)}, "
            f"error={np.round(row['mass_relative_error_evaluator_only'], 4)}"
        )
    if args.live:
        run_live(result, args.cycle_seconds)


if __name__ == "__main__":
    main()
