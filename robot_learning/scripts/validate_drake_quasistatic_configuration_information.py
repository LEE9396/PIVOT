#!/usr/bin/env python3
"""Drake validation of multi-configuration quasi-static density identification.

An articulated object is placed at a sequence of known joint configurations.
At every configuration the joints are held fixed and the robot measures the
gravity wrench at the wrist. Known part geometry makes the measurement linear
in the unknown part-density vector:

    w(q_k) = Y(q_k) rho + epsilon.

This script uses Drake kinematics to build Y, checks its columns against Drake
inverse dynamics, and selects configurations by rank/null-space reduction
before optimizing singular values and posterior information.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import itertools
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from pydrake.all import (
    FixedOffsetFrame,
    MultibodyForces,
    MultibodyPlant,
    PrismaticJoint,
    RevoluteJoint,
    RigidTransform,
    SpatialInertia,
    UnitInertia,
    WeldJoint,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    ROOT
    / "progress/artifacts/2026-07-28/drake_quasistatic_configuration_information"
)

GRAVITY_M_S2 = 9.81
LINK_LENGTHS_M = np.array([0.24, 0.20, 0.17, 0.14])
LINK_WIDTH_M = 0.045
LINK_HEIGHT_M = 0.025
TRUE_DENSITIES_KG_M3 = np.array([1200.0, 2700.0, 7800.0, 1400.0])
PRIOR_MEAN_SCALE = np.array([0.78, 1.22, 0.74, 1.25])
PRIOR_STD_FRACTION = 0.50
FORCE_NOISE_STD_N = 0.15
TORQUE_NOISE_STD_NM = 0.004
ANGLE_GRID_DEG = (-120, -90, -60, -30, 0, 30, 60, 90, 120)
MONTE_CARLO_TRIALS = 1000


@dataclass(frozen=True)
class ChainModel:
    plant: MultibodyPlant
    root_translation: PrismaticJoint
    root_pitch: RevoluteJoint
    internal_joints: tuple[RevoluteJoint, ...]
    link_bodies: tuple
    volumes_m3: np.ndarray
    densities_kg_m3: np.ndarray


@dataclass(frozen=True)
class InformationDiagnostic:
    rank: int
    nullity: int
    singular_values: tuple[float, ...]
    condition_number: float
    information_gain_nats: float
    worst_normalized_posterior_std: float
    nullspace_basis: tuple[tuple[float, ...], ...]


def _box_inertia(
    mass_kg: float,
    length_m: float,
    *,
    com_x_m: float,
) -> SpatialInertia:
    central = mass_kg * UnitInertia.SolidBox(
        length_m,
        LINK_WIDTH_M,
        LINK_HEIGHT_M,
    )
    return SpatialInertia.MakeFromCentralInertia(
        mass_kg,
        np.array([com_x_m, 0.0, 0.0]),
        central,
    )


def build_chain(
    part_count: int,
    densities_kg_m3: np.ndarray | None = None,
) -> ChainModel:
    if part_count not in (2, 3, 4):
        raise ValueError("part_count must be 2, 3, or 4")
    lengths = LINK_LENGTHS_M[:part_count]
    volumes = lengths * LINK_WIDTH_M * LINK_HEIGHT_M
    densities = (
        TRUE_DENSITIES_KG_M3[:part_count].copy()
        if densities_kg_m3 is None
        else np.asarray(densities_kg_m3, dtype=float)
    )
    if densities.shape != (part_count,) or np.any(densities <= 0.0):
        raise ValueError("densities must be positive and cover every link")

    plant = MultibodyPlant(time_step=0.0)
    carrier = plant.AddRigidBody(
        "vertical_carrier",
        _box_inertia(0.05, 0.01, com_x_m=0.0),
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
        _box_inertia(0.05, 0.01, com_x_m=0.0),
    )
    root_pitch = plant.AddJoint(
        RevoluteJoint(
            "wrist_pitch",
            carrier.body_frame(),
            wrist.body_frame(),
            [0.0, 1.0, 0.0],
        )
    )

    bodies = []
    joints = []
    previous_body = wrist
    for index, (length, volume, density) in enumerate(
        zip(lengths, volumes, densities, strict=True)
    ):
        mass = float(volume * density)
        body = plant.AddRigidBody(
            f"part_{index + 1}",
            _box_inertia(mass, float(length), com_x_m=float(length / 2.0)),
        )
        bodies.append(body)
        if index == 0:
            plant.AddJoint(
                WeldJoint(
                    "wrist_to_part_1",
                    wrist.body_frame(),
                    body.body_frame(),
                    RigidTransform(),
                )
            )
        else:
            parent_end = plant.AddFrame(
                FixedOffsetFrame(
                    f"part_{index}_distal",
                    previous_body.body_frame(),
                    RigidTransform([float(lengths[index - 1]), 0.0, 0.0]),
                )
            )
            joint = plant.AddJoint(
                RevoluteJoint(
                    f"joint_{index}",
                    parent_end,
                    body.body_frame(),
                    [0.0, 1.0, 0.0],
                )
            )
            joints.append(joint)
        previous_body = body
    plant.Finalize()
    return ChainModel(
        plant=plant,
        root_translation=root_translation,
        root_pitch=root_pitch,
        internal_joints=tuple(joints),
        link_bodies=tuple(bodies),
        volumes_m3=volumes,
        densities_kg_m3=densities,
    )


def set_configuration(
    model: ChainModel,
    configuration_deg: Iterable[float],
):
    values = tuple(float(value) for value in configuration_deg)
    if len(values) != len(model.internal_joints):
        raise ValueError("configuration has the wrong number of joint angles")
    context = model.plant.CreateDefaultContext()
    model.root_translation.set_translation(context, 0.0)
    model.root_pitch.set_angle(context, 0.0)
    for joint, angle_deg in zip(
        model.internal_joints,
        values,
        strict=True,
    ):
        joint.set_angle(context, math.radians(angle_deg))
    model.plant.SetVelocities(context, np.zeros(model.plant.num_velocities()))
    return context


def part_com_positions(
    model: ChainModel,
    configuration_deg: Iterable[float],
) -> np.ndarray:
    context = set_configuration(model, configuration_deg)
    positions = []
    for body in model.link_bodies:
        pose = model.plant.EvalBodyPoseInWorld(context, body)
        positions.append(
            pose.multiply(
                np.asarray(body.default_spatial_inertia().get_com())
            )
        )
    return np.asarray(positions)


def configuration_regressor(
    model: ChainModel,
    configuration_deg: Iterable[float],
) -> np.ndarray:
    """Return [vertical force, wrist pitch torque] per unit part density."""
    positions = part_com_positions(model, configuration_deg)
    vertical_force = GRAVITY_M_S2 * model.volumes_m3
    pitch_torque = -positions[:, 0] * vertical_force
    return np.vstack((vertical_force, pitch_torque))


def stacked_regressor(
    regressors: dict[tuple[int, ...], np.ndarray],
    configurations: Iterable[tuple[int, ...]],
) -> np.ndarray:
    values = tuple(configurations)
    if not values:
        return np.empty((0, next(iter(regressors.values())).shape[1]))
    return np.vstack([regressors[value] for value in values])


def _whitened_regressor(
    regressor: np.ndarray,
    prior_std_kg_m3: np.ndarray,
) -> np.ndarray:
    row_scale = np.tile(
        np.array([1.0 / FORCE_NOISE_STD_N, 1.0 / TORQUE_NOISE_STD_NM]),
        regressor.shape[0] // 2,
    )
    return (
        row_scale[:, None]
        * regressor
        * prior_std_kg_m3[None, :]
    )


def information_diagnostic(
    regressor: np.ndarray,
    prior_std_kg_m3: np.ndarray,
) -> InformationDiagnostic:
    part_count = prior_std_kg_m3.size
    whitened = _whitened_regressor(regressor, prior_std_kg_m3)
    _, singular_values, vh = np.linalg.svd(whitened, full_matrices=True)
    tolerance = (
        0.0
        if singular_values.size == 0
        else max(whitened.shape)
        * np.finfo(float).eps
        * singular_values[0]
        * 100.0
    )
    rank = int(np.sum(singular_values > tolerance))
    nullity = part_count - rank
    positive = singular_values[:rank]
    condition = (
        math.inf
        if rank < part_count or rank == 0
        else float(positive[0] / positive[-1])
    )
    fisher_standardized = whitened.T @ whitened
    posterior_standardized = np.linalg.inv(
        np.eye(part_count) + fisher_standardized
    )
    sign, logdet = np.linalg.slogdet(
        np.eye(part_count) + fisher_standardized
    )
    if sign <= 0:
        raise FloatingPointError("standardized information is not positive")
    nullspace = vh[rank:, :]
    return InformationDiagnostic(
        rank=rank,
        nullity=nullity,
        singular_values=tuple(float(value) for value in singular_values),
        condition_number=condition,
        information_gain_nats=float(0.5 * logdet),
        worst_normalized_posterior_std=float(
            np.max(np.sqrt(np.diag(posterior_standardized)))
        ),
        nullspace_basis=tuple(
            tuple(float(value) for value in row) for row in nullspace
        ),
    )


def constant_bias_nuisance_diagnostic(
    regressor: np.ndarray,
    prior_std_kg_m3: np.ndarray,
) -> dict[str, object]:
    """Project out one constant force and torque bias shared by all holds."""
    configuration_count = regressor.shape[0] // 2
    row_scale = np.tile(
        np.array([1.0 / FORCE_NOISE_STD_N, 1.0 / TORQUE_NOISE_STD_NM]),
        configuration_count,
    )
    whitened_y = (
        row_scale[:, None]
        * regressor
        * prior_std_kg_m3[None, :]
    )
    bias = np.tile(np.eye(2), (configuration_count, 1))
    whitened_bias = row_scale[:, None] * bias
    projector = (
        np.eye(regressor.shape[0])
        - whitened_bias @ np.linalg.pinv(whitened_bias)
    )
    effective = projector @ whitened_y
    _, singular_values, vh = np.linalg.svd(effective, full_matrices=True)
    tolerance = (
        0.0
        if singular_values.size == 0
        else max(effective.shape)
        * np.finfo(float).eps
        * max(float(singular_values[0]), 1.0)
        * 100.0
    )
    rank = int(np.sum(singular_values > tolerance))
    return {
        "rank": rank,
        "nullity": int(prior_std_kg_m3.size - rank),
        "singular_values": singular_values.tolist(),
        "nullspace_basis": vh[rank:, :].tolist(),
        "interpretation": (
            "One constant averaged force bias and one constant averaged "
            "pitch-torque bias are unknown across all configurations."
        ),
    }


def candidate_configurations(part_count: int) -> tuple[tuple[int, ...], ...]:
    return tuple(
        itertools.product(ANGLE_GRID_DEG, repeat=part_count - 1)
    )


def select_configurations(
    regressors: dict[tuple[int, ...], np.ndarray],
    prior_std_kg_m3: np.ndarray,
    *,
    count: int,
) -> tuple[tuple[int, ...], ...]:
    selected: list[tuple[int, ...]] = []
    for _ in range(count):
        best_configuration = None
        best_objective = None
        for candidate in regressors:
            if candidate in selected:
                continue
            diagnostic = information_diagnostic(
                stacked_regressor(
                    regressors,
                    (*selected, candidate),
                ),
                prior_std_kg_m3,
            )
            positive = diagnostic.singular_values[: diagnostic.rank]
            smallest_positive = positive[-1] if positive else 0.0
            objective = (
                diagnostic.rank,
                smallest_positive,
                diagnostic.information_gain_nats,
            )
            if best_objective is None or objective > best_objective:
                best_objective = objective
                best_configuration = candidate
        if best_configuration is None:
            raise RuntimeError("configuration selection failed")
        selected.append(best_configuration)
    return tuple(selected)


def static_root_wrench(
    model: ChainModel,
    configuration_deg: Iterable[float],
) -> np.ndarray:
    context = set_configuration(model, configuration_deg)
    applied = MultibodyForces(model.plant)
    model.plant.CalcForceElementsContribution(context, applied)
    generalized = model.plant.CalcInverseDynamics(
        context,
        np.zeros(model.plant.num_velocities()),
        applied,
    )
    return np.array(
        [
            generalized[model.root_translation.velocity_start()],
            generalized[model.root_pitch.velocity_start()],
        ]
    )


def validate_regressor_with_inverse_dynamics(
    part_count: int,
    configurations: Iterable[tuple[int, ...]],
) -> dict[str, float]:
    nominal_density = TRUE_DENSITIES_KG_M3[:part_count]
    nominal_model = build_chain(part_count, nominal_density)
    maximum_absolute = 0.0
    maximum_relative = 0.0
    perturbation = 10.0
    for configuration in configurations:
        analytic = configuration_regressor(nominal_model, configuration)
        baseline = static_root_wrench(nominal_model, configuration)
        numerical = np.zeros_like(analytic)
        for part in range(part_count):
            perturbed_density = nominal_density.copy()
            perturbed_density[part] += perturbation
            perturbed_model = build_chain(part_count, perturbed_density)
            numerical[:, part] = (
                static_root_wrench(perturbed_model, configuration) - baseline
            ) / perturbation
        error = numerical - analytic
        maximum_absolute = max(maximum_absolute, float(np.max(np.abs(error))))
        denominator = np.maximum(np.abs(analytic), 1.0e-12)
        maximum_relative = max(
            maximum_relative,
            float(np.max(np.abs(error) / denominator)),
        )
    return {
        "max_absolute_column_error": maximum_absolute,
        "max_relative_column_error": maximum_relative,
    }


def map_density_trials(
    regressor: np.ndarray,
    true_density: np.ndarray,
    prior_mean: np.ndarray,
    prior_std: np.ndarray,
    *,
    trials: int,
    seed: int,
) -> dict[str, object]:
    row_std = np.tile(
        np.array([FORCE_NOISE_STD_N, TORQUE_NOISE_STD_NM]),
        regressor.shape[0] // 2,
    )
    inverse_noise = np.diag(1.0 / np.square(row_std))
    prior_precision = np.diag(1.0 / np.square(prior_std))
    precision = prior_precision + regressor.T @ inverse_noise @ regressor
    covariance = np.linalg.inv(precision)
    deterministic_rhs = prior_precision @ prior_mean
    rng = np.random.default_rng(seed)
    estimates = np.empty((trials, true_density.size))
    noiseless = regressor @ true_density
    for trial in range(trials):
        measurement = noiseless + rng.normal(0.0, row_std)
        rhs = (
            deterministic_rhs
            + regressor.T @ inverse_noise @ measurement
        )
        estimates[trial] = np.linalg.solve(precision, rhs)
    relative_error = (
        estimates - true_density[None, :]
    ) / true_density[None, :]
    return {
        "density_mean_kg_m3": np.mean(estimates, axis=0).tolist(),
        "density_std_kg_m3": np.std(estimates, axis=0, ddof=1).tolist(),
        "relative_rmse": np.sqrt(
            np.mean(np.square(relative_error), axis=0)
        ).tolist(),
        "worst_relative_rmse": float(
            np.max(
                np.sqrt(np.mean(np.square(relative_error), axis=0))
            )
        ),
        "posterior_std_kg_m3": np.sqrt(np.diag(covariance)).tolist(),
    }


def _configuration_record(
    values: tuple[int, ...],
) -> dict[str, object]:
    return {
        "joint_angles_deg": list(values),
        "label": "(" + ", ".join(f"{value}°" for value in values) + ")",
    }


def run_validation(
    *,
    output_dir: Path,
    monte_carlo_trials: int,
    seed: int,
) -> dict[str, object]:
    results = []
    for part_count in (2, 3, 4):
        model = build_chain(part_count)
        true_density = TRUE_DENSITIES_KG_M3[:part_count]
        prior_mean = true_density * PRIOR_MEAN_SCALE[:part_count]
        prior_std = true_density * PRIOR_STD_FRACTION
        configurations = candidate_configurations(part_count)
        regressors = {
            configuration: configuration_regressor(model, configuration)
            for configuration in configurations
        }
        required_count = part_count - 1
        selected = select_configurations(
            regressors,
            prior_std,
            count=required_count,
        )
        repeated = tuple(selected[0] for _ in range(required_count))
        selected_prefixes = []
        for count in range(1, required_count + 1):
            prefix = selected[:count]
            diagnostic = information_diagnostic(
                stacked_regressor(regressors, prefix),
                prior_std,
            )
            selected_prefixes.append(
                {
                    "measurement_count": count,
                    "configurations": [
                        _configuration_record(value) for value in prefix
                    ],
                    "diagnostic": asdict(diagnostic),
                }
            )
        optimized_y = stacked_regressor(regressors, selected)
        repeated_y = stacked_regressor(regressors, repeated)
        optimized_diagnostic = information_diagnostic(
            optimized_y,
            prior_std,
        )
        repeated_diagnostic = information_diagnostic(
            repeated_y,
            prior_std,
        )
        inverse_dynamics = validate_regressor_with_inverse_dynamics(
            part_count,
            selected,
        )
        results.append(
            {
                "part_count": part_count,
                "part_names": [
                    f"part_{index + 1}" for index in range(part_count)
                ],
                "lengths_m": LINK_LENGTHS_M[:part_count].tolist(),
                "volumes_m3": model.volumes_m3.tolist(),
                "true_densities_kg_m3": true_density.tolist(),
                "true_masses_kg": (
                    model.volumes_m3 * true_density
                ).tolist(),
                "minimum_planar_configuration_count": required_count,
                "selected_configurations": [
                    _configuration_record(value) for value in selected
                ],
                "selection_trace": selected_prefixes,
                "optimized_diagnostic": asdict(optimized_diagnostic),
                "unknown_constant_bias_diagnostic": (
                    constant_bias_nuisance_diagnostic(
                        optimized_y,
                        prior_std,
                    )
                ),
                "unknown_constant_bias_all_grid_diagnostic": (
                    constant_bias_nuisance_diagnostic(
                        stacked_regressor(regressors, configurations),
                        prior_std,
                    )
                ),
                "repeated_same_angle_diagnostic": asdict(
                    repeated_diagnostic
                ),
                "optimized_density_trials": map_density_trials(
                    optimized_y,
                    true_density,
                    prior_mean,
                    prior_std,
                    trials=monte_carlo_trials,
                    seed=seed + part_count,
                ),
                "repeated_same_angle_density_trials": map_density_trials(
                    repeated_y,
                    true_density,
                    prior_mean,
                    prior_std,
                    trials=monte_carlo_trials,
                    seed=seed + part_count,
                ),
                "drake_inverse_dynamics_validation": inverse_dynamics,
            }
        )
    payload = {
        "scope": (
            "Planar 2/3/4-link object; multiple known joint configurations; "
            "quasi-static gravity wrench at the robot wrist"
        ),
        "measurement_model": "stack_k w(q_k) = stack_k Y(q_k) rho + noise",
        "selection_rule": (
            "lexicographic: maximize rank, then smallest positive singular "
            "value, then prior-conditioned information gain"
        ),
        "angle_grid_deg": list(ANGLE_GRID_DEG),
        "sensor_noise_after_hold_averaging": {
            "force_std_n": FORCE_NOISE_STD_N,
            "torque_std_nm": TORQUE_NOISE_STD_NM,
        },
        "assumptions": [
            "Part geometry, volume, transforms, and joint angles are known.",
            "Every joint is mechanically held during each static measurement.",
            "Known gripper, fixture, cable, and sensor tare wrench is removed.",
            "The chain is planar and the wrist orientation is fixed.",
            "Part density is uniform within each known part mesh.",
        ],
        "results": results,
        "limitations": [
            "This is an ideal quasi-static wrist-wrench study, not contact/grasp validation.",
            "Quasi-static gravity identifies density-derived part mass/first moments, not an independent rotational-inertia tensor.",
            "Joint holding torque and cable forces are not yet injected as nuisance parameters.",
            "The discrete angle optimum depends on the chosen link geometry and angle limits.",
            "Physical results require calibrated F/T covariance and measured joint repeatability.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "drake_quasistatic_configuration_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_figures(output_dir, payload)
    _write_report(output_dir, payload)
    return payload


def _link_points(
    lengths: np.ndarray,
    configuration_deg: tuple[int, ...],
) -> np.ndarray:
    points = [np.array([0.0, 0.0])]
    orientation = 0.0
    for index, length in enumerate(lengths):
        if index > 0:
            orientation += math.radians(configuration_deg[index - 1])
        step = np.array(
            [length * math.cos(orientation), -length * math.sin(orientation)]
        )
        points.append(points[-1] + step)
    return np.asarray(points)


def _write_figures(output_dir: Path, payload: dict[str, object]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = payload["results"]
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.5))
    for result in results:
        counts = [
            row["measurement_count"] for row in result["selection_trace"]
        ]
        nullity = [
            row["diagnostic"]["nullity"]
            for row in result["selection_trace"]
        ]
        axes[0].plot(
            counts,
            nullity,
            "o-",
            label=f"{result['part_count']}-link",
        )
    axes[0].set_xticks([1, 2, 3])
    axes[0].set_xlabel("number of distinct configurations")
    axes[0].set_ylabel("null-space dimension")
    axes[0].set_title("Each new angle removes ambiguity")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    labels = [f"{result['part_count']}-link" for result in results]
    optimized_condition = [
        result["optimized_diagnostic"]["condition_number"]
        for result in results
    ]
    optimized_rmse = [
        100.0
        * result["optimized_density_trials"]["worst_relative_rmse"]
        for result in results
    ]
    repeated_rmse = [
        100.0
        * result["repeated_same_angle_density_trials"]["worst_relative_rmse"]
        for result in results
    ]
    x = np.arange(len(results))
    axes[1].bar(x - 0.18, optimized_rmse, 0.36, label="selected angles")
    axes[1].bar(x + 0.18, repeated_rmse, 0.36, label="same angle repeated")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("worst part density RMSE [%]")
    axes[1].set_title("Rank diversity matters more than repetition")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()

    axes[2].bar(x, optimized_condition, color="#4c78a8")
    axes[2].set_xticks(x, labels)
    axes[2].set_yscale("log")
    axes[2].set_ylabel("full-rank condition number [log]")
    axes[2].set_title("Condition after minimum angle set")
    axes[2].grid(axis="y", alpha=0.25)
    figure.suptitle(
        "Drake quasi-static configuration information",
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout()
    figure.savefig(
        output_dir / "drake_quasistatic_information_summary.png",
        dpi=180,
    )
    plt.close(figure)

    figure, axes = plt.subplots(3, 3, figsize=(15.0, 9.0))
    for row, result in enumerate(results):
        lengths = np.asarray(result["lengths_m"])
        selected = result["selected_configurations"]
        for column in range(3):
            axis = axes[row, column]
            if column < len(selected):
                angles = tuple(selected[column]["joint_angles_deg"])
                points = _link_points(lengths, angles)
                colors = ("#4c78a8", "#f58518", "#54a24b", "#e45756")
                for part_index in range(len(lengths)):
                    segment = points[part_index : part_index + 2]
                    axis.plot(
                        segment[:, 0],
                        segment[:, 1],
                        "o-",
                        linewidth=5,
                        color=colors[part_index],
                    )
                    midpoint = np.mean(segment, axis=0)
                    axis.text(
                        midpoint[0],
                        midpoint[1] + 0.035,
                        f"P{part_index + 1}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        color=colors[part_index],
                        fontweight="bold",
                    )
                axis.set_title(
                    f"{result['part_count']}-link · Config {column + 1}: ("
                    + ", ".join(f"{value}°" for value in angles)
                    + ")"
                )
            else:
                axis.text(
                    0.5,
                    0.5,
                    "not required",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                    color="#777777",
                )
                axis.set_title(
                    f"{result['part_count']}-link · configuration {column + 1}"
                )
            axis.axhline(0.0, color="#bbbbbb", linewidth=0.8)
            axis.axvline(0.0, color="#bbbbbb", linewidth=0.8)
            axis.set_aspect("equal", adjustable="box")
            axis.set_xlim(-0.30, 0.70)
            axis.set_ylim(-0.55, 0.55)
            axis.grid(alpha=0.20)
            axis.set_xlabel("wrist x [m]")
            axis.set_ylabel("wrist z [m]")
    figure.suptitle(
        "Sequentially selected configurations that close the null space",
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout()
    figure.savefig(
        output_dir / "drake_selected_link_configurations.png",
        dpi=180,
    )
    plt.close(figure)


def _format_angles(result: dict[str, object]) -> str:
    return " → ".join(
        configuration["label"]
        for configuration in result["selected_configurations"]
    )


def _write_report(
    output_dir: Path,
    payload: dict[str, object],
) -> None:
    lines = [
        "# Drake multi-configuration quasi-static information validation",
        "",
        "## 결론",
        "",
        "관절각을 여러 번 바꾸고 각 configuration에서 준정적 wrist F/T를 "
        "누적하면 part density 식별 문제는 stacked linear system으로 정리된다.",
        "",
        r"$$W_K=Y_K\rho+\epsilon,\qquad "
        r"Y_K=[Y(q_1)^T,\ldots,Y(q_K)^T]^T.$$",
        "",
        "모든 part density가 구분되려면 `rank(Y_K)=P`여야 한다. "
        "`null(Y_K)`의 각 vector는 F/T를 바꾸지 않고 함께 변화할 수 있는 "
        "density 조합이다.",
        "",
        "![Summary](drake_quasistatic_information_summary.png)",
        "",
        "![Selected configurations](drake_selected_link_configurations.png)",
        "",
        "## 수치 결과",
        "",
        "| Object | 최소 configuration 수 | 선택된 순서 | calibrated rank/nullity | "
        "unknown-bias rank/nullity | all-grid unknown-bias max rank | condition | 선택각 RMSE | 같은 각도 반복 RMSE | Drake column error |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in payload["results"]:
        optimized = result["optimized_diagnostic"]
        validation = result["drake_inverse_dynamics_validation"]
        lines.append(
            f"| {result['part_count']}-link | "
            f"{result['minimum_planar_configuration_count']} | "
            f"{_format_angles(result)} | "
            f"{optimized['rank']}/{optimized['nullity']} | "
            f"{result['unknown_constant_bias_diagnostic']['rank']}/"
            f"{result['unknown_constant_bias_diagnostic']['nullity']} | "
            f"{result['unknown_constant_bias_all_grid_diagnostic']['rank']}"
            f"/{result['part_count']} | "
            f"{optimized['condition_number']:.2f} | "
            f"{100*result['optimized_density_trials']['worst_relative_rmse']:.2f}% | "
            f"{100*result['repeated_same_angle_density_trials']['worst_relative_rmse']:.2f}% | "
            f"{validation['max_relative_column_error']:.2e} |"
        )
    lines.extend(
        [
            "",
            "## Null-space가 실제로 뜻하는 것",
            "",
            "아래 vector는 prior standard deviation으로 정규화한 density 변화 "
            "`δρ` 방향이다. 이 방향으로 density를 함께 바꾸면 현재까지의 F/T가 "
            "변하지 않는다. 부호보다 중요한 것은 vector가 남아 있다는 사실이다.",
            "",
            "| Object | 누적 configuration | rank/nullity | 남은 normalized null vector |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for result in payload["results"]:
        for trace in result["selection_trace"]:
            diagnostic = trace["diagnostic"]
            basis = diagnostic["nullspace_basis"]
            if basis:
                basis_text = "; ".join(
                    "[" + ", ".join(f"{value:+.3f}" for value in row) + "]"
                    for row in basis
                )
            else:
                basis_text = "none — all density directions observable"
            lines.append(
                f"| {result['part_count']}-link | "
                f"{trace['measurement_count']} | "
                f"{diagnostic['rank']}/{diagnostic['nullity']} | "
                f"`{basis_text}` |"
            )
    lines.extend(
        [
            "",
            "## 왜 최소 각도 수가 P-1인가",
            "",
            "평면 물체를 같은 wrist orientation으로 준정적 측정하면 각 "
            "configuration에서 force row는 항상 total mass 하나이며, 새로 바뀌는 "
            "것은 gravity moment-arm row 하나다. K개 configuration을 쌓아도 "
            "`rank(Y_K) ≤ K+1`이므로 P개 part에는 최소 `K=P-1`개의 서로 다른 "
            "configuration이 필요하다.",
            "",
            "같은 각도를 여러 번 반복하면 noise는 줄지만 row space가 바뀌지 않아 "
            "null-space dimension은 줄지 않는다.",
            "",
            "각 configuration 사이의 관절 이동 구간은 추정에 사용하지 않는다. "
            "목표 각도 도착, 진동 정착, 외부 지지 제거가 확인된 hold 구간의 "
            "평균 wrench만 한 measurement block으로 누적한다.",
            "",
            "## 다음 각도 선택 formulation",
            "",
            "1. 아직 rank deficient이면 rank 증가를 최우선으로 한다.",
            "2. 같은 rank 후보 중 smallest nonzero singular value가 큰 각도를 고른다.",
            "3. full rank 이후에는 condition number와 posterior information gain을 "
            "함께 최적화한다.",
            "4. joint limit, holding stability, robot collision, F/T range를 만족하지 "
            "않는 후보는 선형대수 최적화 전에 제거한다.",
            "",
            "좋은 configuration 집합은 각 link의 projected CoM moment arm이 서로 "
            "다를 뿐 아니라, configuration 사이에서 그 차이의 pattern도 바뀐다. "
            "반대로 모든 link가 비슷한 projected lever arm을 갖거나 같은 angle을 "
            "반복하면 sensitivity column이 평행해져 condition number가 커진다.",
            "",
            "Bayesian form은 다음과 같다.",
            "",
            r"$$\Lambda_{k+1}=\Lambda_k+Y(q)^TR^{-1}Y(q),$$",
            r"$$q_{k+1}=\operatorname*{lex\,max}_{q\in\mathcal Q_{safe}} "
            r"\left(\Delta\operatorname{rank},\;\sigma_{min}^+,\;"
            r"\tfrac12\log\det(\Lambda_{k+1}\Lambda_k^{-1})\right).$$",
            "",
            "## 반드시 포함할 nuisance 처리",
            "",
            "이 결과는 sensor tare와 known fixture wrench를 제거했다. 실제로 bias가 "
            "남으면 `w=Yρ+Bβ+ε`로 확장하고, nuisance column B를 제거한 "
            "`P_B^⊥Y`의 rank와 null space를 평가해야 한다. 특히 모든 각도에서 "
            "동일한 total-force row는 force bias와 겹칠 수 있으므로 F/T zeroing과 "
            "known-mass calibration이 필수다.",
            "",
            r"$$P_B^\perp=I-B(B^TR^{-1}B)^{-1}B^TR^{-1},\qquad "
            r"\operatorname{rank}(P_B^\perp Y_K)=P.$$",
            "",
            "현재 minimum configuration set에서 constant force/torque bias를 "
            "투영하면 모든 object에 null direction이 2개 남았다. 전체 angle grid를 "
            "모두 쌓아도 wrist에 고정된 Part 1의 gravity wrench는 configuration에 "
            "따라 변하지 않아 최대 rank가 `P-1`이었다. 즉, angle 개수를 계속 "
            "늘리는 것으로는 이 문제를 해결할 수 없다. calibrated absolute tare, "
            "wrist orientation 변경, 또는 알려진 translational acceleration 중 "
            "하나가 반드시 필요하다.",
            "",
            "## 한계",
            "",
        ]
    )
    lines.extend(f"- {value}" for value in payload["limitations"])
    (output_dir / "DRAKE_QUASISTATIC_CONFIGURATION_RESULT.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--monte-carlo-trials",
        type=int,
        default=MONTE_CARLO_TRIALS,
    )
    parser.add_argument("--seed", type=int, default=20260728)
    args = parser.parse_args()
    payload = run_validation(
        output_dir=args.output_dir,
        monte_carlo_trials=args.monte_carlo_trials,
        seed=args.seed,
    )
    for result in payload["results"]:
        diagnostic = result["optimized_diagnostic"]
        print(
            f"{result['part_count']}-link: "
            f"configs={_format_angles(result)} "
            f"rank={diagnostic['rank']} "
            f"nullity={diagnostic['nullity']} "
            f"condition={diagnostic['condition_number']:.3f}"
        )


if __name__ == "__main__":
    main()
