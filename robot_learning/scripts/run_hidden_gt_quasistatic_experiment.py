#!/usr/bin/env python3
"""Hidden-GT quasi-static articulated-object experiment.

This runner enforces the same information boundary as the planned real test:

* The simulated world alone owns the true per-part density and sensor bias.
* The robot estimator receives only known part geometry, a VLM-like prior,
  a tare observation, camera-estimated joint angles, and six-axis F/T samples.
* A separate evaluator compares the estimates with GT after every accepted
  held configuration.

The object remains a controlled synthetic calibration object.  The resulting
trace validates estimator wiring and uncertainty calibration in simulation; it
does not predict the final hardware error until real meshes and calibrated
sensor statistics replace the values below.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from functools import lru_cache
import json
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

import validate_drake_quasistatic_configuration_information as base  # noqa: E402


DEFAULT_OUTPUT = (
    ROOT
    / "progress/artifacts/2026-07-28/"
    "hidden_gt_quasistatic_experiment"
)
GRAVITY_WORLD_M_S2 = np.array([0.0, 0.0, -9.81])
TRUE_DENSITIES_KG_M3 = np.array([1180.0, 2760.0, 7420.0, 1560.0])
PRIOR_MEAN_KG_M3 = np.array([900.0, 2250.0, 5200.0, 2050.0])
PRIOR_STD_KG_M3 = np.array([500.0, 950.0, 2400.0, 850.0])
FT_SAMPLE_STD = np.array([0.30, 0.30, 0.30, 0.030, 0.030, 0.030])
HOLD_SAMPLE_COUNT = 100
TARE_SAMPLE_COUNT = 500
BIAS_RUN_STD = np.array([0.40, 0.40, 0.40, 0.040, 0.040, 0.040])
CAMERA_STD_DEG = {
    "D435i-left": 0.80,
    "D435i-right": 0.80,
    "D456-top": 0.50,
}
CAMERA_DROPOUT_PROBABILITY = 0.03
JOINT_SETTING_STD_DEG = 0.18
HOLD_DRIFT_STD_DEG = 0.22
HARD_DRIFT_GATE_DEG = 1.50
MINIMUM_VISIBLE_CAMERAS = 2


@dataclass(frozen=True)
class EstimatorInput:
    part_count: int
    volumes_m3: tuple[float, ...]
    prior_mean_kg_m3: tuple[float, ...]
    prior_covariance: tuple[tuple[float, ...], ...]
    bias_prior_covariance: tuple[tuple[float, ...], ...]
    ft_hold_covariance: tuple[tuple[float, ...], ...]
    ft_tare_covariance: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class HiddenWorld:
    true_density_kg_m3: tuple[float, ...]
    sensor_bias: tuple[float, ...]


@dataclass(frozen=True)
class CameraObservation:
    fused_joint_angles_deg: tuple[float, ...]
    covariance_rad2: tuple[tuple[float, ...], ...]
    visible_cameras: tuple[str, ...]


@dataclass(frozen=True)
class HoldObservation:
    state_index: int
    commanded_joint_angles_deg: tuple[float, ...]
    camera: CameraObservation
    raw_wrench: tuple[float, ...]
    maximum_joint_drift_deg: float
    accepted: bool
    rejection_reason: str | None


@dataclass
class EstimatorState:
    mean: np.ndarray
    covariance: np.ndarray


@lru_cache(maxsize=3)
def _chain_model(part_count: int):
    return base.build_chain(part_count)


@lru_cache(maxsize=3)
def make_estimator_input(part_count: int) -> EstimatorInput:
    model = _chain_model(part_count)
    density_cov = np.diag(PRIOR_STD_KG_M3[:part_count] ** 2)
    return EstimatorInput(
        part_count=part_count,
        volumes_m3=tuple(float(value) for value in model.volumes_m3),
        prior_mean_kg_m3=tuple(
            float(value) for value in PRIOR_MEAN_KG_M3[:part_count]
        ),
        prior_covariance=tuple(
            tuple(float(value) for value in row) for row in density_cov
        ),
        bias_prior_covariance=tuple(
            tuple(float(value) for value in row)
            for row in np.diag(BIAS_RUN_STD**2)
        ),
        ft_hold_covariance=tuple(
            tuple(float(value) for value in row)
            for row in np.diag(FT_SAMPLE_STD**2 / HOLD_SAMPLE_COUNT)
        ),
        ft_tare_covariance=tuple(
            tuple(float(value) for value in row)
            for row in np.diag(FT_SAMPLE_STD**2 / TARE_SAMPLE_COUNT)
        ),
    )


def six_axis_regressor(
    part_count: int,
    configuration_deg,
) -> np.ndarray:
    model = _chain_model(part_count)
    positions = base.part_com_positions(model, configuration_deg)
    regressor = np.zeros((6, part_count))
    for part in range(part_count):
        volume = model.volumes_m3[part]
        force_column = volume * GRAVITY_WORLD_M_S2
        regressor[:3, part] = force_column
        regressor[3:, part] = np.cross(positions[part], force_column)
    return regressor


def _fuse_camera_observation(
    actual_configuration_deg: np.ndarray,
    rng: np.random.Generator,
) -> CameraObservation | None:
    visible = []
    estimates = []
    precisions = []
    for camera, std_deg in CAMERA_STD_DEG.items():
        if rng.random() < CAMERA_DROPOUT_PROBABILITY:
            continue
        visible.append(camera)
        estimates.append(
            actual_configuration_deg
            + rng.normal(0.0, std_deg, actual_configuration_deg.size)
        )
        precisions.append(1.0 / math.radians(std_deg) ** 2)
    if len(visible) < MINIMUM_VISIBLE_CAMERAS:
        return None
    precision_sum = float(sum(precisions))
    weighted = sum(
        precision * np.deg2rad(estimate)
        for precision, estimate in zip(precisions, estimates, strict=True)
    ) / precision_sum
    covariance = np.eye(actual_configuration_deg.size) / precision_sum
    return CameraObservation(
        fused_joint_angles_deg=tuple(
            float(value) for value in np.rad2deg(weighted)
        ),
        covariance_rad2=tuple(
            tuple(float(value) for value in row) for row in covariance
        ),
        visible_cameras=tuple(visible),
    )


def simulate_tare(
    hidden: HiddenWorld,
    estimator_input: EstimatorInput,
    rng: np.random.Generator,
) -> np.ndarray:
    covariance = np.asarray(estimator_input.ft_tare_covariance)
    return np.asarray(hidden.sensor_bias) + rng.multivariate_normal(
        np.zeros(6),
        covariance,
    )


def simulate_hold(
    hidden: HiddenWorld,
    estimator_input: EstimatorInput,
    state_index: int,
    commanded_configuration_deg: tuple[int, ...],
    rng: np.random.Generator,
    *,
    forced_drift_deg: float | None = None,
) -> HoldObservation:
    command = np.asarray(commanded_configuration_deg, dtype=float)
    actual = command + rng.normal(
        0.0,
        JOINT_SETTING_STD_DEG,
        command.size,
    )
    camera = _fuse_camera_observation(actual, rng)
    maximum_drift = (
        float(forced_drift_deg)
        if forced_drift_deg is not None
        else float(abs(rng.normal(0.0, HOLD_DRIFT_STD_DEG)))
    )
    regressor = six_axis_regressor(
        estimator_input.part_count,
        actual,
    )
    covariance = np.asarray(estimator_input.ft_hold_covariance)
    wrench = (
        regressor @ np.asarray(hidden.true_density_kg_m3)
        + np.asarray(hidden.sensor_bias)
        + rng.multivariate_normal(np.zeros(6), covariance)
    )
    reason = None
    if camera is None:
        reason = "fewer_than_two_cameras"
    elif maximum_drift > HARD_DRIFT_GATE_DEG:
        reason = "joint_drift_exceeded_hard_gate"
    return HoldObservation(
        state_index=state_index,
        commanded_joint_angles_deg=tuple(
            float(value) for value in command
        ),
        camera=(
            camera
            if camera is not None
            else CameraObservation((), (), ())
        ),
        raw_wrench=tuple(float(value) for value in wrench),
        maximum_joint_drift_deg=maximum_drift,
        accepted=reason is None,
        rejection_reason=reason,
    )


def initialize_estimator(
    estimator_input: EstimatorInput,
) -> EstimatorState:
    p = estimator_input.part_count
    mean = np.concatenate(
        (np.asarray(estimator_input.prior_mean_kg_m3), np.zeros(6))
    )
    covariance = np.zeros((p + 6, p + 6))
    covariance[:p, :p] = np.asarray(estimator_input.prior_covariance)
    covariance[p:, p:] = np.asarray(
        estimator_input.bias_prior_covariance
    )
    return EstimatorState(mean=mean, covariance=covariance)


def _linear_update(
    state: EstimatorState,
    measurement: np.ndarray,
    matrix: np.ndarray,
    noise_covariance: np.ndarray,
) -> EstimatorState:
    innovation = measurement - matrix @ state.mean
    innovation_covariance = (
        matrix @ state.covariance @ matrix.T + noise_covariance
    )
    gain = (
        state.covariance
        @ matrix.T
        @ np.linalg.inv(innovation_covariance)
    )
    mean = state.mean + gain @ innovation
    identity = np.eye(state.mean.size)
    # Joseph form keeps the covariance positive under finite precision.
    residual = identity - gain @ matrix
    covariance = (
        residual @ state.covariance @ residual.T
        + gain @ noise_covariance @ gain.T
    )
    return EstimatorState(mean=mean, covariance=covariance)


def apply_tare(
    state: EstimatorState,
    estimator_input: EstimatorInput,
    tare_observation: np.ndarray,
) -> EstimatorState:
    p = estimator_input.part_count
    matrix = np.hstack((np.zeros((6, p)), np.eye(6)))
    return _linear_update(
        state,
        tare_observation,
        matrix,
        np.asarray(estimator_input.ft_tare_covariance),
    )


def _angle_induced_covariance(
    estimator_input: EstimatorInput,
    density_linearization: np.ndarray,
    camera: CameraObservation,
) -> np.ndarray:
    q = np.asarray(camera.fused_joint_angles_deg)
    if q.size == 0:
        return np.zeros((6, 6))
    step_deg = 0.05
    jacobian = np.zeros((6, q.size))
    for joint in range(q.size):
        plus = q.copy()
        minus = q.copy()
        plus[joint] += step_deg
        minus[joint] -= step_deg
        plus_wrench = (
            six_axis_regressor(estimator_input.part_count, plus)
            @ density_linearization
        )
        minus_wrench = (
            six_axis_regressor(estimator_input.part_count, minus)
            @ density_linearization
        )
        jacobian[:, joint] = (
            plus_wrench - minus_wrench
        ) / math.radians(2.0 * step_deg)
    return (
        jacobian
        @ np.asarray(camera.covariance_rad2)
        @ jacobian.T
    )


def apply_hold(
    state: EstimatorState,
    estimator_input: EstimatorInput,
    observation: HoldObservation,
) -> EstimatorState:
    if not observation.accepted:
        return state
    p = estimator_input.part_count
    regressor = six_axis_regressor(
        p,
        observation.camera.fused_joint_angles_deg,
    )
    matrix = np.hstack((regressor, np.eye(6)))
    effective_covariance = (
        np.asarray(estimator_input.ft_hold_covariance)
        + _angle_induced_covariance(
            estimator_input,
            state.mean[:p],
            observation.camera,
        )
    )
    return _linear_update(
        state,
        np.asarray(observation.raw_wrench),
        matrix,
        effective_covariance,
    )


@lru_cache(maxsize=3)
def select_configurations(
    estimator_input: EstimatorInput,
) -> tuple[tuple[int, ...], ...]:
    """Select states using visible geometry/prior only, never hidden GT."""
    p = estimator_input.part_count
    model = _chain_model(p)
    regressors = {
        q: base.configuration_regressor(model, q)
        for q in base.candidate_configurations(p)
    }
    prior_std = np.sqrt(
        np.diag(np.asarray(estimator_input.prior_covariance))
    )
    return base.select_configurations(
        regressors,
        prior_std,
        count=p - 1,
    )


def _aggregate_properties(
    part_count: int,
    density: np.ndarray,
    configuration_deg: tuple[int, ...],
) -> dict[str, float | list[float]]:
    model = _chain_model(part_count)
    positions = base.part_com_positions(model, configuration_deg)
    masses = model.volumes_m3 * density
    total_mass = float(np.sum(masses))
    com = np.sum(masses[:, None] * positions, axis=0) / total_mass
    lengths = base.LINK_LENGTHS_M[:part_count]
    central_pitch = masses * (
        lengths**2 + base.LINK_HEIGHT_M**2
    ) / 12.0
    offsets = positions - com[None, :]
    pitch_inertia = float(
        np.sum(
            central_pitch
            + masses * (offsets[:, 0] ** 2 + offsets[:, 2] ** 2)
        )
    )
    return {
        "part_mass_kg": masses.tolist(),
        "total_mass_kg": total_mass,
        "com_world_m": com.tolist(),
        "pitch_inertia_com_kg_m2": pitch_inertia,
    }


def evaluate_state(
    estimator_input: EstimatorInput,
    hidden: HiddenWorld,
    state: EstimatorState,
    reference_configuration: tuple[int, ...],
) -> dict[str, object]:
    p = estimator_input.part_count
    estimate = state.mean[:p]
    std = np.sqrt(np.diag(state.covariance)[:p])
    gt = np.asarray(hidden.true_density_kg_m3)
    relative_density_error = np.abs(estimate - gt) / gt
    estimated_properties = _aggregate_properties(
        p,
        estimate,
        reference_configuration,
    )
    gt_properties = _aggregate_properties(
        p,
        gt,
        reference_configuration,
    )
    estimated_mass = np.asarray(estimated_properties["part_mass_kg"])
    gt_mass = np.asarray(gt_properties["part_mass_kg"])
    com_error = np.linalg.norm(
        np.asarray(estimated_properties["com_world_m"])
        - np.asarray(gt_properties["com_world_m"])
    )
    inertia_gt = float(gt_properties["pitch_inertia_com_kg_m2"])
    return {
        "estimated_density_kg_m3": estimate.tolist(),
        "posterior_std_kg_m3": std.tolist(),
        "density_relative_error": relative_density_error.tolist(),
        "density_gt_inside_95pct_interval": (
            np.abs(estimate - gt) <= 1.96 * std
        ).tolist(),
        "estimated_properties": estimated_properties,
        "gt_properties": gt_properties,
        "part_mass_relative_error": (
            np.abs(estimated_mass - gt_mass) / gt_mass
        ).tolist(),
        "total_mass_relative_error": abs(
            float(estimated_properties["total_mass_kg"])
            - float(gt_properties["total_mass_kg"])
        )
        / float(gt_properties["total_mass_kg"]),
        "com_error_m": float(com_error),
        "pitch_inertia_relative_error": abs(
            float(estimated_properties["pitch_inertia_com_kg_m2"])
            - inertia_gt
        )
        / inertia_gt,
    }


def run_single_experiment(
    part_count: int,
    *,
    seed: int,
    forced_unsafe_state: int | None = None,
) -> dict[str, object]:
    estimator_input = make_estimator_input(part_count)
    rng = np.random.default_rng(seed)
    hidden = HiddenWorld(
        true_density_kg_m3=tuple(
            float(value) for value in TRUE_DENSITIES_KG_M3[:part_count]
        ),
        sensor_bias=tuple(
            float(value)
            for value in rng.normal(0.0, BIAS_RUN_STD)
        ),
    )
    configurations = select_configurations(estimator_input)
    state = initialize_estimator(estimator_input)
    tare = simulate_tare(hidden, estimator_input, rng)
    state = apply_tare(state, estimator_input, tare)
    trace = []
    reference = tuple(0 for _ in range(part_count - 1))
    for index, configuration in enumerate(configurations, start=1):
        observation = simulate_hold(
            hidden,
            estimator_input,
            index,
            configuration,
            rng,
            forced_drift_deg=(
                HARD_DRIFT_GATE_DEG + 0.5
                if forced_unsafe_state == index
                else None
            ),
        )
        state = apply_hold(state, estimator_input, observation)
        trace.append(
            {
                "observation": asdict(observation),
                "robot_estimator_output": {
                    "density_mean_kg_m3": state.mean[:part_count].tolist(),
                    "density_std_kg_m3": np.sqrt(
                        np.diag(state.covariance)[:part_count]
                    ).tolist(),
                    "estimated_sensor_bias": state.mean[part_count:].tolist(),
                },
                "evaluator_only_gt_comparison": evaluate_state(
                    estimator_input,
                    hidden,
                    state,
                    reference,
                ),
            }
        )
    return {
        "schema": "hidden-gt-quasistatic-experiment-v1",
        "validation_level": "synthetic_closed_loop_with_hidden_gt",
        "claim_scope": (
            "Estimator integration and calibration under the declared "
            "synthetic world/noise model; not final hardware performance."
        ),
        "robot_visible_input": asdict(estimator_input),
        "robot_selected_configurations_deg": [
            list(value) for value in configurations
        ],
        "robot_tare_observation": tare.tolist(),
        "trace": trace,
        "evaluator_only_hidden_world": asdict(hidden),
    }


def run_monte_carlo(
    part_count: int,
    *,
    trials: int,
    seed: int,
) -> dict[str, object]:
    final_errors = []
    final_inside = []
    accepted_counts = []
    for trial in range(trials):
        result = run_single_experiment(
            part_count,
            seed=seed + trial,
        )
        accepted_counts.append(
            sum(
                row["observation"]["accepted"]
                for row in result["trace"]
            )
        )
        final = result["trace"][-1]["evaluator_only_gt_comparison"]
        final_errors.append(final["density_relative_error"])
        final_inside.append(final["density_gt_inside_95pct_interval"])
    errors = np.asarray(final_errors)
    inside = np.asarray(final_inside, dtype=float)
    return {
        "trials": trials,
        "density_relative_rmse": np.sqrt(
            np.mean(errors**2, axis=0)
        ).tolist(),
        "density_relative_error_p95": np.quantile(
            errors,
            0.95,
            axis=0,
        ).tolist(),
        "empirical_95pct_interval_coverage": np.mean(
            inside,
            axis=0,
        ).tolist(),
        "accepted_state_count_mean": float(np.mean(accepted_counts)),
        "all_states_accepted_rate": float(
            np.mean(np.asarray(accepted_counts) == part_count - 1)
        ),
    }


def _plot_trace(result: dict[str, object], output: Path) -> None:
    trace = result["trace"]
    p = result["robot_visible_input"]["part_count"]
    gt = np.asarray(
        result["evaluator_only_hidden_world"]["true_density_kg_m3"]
    )
    x = np.arange(len(trace) + 1)
    prior = np.asarray(
        result["robot_visible_input"]["prior_mean_kg_m3"]
    )
    prior_std = np.sqrt(
        np.diag(result["robot_visible_input"]["prior_covariance"])
    )
    means = [prior]
    stds = [prior_std]
    errors = [np.abs(prior - gt) / gt]
    for row in trace:
        output_row = row["robot_estimator_output"]
        means.append(np.asarray(output_row["density_mean_kg_m3"]))
        stds.append(np.asarray(output_row["density_std_kg_m3"]))
        errors.append(
            np.asarray(
                row["evaluator_only_gt_comparison"]["density_relative_error"]
            )
        )
    means = np.asarray(means)
    stds = np.asarray(stds)
    errors = np.asarray(errors)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for part in range(p):
        axes[0].errorbar(
            x,
            means[:, part],
            yerr=1.96 * stds[:, part],
            marker="o",
            capsize=3,
            label=f"part {part + 1} estimate ±95%",
        )
        axes[0].axhline(
            gt[part],
            linestyle="--",
            alpha=0.7,
            label=f"part {part + 1} hidden GT",
        )
        axes[1].plot(
            x,
            100.0 * errors[:, part],
            marker="o",
            label=f"part {part + 1}",
        )
    axes[0].set_xlabel("accepted/configured state count")
    axes[0].set_ylabel("density [kg/m³]")
    axes[0].set_title("Robot estimate vs evaluator-only hidden GT")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("accepted/configured state count")
    axes[1].set_ylabel("absolute relative density error [%]")
    axes[1].set_title("GT error revealed only by evaluator")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def run_all(output_dir: Path, *, trials: int, seed: int) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for part_count in (2, 3, 4):
        single = run_single_experiment(part_count, seed=seed + part_count)
        monte_carlo = run_monte_carlo(
            part_count,
            trials=trials,
            seed=seed + 1000 * part_count,
        )
        row = {
            "part_count": part_count,
            "single_experiment": single,
            "monte_carlo": monte_carlo,
        }
        rows.append(row)
        _plot_trace(
            single,
            output_dir / f"hidden_gt_trace_{part_count}_part.png",
        )
    report = {
        "schema": "hidden-gt-quasistatic-suite-v1",
        "results": rows,
    }
    with (output_dir / "hidden_gt_results.json").open(
        "w",
        encoding="utf-8",
    ) as stream:
        json.dump(report, stream, indent=2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260728)
    args = parser.parse_args()
    report = run_all(
        args.output_dir,
        trials=args.trials,
        seed=args.seed,
    )
    for row in report["results"]:
        final = row["single_experiment"]["trace"][-1][
            "evaluator_only_gt_comparison"
        ]
        print(
            f"{row['part_count']}-part: "
            f"density error={np.round(final['density_relative_error'], 3)}, "
            f"mass error={final['total_mass_relative_error']:.3f}, "
            f"CoM error={1000*final['com_error_m']:.1f} mm, "
            f"I error={final['pitch_inertia_relative_error']:.3f}, "
            f"coverage={np.round(row['monte_carlo']['empirical_95pct_interval_coverage'], 3)}"
        )


if __name__ == "__main__":
    main()
