#!/usr/bin/env python3
"""Validate the grasped L-Phantom as a passive 3-link Drake object."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import numpy as np
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    MeshcatVisualizer,
    Parser,
    RigidTransform,
    RollPitchYaw,
    Simulator,
    StartMeshcat,
)


DEFAULT_ASSET = (
    Path(__file__).resolve().parents[1]
    / "assets/phantom_v3/phantom_v3.urdf"
)
ROOT_LINK = "v3_part0_root"
CASES = (
    ("tilt_a", (35.0, -25.0, 15.0), (75.0, -55.0)),
    ("tilt_b", (-40.0, 30.0, -10.0), (-70.0, 50.0)),
    ("tilt_c", (55.0, 15.0, 40.0), (0.0, 0.0)),
)


@dataclass(frozen=True)
class MotionResult:
    name: str
    initial_deg: list[float]
    final_deg: list[float]
    motion_range_deg: list[float]
    final_window_range_deg: list[float]
    final_window_max_rate_rad_s: list[float]
    maximum_penetration_m: float
    passed: bool


def run_case(
    asset: Path,
    name: str,
    root_rpy_deg: tuple[float, float, float],
    initial_deg: tuple[float, float],
    *,
    duration_s: float,
    meshcat=None,
) -> MotionResult:
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    model = Parser(plant).AddModels(str(asset))[0]
    plant.WeldFrames(
        plant.world_frame(),
        plant.GetFrameByName(ROOT_LINK, model),
        RigidTransform(
            RollPitchYaw(*np.radians(root_rpy_deg)),
            (0.0, 0.0, 1.0),
        ),
    )
    plant.Finalize()
    visualizer = (
        MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
        if meshcat is not None
        else None
    )
    diagram = builder.Build()
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()
    plant_context = plant.GetMyMutableContextFromRoot(context)
    scene_context = scene_graph.GetMyContextFromRoot(context)
    joints = tuple(
        plant.GetJointByName(joint_name, model)
        for joint_name in ("joint1", "joint2")
    )
    for joint, angle_deg in zip(joints, initial_deg, strict=True):
        joint.set_angle(plant_context, math.radians(angle_deg))
        joint.set_angular_rate(plant_context, 0.0)
    simulator.Initialize()
    if visualizer is not None:
        visualizer.StartRecording()

    rows = []
    maximum_penetration_m = 0.0
    steps = int(round(duration_s / 0.01))
    for step in range(steps + 1):
        if step:
            simulator.AdvanceTo(step * 0.01)
        rows.append(
            [
                *(joint.get_angle(plant_context) for joint in joints),
                *(joint.get_angular_rate(plant_context) for joint in joints),
            ]
        )
        penetrations = (
            scene_graph.get_query_output_port()
            .Eval(scene_context)
            .ComputePointPairPenetration()
        )
        maximum_penetration_m = max(
            maximum_penetration_m,
            max((pair.depth for pair in penetrations), default=0.0),
        )

    if visualizer is not None:
        visualizer.StopRecording()
        visualizer.PublishRecording()
    samples = np.asarray(rows)
    angles_deg = np.degrees(samples[:, :2])
    final_count = min(101, len(samples))
    final_angles_deg = angles_deg[-final_count:]
    final_rates = samples[-final_count:, 2:]
    lower = np.array(
        [joint.position_lower_limits()[0] for joint in joints]
    )
    upper = np.array(
        [joint.position_upper_limits()[0] for joint in joints]
    )
    passed = bool(
        np.all(np.ptp(angles_deg, axis=0) >= 5.0)
        and np.all(np.ptp(final_angles_deg, axis=0) <= 0.1)
        and np.all(np.max(np.abs(final_rates), axis=0) <= 0.01)
        and np.all(samples[:, :2] >= lower - 1e-4)
        and np.all(samples[:, :2] <= upper + 1e-4)
        and maximum_penetration_m <= 1e-6
        and plant.num_actuators() == 0
    )
    return MotionResult(
        name=name,
        initial_deg=list(initial_deg),
        final_deg=angles_deg[-1].tolist(),
        motion_range_deg=np.ptp(angles_deg, axis=0).tolist(),
        final_window_range_deg=np.ptp(final_angles_deg, axis=0).tolist(),
        final_window_max_rate_rad_s=np.max(
            np.abs(final_rates), axis=0
        ).tolist(),
        maximum_penetration_m=maximum_penetration_m,
        passed=passed,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", type=Path, default=DEFAULT_ASSET)
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()

    if not args.asset.is_file():
        raise FileNotFoundError(args.asset)
    meshcat = StartMeshcat() if args.live else None
    selected_cases = CASES[:1] if args.live else CASES
    results = [
        run_case(
            args.asset,
            name,
            root_rpy_deg,
            initial_deg,
            duration_s=args.duration,
            meshcat=meshcat,
        )
        for name, root_rpy_deg, initial_deg in selected_cases
    ]
    report = {
        "asset": str(args.asset),
        "root_link_role": "grasped_root_not_heavy_base",
        "authored_link_masses_kg": [0.4, 0.33, 0.27],
        "total_mass_kg": 1.0,
        "pass": all(result.passed for result in results),
        "cases": [asdict(result) for result in results],
    }
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["pass"]:
        raise SystemExit(1)
    if meshcat is not None:
        print(f"Meshcat: {meshcat.web_url()}")
        input("Press Enter to close the live view...")


if __name__ == "__main__":
    main()
