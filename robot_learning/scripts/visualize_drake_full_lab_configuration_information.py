#!/usr/bin/env python3
"""Live Drake view of configuration-wise information acquisition.

The scene mirrors the planned lab: RB5-850E, AFT200 wrist F/T proxy,
PGC-140-50 gripper, 2.3 m x 1.1 m black table, black backdrop, two D435i
cameras, one D456 camera, and a controlled-GT articulated object.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    Box,
    DiagramBuilder,
    FixedOffsetFrame,
    JacobianWrtVariable,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Parser,
    RevoluteJoint,
    Rgba,
    RigidTransform,
    SpatialInertia,
    StartMeshcat,
    UnitInertia,
)


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

import run_hidden_gt_quasistatic_experiment as hidden_gt  # noqa: E402
import validate_drake_quasistatic_configuration_information as base  # noqa: E402
import visualize_drake_rb5_hammer_payload as rb5  # noqa: E402


LAB_CONFIG = (
    ROOT
    / "robot_learning/configs/experiments/"
    "icra_realistic_lab_scene_v1.json"
)
STOP_BUTTON = "종료"
LINK_COLORS = (
    np.array([0.12, 0.45, 0.95, 1.0]),
    np.array([0.96, 0.42, 0.08, 1.0]),
    np.array([0.15, 0.72, 0.38, 1.0]),
    np.array([0.68, 0.26, 0.88, 1.0]),
)
LAB_ARM_POSITION_RAD = rb5.INITIAL_ARM_POSITION_RAD.copy()
LAB_ARM_POSITION_RAD[0] = math.pi


@dataclass(frozen=True)
class Scene:
    diagram: object
    plant: object
    robot: rb5.ArmRecord
    internal_joints: tuple[RevoluteJoint, ...]
    part_bodies: tuple[object, ...]
    cameras: tuple[dict[str, object], ...]


def _part_spatial_inertia(part_index: int) -> SpatialInertia:
    length = float(base.LINK_LENGTHS_M[part_index])
    volume = length * base.LINK_WIDTH_M * base.LINK_HEIGHT_M
    mass = volume * float(hidden_gt.TRUE_DENSITIES_KG_M3[part_index])
    return SpatialInertia.MakeFromCentralInertia(
        mass,
        [length / 2.0, 0.0, 0.0],
        mass
        * UnitInertia.SolidBox(
            length,
            base.LINK_WIDTH_M,
            base.LINK_HEIGHT_M,
        ),
    )


def _part1_assembly() -> rb5.HammerAssembly:
    inertia = _part_spatial_inertia(0)
    mass = float(inertia.get_mass())
    com = tuple(float(value) for value in inertia.get_com())
    central = np.asarray(
        (
            mass
            * UnitInertia.SolidBox(
                float(base.LINK_LENGTHS_M[0]),
                base.LINK_WIDTH_M,
                base.LINK_HEIGHT_M,
            )
        ).CopyToFullMatrix3()
    )
    density = float(hidden_gt.TRUE_DENSITIES_KG_M3[0])
    return rb5.HammerAssembly(
        spatial_inertia=inertia,
        mass_kg=mass,
        com_h_m=com,
        central_inertia_h_kg_m2=tuple(
            tuple(float(value) for value in row) for row in central
        ),
        part_masses_kg=(mass, 0.0, 0.0, 0.0),
        effective_densities_kg_m3=(density, 0.0, 0.0, 0.0),
    )


def _register_part1_visual(plant, body, spec) -> None:
    length = float(base.LINK_LENGTHS_M[0])
    plant.RegisterVisualGeometry(
        body,
        RigidTransform([length / 2.0, 0.0, 0.0]),
        Box(length, base.LINK_WIDTH_M, base.LINK_HEIGHT_M),
        f"{spec.name}_gt_part_1",
        LINK_COLORS[0],
    )


def _ignore_geometry(*_args) -> None:
    return None


def _register_lab_visuals(plant, lab: dict[str, object]) -> None:
    table = lab["table"]
    size = np.asarray(table["size_xyz_m"], dtype=float)
    center_z = float(table["top_height_m"]) - size[2] / 2.0
    plant.RegisterVisualGeometry(
        plant.world_body(),
        RigidTransform([*table["center_xy_m"], center_z]),
        Box(*size),
        "black_experiment_table",
        np.array([*table["surface_rgb"], 1.0]),
    )
    background = lab["background"]
    color = np.array([*background["rgb"], 1.0])
    for name, center_key, size_key in (
        ("backdrop", "backdrop_center_xyz_m", "backdrop_size_xyz_m"),
        ("left_wing", "left_wing_center_xyz_m", "wing_size_xyz_m"),
        ("right_wing", "right_wing_center_xyz_m", "wing_size_xyz_m"),
    ):
        plant.RegisterVisualGeometry(
            plant.world_body(),
            RigidTransform(background[center_key]),
            Box(*background[size_key]),
            f"black_cloth_{name}",
            color,
        )
    for camera in lab["cameras"]:
        camera_color = (
            np.array([0.10, 0.56, 0.95, 1.0])
            if "D435i" in camera["model"]
            else np.array([0.06, 0.84, 0.72, 1.0])
        )
        plant.RegisterVisualGeometry(
            plant.world_body(),
            RigidTransform(camera["position_xyz_m"]),
            Box(0.09, 0.035, 0.03),
            f"{camera['id']}_body",
            camera_color,
        )


def build_scene(part_count: int, *, meshcat=None) -> Scene:
    with LAB_CONFIG.open(encoding="utf-8") as stream:
        lab = json.load(stream)
    rb5_description, urdf_path, pgc_urdf_path, drake_assets = (
        rb5.validate_htd_source(rb5.DEFAULT_HTD_ROOT)
    )
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(
        builder,
        time_step=0.0,
    )
    plant.mutable_gravity_field().set_gravity_vector([0.0, 0.0, -9.81])
    parser = Parser(plant)
    parser.package_map().Add("rbpodo_description", str(rb5_description))
    parser.package_map().Add("htd", str(rb5.DEFAULT_HTD_ROOT.resolve()))
    variant = rb5.VariantSpec(
        name=f"gt_{part_count}_part",
        left_insert_mass_kg=rb5.LIGHT_INSERT_MASS_KG,
        right_insert_mass_kg=rb5.LIGHT_INSERT_MASS_KG,
        base_y_m=float(lab["robot"]["base_xyz_m"][1]),
    )
    robot = rb5.add_rb5_with_payload(
        plant,
        parser,
        urdf_path,
        pgc_urdf_path,
        drake_assets,
        variant,
        include_visuals=meshcat is not None,
        assembly_override=_part1_assembly(),
        controller_kp=0.0,
        controller_kd=0.0,
        payload_body_name="gt_part_1",
        # The first part passes between the opposing fingertips.  Its local
        # y thickness is 45 mm; the fixed-jaw proxy has about a 41.5 mm inner
        # gap, giving a small visual/contact preload instead of placing the
        # payload through the gripper base.
        payload_attachment=RigidTransform([0.0, 0.0, rb5.PGC_TCP_Z_M]),
        payload_visual_registrar=_register_part1_visual,
        payload_collision_registrar=_ignore_geometry,
    )
    parts = [robot.payload]
    internal_joints = []
    previous = robot.payload
    for index in range(1, part_count):
        model = plant.AddModelInstance(f"gt_part_{index + 1}_model")
        body = plant.AddRigidBody(
            f"gt_part_{index + 1}",
            model,
            _part_spatial_inertia(index),
        )
        length = float(base.LINK_LENGTHS_M[index])
        if meshcat is not None:
            plant.RegisterVisualGeometry(
                body,
                RigidTransform([length / 2.0, 0.0, 0.0]),
                Box(length, base.LINK_WIDTH_M, base.LINK_HEIGHT_M),
                f"gt_part_{index + 1}_visual",
                LINK_COLORS[index],
            )
        parent_frame = plant.AddFrame(
            FixedOffsetFrame(
                f"gt_joint_{index}_parent",
                previous.body_frame(),
                RigidTransform(
                    [float(base.LINK_LENGTHS_M[index - 1]), 0.0, 0.0]
                ),
            )
        )
        joint = plant.AddJoint(
            RevoluteJoint(
                f"gt_joint_{index}",
                parent_frame,
                body.body_frame(),
                [0.0, 1.0, 0.0],
            )
        )
        internal_joints.append(joint)
        parts.append(body)
        previous = body
    if meshcat is not None:
        _register_lab_visuals(plant, lab)
    plant.Finalize()
    if meshcat is not None:
        MeshcatVisualizer.AddToBuilder(
            builder,
            scene_graph,
            meshcat,
            MeshcatVisualizerParams(publish_period=1.0 / 30.0),
        )
    return Scene(
        diagram=builder.Build(),
        plant=plant,
        robot=robot,
        internal_joints=tuple(internal_joints),
        part_bodies=tuple(parts),
        cameras=tuple(lab["cameras"]),
    )


def _set_robot(scene: Scene, context, q: np.ndarray) -> None:
    for joint, value in zip(scene.robot.joints, q, strict=True):
        joint.set_angle(context, float(value))
        joint.set_angular_rate(context, 0.0)


def _lift_path(scene: Scene, context, lift_m: float = 0.10) -> np.ndarray:
    initial = LAB_ARM_POSITION_RAD.copy()
    _set_robot(scene, context, initial)
    samples = [initial.copy()]
    current = initial.copy()
    for _ in range(60):
        _set_robot(scene, context, current)
        jacobian = scene.plant.CalcJacobianSpatialVelocity(
            context,
            JacobianWrtVariable.kV,
            scene.robot.payload.body_frame(),
            np.zeros(3),
            scene.plant.world_frame(),
            scene.plant.world_frame(),
        )
        columns = np.column_stack(
            [jacobian[:, joint.velocity_start()] for joint in scene.robot.joints]
        )
        desired = np.array([0.0, 0.0, 0.0, 0.0, 0.0, lift_m / 60.0])
        delta, *_ = np.linalg.lstsq(columns, desired, rcond=None)
        delta = np.clip(delta, -0.015, 0.015)
        current = current + delta
        samples.append(current.copy())
    _set_robot(scene, context, initial)
    return np.asarray(samples)


def _set_camera_sight_lines(meshcat, cameras) -> None:
    for camera in cameras:
        points = np.column_stack(
            (
                np.asarray(camera["position_xyz_m"], dtype=float),
                np.asarray(camera["look_at_xyz_m"], dtype=float),
            )
        )
        color = Rgba(0.12, 0.62, 1.0, 0.65)
        meshcat.SetLine(f"/camera_rays/{camera['id']}", points, 2.0, color)


def _state_label(result: dict[str, object], state_index: int) -> str:
    row = result["trace"][state_index]
    observation = row["observation"]
    robot = row["robot_estimator_output"]
    evaluation = row["evaluator_only_gt_comparison"]
    q = ",".join(
        f"{value:.1f}°"
        for value in observation["camera"]["fused_joint_angles_deg"]
    )
    estimate = ",".join(
        f"{value:.0f}" for value in robot["density_mean_kg_m3"]
    )
    std = ",".join(
        f"{value:.0f}" for value in robot["density_std_kg_m3"]
    )
    error = ",".join(
        f"{100.0 * value:.1f}%"
        for value in evaluation["density_relative_error"]
    )
    return (
        f"STATE {state_index + 1} | camera q=({q}) | "
        f"accepted={observation['accepted']} | "
        f"ROBOT ρ=({estimate}) ±σ=({std}) | "
        f"EVAL-ONLY GT error=({error})"
    )


def run_live(part_count: int, cycle_seconds: float) -> None:
    result = hidden_gt.run_single_experiment(
        part_count,
        seed=20260728 + part_count,
    )
    meshcat = StartMeshcat()
    scene = build_scene(part_count, meshcat=meshcat)
    context = scene.diagram.CreateDefaultContext()
    plant_context = scene.plant.GetMyMutableContextFromRoot(context)
    _set_robot(scene, plant_context, LAB_ARM_POSITION_RAD)
    lift_path = _lift_path(scene, plant_context)
    _set_camera_sight_lines(meshcat, scene.cameras)
    meshcat.AddButton(STOP_BUTTON)
    current_label = None
    print(f"Meshcat: {meshcat.web_url()}", flush=True)
    print("파란 선: 3대 RealSense의 관측선", flush=True)
    while meshcat.GetButtonClicks(STOP_BUTTON) < 1:
        for state_index, row in enumerate(result["trace"]):
            if meshcat.GetButtonClicks(STOP_BUTTON) >= 1:
                break
            if current_label is not None:
                meshcat.DeleteButton(current_label)
            current_label = _state_label(result, state_index)
            meshcat.AddButton(current_label)
            observed_angles = row["observation"]["camera"][
                "fused_joint_angles_deg"
            ]
            if not observed_angles:
                observed_angles = row["observation"][
                    "commanded_joint_angles_deg"
                ]
            for joint, degrees in zip(
                scene.internal_joints,
                observed_angles,
                strict=True,
            ):
                joint.set_angle(plant_context, math.radians(degrees))
                joint.set_angular_rate(plant_context, 0.0)
            # Lift, hold for the quasi-static average, and lower.
            for q in lift_path:
                _set_robot(scene, plant_context, q)
                scene.diagram.ForcedPublish(context)
                time.sleep(cycle_seconds / (3.0 * len(lift_path)))
            hold_end = time.time() + cycle_seconds / 3.0
            while time.time() < hold_end:
                scene.diagram.ForcedPublish(context)
                time.sleep(1.0 / 30.0)
            for q in reversed(lift_path):
                _set_robot(scene, plant_context, q)
                scene.diagram.ForcedPublish(context)
                time.sleep(cycle_seconds / (3.0 * len(lift_path)))
    if current_label is not None:
        meshcat.DeleteButton(current_label)
    meshcat.DeleteButton(STOP_BUTTON)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part-count", type=int, choices=(2, 3, 4), default=4)
    parser.add_argument("--cycle-seconds", type=float, default=6.0)
    parser.add_argument("--test-only", action="store_true")
    args = parser.parse_args()
    if args.test_only:
        scene = build_scene(args.part_count)
        context = scene.diagram.CreateDefaultContext()
        plant_context = scene.plant.GetMyMutableContextFromRoot(context)
        path = _lift_path(scene, plant_context)
        print(
            f"built {args.part_count}-part scene: "
            f"positions={scene.plant.num_positions()}, lift_samples={len(path)}"
        )
        return
    run_live(args.part_count, args.cycle_seconds)


if __name__ == "__main__":
    main()
