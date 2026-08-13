#!/usr/bin/env python3
"""Visualize the custom-CAD 2/3/4-link hidden-GT experiment in Drake."""

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
    DiagramBuilder,
    FixedOffsetFrame,
    Mesh,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Parser,
    RevoluteJoint,
    RigidTransform,
    SpatialInertia,
    StartMeshcat,
    UnitInertia,
)


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

import run_custom_cad_configuration_information_experiment as experiment  # noqa: E402
import visualize_drake_full_lab_configuration_information as lab  # noqa: E402
import visualize_drake_rb5_hammer_payload as rb5  # noqa: E402


CAD_ROOT = ROOT / "custom_object_cad"
DRAKE_CAD_ROOT = CAD_ROOT / "drake"
DEFAULT_RESULTS = (
    ROOT
    / "progress/artifacts/2026-07-28/"
    "custom_cad_configuration_information/custom_cad_results.json"
)
STOP_BUTTON = "종료"
LID_Z_M = -experiment.BOX_HEIGHT_M / 2.0 + 0.004 / 2.0


@dataclass(frozen=True)
class Scene:
    diagram: object
    plant: object
    robot: rb5.ArmRecord
    internal_joints: tuple[RevoluteJoint, ...]
    cameras: tuple[dict[str, object], ...]


def _box_inertia(
    mass_kg: float,
    length_m: float,
    com_m,
) -> SpatialInertia:
    return SpatialInertia.MakeFromCentralInertia(
        mass_kg,
        np.asarray(com_m),
        mass_kg
        * UnitInertia.SolidBox(
            length_m,
            experiment.BOX_DEPTH_M,
            experiment.BOX_HEIGHT_M,
        ),
    )


def _parent_assembly() -> rb5.HammerAssembly:
    mass = float(
        experiment.PRINTED_SOLID_VOLUME_M3[0]
        * experiment.PLA_DENSITY_KG_M3
        + experiment.HIDDEN_INSERT_MASS_KG[0]
    )
    inertia = _box_inertia(
        mass,
        experiment.PARENT_LENGTH_M,
        np.zeros(3),
    )
    central = (
        mass
        * UnitInertia.SolidBox(
            experiment.PARENT_LENGTH_M,
            experiment.BOX_DEPTH_M,
            experiment.BOX_HEIGHT_M,
        )
    ).CopyToFullMatrix3()
    density = mass / (
        experiment.PARENT_LENGTH_M
        * experiment.BOX_DEPTH_M
        * experiment.BOX_HEIGHT_M
    )
    return rb5.HammerAssembly(
        spatial_inertia=inertia,
        mass_kg=mass,
        com_h_m=(0.0, 0.0, 0.0),
        central_inertia_h_kg_m2=tuple(
            tuple(float(value) for value in row) for row in central
        ),
        part_masses_kg=(mass, 0.0, 0.0, 0.0),
        effective_densities_kg_m3=(density, 0.0, 0.0, 0.0),
    )


def _register_parent_visual(plant, body, _spec) -> None:
    plant.RegisterVisualGeometry(
        body,
        RigidTransform(),
        Mesh(DRAKE_CAD_ROOT / "parent_body.obj", 0.001),
        "custom_parent_body_blue",
        np.array([0.08, 0.24, 0.62, 1.0]),
    )
    plant.RegisterVisualGeometry(
        body,
        RigidTransform([0.0, 0.0, LID_Z_M]),
        Mesh(DRAKE_CAD_ROOT / "parent_lid.obj", 0.001),
        "custom_parent_lid_gray",
        np.array([0.34, 0.38, 0.45, 1.0]),
    )


def _ignore_geometry(*_args) -> None:
    return None


def _register_child_visual(plant, body, index: int) -> None:
    center = np.array(
        [
            experiment.HINGE_GAP_M / 2.0
            + experiment.CHILD_LENGTH_M / 2.0,
            0.0,
            -experiment.BOX_HEIGHT_M / 2.0,
        ]
    )
    colors = (
        np.array([0.08, 0.58, 0.46, 1.0]),
        np.array([0.86, 0.44, 0.08, 1.0]),
        np.array([0.55, 0.22, 0.75, 1.0]),
    )
    plant.RegisterVisualGeometry(
        body,
        RigidTransform(center),
        Mesh(DRAKE_CAD_ROOT / "child_body.obj", 0.001),
        f"custom_child_{index}_body",
        colors[index - 1],
    )
    plant.RegisterVisualGeometry(
        body,
        RigidTransform(center + [0.0, 0.0, LID_Z_M]),
        Mesh(DRAKE_CAD_ROOT / "child_lid.obj", 0.001),
        f"custom_child_{index}_lid",
        np.array([0.34, 0.38, 0.45, 1.0]),
    )


def build_scene(part_count: int, *, meshcat=None) -> Scene:
    with lab.LAB_CONFIG.open(encoding="utf-8") as stream:
        lab_config = json.load(stream)
    rb5_description, urdf_path, pgc_urdf_path, drake_assets = (
        rb5.validate_htd_source(rb5.DEFAULT_HTD_ROOT)
    )
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    plant.mutable_gravity_field().set_gravity_vector([0.0, 0.0, -9.81])
    parser = Parser(plant)
    parser.package_map().Add("rbpodo_description", str(rb5_description))
    parser.package_map().Add("htd", str(rb5.DEFAULT_HTD_ROOT.resolve()))
    variant = rb5.VariantSpec(
        name=f"custom_cad_{part_count}_link",
        left_insert_mass_kg=rb5.LIGHT_INSERT_MASS_KG,
        right_insert_mass_kg=rb5.LIGHT_INSERT_MASS_KG,
        base_y_m=float(lab_config["robot"]["base_xyz_m"][1]),
    )
    robot = rb5.add_rb5_with_payload(
        plant,
        parser,
        urdf_path,
        pgc_urdf_path,
        drake_assets,
        variant,
        include_visuals=meshcat is not None,
        assembly_override=_parent_assembly(),
        controller_kp=0.0,
        controller_kd=0.0,
        payload_body_name="custom_parent",
        payload_attachment=RigidTransform([0.0, 0.0, rb5.PGC_TCP_Z_M]),
        payload_visual_registrar=_register_parent_visual,
        payload_collision_registrar=_ignore_geometry,
        # PGC has 50 mm total stroke.  20 mm per jaw changes the nominal
        # 140 mm opening to about 100 mm for the Parent depth.
        pgc_joint_position_m=0.020,
    )
    joints = []
    previous_body = robot.payload
    parent_joint_frame = plant.AddFrame(
        FixedOffsetFrame(
            "custom_joint_1_parent",
            previous_body.body_frame(),
            RigidTransform(
                [
                    experiment.PARENT_LENGTH_M / 2.0
                    + experiment.HINGE_GAP_M / 2.0,
                    0.0,
                    experiment.BOX_HEIGHT_M / 2.0,
                ]
            ),
        )
    )
    for child_index in range(1, part_count):
        mass = float(
            experiment.PRINTED_SOLID_VOLUME_M3[child_index]
            * experiment.PLA_DENSITY_KG_M3
            + experiment.HIDDEN_INSERT_MASS_KG[child_index]
        )
        com = np.array(
            [
                experiment.HINGE_GAP_M / 2.0
                + experiment.CHILD_LENGTH_M / 2.0,
                0.0,
                -experiment.BOX_HEIGHT_M / 2.0,
            ]
        )
        model_instance = plant.AddModelInstance(
            f"custom_child_{child_index}_model"
        )
        body = plant.AddRigidBody(
            f"custom_child_{child_index}",
            model_instance,
            _box_inertia(mass, experiment.CHILD_LENGTH_M, com),
        )
        if meshcat is not None:
            _register_child_visual(plant, body, child_index)
        joint = plant.AddJoint(
            RevoluteJoint(
                f"custom_hinge_{child_index}",
                parent_joint_frame,
                body.body_frame(),
                [0.0, 1.0, 0.0],
            )
        )
        joints.append(joint)
        if child_index < part_count - 1:
            parent_joint_frame = plant.AddFrame(
                FixedOffsetFrame(
                    f"custom_joint_{child_index + 1}_parent",
                    body.body_frame(),
                    RigidTransform(
                        [
                            experiment.HINGE_GAP_M
                            + experiment.CHILD_LENGTH_M,
                            0.0,
                            0.0,
                        ]
                    ),
                )
            )
        previous_body = body
    if meshcat is not None:
        lab._register_lab_visuals(plant, lab_config)
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
        internal_joints=tuple(joints),
        cameras=tuple(lab_config["cameras"]),
    )


def _load_result(path: Path, part_count: int) -> dict[str, object]:
    report = json.loads(path.read_text(encoding="utf-8"))
    return next(
        row["nominal_trace"]
        for row in report["results"]
        if row["part_count"] == part_count
    )


def _label(row: dict[str, object]) -> str:
    output = row["robot_output"]
    evaluation = row["evaluator_only"]
    q = ",".join(
        f"{value:.1f}°" for value in row["camera_opening_angles_deg"]
    )
    density = ",".join(
        f"{value:.0f}" for value in output["density_mean_kg_m3"]
    )
    std = ",".join(
        f"{value:.0f}" for value in output["density_std_kg_m3"]
    )
    error = ",".join(
        f"{100.0 * value:.1f}%"
        for value in evaluation["density_relative_error"]
    )
    return (
        f"STATE {row['state_index']} | opening=({q}) | "
        f"rank/null={output['data_rank']}/{output['nullity']} | "
        f"ΔI={output['incremental_information_gain_nats']:.2f} nat | "
        f"ROBOT ρ=({density}) σ=({std}) | EVAL error=({error})"
    )


def run_live(
    part_count: int,
    *,
    results_path: Path,
    cycle_seconds: float,
) -> None:
    result = _load_result(results_path, part_count)
    meshcat = StartMeshcat()
    scene = build_scene(part_count, meshcat=meshcat)
    context = scene.diagram.CreateDefaultContext()
    plant_context = scene.plant.GetMyMutableContextFromRoot(context)
    lab._set_robot(scene, plant_context, lab.LAB_ARM_POSITION_RAD)
    lift_path = lab._lift_path(scene, plant_context, lift_m=0.10)
    lab._set_camera_sight_lines(meshcat, scene.cameras)
    meshcat.AddButton(STOP_BUTTON)
    current_label = None
    print(f"Meshcat: {meshcat.web_url()}", flush=True)
    while meshcat.GetButtonClicks(STOP_BUTTON) < 1:
        for row in result["trace"]:
            if meshcat.GetButtonClicks(STOP_BUTTON) >= 1:
                break
            if current_label is not None:
                meshcat.DeleteButton(current_label)
            current_label = _label(row)
            meshcat.AddButton(current_label)
            opening = (
                row["camera_opening_angles_deg"]
                or row["commanded_opening_angles_deg"]
            )
            for joint, opening_deg in zip(
                scene.internal_joints,
                opening,
                strict=True,
            ):
                bend_rad = -math.radians(180.0 - opening_deg)
                joint.set_angle(plant_context, bend_rad)
                joint.set_angular_rate(plant_context, 0.0)
            for robot_q in lift_path:
                lab._set_robot(scene, plant_context, robot_q)
                scene.diagram.ForcedPublish(context)
                time.sleep(cycle_seconds / (3.0 * len(lift_path)))
            hold_end = time.time() + cycle_seconds / 3.0
            while time.time() < hold_end:
                scene.diagram.ForcedPublish(context)
                time.sleep(1.0 / 30.0)
            for robot_q in reversed(lift_path):
                lab._set_robot(scene, plant_context, robot_q)
                scene.diagram.ForcedPublish(context)
                time.sleep(cycle_seconds / (3.0 * len(lift_path)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part-count", type=int, choices=(2, 3, 4), default=4)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--cycle-seconds", type=float, default=6.0)
    parser.add_argument("--test-only", action="store_true")
    args = parser.parse_args()
    if args.test_only:
        scene = build_scene(args.part_count)
        context = scene.diagram.CreateDefaultContext()
        plant_context = scene.plant.GetMyMutableContextFromRoot(context)
        lab._set_robot(scene, plant_context, lab.LAB_ARM_POSITION_RAD)
        path = lab._lift_path(scene, plant_context)
        print(
            f"custom CAD {args.part_count}-link scene: "
            f"positions={scene.plant.num_positions()}, lift={len(path)}"
        )
        return
    run_live(
        args.part_count,
        results_path=args.results,
        cycle_seconds=args.cycle_seconds,
    )


if __name__ == "__main__":
    main()
