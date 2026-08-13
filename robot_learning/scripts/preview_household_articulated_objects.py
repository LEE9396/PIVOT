#!/usr/bin/env python3
"""Preview frictionless articulated desk-lamp and jewelry-box objects."""

from __future__ import annotations

import argparse
import itertools
import math
import time

import numpy as np
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    Box,
    CoulombFriction,
    DiagramBuilder,
    FixedOffsetFrame,
    MeshcatVisualizer,
    RevoluteJoint,
    RigidTransform,
    RotationMatrix,
    Simulator,
    SpatialInertia,
    StartMeshcat,
    UnitInertia,
)


def inertia(
    mass: float,
    size: tuple[float, float, float],
    com=(0.0, 0.0, 0.0),
) -> SpatialInertia:
    return SpatialInertia.MakeFromCentralInertia(
        mass,
        np.asarray(com),
        mass * UnitInertia.SolidBox(*size),
    )


def add_box(plant, body, pose, size, name, color, *, collision=False) -> None:
    plant.RegisterVisualGeometry(body, pose, Box(*size), name, color)
    if collision:
        plant.RegisterCollisionGeometry(
            body,
            pose,
            Box(*size),
            f"{name}_collision",
            CoulombFriction(0.8, 0.7),
        )


def add_jewelry_box(plant):
    model = plant.AddModelInstance("jewelry_box")
    size = (0.28, 0.20, 0.045)
    base = plant.AddRigidBody("base", model, inertia(0.75, size))
    plant.WeldFrames(
        plant.world_frame(),
        base.body_frame(),
        RigidTransform([-0.32, 0.0, 0.76]),
    )
    add_box(
        plant,
        base,
        RigidTransform(),
        size,
        "velvet_base",
        [0.18, 0.32, 0.55, 1],
        collision=True,
    )
    add_box(
        plant,
        base,
        RigidTransform([0, 0, size[2] / 2 + 0.004]),
        (0.245, 0.165, 0.008),
        "velvet_insert",
        [0.82, 0.83, 0.85, 1],
    )

    lid_size = (0.28, 0.20, 0.015)
    lid_com = (0.0, -size[1] / 2, 0.003 + lid_size[2] / 2)
    lid = plant.AddRigidBody("lid", model, inertia(0.28, lid_size, lid_com))
    hinge_parent = plant.AddFrame(
        FixedOffsetFrame(
            "lid_hinge_parent",
            base.body_frame(),
            RigidTransform([0, size[1] / 2, size[2] / 2]),
        )
    )
    hinge = plant.AddJoint(
        RevoluteJoint(
            "lid_hinge",
            hinge_parent,
            lid.body_frame(),
            [1, 0, 0],
            damping=0.0,
        )
    )
    hinge.set_position_limits(np.array([-math.radians(120)]), np.array([0.0]))
    add_box(
        plant,
        lid,
        RigidTransform(lid_com),
        lid_size,
        "lid_frame",
        [0.62, 0.28, 0.58, 1],
        collision=True,
    )
    add_box(
        plant,
        lid,
        RigidTransform([0, -size[1] / 2, 0.003]),
        (0.245, 0.165, 0.004),
        "lid_inner",
        [0.88, 0.89, 0.91, 1],
    )
    return (hinge,), (-105.0,)


def add_desk_lamp(plant):
    model = plant.AddModelInstance("desk_lamp")
    base_size = (0.30, 0.20, 0.035)
    base = plant.AddRigidBody("base", model, inertia(0.95, base_size))
    plant.WeldFrames(
        plant.world_frame(),
        base.body_frame(),
        RigidTransform(
            RotationMatrix.MakeZRotation(math.pi),
            [0.30, 0.0, 0.7475],
        ),
    )
    add_box(
        plant,
        base,
        RigidTransform(),
        base_size,
        "lamp_base",
        [0.08, 0.28, 0.68, 1],
        collision=True,
    )
    hinge_x = -base_size[0] / 2 + 0.035 / 2
    pedestal_height = 0.07
    add_box(
        plant,
        base,
        RigidTransform(
            [hinge_x, 0.0, base_size[2] / 2 + pedestal_height / 2]
        ),
        (0.035, 0.05, pedestal_height),
        "lamp_hinge_pedestal",
        [0.08, 0.28, 0.68, 1],
        collision=True,
    )

    arm_length = 0.40
    arm_size = (arm_length, 0.035, 0.035)
    arm_link_rotation = RotationMatrix()
    arm_com = arm_link_rotation @ np.array([arm_length / 2, 0.0, 0.0])
    arm = plant.AddRigidBody(
        "lower_arm", model, inertia(0.38, arm_size, arm_com)
    )
    lower_parent = plant.AddFrame(
        FixedOffsetFrame(
            "lower_hinge_parent",
            base.body_frame(),
            RigidTransform(
                [
                    hinge_x,
                    0,
                    base_size[2] / 2
                    + pedestal_height
                    + arm_size[2] / 2
                    + 0.003,
                ]
            ),
        )
    )
    lower = plant.AddJoint(
        RevoluteJoint(
            "lower_hinge",
            lower_parent,
            arm.body_frame(),
            [0, 1, 0],
            damping=0.0,
        )
    )
    lower.set_position_limits(
        np.array([-math.radians(120)]), np.array([0.0])
    )
    add_box(
        plant,
        arm,
        RigidTransform(arm_link_rotation, arm_com),
        arm_size,
        "lamp_arm",
        [0.05, 0.62, 0.58, 1],
        collision=True,
    )

    head_size = (0.283, 0.055, 0.025)
    head_com = (-head_size[0] / 2, 0.0, 0.0)
    head = plant.AddRigidBody(
        "lamp_head", model, inertia(0.32, head_size, head_com)
    )
    upper_parent = plant.AddFrame(
        FixedOffsetFrame(
            "upper_hinge_parent",
            arm.body_frame(),
            RigidTransform(
                arm_link_rotation,
                arm_link_rotation
                @ np.array([
                    arm_length,
                    0,
                    -(arm_size[2] / 2 + head_size[2] / 2 + 0.003),
                ]),
            ),
        )
    )
    upper = plant.AddJoint(
        RevoluteJoint(
            "upper_hinge",
            upper_parent,
            head.body_frame(),
            [0, 1, 0],
            damping=0.0,
        )
    )
    upper.set_position_limits(np.array([-math.radians(90)]), np.array([0.0]))
    add_box(
        plant,
        head,
        RigidTransform(head_com),
        head_size,
        "lamp_head",
        [0.95, 0.65, 0.08, 1],
        collision=True,
    )
    add_box(
        plant,
        head,
        RigidTransform([-head_size[0] / 2, 0, -head_size[2] / 2 - 0.003]),
        (0.25, 0.035, 0.004),
        "light_panel",
        [1.0, 0.82, 0.40, 1],
    )
    return (lower, upper), (-120.0, 0.0)


def build(meshcat=None):
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 0.0001)
    plant.set_penetration_allowance(1e-4)
    plant.RegisterVisualGeometry(
        plant.world_body(),
        RigidTransform([0, 0, 0.70]),
        Box(1.10, 0.65, 0.06),
        "table",
        [0.54, 0.35, 0.18, 1],
    )
    plant.RegisterCollisionGeometry(
        plant.world_body(),
        RigidTransform([0, 0, 0.70]),
        Box(1.10, 0.65, 0.06),
        "table_collision",
        CoulombFriction(0.8, 0.7),
    )
    box_joints, box_angles = add_jewelry_box(plant)
    lamp_joints, lamp_angles = add_desk_lamp(plant)
    plant.Finalize()
    if meshcat is not None:
        MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    diagram = builder.Build()
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()
    plant_context = plant.GetMyMutableContextFromRoot(context)
    for joint, degrees in zip(
        box_joints + lamp_joints, box_angles + lamp_angles, strict=True
    ):
        joint.set_angle(plant_context, math.radians(degrees))
    simulator.Initialize()
    diagram.ForcedPublish(context)
    return simulator, box_joints, lamp_joints


def minimum_lamp_clearances(simulator, lamp_joints):
    diagram = simulator.get_system()
    context = simulator.get_mutable_context()
    plant = diagram.GetSubsystemByName("plant")
    plant_context = plant.GetMyMutableContextFromRoot(context)
    scene_graph = diagram.GetSubsystemByName("scene_graph")
    scene_context = scene_graph.GetMyContextFromRoot(context)
    inspector = scene_graph.model_inspector()
    ids = [
        geometry_id
        for geometry_id in inspector.GetAllGeometryIds()
        if "collision" in inspector.GetName(geometry_id)
    ]
    part_clearance = table_clearance = math.inf
    for lower_deg, upper_deg in itertools.product(
        range(0, 121, 5), range(0, 91, 5)
    ):
        lamp_joints[0].set_angle(
            plant_context, -math.radians(lower_deg)
        )
        lamp_joints[1].set_angle(plant_context, -math.radians(upper_deg))
        query = scene_graph.get_query_output_port().Eval(scene_context)
        for a, b in itertools.combinations(ids, 2):
            name_a, name_b = inspector.GetName(a), inspector.GetName(b)
            if {name_a, name_b} == {
                "desk_lamp::lamp_arm_collision",
                "desk_lamp::lamp_head_collision",
            }:
                part_clearance = min(
                    part_clearance,
                    query.ComputeSignedDistancePairClosestPoints(a, b).distance,
                )
            if (
                "table" in name_a
                and name_b == "desk_lamp::lamp_base_collision"
                or "table" in name_b
                and name_a == "desk_lamp::lamp_base_collision"
            ):
                table_clearance = min(
                    table_clearance,
                    query.ComputeSignedDistancePairClosestPoints(a, b).distance,
                )
    return part_clearance, table_clearance


def minimum_dynamic_self_clearances(simulator):
    diagram = simulator.get_system()
    context = simulator.get_context()
    scene_graph = diagram.GetSubsystemByName("scene_graph")
    scene_context = scene_graph.GetMyContextFromRoot(context)
    inspector = scene_graph.model_inspector()
    ids = [
        geometry_id
        for geometry_id in inspector.GetAllGeometryIds()
        if "collision" in inspector.GetName(geometry_id)
    ]
    clearances = {"jewelry_box": math.inf, "desk_lamp": math.inf}
    for step in range(1, 501):
        simulator.AdvanceTo(step / 100)
        query = scene_graph.get_query_output_port().Eval(scene_context)
        for a, b in itertools.combinations(ids, 2):
            name_a, name_b = inspector.GetName(a), inspector.GetName(b)
            for prefix in clearances:
                if (
                    prefix in name_a
                    and prefix in name_b
                    and inspector.GetFrameId(a) != inspector.GetFrameId(b)
                ):
                    clearances[prefix] = min(
                        clearances[prefix],
                        query.ComputeSignedDistancePairClosestPoints(a, b).distance,
                    )
    return clearances


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--physics", action="store_true")
    args = parser.parse_args()
    meshcat = None if args.test else StartMeshcat()
    simulator, box_joints, lamp_joints = build(meshcat)
    assert len(box_joints) == 1 and len(lamp_joints) == 2
    assert math.isclose(box_joints[0].position_lower_limits()[0], -math.radians(120))
    assert np.allclose(
        [
            lamp_joints[0].position_lower_limits()[0],
            lamp_joints[1].position_lower_limits()[0],
        ],
        np.radians([-120, -90]),
    )
    if args.test:
        part_clearance, table_clearance = minimum_lamp_clearances(
            simulator, lamp_joints
        )
        assert part_clearance >= 0.0
        assert table_clearance >= -1e-9
        dynamic_clearance = minimum_dynamic_self_clearances(build()[0])
        assert dynamic_clearance["jewelry_box"] >= 0.0
        assert dynamic_clearance["desk_lamp"] >= -1e-4
        print(
            "PASS: jewelry box 120°, desk lamp 120°/90°, "
            f"part/table clearance={part_clearance:.4f}/{table_clearance:.4f}m, "
            f"dynamic={dynamic_clearance}"
        )
        return
    print(f"Meshcat: {meshcat.web_url()}", flush=True)
    if args.physics:
        print("3초 후 중력 자유운동을 시작합니다.", flush=True)
        time.sleep(3.0)
        simulator.set_target_realtime_rate(0.5)
        simulator.AdvanceTo(5.0)
        context = simulator.get_context()
        plant = simulator.get_system().GetSubsystemByName("plant")
        plant_context = plant.GetMyContextFromRoot(context)
        print(
            "완료: jewelry={:.2f}°, lamp upper/lower=({:.2f}°, {:.2f}°)".format(
                -math.degrees(box_joints[0].get_angle(plant_context)),
                -math.degrees(lamp_joints[1].get_angle(plant_context)),
                -math.degrees(lamp_joints[0].get_angle(plant_context)),
            ),
            flush=True,
        )
    input("Press Enter to close...")


if __name__ == "__main__":
    main()
