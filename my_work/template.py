"""내 알고리즘을 Drake에서 테스트하기 위한 시작 템플릿.

실행:
    cd ~/Desktop/PIVOT/robot_learning
    ./scripts/run_drake_env.sh python ../my_work/template.py
    ./scripts/run_drake_env.sh python ../my_work/template.py --live   # Meshcat 시각화
"""

import argparse

import numpy as np
from pydrake.geometry import Box, StartMeshcat
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph, CoulombFriction
from pydrake.multibody.tree import SpatialInertia, UnitInertia
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.visualization import AddDefaultVisualization

TIME_STEP_S = 0.001
FRICTION = CoulombFriction(0.9, 0.8)


def build():
    """물체 하나가 바닥으로 떨어지는 최소 장면. 여기를 내 문제로 교체한다."""
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, TIME_STEP_S)

    # 바닥
    plant.RegisterCollisionGeometry(
        plant.world_body(),
        RigidTransform([0.0, 0.0, -0.05]),
        Box(4.0, 4.0, 0.1),
        "ground_collision",
        FRICTION,
    )
    plant.RegisterVisualGeometry(
        plant.world_body(),
        RigidTransform([0.0, 0.0, -0.05]),
        Box(4.0, 4.0, 0.1),
        "ground_visual",
        [0.6, 0.6, 0.6, 1.0],
    )

    # 자유롭게 움직이는 상자 하나
    mass_kg = 0.5
    size_m = 0.1
    body = plant.AddRigidBody(
        "block",
        SpatialInertia(
            mass_kg,
            np.zeros(3),
            UnitInertia.SolidBox(size_m, size_m, size_m),
        ),
    )
    shape = Box(size_m, size_m, size_m)
    plant.RegisterCollisionGeometry(
        body,
        RigidTransform(),
        shape,
        "block_collision",
        FRICTION,
    )
    plant.RegisterVisualGeometry(
        body, RigidTransform(), shape, "block_visual", [0.2, 0.5, 0.9, 1.0]
    )

    return builder, plant, scene_graph, body


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Meshcat 시각화")
    parser.add_argument("--duration-s", type=float, default=2.0)
    args = parser.parse_args()

    builder, plant, scene_graph, body = build()
    plant.Finalize()

    meshcat = None
    if args.live:
        meshcat = StartMeshcat()
        AddDefaultVisualization(builder, meshcat)

    diagram = builder.Build()
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()
    plant_context = plant.GetMyMutableContextFromRoot(context)

    # 초기 상태: 0.5 m 높이에서 살짝 기울여 놓기
    plant.SetFreeBodyPose(
        plant_context,
        body,
        RigidTransform(RotationMatrix.MakeXRotation(0.2), [0.0, 0.0, 0.5]),
    )

    if args.live:
        simulator.set_target_realtime_rate(1.0)
        input("Meshcat을 브라우저에서 연 뒤 Enter를 누르세요...")

    simulator.AdvanceTo(args.duration_s)

    pose = plant.EvalBodyPoseInWorld(plant_context, body)
    print(f"{args.duration_s}s 후 블록 높이: {pose.translation()[2]:.4f} m")

    if args.live:
        input("Meshcat 확인 후 Enter를 누르면 종료합니다...")


if __name__ == "__main__":
    main()
