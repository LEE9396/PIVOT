"""density_id_drake.py가 고른 관절 자세들을 눈으로 확인하기 위한 뷰어.

두 가지 방식:
  --live      Meshcat. 브라우저에서 돌려보고 확대/회전 가능
  --render    PNG로 저장. 브라우저 없이 확인 (frames/ 폴더)

실행:
    cd ~/Desktop/PIVOT/my_work
    ../robot_learning/scripts/run_drake_env.sh python view_density_id.py --live
    ../robot_learning/scripts/run_drake_env.sh python view_density_id.py --render
"""

import argparse
import time
from pathlib import Path

import numpy as np
from pydrake.geometry import (
    ClippingRange,
    DepthRange,
    DepthRenderCamera,
    MakeRenderEngineVtk,
    RenderCameraCore,
    RenderEngineVtkParams,
    Rgba,
    RenderEngineVtkParams,
    StartMeshcat,
)
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.sensors import CameraInfo, RgbdSensor
from pydrake.visualization import AddDefaultVisualization

import density_id_drake as alg

# 팔이 완전히 펴지면 기준점에서 반경 약 0.5 m까지 뻗으므로,
# 6개 자세를 모두 담을 수 있는 거리에 카메라를 둔다.
CAMERA_EYE = np.array([0.80, -1.10, 0.35])
CAMERA_TARGET = np.array([0.0, 0.0, 0.05])


def look_at(eye, target):
    """Drake 카메라 규약(z=전방, y=아래)에 맞춘 pose."""
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return RigidTransform(RotationMatrix(np.column_stack([x, y, z])), eye)


def build(with_camera=False):
    builder = DiagramBuilder()
    plant, bodies = alg.build_lamp_plant(alg.TRUE_RHO, builder=builder)
    scene_graph = builder.GetSubsystemByName("scene_graph")
    return builder, plant, scene_graph


def selected_configurations():
    """알고리즘이 실제로 고른 자세 순서 (main()과 동일한 seed)."""
    history = alg.run_loop(alg.select_active, n_rounds=6, seed=1)
    return [h["theta"] for h in history]


def interpolate(thetas, steps=40):
    """자세 사이를 부드럽게 이어 붙여 애니메이션용 경로를 만든다."""
    path = [np.asarray(thetas[0])]
    for a, b in zip(thetas[:-1], thetas[1:]):
        for s in np.linspace(0.0, 1.0, steps)[1:]:
            path.append((1.0 - s) * np.asarray(a) + s * np.asarray(b))
    return path


def run_live(thetas):
    meshcat = StartMeshcat()
    builder, plant, scene_graph = build()
    AddDefaultVisualization(builder, meshcat)
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyMutableContextFromRoot(context)

    print(f"\nMeshcat: {meshcat.web_url()}")
    input("브라우저에서 위 주소를 연 뒤 Enter를 누르세요...")

    for cycle in range(100):
        for index, theta in enumerate(interpolate(thetas)):
            plant.SetPositions(plant_context, theta)
            diagram.ForcedPublish(context)
            time.sleep(0.02)
        print(f"  재생 {cycle + 1}회 완료 (Ctrl-C로 종료)")


def run_render(thetas, out_dir):
    builder, plant, scene_graph = build()
    scene_graph.AddRenderer("vtk", MakeRenderEngineVtk(RenderEngineVtkParams()))
    camera = builder.AddSystem(
        RgbdSensor(
            scene_graph.world_frame_id(),
            look_at(CAMERA_EYE, CAMERA_TARGET),
            DepthRenderCamera(
                RenderCameraCore(
                    "vtk",
                    CameraInfo(800, 600, np.pi / 4),
                    ClippingRange(0.05, 10.0),
                    RigidTransform(),
                ),
                DepthRange(0.05, 10.0),
            ),
        )
    )
    builder.Connect(
        scene_graph.get_query_output_port(), camera.query_object_input_port()
    )
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyMutableContextFromRoot(context)
    camera_context = camera.GetMyContextFromRoot(context)

    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    for index, theta in enumerate(thetas):
        plant.SetPositions(plant_context, theta)
        image = camera.color_image_output_port().Eval(camera_context)
        path = out_dir / f"round{index + 1}_J1={theta[0]:+.2f}_J2={theta[1]:+.2f}.png"
        Image.fromarray(image.data[:, :, :3]).save(path)
        print(f"  저장 {path.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Meshcat 시각화")
    parser.add_argument("--render", action="store_true", help="PNG 저장")
    parser.add_argument("--out-dir", type=Path, default=Path("frames"))
    args = parser.parse_args()

    if not (args.live or args.render):
        parser.error("--live 또는 --render 중 하나를 지정하세요")

    thetas = selected_configurations()
    print("알고리즘이 선택한 자세 (J1, J2) [rad]:")
    for index, theta in enumerate(thetas):
        print(f"  round {index + 1}: {np.round(theta, 3)}")

    if args.render:
        run_render(thetas, args.out_dir)
    if args.live:
        run_live(thetas)


if __name__ == "__main__":
    main()
