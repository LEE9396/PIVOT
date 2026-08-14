"""관절 축을 카메라에 어떻게 두어야 각도가 잘 보이는지 **재본다**.

왜 재나
-------
"관절이 카메라를 향해야 한다" 는 말은 두 가지로 읽힌다.

    (가) 관절 축이 카메라를 **향한다**   (축 ∥ 시선)  -> 회전이 화면 안에서 돈다
    (나) 관절 축이 시선과 **직각이다**   (축 ⊥ 시선)  -> 회전이 화면 밖으로 나간다

어느 쪽이 각도를 잘 재는지는 말로 정할 일이 아니다. FoundationPose 는 결국
**화면에 보이는 변화**로 자세를 맞추므로, 각도를 1도 돌렸을 때 그림이 얼마나
바뀌는지가 그대로 각도 감도다. 그림이 거의 안 바뀌면 그 각도는 못 재는 것이다.

무엇을 재나
-----------
실험실 설정의 D456 카메라를 그대로 세우고, 물체를 여러 방향으로 돌려 가며

    축과 시선이 이루는 각 alpha = 0, 10, ..., 90 deg

마다 관절을 2도 돌린 전후의 **딱지 그림(label image)** 을 찍는다. 움직이는
링크의 실루엣이 얼마나 달라졌는지를

    민감도 = (달라진 화소 수) / (원래 실루엣 화소 수) / (돌린 각도)

로 잰다. 이 값이 클수록 그 자세에서 각도를 잘 잰다.

실행:
  ../robot_learning/scripts/run_drake_env.sh python study_startpose.py
  ../robot_learning/scripts/run_drake_env.sh python study_startpose.py --object 2link
"""

import argparse

import numpy as np
from pydrake.geometry import (ClippingRange, ColorRenderCamera, DepthRange,
                              DepthRenderCamera, MakeRenderEngineVtk,
                              RenderCameraCore, RenderEngineVtkParams)
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.sensors import CameraInfo, RgbdSensor

import density_id_objects as obj
import explore_view as ev
import robot_scene as rs

DELTA_DEG = 2.0                     # 감도를 잴 때 돌리는 각도
PRESENT_XYZ = (0.0, -0.08, 1.15)    # 물체를 들고 있을 자리 (제시 위치)


def camera_from_config(camera=rs.CAMERA):
    """실험실 설정의 D456 를 Drake 카메라로."""
    intrinsics = camera["depth_intrinsics"]
    width, height = camera["resolution"]
    info = CameraInfo(width, height, intrinsics["fx"], intrinsics["fy"],
                      intrinsics["cx"], intrinsics["cy"])
    core = RenderCameraCore("vtk", info, ClippingRange(0.05, 6.0),
                            RigidTransform())
    return core, rs.camera_pose(camera)


def view_direction(camera=rs.CAMERA, at=PRESENT_XYZ):
    """카메라에서 물체를 향하는 단위벡터."""
    direction = np.asarray(at, float) - rs.camera_pose(camera).translation()
    return direction / np.linalg.norm(direction)


def rotation_taking(a, b):
    """단위벡터 a 를 b 로 보내는 회전."""
    a = np.asarray(a, float) / np.linalg.norm(a)
    b = np.asarray(b, float) / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(a @ b)
    if np.linalg.norm(v) < 1e-12:
        if c > 0:
            return np.eye(3)
        helper = np.eye(3)[int(np.argmin(np.abs(a)))]
        axis = np.cross(a, helper)
        axis /= np.linalg.norm(axis)
        return RotationMatrix(np.array([axis, np.cross(a, axis), a]).T).matrix() @ -np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1.0 + c)


class Scene:
    """물체 하나와 카메라 하나. 로봇은 없다 — 여기서는 기하만 본다."""

    def __init__(self, spec):
        self.spec = spec
        densities = obj.bind_object(spec)
        builder = DiagramBuilder()
        self.plant, self.bodies = ev.build_floating(spec, densities, builder)
        scene_graph = builder.GetMutableSystems()[1]
        scene_graph.AddRenderer("vtk", MakeRenderEngineVtk(
            RenderEngineVtkParams()))
        core, pose = camera_from_config()
        sensor = builder.AddSystem(RgbdSensor(
            self.plant.GetBodyFrameIdOrThrow(self.plant.world_body().index()),
            pose, ColorRenderCamera(core, False),
            DepthRenderCamera(core, DepthRange(0.05, 6.0))))
        builder.Connect(scene_graph.get_query_output_port(),
                        sensor.query_object_input_port())
        self.sensor = sensor
        self.diagram = builder.Build()
        self.root = self.diagram.CreateDefaultContext()
        self.context = self.plant.GetMyMutableContextFromRoot(self.root)
        self.base = self.bodies[spec.parts[0].name]

    def place(self, R_WB, theta_deg):
        """바탕 링크의 방향을 R_WB 로 두고 관절각을 맞춘다."""
        self.plant.SetFreeBodyPose(
            self.context,
            self.base,
            RigidTransform(RotationMatrix(R_WB), np.array(PRESENT_XYZ)))
        for joint, value in zip(self.spec.joints, np.atleast_1d(theta_deg)):
            self.plant.GetJointByName(joint.name).set_angle(
                self.context, float(np.radians(value)))

    def axis_in_base(self, index, theta_deg):
        """관절 index 의 축을 **바탕 링크 좌표계**로 옮긴다.

        축은 그 관절의 부모 링크에 붙어 있으므로, 앞선 관절각에 따라
        바탕 기준 방향이 달라진다.
        """
        self.place(np.eye(3), theta_deg)
        joint = self.spec.joints[index]
        parent = self.bodies[joint.parent]
        R_WP = parent.body_frame().CalcRotationMatrixInWorld(
            self.context).matrix()
        R_WB = self.base.body_frame().CalcRotationMatrixInWorld(
            self.context).matrix()
        return R_WB.T @ R_WP @ np.asarray(joint.axis, float)

    def label(self):
        image = self.sensor.label_image_output_port().Eval(
            self.sensor.GetMyContextFromRoot(self.root))
        # 반드시 복사해야 한다. 반환값은 센서가 다시 그릴 때 덮어쓰는 버퍼를
        # 가리키므로, 복사하지 않으면 '전'과 '후'가 같은 그림이 된다.
        return np.array(image.data[:, :, 0], copy=True)

    def moving_mask(self, index):
        """관절 index 보다 아래쪽(움직이는) 링크들의 딱지 값."""
        names = [p.name for p in self.spec.parts][index + 1:]
        return names


def pixel_speed(scene, index, theta_deg, R_WB, camera=rs.CAMERA):
    """관절을 1도 돌릴 때 움직이는 링크가 화면에서 몇 화소 움직이나.

    실루엣 변화는 물체가 얼마나 크게 보이는지에 함께 휘둘린다. 이 값은
    그렇지 않다. 회전축 a 둘레로 점 p 가 도는 속도는

        v = a x (p - o)          (o = 축 위의 점)

    이고, 카메라가 보는 것은 그중 **시선에 수직인 성분**뿐이다. 시선 방향
    성분은 깊이로만 나타나 거의 안 보인다. 그래서 화면 속도는

        (f / z) * |v - (v . z_cam) z_cam|

    이다. 축이 시선과 나란하면 v 가 통째로 화면 안에 들어오고, 축이 시선과
    직각이면 절반이 시선 방향으로 새어 나간다.
    """
    scene.place(R_WB, theta_deg)
    joint = scene.spec.joints[index]
    context = scene.context
    parent = scene.bodies[joint.parent].body_frame()
    axis_W = (parent.CalcRotationMatrixInWorld(context).matrix()
              @ np.asarray(joint.axis, float))
    axis_W /= np.linalg.norm(axis_W)
    child_frame = scene.plant.GetFrameByName(f"{joint.name}_child")
    origin_W = child_frame.CalcPoseInWorld(context).translation()

    X_WC = rs.camera_pose(camera)
    R_CW = X_WC.rotation().matrix().T
    eye = X_WC.translation()
    focal = camera["depth_intrinsics"]["fx"]

    # 움직이는 링크들의 표본점: 각 부위 상자의 꼭짓점 8개.
    speeds = []
    for part in scene.spec.parts[index + 1:]:
        body = scene.bodies[part.name]
        X_WB = body.body_frame().CalcPoseInWorld(context)
        half = np.array(part.bbox_mm, float) * 0.5e-3
        for sign in np.ndindex(2, 2, 2):
            corner = half * (np.array(sign) * 2 - 1)
            p_W = X_WB @ corner
            v_W = np.cross(axis_W, p_W - origin_W)      # 1 rad/s 일 때 속도
            p_C = R_CW @ (p_W - eye)
            v_C = R_CW @ v_W
            depth = max(p_C[2], 1e-3)
            # 화면 속도 [px/rad] — 원근투영의 야코비안
            image_v = focal * (v_C[:2] - p_C[:2] * v_C[2] / depth) / depth
            speeds.append(np.linalg.norm(image_v))
    return float(np.mean(speeds)) * np.pi / 180.0      # px per degree


def sensitivity(scene, index, theta_deg, R_WB):
    """이 자세에서 관절 index 를 DELTA_DEG 돌렸을 때 실루엣이 얼마나 변하나."""
    theta = np.array(theta_deg, dtype=float)
    scene.place(R_WB, theta)
    before = scene.label()
    theta_after = theta.copy()
    theta_after[index] += DELTA_DEG
    scene.place(R_WB, theta_after)
    after = scene.label()

    # 물체 화소 = 배경·지면이 아닌 것. 움직이는 링크만 보려고 딱지를 쓰지 않고
    # '물체 전체 실루엣의 변화' 로 재는 편이 FoundationPose 가 보는 것에 가깝다.
    background = np.bincount(before.ravel()).argmax()
    mask_before = before != background
    mask_after = after != background
    changed = np.logical_xor(mask_before, mask_after).sum()
    area = mask_before.sum()
    if area == 0:
        return 0.0, 0
    return float(changed) / float(area) / DELTA_DEG, int(area)


def sweep(spec, index, theta_deg, n_alpha=10, verbose=True):
    """축과 시선이 이루는 각 alpha 를 0~90 도로 바꿔가며 감도를 잰다."""
    scene = Scene(spec)
    view = view_direction()
    axis_B = scene.axis_in_base(index, theta_deg)

    # alpha 를 만들 때 축을 어느 평면에서 기울일지 정한다. 시선과 직교하는
    # 아무 방향이나 잡으면 된다 (물체를 그 축 둘레로 돌리는 것과 같다).
    helper = np.eye(3)[int(np.argmin(np.abs(view)))]
    perp = np.cross(view, helper)
    perp /= np.linalg.norm(perp)

    rows = []
    for alpha in np.linspace(0.0, 90.0, n_alpha):
        target = (np.cos(np.radians(alpha)) * view
                  + np.sin(np.radians(alpha)) * perp)
        R_WB = rotation_taking(axis_B, target)
        speed = pixel_speed(scene, index, theta_deg, R_WB)
        value, area = sensitivity(scene, index, theta_deg, R_WB)
        rows.append((alpha, speed, value, area))
        if verbose:
            print(f"    alpha {alpha:5.1f} deg  |cos| {abs(np.cos(np.radians(alpha))):.3f}"
                  f"   화면속도 {speed:6.2f} px/deg"
                  f"   실루엣변화 {value:.5f} /deg  ({area:6d} px)")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--object", choices=("2link", "3link"), default="3link")
    parser.add_argument("--theta-deg", type=float, nargs="*", default=None)
    parser.add_argument("--plot", default="figures/study_startpose.png")
    args = parser.parse_args()

    spec = obj.OBJECTS[args.object]
    theta = (args.theta_deg if args.theta_deg is not None
             else [90.0] * len(spec.joints))
    obj.set_measurement_averaging()

    print(f"{spec.label}   관절각 {np.round(theta, 1)} deg")
    print(f"카메라 {rs.CAMERA_ID}  {rs.CAMERA['source']}"
          f"  위치 {np.round(rs.camera_pose(rs.CAMERA).translation(), 3)}")
    print(f"시선 방향 {np.round(view_direction(), 3)}\n")

    results = {}
    for index, joint in enumerate(spec.joints):
        print(f"  [{joint.name}]  축 {joint.axis} ({joint.parent} 기준)")
        results[joint.name] = sweep(spec, index, theta)
        rows = results[joint.name]
        best = max(rows, key=lambda row: row[1])
        worst = min(rows, key=lambda row: row[1])
        sil_best = max(rows, key=lambda row: row[2])
        print(f"    -> 화면속도: 가장 좋은 각 {best[0]:.0f} deg ({best[1]:.2f} px/deg),"
              f" 가장 나쁜 각 {worst[0]:.0f} deg ({worst[1]:.2f}),"
              f" 비 {best[1] / max(worst[1], 1e-9):.1f} 배")
        print(f"       실루엣변화: 가장 좋은 각 {sil_best[0]:.0f} deg\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for name, rows in results.items():
            alpha = [r[0] for r in rows]
            axes[0].plot(alpha, [r[1] for r in rows], "o-", label=name)
            axes[1].plot(alpha, [r[2] for r in rows], "o-", label=name)
        axes[0].set_ylabel("image speed of moving link [px/deg]")
        axes[1].set_ylabel("silhouette change [1/deg]")
        for ax in axes:
            ax.set_xlabel("angle between joint axis and camera ray [deg]")
            ax.legend()
            ax.grid(alpha=0.3)
        axes[0].set_title(f"angle observability — {args.object}")
        axes[1].set_title("cross-check (rendered)")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=130)
        print(f"그림 저장 -> {args.plot}")
    except Exception as exc:                          # noqa: BLE001
        print(f"(그림은 건너뜀: {exc})")


if __name__ == "__main__":
    main()
