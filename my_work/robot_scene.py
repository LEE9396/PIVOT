"""RB5 + AFT200 + PGC 그리퍼가 커스텀 물체를 실제로 잡은 Drake 씬.

지금까지의 검증은 물체만 공중에 띄워 놓고 돌렸다. 실제 실험에서는 로봇이
물체를 잡고 손목을 돌려 자세를 만들므로, 다음 두 가지가 추가로 걸린다.

  (a) 도달 가능성 — 그 자세를 만드는 팔 자세가 존재하는가
  (b) 충돌       — 팔·그리퍼·물체·테이블·카메라가 서로 부딪히지 않는가

실험실 배치(테이블, 로봇 받침대, D456 카메라 1대, 안전 촬영 영역)는
configs/experiments/icra_realistic_lab_scene_v1.json 값을 그대로 쓴다.
관절각은 다른 팀원이 맡은 FoundationPose 로 관측된다고 가정하고,
여기서는 카메라를 **장애물이자 가시성 판정 기준**으로만 다룬다.

이 모듈은 두 조건을 Drake의 역기구학과 최소거리 제약으로 직접 판정하고,
통과한 자세만 추정 파이프라인의 후보로 넘긴다.

기하 출처: third_party/HTD (RB5-850E URDF, PGC-140-50 URDF, AFT200 치수)

실행:
    cd ~/Desktop/PIVOT/my_work
    ../robot_learning/scripts/run_drake_env.sh python robot_scene.py --object 3link
"""

import argparse
import itertools
import os
import sys
from pathlib import Path

import numpy as np
from pydrake.geometry import (Box, CollisionFilterDeclaration, Convex,
                              Cylinder, GeometrySet, Mesh)
from pydrake.math import RigidTransform, RollPitchYaw, RotationMatrix
from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph, CoulombFriction
from pydrake.multibody.tree import (
    FixedOffsetFrame,
    RevoluteJoint,
    SpatialInertia,
    UnitInertia,
)
from pydrake.solvers import Solve
from pydrake.systems.framework import DiagramBuilder

import density_id_drake as alg
import density_id_objects as obj
import grippers as gr
import path_planning as pp

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "robot_learning" / "scripts"))
import visualize_drake_rb5_hammer_payload as rb5  # noqa: E402

MM = 1e-3
ARM_JOINT_NAMES = rb5.ARM_JOINT_NAMES

# AFT200 마운트: 브래킷 + 센서 퍽을 합친 원기둥 하나로 모형화한다.
AFT_TOTAL_H_M = 0.0523
AFT_DIAMETER_M = rb5.AFT200_DIAMETER_M
AFT_MASS_KG = rb5.AFT200_BRACKET_MASS_KG + rb5.AFT200_SENSOR_MASS_KG
AFT_CABLE_LENGTH_M = 0.050
AFT_CABLE_RADIUS_M = 0.010

PGC_TCP_Z_M = rb5.PGC_TCP_Z_M          # 0.125 — 그리퍼 베이스에서 파지점까지
PGC_FINGER_ORIGIN_Y_M = 0.0265         # 손가락 원점의 y (열림 최대)
PGC_STROKE_M = 0.025                   # 손가락당 행정
GRIPPER_MOUNT_YAW_RAD = np.pi / 2.0    # 실물 AFT 기준 그리퍼 장착 방향

# ---------------------------------------------------------------------------
# 실험실 배치 — configs/experiments/icra_realistic_lab_scene_v1.json 그대로.
# 좌표계 lab_world: x = 테이블 긴 축, y = 짧은 축(로봇 반대쪽이 +), z = 위.
# ---------------------------------------------------------------------------
LAB_SCENE = (WORKSPACE / "robot_learning" / "configs" / "experiments"
             / "icra_realistic_lab_scene_v1.json")


def load_lab_scene(path=LAB_SCENE):
    import json
    return json.loads(Path(path).read_text())


_LAB = load_lab_scene()

TABLE_SIZE_M = tuple(_LAB["table"]["size_xyz_m"])
TABLE_TOP_Z_M = _LAB["table"]["top_height_m"]
TABLE_CENTER_M = (_LAB["table"]["center_xy_m"][0],
                  _LAB["table"]["center_xy_m"][1],
                  TABLE_TOP_Z_M - TABLE_SIZE_M[2] / 2.0)

ROBOT_BASE_XYZ_M = tuple(_LAB["robot"]["base_xyz_m"])
ROBOT_BASE_RPY_DEG = tuple(_LAB["robot"]["base_rpy_deg"])


# ---------------------------------------------------------------------------
# 테이블은 **잰 값**을 쓴다 (있으면).
#
# 위의 TABLE_* 는 도면상의 명목값이다. 실물 테이블은 그 자리에 그 높이로
# 있지 않다. 명목이 실제보다 낮으면 계획한 경로가 진짜 테이블을 긁고,
# 높으면 갈 수 있는 자세가 막힌다. 램프처럼 여유가 3 mm 인 물체에서는
# 이 차이가 그대로 성패를 가른다.
#
# my_work/calibrate_table_rgbd.py 가 D456 depth + 손-눈 변환으로 RB5 기준
# 테이블 평면을 재서 calibration/rb5_table_current.json 에 쓴다. 그 파일이
# 있으면 그것을 쓰고, 없으면 명목값으로 간다 — 어느 쪽인지 반드시 찍는다.
# 조용히 명목으로 도는 것이 이 저장소에서 가장 비싼 실패 방식이었다.
#
#   1) 환경변수 TABLE_CALIBRATION
#   2) calibration/rb5_table_current.json
TABLE_CALIBRATION_NAME = "rb5_table_current.json"


def table_calibration_path():
    override = os.environ.get("TABLE_CALIBRATION")
    return (Path(override) if override
            else CALIBRATION_DIR / TABLE_CALIBRATION_NAME)


def _robot_base_pose():
    """lab_world <- RB5 베이스."""
    from pydrake.math import RollPitchYaw
    return RigidTransform(
        RollPitchYaw(np.deg2rad(np.asarray(ROBOT_BASE_RPY_DEG, dtype=float))),
        np.asarray(ROBOT_BASE_XYZ_M, dtype=float))


def load_table_calibration(path=None):
    """잰 테이블 평면을 lab_world 좌표의 (법선, 원점거리) 로 준다.

    파일이 없거나 status 가 valid 가 아니면 None. 캘리브레이션이 스스로
    "못 믿겠다" 고 적어 둔 값을 몰래 쓰면 명목보다 나쁘다.
    """
    import json

    path = Path(path) if path else table_calibration_path()
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    if data.get("status") != "valid":
        print(f"  [주의] 테이블 캘리브레이션 {path.name} 의 status 가"
              f" '{data.get('status')}' 라 쓰지 않습니다 — 명목값으로 갑니다")
        return None
    plane = data["plane_in_robot_base"]
    normal_B = np.asarray(plane["normal"], dtype=float)
    normal_B /= np.linalg.norm(normal_B)
    offset_B = float(plane["equation"][3])

    # 베이스 좌표의 평면 n_B.p_B + d_B = 0 을 lab_world 로 옮긴다.
    #   p_B = R_WB^T (p_W - t_WB)  =>  n_W = R_WB n_B,  d_W = d_B - n_W.t_WB
    X_WB = _robot_base_pose()
    normal_W = X_WB.rotation().matrix() @ normal_B
    offset_W = offset_B - float(normal_W @ X_WB.translation())
    if normal_W[2] < 0.0:
        normal_W, offset_W = -normal_W, -offset_W
    return dict(normal=normal_W, offset=offset_W, source=str(path),
                tilt_deg=float(plane.get("tilt_deg", 0.0)),
                rms_mm=float(data.get("quality", {}).get("rms_mm", float("nan"))),
                bounds=data.get("selected_xy_bounds_in_robot_base_m"))


def table_box_pose(calibration=None):
    """테이블 상자의 (크기, lab_world 자세). 캘리브레이션이 있으면 반영한다.

    상자의 **윗면**이 잰 평면 위에 오도록 놓는다. 상자 중심은 법선 반대쪽으로
    두께의 절반만큼 내린다.
    """
    from pydrake.math import RotationMatrix

    size = np.asarray(TABLE_SIZE_M, dtype=float)
    if calibration is None:
        return tuple(size), RigidTransform(np.asarray(TABLE_CENTER_M, float))

    normal = calibration["normal"]
    # 명목 중심의 x,y 를 유지한 채, 그 자리에서 잰 평면의 높이를 구한다.
    x, y = TABLE_CENTER_M[0], TABLE_CENTER_M[1]
    z_top = -(calibration["offset"] + normal[0] * x + normal[1] * y) / normal[2]
    top = np.array([x, y, z_top])
    # z 축이 법선을 향하는 회전 (기울기 반영).
    z_axis = normal / np.linalg.norm(normal)
    helper = np.array([1.0, 0.0, 0.0])
    if abs(z_axis @ helper) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    x_axis = np.cross(helper, z_axis); x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    rotation = RotationMatrix(np.column_stack([x_axis, y_axis, z_axis]))
    centre = top - z_axis * (size[2] / 2.0)
    return tuple(size), RigidTransform(rotation, centre)

# 측정 자세를 둘 안전 촬영 영역 (설정 파일의 safe_capture_box).
_BOX = np.array(_LAB["workspace"]["safe_capture_box_xyz_m"])
_BOX_C = np.array(_LAB["workspace"]["safe_capture_box_center_xyz_m"])
WORKSPACE_LOWER_M = _BOX_C - _BOX / 2.0
WORKSPACE_UPPER_M = _BOX_C + _BOX / 2.0

# 카메라는 Intel RealSense D456 한 대만 쓴다. 실제로 자리를 차지하는
# 장애물이므로 팔이 치면 안 되고, 한 대뿐이라 팔에 가리면 관측이 끊긴다.
CAMERA_ID = "cam_d456_front"
CAMERAS = [c for c in _LAB["cameras"] if c["id"] == CAMERA_ID]
if not CAMERAS:
    raise RuntimeError(f"{CAMERA_ID} 를 실험실 설정에서 찾지 못했다")

# 카메라 자세는 두 출처가 있다.
#
#   설정값   icra_realistic_lab_scene_v1.json 의 명목 위치. 도면상의 값이다.
#   실측값   실제로 손-눈 캘리브레이션을 한 뒤 나온 값.
#
# 각도 측정 자세는 **카메라가 어디서 보느냐**로 정해지므로, 캘리브레이션을
# 했으면 반드시 그 값을 써야 한다. 명목 위치로 계산한 자세는 실제 카메라
# 기준으로는 최적이 아니고, 심하면 물체가 화면 밖으로 나간다.
# 카메라는 장애물이기도 해서 충돌 판정도 같이 달라진다.
#
# 파일 위치는 이 순서로 찾는다.
#   1) 환경변수 CAMERA_CALIBRATION
#   2) calibration/camera_<id>.json   <- 캘리브레이션 결과를 여기에 둔다
#   3) 없으면 설정값 (그리고 그렇다고 알려 준다)
CALIBRATION_DIR = WORKSPACE / "calibration"


def calibration_path(camera_id=CAMERA_ID):
    import os
    override = os.environ.get("CAMERA_CALIBRATION")
    return Path(override) if override else (CALIBRATION_DIR
                                            / f"camera_{camera_id}.json")


def load_camera(camera_id=CAMERA_ID, path=None, announce=False):
    """카메라 정보를 돌려준다. 캘리브레이션 파일이 있으면 그것이 이긴다.

    파일 형식 (calibration/camera_cam_d456_front.json):

        {"id": "cam_d456_front",
         "X_WC": [[...4x4 행렬...]],          # 월드 <- 카메라 (Drake 규약)
         "depth_intrinsics": {"fx":..., "fy":..., "cx":..., "cy":...},
         "resolution": [1280, 720],
         "calibrated_at": "2026-08-14", "rms_px": 0.31, "method": "..."}

    X_WC 대신 position_xyz_m + look_at_xyz_m 만 적어도 된다 (대충 맞출 때).
    Drake 카메라 규약은 z 가 전방, y 가 아래다 — look_at_pose 가 그렇게 만든다.
    """
    import json

    base = dict(next(c for c in _LAB["cameras"] if c["id"] == camera_id))
    base["source"] = "설정값 (명목 위치)"
    path = Path(path) if path else calibration_path(camera_id)
    if not path.is_file():
        if announce:
            print(f"  카메라: {base['source']} — 캘리브레이션 파일이 없습니다"
                  f" ({path})")
        return base
    data = json.loads(path.read_text())
    base.update({k: v for k, v in data.items() if k != "source"})
    base["source"] = f"캘리브레이션 {path.name}"
    intrinsics_path = os.environ.get("PIVOT_CAMERA_INTRINSICS")
    if intrinsics_path and Path(intrinsics_path).is_file():
        live = json.loads(Path(intrinsics_path).read_text())
        base["depth_intrinsics"] = {
            key: float(live[key]) for key in ("fx", "fy", "cx", "cy")}
        base["resolution"] = [int(live["width"]), int(live["height"])]
        base["source"] += f" + 현재 스트림 {Path(intrinsics_path).name}"
    if announce:
        stamp = data.get("calibrated_at", "날짜 없음")
        rms = data.get("rms_px")
        print(f"  카메라: {base['source']} ({stamp}"
              + (f", 잔차 {rms:.2f} px)" if rms is not None else ")"))
    return base


def camera_pose(camera):
    """카메라의 월드 자세 X_WC. 캘리브레이션 행렬이 있으면 그것을 쓴다."""
    matrix = camera.get("X_WC")
    if matrix is not None:
        matrix = np.asarray(matrix, dtype=float)
        return RigidTransform(RotationMatrix(matrix[:3, :3]), matrix[:3, 3])
    return look_at_pose(camera["position_xyz_m"], camera["look_at_xyz_m"])


def camera_view(camera):
    """Meshcat용 카메라 위치와 광축 위 목표점."""
    X_WC = camera_pose(camera)
    position = X_WC.translation()
    return position, position + X_WC.rotation().matrix()[:, 2]


def save_calibration(X_WC, camera_id=CAMERA_ID, path=None, **extra):
    """캘리브레이션 결과를 파일로 남긴다. 실물 절차에서 이걸 부르면 된다."""
    import json

    path = Path(path) if path else calibration_path(camera_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.eye(4)
    matrix[:3, :3] = X_WC.rotation().matrix()
    matrix[:3, 3] = X_WC.translation()
    payload = dict(id=camera_id, X_WC=matrix.tolist(), **extra)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return path


CAMERA = load_camera()
CAMERA_BODY_SIZE_M = (0.124, 0.026, 0.029)   # D456 외형 (124 x 26 x 29 mm)
CAMERA_MOUNT_RADIUS_M = 0.016                # 지지대 봉
CAMERA_FOV_MARGIN_PX = int(os.environ.get("PIVOT_FOV_MARGIN_PX", "20"))


def _image_bounds(points_W, camera=CAMERA):
    """월드 점들의 영상 경계 (u_min, v_min, u_max, v_max)."""
    points_W = np.asarray(points_W, dtype=float)
    X_WC = camera_pose(camera)
    points_C = (X_WC.rotation().matrix().T
                @ (points_W - X_WC.translation()).T).T
    if not len(points_C) or np.any(points_C[:, 2] <= 0.0):
        return None
    intr = camera["depth_intrinsics"]
    uv = points_C[:, :2] / points_C[:, 2, None]
    uv[:, 0] = intr["fx"] * uv[:, 0] + intr["cx"]
    uv[:, 1] = intr["fy"] * uv[:, 1] + intr["cy"]
    return np.array([uv[:, 0].min(), uv[:, 1].min(),
                     uv[:, 0].max(), uv[:, 1].max()])


_PART_CORNERS = {}


def _part_corners(part):
    """FOV 판정에 쓸 실제 CoACD 메시 정점. 없으면 visual/AABB 순으로 대체."""
    key = (tuple(part.collision_meshes), part.visual_mesh,
           tuple(part.mesh_offset_m), tuple(part.bbox_mm))
    if key not in _PART_CORNERS:
        vertices = []
        paths = tuple(part.collision_meshes) or ((part.visual_mesh,)
                                                 if part.visual_mesh else ())
        for path in paths:
            try:
                import mesh_props
                points, _ = mesh_props.read_mesh(path)
                vertices.append(np.asarray(points, dtype=float))
            except (OSError, ValueError):
                pass
        if vertices:
            _PART_CORNERS[key] = (np.vstack(vertices)
                                  + np.asarray(part.mesh_offset_m, dtype=float))
        else:
            half = 0.5 * np.asarray(part.bbox_mm, dtype=float) * MM
            _PART_CORNERS[key] = np.array([
                [x, y, z] for x in (-half[0], half[0])
                for y in (-half[1], half[1]) for z in (-half[2], half[2])])
    return _PART_CORNERS[key]


def object_image_bounds(plant, context, spec, model, camera=CAMERA):
    """현재 자세의 물체 전체 visual mesh가 차지하는 영상 경계."""
    points = []
    for part in spec.parts:
        X_WB = plant.GetBodyByName(part.name, model).body_frame().CalcPoseInWorld(
            context)
        points.extend(X_WB @ corner for corner in _part_corners(part))
    return _image_bounds(points, camera)


def object_in_camera(plant, context, spec, model, camera=CAMERA,
                     margin_px=CAMERA_FOV_MARGIN_PX):
    """현재 자세에서 물체 전체가 보정된 카메라 영상 안에 있는가."""
    bounds = object_image_bounds(plant, context, spec, model, camera)
    if bounds is None:
        return False
    width, height = camera.get("resolution", (1280, 720))
    return bool(bounds[0] >= margin_px and bounds[1] >= margin_px
                and bounds[2] <= width - margin_px
                and bounds[3] <= height - margin_px)


def fov_self_test():
    camera = dict(X_WC=np.eye(4).tolist(),
                  depth_intrinsics=dict(fx=100, fy=100, cx=50, cy=40),
                  resolution=[100, 80])
    assert np.allclose(_image_bounds([[0, 0, 1]], camera), [50, 40, 50, 40])
    assert np.allclose(_image_bounds([[-.4, -.3, 1], [.4, .3, 1]], camera),
                       [10, 10, 90, 70])
    assert _image_bounds([[0, 0, -1]], camera) is None
    print("카메라 FOV 투영 자기검사 통과")

# 이 값보다 가까워지면 충돌로 본다. **자세와 경로 모두** 이 값을 지킨다.
#
# 실물에서는 6 mm 로 부족하다. 시뮬레이션에서 재는 간격은 '모형 사이의'
# 간격인데, 실물에는 모형에 없는 오차가 겹쳐 쌓이기 때문이다.
#
#   손-눈 보정 잔차        2~3 mm     (calibrate_camera 를 돌려도 남는다)
#   FoundationPose 자세     2~3 mm     (관절각 ±5% 가 끝단에서 만드는 어긋남)
#   파지점 어긋남           ~5 mm      (GRASP_SIGMA_M, 지그를 대도 남는다)
#   스캔 형상 오차          1~2 mm
#
# 이 중 파지점 어긋남은 이미 추정기가 미지수로 풀지만(design_core.grasp_columns),
# 그건 **렌치를 고치는** 것이지 충돌 검사에 반영되지는 않는다. 충돌 검사는
# 명목 형상만 본다. 그래서 여유를 형상 쪽에 따로 주어야 한다.
#
# 10 mm 에서 20 mm 로 올린다. 이유 둘.
#   1) 위 오차 예산의 합이 이미 10~13 mm 다. 10 mm 는 여유가 아니라 본전이었다.
#   2) 실물이 움직일 때 10 mm 는 사람 눈으로 위험 여부를 판단할 수 없다.
#      20 mm 면 보인다. 사람이 옆에 서 있는 실험이므로 이게 안전 조건이다.
# 후보가 줄지 않는 것은 확인했다 — 램프 4x4 격자에서 10 mm 와 20 mm 가 똑같이
# 8/16 을 통과한다. 22 mm 에서 9/16, 25 mm 에서 0/16 이므로 20 mm 는 절벽 앞
# 안전한 자리다. 가장 빠듯한 쌍은 link_2(램프 머리) <-> base_floor 였다.
MIN_DISTANCE_M = 0.020

# IK 에만 얹는 추가 여유.
#
# IK 는 최소거리 제약을 **경계에 딱 붙여** 푼다. 문턱을 10 mm 로 주면 해가
# 10.0000 mm 로 나오고, 그러면 같은 문턱을 쓰는 경로계획기와 이동 감시가
# 부동소수점 차이만으로 그 해를 탈락시킨다.
#
# 예전에는 계획기 문턱을 10% 낮춰(PLANNER_MARGIN_RATIO) 피했는데, 그러면
# 실제로 보장되는 간격이 문턱보다 작아진다 — 20 mm 를 요구해도 경로는
# 9 mm 까지 파고들 수 있었다. 여유는 **지켜야 하는 쪽을 낮추는 게 아니라
# 만드는 쪽을 높여서** 주는 것이 맞다. 그래서 IK 만 1 mm 더 요구한다.
IK_SLACK_M = 0.001

ANGLE_TOL_RAD = np.deg2rad(1.0)

# 물체 관절각을 얼마나 못 믿는가 (FoundationPose 실측 오차).
#
# 창 2 가 읽어 주는 각도에는 1~3 도의 오차가 있다. 충돌 검사는 명목 각도
# 하나만 보므로, 그 오차만큼 부위 끝단이 움직인 자리는 검사되지 않는다.
# 램프에서 얼마나 움직이는지 재보면:
#
#     head (joint_2 에서 최대 지레팔 304 mm)   1도 5.3 mm   2도 10.6   3도 15.9
#     base (joint_1 에서 최대 지레팔 151 mm)   1도 2.6 mm   2도  5.3   3도  7.9
#
# 3 도면 head 끝이 15.9 mm 움직인다 — MIN_DISTANCE_M(20 mm) 에 육박한다.
# 즉 "여유 20 mm" 라고 판정한 자세가 실제로는 4 mm 까지 줄어들 수 있다.
#
# 그래서 IK 로 찾은 팔 자세를 물체 각도 +-여유의 **모서리에서 다시 검사**한다.
# IK 를 다시 풀지 않고 arm_pose_is_clear 로 간격만 재므로 값이 싸다.
# 0 으로 두면 예전 동작(명목 각도만 검사).
ANGLE_MARGIN_RAD = np.deg2rad(
    float(os.environ.get("PIVOT_ANGLE_MARGIN_DEG", "0.0")))

# 물체를 그리퍼에 어떻게 물릴지.
#
# 죠는 그리퍼의 x 방향으로 열린다. 그러므로 물체에서 **죠가 물어야 할 축**
# (가장 좁은 단면 방향) 을 그리퍼 x 로 보내야 한다. 예전에는 roll 하나만
# 주었는데, 그러면 물체의 x 축이 그대로 죠 방향에 남는다. 3-link 는 그
# 축이 150 mm 길이 방향이라, 44 mm 로 오므린 패드가 물체 옆구리에 걸려
# 화면에서 물체가 손가락 위에 떠 있는 것처럼 보였다.
#
# 가장 긴 축은 죠 사이를 가로지르도록 그리퍼 y 로 보낸다. 패드가 그 방향으로
# 넓어서(2F-85 는 22 mm) 길쭉한 물체를 안정적으로 문다.
GRASP_ROLL_DEG = {}          # 남겨 둔다: 특정 물체만 손으로 돌리고 싶을 때


def grasp_axes(spec):
    """(죠가 무는 축, 길이 축) 을 물체 좌표계의 축 번호로 돌려준다.

    사양이 파지 축을 알려주면 그것을 쓴다 (스캔 물체는 메시로 단면을 재서
    어느 축이 가장 좁은지 함께 알려준다). 없으면 AABB 에서 고른다.
    """
    part = spec.parts[0]
    dims = np.asarray(part.bbox_mm, dtype=float)
    jaw = getattr(part, "grasp_axis", None)
    length = getattr(part, "grasp_long_axis", None)
    if jaw is None:
        jaw = int(np.argmin(dims))
    if length is None:
        length = int(np.argmax(dims))
        if length == jaw:                   # 정육면체 같은 경우
            length = int(np.argmax([d if k != jaw else -1.0
                                    for k, d in enumerate(dims)]))
    # 축 번호로 준 것은 단위벡터로 바꾼다. 이후는 벡터 하나로 다룬다.
    jaw = np.eye(3)[jaw] if np.isscalar(jaw) else np.asarray(jaw, float)
    length = (np.eye(3)[length] if np.isscalar(length)
              else np.asarray(length, float))
    return jaw, length


# 길쭉한 축을 그리퍼의 어느 방향으로 둘 것인가.
#   "z"  손목 축을 따라 뻗는다 (드라이버를 쥔 모양). 팔이 훨씬 자유롭다.
#   "y"  죠를 가로질러 옆으로 뻗는다. 물체가 옆으로 튀어나와 자주 막힌다.
# 재보니 3-link 는 y 로 두면 안전한 시작 자세가 아예 없고, z 로 두면
# 25/25 가 통과했다. 그래서 기본을 z 로 둔다.
GRASP_LONG_AXIS = "z"

# 그런데 "z" 는 **사슬의 끝을 잡을 때만** 맞는 규약이다. 파지점 뒤로는
# 그리퍼 몸통(0~147 mm)과 AFT200 마운트(147~199 mm)가 있으므로, 파지점
# 뒤쪽에 물체가 있으면 안 된다.
#
#   2link  parent  75 mm, 파지점이 끝면   -> 뒤쪽 0 mm      z 로 안전
#   3link  link0  150 mm, 파지점이 끝면   -> 뒤쪽 0 mm      z 로 안전
#   램프   support 330 mm, 파지점이 한가운데, 게다가 base 와 head 가
#          **양쪽으로** 갈라진 트리 -> 부호를 어떻게 잡아도 한쪽이 뒤로 간다
#            head 를 +z 로 두면 base 가 마운트 안으로 112 점
#            base 를 +z 로 두면 head 가 그리퍼 몸통 안으로 481 점
#
# 갈래 구조는 장축을 죠에 **가로질러**(y) 두어야 양쪽이 다 옆으로 빠진다.
# 전역으로 y 를 주면 안 된다 — 한 방향 사슬인 3link 는 y 로 두면 안전한
# 시작 자세가 아예 없었다 (위 주석).
GRASP_LONG_AXIS_BY_OBJECT = {"desklamp": "y"}


def grasp_rotation(spec, long_axis=None):
    """물체 좌표계를 그리퍼 좌표계로 보내는 회전 R_GO.

    죠 축은 반드시 그리퍼 x (죠가 열리는 방향) 로 간다. 길이 축은
    GRASP_LONG_AXIS 에 따라 y 또는 z 로 보낸다.
    """
    jaw, length = grasp_axes(spec)
    long_axis = (long_axis
                 or GRASP_LONG_AXIS_BY_OBJECT.get(getattr(spec, "key", None))
                 or GRASP_LONG_AXIS)
    jaw_vector = np.asarray(jaw, dtype=float)
    long_vector = np.asarray(length, dtype=float)
    jaw_vector = jaw_vector / np.linalg.norm(jaw_vector)
    # 길이 축을 죠 축에 직교화한다 (볼록 조각의 주축은 정확히 직교하지 않는다)
    long_vector = long_vector - (long_vector @ jaw_vector) * jaw_vector
    long_vector = long_vector / np.linalg.norm(long_vector)
    basis = np.zeros((3, 3))
    basis[:, 0] = jaw_vector
    if long_axis == "z":
        basis[:, 2] = long_vector
        basis[:, 1] = np.cross(basis[:, 2], basis[:, 0])
    else:
        basis[:, 1] = long_vector
        basis[:, 2] = np.cross(basis[:, 0], basis[:, 1])
    roll = np.deg2rad(GRASP_ROLL_DEG.get(spec.key, 0.0))
    return RotationMatrix.MakeXRotation(roll) @ RotationMatrix(basis.T)


def load_measured_grasp(explicit=None):
    """잰 파지 변환 X_G_O (4x4) 를 찾는다. 없으면 None -> 짐작으로 간다.

    찾는 순서
      1) 인자로 직접 준 것 (4x4 배열 또는 파일 경로)
      2) 환경변수 PIVOT_GRASP_FILE
      3) 환경변수 PIVOT_SESSION 아래의 grasp.json
    """
    import json

    if explicit is not None and not isinstance(explicit, (str, Path)):
        return np.asarray(explicit, dtype=float).reshape(4, 4)
    candidates = []
    if explicit is not None:
        candidates.append(Path(explicit))
    env = os.environ.get("PIVOT_GRASP_FILE")
    if env:
        candidates.append(Path(env))
    session = os.environ.get("PIVOT_SESSION")
    if session:
        candidates.append(Path(session) / "grasp.json")
    for path in candidates:
        if not path.is_file():
            continue
        data = json.loads(path.read_text())
        matrix = data.get("X_G_O", data) if isinstance(data, dict) else data
        matrix = np.asarray(matrix, dtype=float).reshape(4, 4)
        print(f"  파지 변환: **잰 값** {path}"
              f"  위치 {np.round(1000 * matrix[:3, 3], 1)} mm")
        return matrix
    return None


def look_at_pose(eye, target):
    """Drake 카메라 규약(z=전방, y=아래)에 맞춘 pose."""
    z = np.asarray(target, dtype=float) - np.asarray(eye, dtype=float)
    z /= np.linalg.norm(z)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return RigidTransform(RotationMatrix(np.column_stack([x, y, z])),
                          np.asarray(eye, dtype=float))


# ---------------------------------------------------------------------------
def _add_object(plant, spec, densities, joint_limits_rad):
    """커스텀 물체를 plant 에 추가한다. body frame 원점 = 외형 중심."""
    model = plant.AddModelInstance("payload")
    parts = {p.name: p for p in spec.parts}
    bodies = {}
    # 힌지 질량은 여기서 다루지 않는다. 이 plant 는 도달·충돌 판정 전용이라
    # 질량이 결과를 바꾸지 않고, 힌지 실물은 링크 단면 안쪽에 들어가 있어
    # 충돌 형상을 따로 둘 필요도 없다. 밀도 벡터 뒤쪽(힌지분)은 무시한다.
    for part, rho in zip(spec.parts, densities[:len(spec.parts)]):
        dims_m = tuple(d * MM for d in part.bbox_mm)
        body = plant.AddRigidBody(
            part.name, model,
            SpatialInertia(rho * part.volume_m3, np.zeros(3),
                           UnitInertia.SolidBox(*dims_m)),
        )
        # 화면: 스캔 색 조각 -> 스캔 메시 -> bbox 상자 순으로 그린다.
        X_mesh = RigidTransform(np.array(part.mesh_offset_m))
        obj.register_part_visual(plant, body, part, dims_m)
        # 충돌: 볼록 조각들이 있으면 그것을 쓴다. AABB 상자는 실물보다
        # 훨씬 뚱뚱해서 (램프 Arm 은 3배) 갈 수 있는 자세를 과하게 막는다.
        if part.collision_meshes:
            for index, path in enumerate(part.collision_meshes):
                plant.RegisterCollisionGeometry(
                    body, X_mesh, Convex(path, 1.0),
                    f"{part.name}_collision_{index}",
                    CoulombFriction(0.9, 0.8))
        else:
            plant.RegisterCollisionGeometry(
                body, RigidTransform(), Box(*dims_m),
                f"{part.name}_collision", CoulombFriction(0.9, 0.8))
        bodies[part.name] = body

    for joint, limits in zip(spec.joints, joint_limits_rad):
        origin = np.array(joint.origin_in_parent_link_mm)
        on_parent = (origin
                     - np.array(parts[joint.parent].bbox_center_in_link_mm)) * MM
        child_origin = np.array(joint.origin_in_child_link_mm
                                if joint.origin_in_child_link_mm is not None
                                else (0.0, 0.0, 0.0))
        on_child = (child_origin
                    - np.array(parts[joint.child].bbox_center_in_link_mm)) * MM
        revolute = plant.AddJoint(
            RevoluteJoint(
                joint.name,
                plant.AddFrame(FixedOffsetFrame(
                    f"{joint.name}_parent", bodies[joint.parent].body_frame(),
                    RigidTransform(on_parent))),
                plant.AddFrame(FixedOffsetFrame(
                    f"{joint.name}_child", bodies[joint.child].body_frame(),
                    RigidTransform(on_child))),
                joint.axis, damping=0.0,
            )
        )
        revolute.set_position_limits([limits[0]], [limits[1]])

    # 물체의 센서 프레임 = base part 의 링크 프레임 원점.
    sensor_frame = plant.AddFrame(FixedOffsetFrame(
        "obj_sensor", bodies[spec.parts[0].name].body_frame(),
        RigidTransform(-np.array(spec.base_bbox_center_in_sensor_mm) * MM),
    ))
    return model, bodies, sensor_frame


def build_scene(spec, densities=None, joint_limits_rad=None,
                builder=None, include_visuals=True, gripper="robotiq2f85",
                grasp_transform=None, payload_pose_tcp=None,
                include_aft_cable=False):
    """RB5 + AFT200 + 그리퍼 + 물체(그리퍼에 고정)를 한 plant 로 만든다.

    gripper 는 grippers.GRIPPERS 의 키다. 개구량이 달라 잡을 수 있는 물체가
    갈린다 (PGC 53 mm vs Robotiq 78 mm).
    """
    gripper_spec = gr.GRIPPERS[gripper]
    if densities is None:
        densities = [row["rho_gt"] for row in obj.body_table(spec)]
    if joint_limits_rad is None:
        joint_limits_rad = [j.limits_rad for j in spec.joints]

    htd = rb5.DEFAULT_HTD_ROOT
    rb5.validate_htd_source(htd)
    urdf = htd / rb5.RB5_URDF_RELATIVE
    pgc_urdf = htd / rb5.PGC_URDF_RELATIVE

    if builder is None:
        builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    parser = Parser(plant)
    parser.package_map().Add("htd", str(htd))

    # --- 로봇 (받침대 위, yaw 90도로 테이블을 바라본다) ---
    arm = parser.AddModelsFromString(
        rb5.rb5_urdf_string(urdf, "rb5", include_visuals), "urdf")[0]
    base_rpy = np.deg2rad(ROBOT_BASE_RPY_DEG)
    X_base = RigidTransform(
        RotationMatrix(RollPitchYaw(*base_rpy)), np.array(ROBOT_BASE_XYZ_M))
    plant.WeldFrames(plant.world_frame(),
                     plant.GetFrameByName("link0", arm), X_base)

    # --- AFT200 마운트 ---
    mount_model = plant.AddModelInstance("aft200")
    mount = plant.AddRigidBody(
        "ft_mount", mount_model,
        SpatialInertia(AFT_MASS_KG, np.zeros(3),
                       UnitInertia.SolidCylinder(AFT_DIAMETER_M / 2.0,
                                                 AFT_TOTAL_H_M, [0, 0, 1])),
    )
    shape = Cylinder(AFT_DIAMETER_M / 2.0, AFT_TOTAL_H_M)
    if include_visuals:
        plant.RegisterVisualGeometry(mount, RigidTransform(), shape,
                                     "ft_mount_visual", [0.2, 0.2, 0.22, 1.0])
    plant.RegisterCollisionGeometry(mount, RigidTransform(), shape,
                                    "ft_mount_collision", CoulombFriction(0.9, 0.8))
    # RB5 flange extends along TCP -Y; rotate the tool so its +Z follows it.
    plant.WeldFrames(
        plant.GetFrameByName("tcp", arm), mount.body_frame(),
        RigidTransform(RotationMatrix(
                           RotationMatrix.MakeXRotation(np.pi / 2.0).matrix()
                           @ RotationMatrix.MakeZRotation(
                               GRIPPER_MOUNT_YAW_RAD).matrix()),
                       [0.0, -AFT_TOTAL_H_M / 2.0, 0.0]))

    # 케이블은 실행 경로(RRT) 장면에만 추가한다. 후보 IK와 기존 로봇 충돌
    # 모델은 케이블을 넣기 전과 완전히 같게 유지한다.
    cable = None
    if include_aft_cable:
        cable_model = plant.AddModelInstance("aft200_cable")
        cable = plant.AddRigidBody(
            "ft_cable_keepout", cable_model,
            SpatialInertia(0.01, np.zeros(3),
                           UnitInertia.SolidCylinder(AFT_CABLE_RADIUS_M,
                                                     AFT_CABLE_LENGTH_M,
                                                     [0, 0, 1])))
        cable_shape = Cylinder(AFT_CABLE_RADIUS_M, AFT_CABLE_LENGTH_M)
        if include_visuals:
            plant.RegisterVisualGeometry(cable, RigidTransform(), cable_shape,
                                         "ft_cable_keepout_visual",
                                         [1.0, 0.35, 0.05, 0.65])
        plant.RegisterCollisionGeometry(
            cable, RigidTransform(), cable_shape, "ft_cable_keepout_collision",
            CoulombFriction(0.9, 0.8))
        plant.WeldFrames(
            mount.body_frame(), cable.body_frame(),
            RigidTransform(RotationMatrix.MakeYRotation(-np.pi / 2.0),
                           [-(AFT_DIAMETER_M / 2.0
                              + AFT_CABLE_LENGTH_M / 2.0), 0.0, 0.0]))

    # --- 그리퍼 ---
    opening = jaw_opening_for(spec, gripper_spec)
    if gripper_spec.key == "pgc140":
        gripper_urdf = rb5.pgc_movable_urdf_string(pgc_urdf, "pgc",
                                                   include_visuals)
        tcp_z = gripper_spec.tcp_z_m
    else:
        gripper_urdf = gr.robotiq_urdf_string(opening, include_visuals)
        tcp_z = gr.robotiq_tcp_z(opening)
    gripper_model = parser.AddModelsFromString(gripper_urdf, "urdf")[0]
    plant.WeldFrames(mount.body_frame(),
                     plant.GetFrameByName(gripper_spec.base_frame,
                                          gripper_model),
                     RigidTransform([0.0, 0.0, AFT_TOTAL_H_M / 2.0]))
    gripper = gripper_model

    # --- 물체를 그리퍼 파지점에 고정 (운동학적 파지) ---
    #
    # 여기가 이 파일에서 제일 조용히 틀리는 자리다. 물체가 그리퍼의
    # **어디에** 붙어 있는지를 정하는데, 기본값은 pinch_grasp() 가 볼록
    # 조각에서 짐작한 값이다. 시뮬에서는 그 짐작이 곧 정답이라 절대 안
    # 틀리지만(렌치도 같은 용접에서 나온다), 실물에서는 사람이 놓은 자리와
    # 맞아야 한다. 2 mm 어긋나면 밀도가 113 % 틀린다 (study_grasp.py).
    #
    # 그래서 **잰 값이 있으면 그것이 이긴다.** tools/grasp_measure.py 가
    # 창 2 의 물체 자세 + 핸드아이 + 로봇 q 로 X_G_O 를 계산해 세션에
    # 남긴다. 그 값이 오면 grasp_rotation / GRASP_LONG_AXIS /
    # base_bbox_center_in_sensor_mm 같은 짐작이 전부 안 쓰인다.
    payload, parts, sensor_frame = _add_object(
        plant, spec, densities, joint_limits_rad)
    measured = load_measured_grasp(grasp_transform)
    if payload_pose_tcp is not None:
        matrix = np.asarray(payload_pose_tcp, dtype=float).reshape(4, 4)
        X_tcp_object = RigidTransform(
            RotationMatrix(matrix[:3, :3]), matrix[:3, 3])
        plant.WeldFrames(plant.GetFrameByName("tcp", arm),
                         parts[spec.parts[0].name].body_frame(),
                         X_tcp_object)
    elif measured is not None:
        X_gripper_object = RigidTransform(
            RotationMatrix(measured[:3, :3]), measured[:3, 3])
        plant.WeldFrames(plant.GetFrameByName(gripper_spec.base_frame, gripper),
                         parts[spec.parts[0].name].body_frame(),
                         X_gripper_object)
    else:
        X_pgc_sensor = RigidTransform(grasp_rotation(spec), [0.0, 0.0, tcp_z])
        X_sensor_base = RigidTransform(
            np.array(spec.base_bbox_center_in_sensor_mm) * MM)
        X_gripper_object = X_pgc_sensor @ X_sensor_base
        plant.WeldFrames(plant.GetFrameByName(gripper_spec.base_frame, gripper),
                         parts[spec.parts[0].name].body_frame(),
                         X_gripper_object)

    # --- 실험실 고정물: 테이블, 로봇 받침대, 카메라 3대 ---
    fixtures = plant.AddModelInstance("lab")

    def add_fixture(name, shape, pose, rgba, collide=True):
        body = plant.AddRigidBody(
            name, fixtures,
            SpatialInertia(1.0, np.zeros(3), UnitInertia.SolidBox(0.1, 0.1, 0.1)))
        if include_visuals:
            plant.RegisterVisualGeometry(body, RigidTransform(), shape,
                                         f"{name}_visual", rgba)
        if collide:
            plant.RegisterCollisionGeometry(
                body, RigidTransform(), shape, f"{name}_collision",
                CoulombFriction(0.9, 0.8))
        plant.WeldFrames(plant.world_frame(), body.body_frame(), pose)
        return body

    table_calibration = load_table_calibration()
    table_size, table_pose = table_box_pose(table_calibration)
    if table_calibration is None:
        print(f"  [주의] 테이블이 **명목값**입니다 (도면 {1000*TABLE_TOP_Z_M:.0f} mm)."
              f" 실측을 쓰려면 calibrate_table_rgbd.py 로"
              f" {table_calibration_path()} 를 만드세요")
    else:
        print(f"  테이블 실측 반영: 윗면 z {1000*table_pose.translation()[2] + 500*table_size[2]:.1f} mm"
              f" (명목 {1000*TABLE_TOP_Z_M:.0f}), 기울기"
              f" {table_calibration['tilt_deg']:.2f} deg,"
              f" rms {table_calibration['rms_mm']:.2f} mm")
    add_fixture("table", Box(*table_size), table_pose,
                [*_LAB["table"]["surface_rgb"], 1.0])
    # 받침대 형상 대신 base 장착면 아래 전체를 금지한다. 따라서 움직이는
    # 링크뿐 아니라 그리퍼와 잡힌 물체도 이 높이 아래로 내려갈 수 없다.
    base_floor_depth = 2.0
    base_floor_width = 2.0 * 0.85  # RB5 최대 도달 반경만 덮는다.
    base_floor = add_fixture(
        "base_floor", Box(base_floor_width, base_floor_width, base_floor_depth),
        RigidTransform([ROBOT_BASE_XYZ_M[0], ROBOT_BASE_XYZ_M[1],
                        ROBOT_BASE_XYZ_M[2] - base_floor_depth / 2.0]),
        [0.30, 0.31, 0.33, 1.0])

    # 카메라 본체와 지지봉. 팔이 여기에 닿으면 안 되므로 충돌 대상이다.
    # 캘리브레이션을 했으면 **실측 자세**로 세운다. 명목 위치로 세워 두면
    # 실제로는 부딪히는 자세를 통과시키게 된다.
    for camera in ([CAMERA] if CAMERA["id"] == CAMERA_ID else CAMERAS):
        pose = camera_pose(camera)
        position = pose.translation()
        add_fixture(camera["id"], Box(*CAMERA_BODY_SIZE_M), pose,
                    [0.12, 0.13, 0.15, 1.0])
        # 바닥에서 카메라까지 올라오는 지지봉
        mast_h = position[2]
        add_fixture(
            f"{camera['id']}_mast",
            Cylinder(CAMERA_MOUNT_RADIUS_M, mast_h),
            RigidTransform([position[0], position[1], mast_h / 2.0]),
            [0.22, 0.23, 0.25, 1.0])

    plant.Finalize()

    # --- 의도된 접촉은 충돌 판정에서 제외 ---
    # 그리퍼가 물체를 잡고 있으므로 손가락·그리퍼 몸통·마운트와 base part 는
    # 닿아 있는 것이 정상이다. 나머지 쌍은 모두 검사한다.
    # 엔드이펙터 쪽: 마운트·그리퍼·잡힌 base part 는 서로 붙어 있는 것이 정상.
    end_effector = [mount]
    end_effector += [plant.GetBodyByName(n, gripper)
                     for n in gripper_spec.body_names]
    end_effector += [parts[spec.parts[0].name]]
    # 손목 링크: AFT200 이 link6 플랜지에 볼트로 붙으므로 항상 겹친다.
    wrist = [plant.GetBodyByName(n, arm) for n in ("link5", "link6")]

    def ids(bodies):
        out = []
        for body in bodies:
            out += list(plant.GetCollisionGeometriesForBody(body))
        return GeometrySet(out)

    manager = scene_graph.collision_filter_manager()
    manager.Apply(CollisionFilterDeclaration().ExcludeWithin(ids(end_effector)))
    manager.Apply(CollisionFilterDeclaration().ExcludeBetween(
        ids(end_effector), ids(wrist)))
    if cable is not None:
        manager.Apply(CollisionFilterDeclaration().ExcludeBetween(
            ids([cable]), ids([mount])))
    manager.Apply(CollisionFilterDeclaration().ExcludeBetween(
        ids([base_floor]), ids([plant.GetBodyByName("link0", arm)])))
    # link0..link4 와 물체의 나머지 부위, 테이블은 그대로 검사한다.

    return dict(
        builder=builder, plant=plant, scene_graph=scene_graph,
        arm=arm, gripper=gripper, payload=payload, parts=parts,
        mount=mount, cable=cable, sensor_frame=sensor_frame, spec=spec,
        gripper_spec=gripper_spec, jaw_opening_m=opening, tcp_z_m=tcp_z,
    )


# ---------------------------------------------------------------------------
def jaw_dimension_m(spec):
    """죠가 물어야 하는 물체 단면 [m].

    사양이 파지 단면을 직접 알려주면 그걸 쓴다. 없으면 AABB 로 짐작한다.
    직육면체 부품은 AABB 가 곧 단면이라 맞지만, 굽은 팔 같은 형상은
    AABB 가 실제 단면보다 훨씬 크다.
    """
    part = spec.parts[0]
    if part.grasp_width_mm is not None:
        return part.grasp_width_mm * MM
    jaw, _ = grasp_axes(spec)
    # 벡터 축이면 AABB 를 그 방향으로 투영해 대략의 폭을 쓴다.
    return float(np.abs(np.asarray(part.bbox_mm, float) @ np.abs(jaw))) * MM


def jaw_opening_for(spec, gripper_spec):
    """그 물체를 물었을 때의 개구량 [m].

    단면이 개구량보다 크면 실물에서는 못 잡는다. 시뮬레이션은 파지를
    용접으로 모형화하므로 그대로 돌긴 하지만, 여기서 알려는 준다.
    """
    needed = jaw_dimension_m(spec)
    if needed > gripper_spec.max_opening_m:
        print(f"  [경고] {spec.key} 단면 {1000*needed:.0f} mm 가 "
              f"{gripper_spec.label} 개구 {1000*gripper_spec.max_opening_m:.0f} mm"
              f" 를 넘습니다 — 실물에서는 못 잡습니다")
    return float(np.clip(needed, 0.0, gripper_spec.max_opening_m))


def finger_position_for(spec):
    """PGC 손가락 관절 위치 [m]. Robotiq 은 손가락이 굳어 있어 안 쓴다."""
    half = jaw_dimension_m(spec) / 2.0
    return float(np.clip(PGC_FINGER_ORIGIN_Y_M - half - 0.001,
                         0.0, PGC_STROKE_M))


class PoseChecker:
    """자세 (물체 관절각 θ, 중력방향 ĝ) 가 로봇으로 실현 가능한지 판정."""

    def __init__(self, spec, densities=None, joint_limits_rad=None,
                 min_distance_m=MIN_DISTANCE_M, seed_q=None, ik_restarts=4,
                 gripper="robotiq2f85", angle_margin_rad=ANGLE_MARGIN_RAD,
                 payload_pose_tcp=None):
        scene = build_scene(spec, densities, joint_limits_rad,
                            include_visuals=False, gripper=gripper,
                            payload_pose_tcp=payload_pose_tcp)
        self.spec = spec
        self.plant = scene["plant"]
        self.arm = scene["arm"]
        self.gripper = scene["gripper"]
        self.payload = scene["payload"]
        self.sensor_frame = scene["sensor_frame"]
        # min_distance_m 은 '지켜야 하는 값', ik_distance_m 은 'IK 에게
        # 요구하는 값'. 둘을 나눠 두어야 IK 해가 검사를 통과한다.
        self.min_distance_m = min_distance_m
        self.ik_distance_m = min_distance_m + IK_SLACK_M
        # 물체 각도를 못 믿는 폭. 창 2 (FoundationPose) 의 실측 오차를 넣는다.
        self.angle_margin_rad = float(angle_margin_rad)
        # 최소거리 제약은 SceneGraph 질의가 필요하므로 diagram 을 완성한 뒤
        # 그 안의 plant 서브컨텍스트를 써야 한다.
        self.diagram = scene["builder"].Build()
        self.diagram_context = self.diagram.CreateDefaultContext()
        self.context = self.plant.GetMyContextFromRoot(self.diagram_context)

        self.arm_joints = [self.plant.GetJointByName(n, self.arm)
                           for n in ARM_JOINT_NAMES]
        self.gripper_spec = scene["gripper_spec"]
        # Robotiq 은 손가락을 굳혀 붙이므로 관절이 없다.
        self.finger_joints = [self.plant.GetJointByName(n, self.gripper)
                              for n in self.gripper_spec.finger_joint_names]
        self.object_joints = [self.plant.GetJointByName(j.name, scene["payload"])
                              for j in spec.joints]
        # PGC 는 손가락 관절이 있고, Robotiq 은 굳혀 붙였다. 후자면 이 값은
        # 화면 표시용 개구량일 뿐 IK 제약으로 안 쓰인다.
        self.finger_value = (finger_position_for(spec)
                             if self.gripper_spec.finger_joint_names
                             else scene["jaw_opening_m"])
        self.jaw_opening_m = scene["jaw_opening_m"]
        self.tcp_z_m = scene["tcp_z_m"]
        # 팔꿈치를 세운 자연스러운 초기 자세.
        self.seed_q = (np.array(seed_q) if seed_q is not None
                       else np.deg2rad([0.0, -50.0, 100.0, -50.0, -90.0, 0.0]))
        self._last_solution = None
        # IK 가 실패했을 때 초기추측을 흔들어 다시 시도하는 횟수.
        # 0 이면 예전 동작(한 번만 시도).
        self.ik_restarts = ik_restarts

    # ------------------------------------------------------------------
    def solve(self, theta, g_hat, warm_start=True, workspace=None):
        """IK 를 풀어 팔 자세를 찾는다. 실패하면 None."""
        ik = InverseKinematics(self.plant, self.context,
                               with_joint_limits=True)
        prog = ik.prog()
        q = ik.q()

        # 물체 관절각과 손가락은 고정값으로 묶는다.
        for joint, value in zip(self.object_joints, np.atleast_1d(theta)):
            index = joint.position_start()
            prog.AddBoundingBoxConstraint(value, value, q[index])
        for joint in self.finger_joints:
            index = joint.position_start()
            prog.AddBoundingBoxConstraint(self.finger_value, self.finger_value,
                                          q[index])

        # 중력 방향 조건: 월드 아래 방향이 센서 좌표계에서 ĝ 가 되어야 한다.
        ik.AddAngleBetweenVectorsConstraint(
            self.plant.world_frame(), np.array([0.0, 0.0, -1.0]),
            self.sensor_frame, np.asarray(g_hat, dtype=float),
            0.0, ANGLE_TOL_RAD,
        )
        # 작업 공간 상자 안에 물체를 둔다.
        lower, upper = workspace if workspace is not None else (
            WORKSPACE_LOWER_M, WORKSPACE_UPPER_M)
        ik.AddPositionConstraint(
            self.sensor_frame, np.zeros(3), self.plant.world_frame(),
            lower, upper,
        )
        # 충돌 회피 — 이것이 "motion planning 으로 충돌 없는 자세만" 조건.
        # 문턱보다 IK_SLACK_M 만큼 더 요구한다. 해가 경계에 붙어 나오므로,
        # 딱 문턱으로 주면 검증하는 쪽에서 부동소수점 차이로 떨어진다.
        ik.AddMinimumDistanceLowerBoundConstraint(self.ik_distance_m, 0.01)

        guess = self.plant.GetPositions(self.context).copy()
        if warm_start and self._last_solution is not None:
            guess = self._last_solution.copy()
        for joint, value in zip(self.arm_joints, self.seed_q):
            guess[joint.position_start()] = (
                guess[joint.position_start()] if warm_start
                and self._last_solution is not None else value)
        for joint, value in zip(self.object_joints, np.atleast_1d(theta)):
            guess[joint.position_start()] = value
        for joint in self.finger_joints:
            guess[joint.position_start()] = self.finger_value
        prog.SetInitialGuess(q, guess)

        result = Solve(prog)
        if not result.is_success():
            return None
        solution = result.GetSolution(q)
        if not self.arm_pose_is_clear(solution, theta):
            return None
        self._last_solution = solution.copy()
        return solution

    # ------------------------------------------------------------------
    def solve_at(self, theta, g_hat, position, tol_m=0.01):
        """제시 위치를 지정해서 푸는 변형. 시작 자세를 찾을 때 쓴다."""
        box = (np.array(position) - tol_m, np.array(position) + tol_m)
        return self.solve(theta, g_hat, warm_start=False, workspace=box)

    def solve_oriented(self, theta, R_WS, position, tol_m=0.04,
                       tol_rad=np.deg2rad(8.0)):
        """물체의 **방향까지** 지정해서 푼다. 각도 관측용 자세에 쓴다.

        중력 방향만 묶는 solve() 와 달리 여기서는 세 축을 다 묶는다. 관측성은
        축이 카메라에 대해 어떻게 놓였나로 정해지므로 방향 전체가 중요하다.
        허용 오차를 8도로 넉넉히 둔 것은, 관측성이 그 정도 어긋남에는 거의
        변하지 않는데 (재보니 10도에 3% 안쪽) 딱 맞추려 들면 IK 가 자주
        실패하기 때문이다. 대신 성공한 자세에서 관측성을 **다시 잰다**.
        """
        ik = InverseKinematics(self.plant, self.context, with_joint_limits=True)
        prog = ik.prog()
        q = ik.q()
        for joint, value in zip(self.object_joints, np.atleast_1d(theta)):
            index = joint.position_start()
            prog.AddBoundingBoxConstraint(value, value, q[index])
        for joint in self.finger_joints:
            index = joint.position_start()
            prog.AddBoundingBoxConstraint(self.finger_value, self.finger_value,
                                          q[index])
        ik.AddOrientationConstraint(
            self.plant.world_frame(), RotationMatrix(np.asarray(R_WS, float)),
            self.sensor_frame, RotationMatrix(), tol_rad)
        ik.AddPositionConstraint(
            self.sensor_frame, np.zeros(3), self.plant.world_frame(),
            np.asarray(position, float) - tol_m,
            np.asarray(position, float) + tol_m)
        ik.AddMinimumDistanceLowerBoundConstraint(self.ik_distance_m, 0.01)

        guess = self.plant.GetPositions(self.context).copy()
        for joint, value in zip(self.arm_joints, self.seed_q):
            guess[joint.position_start()] = value
        for joint, value in zip(self.object_joints, np.atleast_1d(theta)):
            guess[joint.position_start()] = value
        for joint in self.finger_joints:
            guess[joint.position_start()] = self.finger_value
        prog.SetInitialGuess(q, guess)
        result = Solve(prog)
        if not result.is_success():
            return None
        solution = result.GetSolution(q)
        return solution if self.arm_pose_is_clear(solution, theta) else None

    def arm_pose_is_clear(self, arm_q, theta):
        """팔 자세를 고정한 채 충돌이 없고 물체 전체가 보이는가.

        작업자가 물체를 손으로 돌리는 동안 로봇은 멈춰 있으므로,
        이 검사가 시작 자세의 안전 조건이다.
        """
        return self.pose_block_reason(arm_q, theta) is None

    def pose_block_reason(self, arm_q, theta):
        """막힌 이유를 사람이 읽을 수 있게 돌려준다. 안 막혔으면 None.

        왜 이유를 따로 돌려주나
        -----------------------
        예전에는 참/거짓만 돌려주어서 화면에 "충돌/FOV 경로 실패" 라고만 떴다.
        둘 중 어느 쪽인지, 어디가 몇 mm 모자란지 알 수 없으니 사람이 고칠 수가
        없다. 시작 자세를 사람이 직접 정하게 된 뒤로는 더 그렇다 —
        무엇을 얼마나 옮기면 되는지 알려주어야 한다.
        """
        q = np.asarray(arm_q).copy()
        for joint, value in zip(self.object_joints, np.atleast_1d(theta)):
            q[joint.position_start()] = value
        self.plant.SetPositions(self.context, q)
        query = self.plant.get_geometry_query_input_port().Eval(self.context)
        if not pp.collision_free(self.plant, query, self.min_distance_m,
                                 self.arm):
            return f"충돌 ({self.closest_pair_text(query)})"
        if not object_in_camera(self.plant, self.context, self.spec,
                                self.payload):
            return "카메라 시야 밖 (물체 전체가 화면 안에 안 들어옴)"
        return None

    def closest_pair_text(self, query):
        """가장 빠듯한 충돌쌍을 "A <-> B, 필요 20 mm 현재 4 mm" 로 적는다."""
        inspector = query.inspector()
        worst, text = np.inf, "가장 가까운 쌍을 못 찾음"
        for pair in query.ComputeSignedDistancePairwiseClosestPoints(
                self.min_distance_m):
            bodies = [self.plant.GetBodyFromFrameId(inspector.GetFrameId(gid))
                      for gid in (pair.id_A, pair.id_B)]
            required = pp._required_clearance(bodies[0].model_instance(),
                                              bodies[1].model_instance(),
                                              self.arm, self.min_distance_m)
            slack = pair.distance - required
            if slack < worst:
                worst = slack
                text = (f"{bodies[0].name()} <-> {bodies[1].name()},"
                        f" 필요 {1000*required:.0f} mm"
                        f" 현재 {1000*pair.distance:.1f} mm")
        return text

    def solve_robust(self, theta, g_hat, workspace=None):
        """초기추측을 바꿔가며 IK 를 여러 번 시도한다.

        IK 는 초기추측에서 출발해 근처 해를 찾는 국소 최적화다. 그래서 성공
        여부가 '그 각도'만의 성질이 아니라 '어디서 출발했나'에도 달려 있다.
        직전 해를 물려받는 방식은 보통 이득이지만, 한 번 이상한 자세로 빠지면
        그 뒤 점들이 줄줄이 같이 실패한다. 격자를 순서대로 훑으면 18/25 만
        통과하고, 순서를 뒤집으면 20/25, 점마다 초기추측을 지우면 24/25 가
        통과했다. 각도는 그대로인데 판정만 달라진 것이다.

        여기서는 (1) 직전 해, (2) 기본 시드, (3) 무작위 자세 몇 개 순으로
        시도한다. 하나라도 풀리면 그 각도는 실현 가능한 것이다.
        """
        if self._last_solution is not None:
            found = self.solve(theta, g_hat, warm_start=True,
                               workspace=workspace)
            if found is not None:
                return found
        self._last_solution = None
        found = self.solve(theta, g_hat, warm_start=False, workspace=workspace)
        if found is not None:
            return found
        rng = np.random.default_rng(abs(hash((tuple(np.atleast_1d(theta)),
                                              tuple(np.asarray(g_hat))))) % 2**32)
        saved = np.array(self.seed_q, dtype=float).copy()
        try:
            for _ in range(self.ik_restarts):
                self.seed_q = saved + rng.uniform(-0.6, 0.6, size=saved.shape)
                self._last_solution = None
                found = self.solve(theta, g_hat, warm_start=False,
                                   workspace=workspace)
                if found is not None:
                    return found
        finally:
            self.seed_q = saved
        self._last_solution = None
        return None

    def is_reachable(self, theta):
        """중력 3방향 모두에서 도달 가능하고 충돌이 없어야 한다."""
        for g_hat in alg.G_DIRS:
            if self.solve_robust(theta, g_hat) is None:
                return False
        return True

    def object_self_is_clear(self, theta):
        """로봇 IK 전에 물체 부위끼리의 충돌 조합을 싸게 거른다."""
        q = self.plant.GetPositions(self.context).copy()
        for joint, value in zip(self.object_joints, np.atleast_1d(theta)):
            q[joint.position_start()] = value
        self.plant.SetPositions(self.context, q)
        query = self.plant.get_geometry_query_input_port().Eval(self.context)
        inspector = query.inspector()
        for pair in query.ComputeSignedDistancePairwiseClosestPoints(
                self.min_distance_m):
            bodies = [self.plant.GetBodyFromFrameId(inspector.GetFrameId(gid))
                      for gid in (pair.id_A, pair.id_B)]
            if (all(body.model_instance() == self.payload for body in bodies)
                    and pair.distance < self.min_distance_m):
                return False
        return True

    def angle_corners(self, theta):
        """물체 각도 오차의 모서리들. 여유가 0 이면 명목 각도 하나뿐."""
        theta = np.atleast_1d(np.asarray(theta, dtype=float))
        if self.angle_margin_rad <= 0.0:
            return [theta]
        deltas = itertools.product(*[(-self.angle_margin_rad,
                                      self.angle_margin_rad)] * theta.size)
        return [theta] + [theta + np.asarray(d) for d in deltas]

    def solutions_for(self, theta):
        """중력 3방향의 팔 자세. 각도만으로 결과가 정해지도록 푼다.

        angle_margin_rad > 0 이면, 찾은 팔 자세가 물체 각도 +-여유의
        모든 모서리에서도 간격을 지켜야 통과시킨다 (ANGLE_MARGIN_RAD 설명).
        """
        self._last_solution = None
        if not self.object_self_is_clear(theta):
            return {tuple(g): None for g in alg.G_DIRS}
        found = {}
        for g in alg.G_DIRS:
            arm_q = self.solve_robust(theta, g)
            if arm_q is not None and self.angle_margin_rad > 0.0:
                if not all(self.arm_pose_is_clear(arm_q, corner)
                           for corner in self.angle_corners(theta)):
                    arm_q = None
            found[tuple(g)] = arm_q
        return found


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 준정적 이동
#
# 관절이 고정된다는 가정은 "중력만 버티면 된다"는 뜻이 아니다. 로봇이 물체를
# 옮기는 동안 가속도가 붙으면 관성 반력이 힌지에 추가로 걸린다. 너무 빨리
# 움직이면 그 관성 토크만으로 관절이 흐를 수 있다.
#
# 그래서 "느리게"를 감이 아니라 역동역학으로 정한다. 계획한 궤적을 따라
# CalcInverseDynamics 로 물체 관절에 필요한 토크를 직접 구하고, 그것이 힌지
# 유지토크의 일정 비율 아래가 되는 최소 이동 시간을 찾는다.
# ---------------------------------------------------------------------------
def _cycloidal(fraction):
    """부드러운 이동 프로파일. 시작·끝 속도와 가속도가 0."""
    s = fraction - np.sin(2.0 * np.pi * fraction) / (2.0 * np.pi)
    ds = (1.0 - np.cos(2.0 * np.pi * fraction))
    dds = 2.0 * np.pi * np.sin(2.0 * np.pi * fraction)
    return s, ds, dds


def peak_payload_torque(plant, context, payload_joints, q_start, q_end,
                        duration_s, samples=40):
    """이 궤적을 이 시간에 소화하려면 힌지가 내야 하는 최대 토크."""
    from pydrake.multibody.tree import MultibodyForces

    q_start = np.asarray(q_start, dtype=float)
    delta = np.asarray(q_end, dtype=float) - q_start
    indices = [joint.velocity_start() for joint in payload_joints]
    worst = 0.0
    for fraction in np.linspace(0.0, 1.0, samples):
        s, ds, dds = _cycloidal(fraction)
        q = q_start + s * delta
        v = delta * ds / duration_s
        vdot = delta * dds / duration_s ** 2
        plant.SetPositions(context, q)
        plant.SetVelocities(context, v)
        forces = MultibodyForces(plant)
        plant.CalcForceElementsContribution(context, forces)
        tau = plant.CalcInverseDynamics(context, vdot, forces)
        worst = max(worst, float(np.max(np.abs(tau[indices]))))
    return worst


def quasi_static_duration(plant, context, payload_joints, q_start, q_end,
                          torque_limit_nm, fraction=0.25,
                          lo_s=0.2, hi_s=60.0, tolerance=0.02):
    """관성 토크가 유지토크의 fraction 이하가 되는 최소 이동 시간 [s].

    fraction 은 중력 몫을 남겨두기 위한 여유다. 정지 상태에서 이미 중력
    토크가 걸려 있으므로, 이동 중 추가분을 그 이하로 묶는다.
    """
    target = torque_limit_nm * fraction
    if peak_payload_torque(plant, context, payload_joints,
                           q_start, q_end, hi_s) > target:
        return hi_s, False            # 최대 시간으로도 못 맞춘다
    while hi_s - lo_s > tolerance:
        mid = 0.5 * (lo_s + hi_s)
        if peak_payload_torque(plant, context, payload_joints,
                               q_start, q_end, mid) > target:
            lo_s = mid
        else:
            hi_s = mid
    return hi_s, True


def _body(plant, name, model=None):
    return (plant.GetBodyByName(name) if model is None
            else plant.GetBodyByName(name, model))


def _frame(plant, name, model=None):
    return (plant.GetFrameByName(name) if model is None
            else plant.GetFrameByName(name, model))


def observability_px_per_deg(plant, context, spec, index, model=None,
                             camera=CAMERA):
    """관절 index 를 1도 돌릴 때 움직이는 부위가 화면에서 몇 화소 움직이나.

    FoundationPose 는 결국 **화면에서 보이는 변화**로 자세를 맞춘다. 그러니
    각도를 조금 돌렸을 때 그림이 거의 안 바뀌는 자세라면, 그 각도는 못 재는
    것이다. 이 값이 그대로 각도 관측성이다.

        축 a 둘레로 도는 점 p 의 속도   v = a x (p - o)
        카메라가 보는 것은 시선에 수직인 성분뿐  -> (f/z) |v - (v.z_cam) z_cam|

    축이 시선과 직각이면 회전이 화면 밖(깊이 방향)으로 나가 거의 안 보인다.
    실제로 3-link 를 돌려가며 재보니 그 자세가 가장 나빴다 (study_startpose.py:
    joint2 기준 0.36 px/deg vs 최대 0.99 px/deg, 2.8배).

    돌려주는 값의 단위는 화소/도. 클수록 좋다.
    """
    joint = spec.joints[index]
    parent = _body(plant, joint.parent, model).body_frame()
    axis_W = (parent.CalcRotationMatrixInWorld(context).matrix()
              @ np.asarray(joint.axis, float))
    axis_W /= np.linalg.norm(axis_W)
    origin_W = _frame(plant, f"{joint.name}_child", model).CalcPoseInWorld(
        context).translation()

    X_WC = camera_pose(camera)
    R_CW = X_WC.rotation().matrix().T
    eye = X_WC.translation()
    focal = camera["depth_intrinsics"]["fx"]

    speeds = []
    for part in spec.parts[index + 1:]:            # 이 관절보다 아래쪽 = 움직이는 쪽
        X_WB = _body(plant, part.name, model).body_frame().CalcPoseInWorld(
            context)
        half = np.array(part.bbox_mm, float) * 0.5e-3
        for sign in np.ndindex(2, 2, 2):
            p_W = X_WB @ (half * (np.array(sign) * 2 - 1))
            v_C = R_CW @ np.cross(axis_W, p_W - origin_W)
            p_C = R_CW @ (p_W - eye)
            depth = max(float(p_C[2]), 1e-3)
            image_v = focal * (v_C[:2] - p_C[:2] * v_C[2] / depth) / depth
            speeds.append(float(np.linalg.norm(image_v)))
    if not speeds:
        return 0.0
    return float(np.mean(speeds)) * np.pi / 180.0


def axis_view_angle_deg(plant, context, spec, index, model=None, camera=CAMERA):
    """관절 축과 카메라 시선이 이루는 각 [deg]. 보고용."""
    joint = spec.joints[index]
    parent = _body(plant, joint.parent, model).body_frame()
    axis_W = (parent.CalcRotationMatrixInWorld(context).matrix()
              @ np.asarray(joint.axis, float))
    axis_W /= np.linalg.norm(axis_W)
    origin_W = _frame(plant, f"{joint.name}_child", model).CalcPoseInWorld(
        context).translation()
    view = origin_W - camera_pose(camera).translation()
    view /= np.linalg.norm(view)
    return float(np.degrees(np.arccos(np.clip(abs(axis_W @ view), 0.0, 1.0))))


class ViewScorer:
    """로봇 없이 물체만 띄운 가벼운 plant. 방향 후보의 점수를 매긴다.

    IK 는 한 번 푸는 데 0.2~1 초라 방향 후보 수십 개를 다 풀 수는 없다.
    관측성은 **물체가 어떻게 놓였나** 만으로 정해지므로, 먼저 여기서 공짜로
    점수를 매겨 순위를 내고, 위쪽 몇 개만 실제로 IK 를 풀어 본다.
    """

    def __init__(self, spec, densities=None):
        import explore_view as ev          # 순환 참조가 없다 (ev 는 rs 를 모름)
        if densities is None:
            densities = obj.bind_object(spec)
        builder = DiagramBuilder()
        self.spec = spec
        self.plant, self.bodies = ev.build_floating(spec, densities, builder)
        self.diagram = builder.Build()
        self.root = self.diagram.CreateDefaultContext()
        self.context = self.plant.GetMyMutableContextFromRoot(self.root)
        self.base = self.bodies[spec.parts[0].name]
        self.offset = np.array(spec.base_bbox_center_in_sensor_mm) * MM

    def place(self, R_WS, position):
        """센서 프레임(=파지점)이 position 에 오고 방향이 R_WS 가 되도록."""
        X_WS = RigidTransform(RotationMatrix(R_WS), np.asarray(position, float))
        self.plant.SetFreeBodyPose(self.context, self.base,
                                   X_WS @ RigidTransform(self.offset))

    def set_theta(self, theta_rad):
        for joint, value in zip(self.spec.joints, np.atleast_1d(theta_rad)):
            self.plant.GetJointByName(joint.name).set_angle(self.context,
                                                            float(value))

    def score(self, R_WS, position, theta_rad, index):
        self.place(R_WS, position)
        self.set_theta(theta_rad)
        return observability_px_per_deg(self.plant, self.context, self.spec,
                                        index)

    def score_group(self, R_WS, position, theta_rad, indices):
        """묶인 관절들을 한 자세로 볼 때의 점수 = 그중 **가장 나쁜** 값.

        평균을 쓰면 하나가 아주 잘 보이고 다른 하나가 안 보이는 자세가
        뽑힐 수 있다. 우리가 원하는 것은 '둘 다 읽을 수 있는' 자세다.
        """
        self.place(R_WS, position)
        self.set_theta(theta_rad)
        return min(observability_px_per_deg(self.plant, self.context,
                                            self.spec, index)
                   for index in indices)

    def axis_in_base(self, index, theta_rad):
        """관절 index 의 축을 바탕 링크 좌표계로. 나란한지 비교할 때 쓴다."""
        self.place(np.eye(3), (0.0, 0.0, 1.0))
        self.set_theta(theta_rad)
        joint = self.spec.joints[index]
        R_WP = self.bodies[joint.parent].body_frame().CalcRotationMatrixInWorld(
            self.context).matrix()
        R_WB = self.base.body_frame().CalcRotationMatrixInWorld(
            self.context).matrix()
        axis = R_WB.T @ R_WP @ np.asarray(joint.axis, float)
        return axis / np.linalg.norm(axis)


def parallel_groups(scorer, theta_rad, tol_deg=15.0):
    """축이 나란한 관절끼리 묶는다.

    축이 나란하면 한 자세에서 두 관절이 똑같이 잘 보인다. 굳이 자세를 나눌
    이유가 없으므로 묶어서 한 번만 간다 (이동 시간과 파지 흔들림이 준다).
    부호는 무시한다 — 축이 반대로 향해도 회전은 같은 평면에서 일어난다.

    3-link 는 joint1 이 z, joint2 가 -y 라 90도 어긋나 안 묶인다.
    4-link 처럼 z, -y, z, -y 로 번갈아 가는 물체는 두 무리가 된다.
    """
    axes = [scorer.axis_in_base(index, theta_rad)
            for index in range(len(scorer.spec.joints))]
    threshold = np.cos(np.radians(tol_deg))
    groups = []
    for index, axis in enumerate(axes):
        for group in groups:
            if abs(float(axis @ axes[group[0]])) >= threshold:
                group.append(index)
                break
        else:
            groups.append([index])
    return groups


def candidate_rotations(n=48, seed=0):
    """물체를 들 방향 후보. 균일하게 흩뿌린 회전들.

    무작위지만 seed 를 고정해 매번 같은 후보를 본다. 판정이 실행할 때마다
    달라지면 '되던 자세가 안 되는' 일이 생긴다.
    """
    rng = np.random.default_rng(seed)
    out = [np.eye(3)]
    for _ in range(n - 1):
        # 균일 무작위 회전 (Shoemake). 축 하나만 흔들면 한쪽으로 몰린다.
        u1, u2, u3 = rng.random(3)
        quaternion = np.array([
            np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
            np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
            np.sqrt(u1) * np.sin(2 * np.pi * u3),
            np.sqrt(u1) * np.cos(2 * np.pi * u3)])
        out.append(RotationMatrix(
            _quaternion_to_rotation(quaternion)).matrix())
    return out


def _quaternion_to_rotation(q):
    from pydrake.common.eigen_geometry import Quaternion
    q = np.asarray(q, float)
    q = q / np.linalg.norm(q)
    return Quaternion(w=q[3], x=q[0], y=q[1], z=q[2])


def find_viewing_poses(checker, theta_rad, scorer=None, presentations=None,
                       n_candidates=48, n_ik=6, verbose=False,
                       merge_parallel=True, parallel_tol_deg=15.0):
    """관절마다 하나씩, **그 관절 각도가 가장 잘 보이는** 팔 자세를 찾는다.

    왜 관절마다 따로 두나
    ---------------------
    관절 축의 방향은 관절마다 다르다. 3-link 는 joint1 이 z, joint2 가 -y 라
    한 자세로 둘 다 잘 보이게 할 수 없다. 카메라는 축이 시선과 직각일 때
    그 회전을 거의 못 본다 (재보니 2.8~6.8 배 나빠진다, study_startpose.py).
    그래서 **관절 수만큼** 자세를 만들고, 각 관절 각도는 자기 자세에서 읽는다.

    어떻게 고르나
    -------------
      1) 가벼운 plant 에서 방향 후보 48개의 관측성을 공짜로 잰다
      2) 점수가 높은 순으로 IK 를 풀어 본다 (도달·충돌 통과할 때까지)
      3) 처음 성공한 것을 그 관절의 자세로 삼는다

    돌려주는 것: [{"joint", "arm_q", "observability", "axis_view_deg",
                   "position", "rotation"}, ...] — 관절 순서대로.
    실패한 관절은 arm_q 가 None 이다 (그 각도는 파지 자세에서 읽어야 한다).
    """
    spec = checker.spec
    theta = np.atleast_1d(np.asarray(theta_rad, float))
    if scorer is None:
        scorer = ViewScorer(spec)
    if presentations is None:
        center = _BOX_C
        presentations = [center + np.array([0.0, dy, dz])
                         for dz in (0.20, 0.12) for dy in (-0.10, 0.0)]
    rotations = candidate_rotations(n_candidates)

    groups = (parallel_groups(scorer, theta, parallel_tol_deg)
              if merge_parallel else [[k] for k in range(len(spec.joints))])
    if verbose and merge_parallel:
        merged = [g for g in groups if len(g) > 1]
        if merged:
            print("    축이 나란해 한 자세로 묶은 관절: "
                  + ", ".join("+".join(spec.joints[k].name for k in g)
                              for g in merged))

    out = [None] * len(spec.joints)
    for number, group in enumerate(groups):
        ranked = []
        for position in presentations:
            for R in rotations:
                ranked.append((scorer.score_group(R, position, theta, group),
                               position, R))
        ranked.sort(key=lambda row: -row[0])
        arm_q = None
        for score, position, R in ranked[:n_ik]:
            checker._last_solution = None
            arm_q = checker.solve_oriented(theta, R, position)
            if arm_q is not None:
                break
        for index in group:
            joint = spec.joints[index]
            if arm_q is None:
                out[index] = dict(
                    joint=joint.name, arm_q=None, observability=0.0,
                    axis_view_deg=float("nan"), position=None, rotation=None,
                    group=number, predicted=ranked[0][0] if ranked else 0.0)
                continue
            # 실제로 팔이 잡은 상태의 관측성을 다시 잰다. 물체가 IK 허용
            # 오차만큼 틀어져 있을 수 있으므로 예측 점수를 믿지 않는다.
            checker.plant.SetPositions(checker.context, arm_q)
            out[index] = dict(
                joint=joint.name, arm_q=arm_q, group=number,
                observability=observability_px_per_deg(
                    checker.plant, checker.context, spec, index,
                    checker.payload),
                axis_view_deg=axis_view_angle_deg(
                    checker.plant, checker.context, spec, index,
                    checker.payload),
                position=np.asarray(position, float), rotation=R,
                predicted=score)
        if verbose:
            for index in group:
                chosen = out[index]
                if chosen["arm_q"] is None:
                    print(f"    {chosen['joint']}: 자세를 못 찾음 "
                          f"(예상 최고 {chosen['predicted']:.2f} px/deg)")
                else:
                    print(f"    {chosen['joint']}: {chosen['observability']:.2f}"
                          f" px/deg, 축-시선 {chosen['axis_view_deg']:.0f} deg"
                          f"  [자세 {number + 1}/{len(groups)}]")
    return out


def find_starting_pose(checker, thetas, presentations=None):
    """작업자가 물체 관절을 손으로 조정하는 동안 로봇이 멈춰 있을 자세.

    작업자가 관절을 움직이므로, 이 자세는 **구동범위 전 구간에서** 충돌이
    없어야 한다. 한 자세라도 걸리면 다른 제시 위치로 다시 시도한다.
    """
    if presentations is None:
        # 안전 촬영 영역 안에서 제시 위치를 격자로 훑는다. 물체가 길어
        # 테이블(상판 0.75 m)에 닿기 쉬우므로 높은 쪽부터 시도한다.
        center = _BOX_C
        presentations = []
        for dz in (0.22, 0.17, 0.12, 0.07, 0.02):
            for dy in (-0.12, -0.06, 0.0):
                point = center + np.array([0.0, dy, dz])
                if np.all(point >= WORKSPACE_LOWER_M) and \
                        np.all(point <= WORKSPACE_UPPER_M):
                    presentations.append(point)
    down = np.array([0.0, 0.0, -1.0])
    best = (None, None, len(thetas) + 1)
    for target in presentations:
        checker._last_solution = None
        q = checker.solve_at(thetas[0], down, position=target, tol_m=0.02)
        if q is None:
            continue
        # 전 구동범위에서 이 팔 자세가 안전한가
        blocked = [th for th in thetas
                   if not checker.arm_pose_is_clear(q, th)]
        if not blocked:
            return q, target
        if len(blocked) < best[2]:
            best = (q, target, len(blocked))
    if best[0] is not None:
        print(f"  가장 좋은 후보도 {best[2]}/{len(thetas)} 자세에서 간섭"
              f" (제시 위치 {np.round(best[1], 3)} m)")
    return None, None


def plan_experiment(spec, hinge, joint_limits_rad, safety=obj.DEFAULT_SAFETY,
                    steps=5, n_rounds=4, min_distance_m=MIN_DISTANCE_M,
                    density_scale=1.0):
    """Drake 시뮬레이션으로 후보 자세를 찾고, 탐색 순서를 계획으로 만든다.

    반환한 계획을 operator_ui.py 가 그대로 실행한다.
    """
    rho_gt = obj.bind_object(spec, hinge=hinge, safety=safety,
                             density_scale=density_scale)
    # 물체 관절 구동범위를 사용자가 지정한 값으로 교체한다.
    alg.JOINT_LIMITS = list(joint_limits_rad)

    hinge_ok = obj.make_is_feasible(spec, hinge, safety, rho_gt)
    checker = PoseChecker(spec, densities=rho_gt,
                          joint_limits_rad=joint_limits_rad,
                          min_distance_m=min_distance_m)

    axes = [np.linspace(lo, hi, steps) for lo, hi in joint_limits_rad]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(
        -1, len(spec.joints))

    print(f"후보 격자 {len(grid)}개를 걸러낸다")
    reachable, arm_solutions = [], {}
    n_hinge = 0
    for theta in grid:
        if not hinge_ok(theta):
            continue
        n_hinge += 1
        solutions = checker.solutions_for(theta)
        if any(v is None for v in solutions.values()):
            continue
        reachable.append(theta)
        arm_solutions[tuple(np.round(theta, 9))] = solutions
    print(f"  힌지가 버티는 자세 {n_hinge}/{len(grid)}")
    print(f"  로봇이 도달 가능하고 충돌 없는 자세 {len(reachable)}/{n_hinge}")
    if not reachable:
        return None

    # 걸러낸 자세만 추정 파이프라인의 후보로 준다.
    keys = {tuple(np.round(t, 9)) for t in reachable}
    alg.is_feasible = lambda th: tuple(np.round(np.asarray(th), 9)) in keys

    start_q, presentation = find_starting_pose(checker, reachable)
    if start_q is None:
        print("  경고: 전 구동범위에서 안전한 시작 자세를 찾지 못했다")
        return None
    print(f"  시작 자세 확보 — 제시 위치 {np.round(presentation, 3)} m,"
          f" 구동범위 전 구간 충돌 없음")

    history = alg.run_loop(alg.select_active, n_rounds, seed=1)
    rounds = []
    limit_nm = hinge.holding_torque_nm / safety
    for index, entry in enumerate(history):
        theta = np.asarray(entry["theta"])
        solutions = arm_solutions[tuple(np.round(theta, 9))]
        # 관절각은 시작 지점에서 사람이 한 번 고정한 뒤 그대로 유지된다고
        # 가정한다. 실시간 추적이 없으므로, 그 가정을 지켜 주는 것은 힌지의
        # 유지토크뿐이다. 이번 라운드에서 힌지가 견뎌야 할 최악 토크를 함께
        # 기록해 두어야 가정의 근거를 나중에 확인할 수 있다.
        worst = obj.max_joint_torques(spec, [theta], rho_gt)[0]
        rounds.append(dict(
            round=index + 1,
            object_joint_deg=[float(v) for v in np.degrees(theta)],
            arm_q_measure=[[float(v) for v in solutions[tuple(g)]]
                           for g in alg.G_DIRS],
            gravity_dirs=[[float(v) for v in g] for g in alg.G_DIRS],
            hinge_torque_nm=[float(v) for v in worst],
            hinge_margin=float(hinge.holding_torque_nm / max(worst.max(), 1e-9)),
            hinge_limit_nm=float(limit_nm),
            rho_estimate=[float(v) for v in entry["rho"]],
            rmse=float(entry["rmse"]),
        ))

    return dict(
        object=spec.key,
        label=spec.label,
        joint_names=[j.name for j in spec.joints],
        joint_range_deg=[[float(np.degrees(lo)), float(np.degrees(hi))]
                         for lo, hi in joint_limits_rad],
        hinge=dict(label=hinge.label, torque_nm=hinge.holding_torque_nm,
                   safety=safety),
        finger_position_m=float(checker.finger_value),
        density_gt=[float(v) for v in rho_gt],
        arm_q_start=[float(v) for v in start_q],
        n_candidates=len(grid),
        n_reachable=len(reachable),
        rounds=rounds,
    )


def parse_joint_range(spec, values):
    """--joint-range-deg 로 받은 값을 관절별 (lo, hi) [rad] 로 바꾼다."""
    if not values:
        return [j.limits_rad for j in spec.joints]
    if len(values) == 2:
        pairs = [(values[0], values[1])] * len(spec.joints)
    elif len(values) == 2 * len(spec.joints):
        pairs = [(values[2 * i], values[2 * i + 1])
                 for i in range(len(spec.joints))]
    else:
        raise ValueError(
            f"--joint-range-deg 는 2개 또는 {2 * len(spec.joints)}개를 받는다")
    # 물리한계를 아주 조금 넘는 것은 단위 환산 때문이지 사용자 실수가 아니다.
    # 예: 스캔 URDF 가 한계를 1.57 rad 로 적어 두면 그건 89.954 deg 다.
    # 여기에 -90 90 을 넣으면 0.046 deg 넘치는데, 이건 잘라 주는 게 맞다.
    SNAP_DEG = 0.5

    limits = []
    for (lo, hi), joint in zip(pairs, spec.joints):
        hard_lo, hard_hi = joint.limits_rad
        rad = [np.deg2rad(lo), np.deg2rad(hi)]
        for index, (value, hard) in enumerate(((rad[0], hard_lo),
                                               (rad[1], hard_hi))):
            over = (hard - value) if index == 0 else (value - hard)
            if over <= 1e-9:
                continue
            if over <= np.deg2rad(SNAP_DEG):
                print(f"  {joint.name}: 요청 {(lo, hi)[index]:.3f} deg 가 물리한계"
                      f" {np.degrees(hard):.3f} deg 를 {np.degrees(over):.3f} deg"
                      f" 넘어 한계로 맞춥니다")
                rad[index] = hard
            else:
                raise ValueError(
                    f"{joint.name}: 요청 {lo:.3f}~{hi:.3f} deg 가 힌지 물리한계 "
                    f"{np.degrees(hard_lo):.3f}~{np.degrees(hard_hi):.3f} deg 를 "
                    f"{np.degrees(over):.3f} deg 벗어난다")
        if rad[0] >= rad[1]:
            raise ValueError(
                f"{joint.name}: 하한 {np.degrees(rad[0]):.3f} 이 상한 "
                f"{np.degrees(rad[1]):.3f} 보다 크거나 같다")
        limits.append(tuple(rad))
    return limits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test-fov", action="store_true")
    parser.add_argument("--object", choices=tuple(obj.OBJECTS), default="3link")
    parser.add_argument("--joint-range-deg", type=float, nargs="+", default=None,
                        help="관절 구동범위. 2개(전 관절 공통) 또는 관절당 2개")
    parser.add_argument("--steps", type=int, default=5,
                        help="관절당 격자 점 수")
    parser.add_argument("--min-distance-mm", type=float,
                        default=MIN_DISTANCE_M / MM,
                        help="충돌로 보는 최소 간격. 자세와 경로 모두 지킨다.")
    parser.add_argument("--plan", type=Path, default=None,
                        help="탐색 계획을 JSON 으로 저장한다")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--hinge-torque", type=float,
                        default=obj.MG_PLASTIC_DEFAULT_NM)
    parser.add_argument("--safety", type=float, default=obj.DEFAULT_SAFETY)
    parser.add_argument("--auto-scale", action="store_true")
    args = parser.parse_args()

    if args.self_test_fov:
        fov_self_test()
        return

    if args.plan is not None:
        import json
        spec = obj.OBJECTS[args.object]
        limits = parse_joint_range(spec, args.joint_range_deg)
        hinge = obj.Hinge(obj.HINGES["mg_plastic"].label, args.hinge_torque,
                          obj.HINGES["mg_plastic"].note)
        scale = (min(1.0, obj.max_feasible_density_scale(spec, hinge, args.safety))
                 if args.auto_scale else 1.0)
        print(f"{spec.label} — 탐색 계획 생성")
        for joint, (lo, hi) in zip(spec.joints, limits):
            print(f"  {joint.name} 구동범위 "
                  f"{np.degrees(lo):.0f} ~ {np.degrees(hi):.0f} deg")
        plan = plan_experiment(
            spec, hinge, limits, safety=args.safety, steps=args.steps,
            n_rounds=args.rounds, min_distance_m=args.min_distance_mm * MM,
            density_scale=scale)
        if plan is None:
            print("계획을 만들 수 없다.")
            return
        args.plan.write_text(json.dumps(plan, indent=2, ensure_ascii=False))
        print(f"\n계획 저장 -> {args.plan}")
        for entry in plan["rounds"]:
            print(f"  round {entry['round']}: 물체 관절각 "
                  f"{np.round(entry['object_joint_deg'], 1)} deg")
        return

    spec = obj.OBJECTS[args.object]
    limits = parse_joint_range(spec, args.joint_range_deg)
    print(f"{spec.label}")
    for joint, (lo, hi) in zip(spec.joints, limits):
        print(f"  {joint.name} 구동범위 {np.degrees(lo):.0f} ~ {np.degrees(hi):.0f} deg")

    checker = PoseChecker(spec, joint_limits_rad=limits,
                          min_distance_m=args.min_distance_mm * MM)
    print(f"  손가락 위치 {1000 * checker.finger_value:.1f} mm"
          f" (단면에 맞춰 닫음)")
    print(f"  plant 자유도 {checker.plant.num_positions()}"
          f" = 팔 6 + 손가락 2 + 물체 {len(spec.joints)}")

    axes = [np.linspace(lo, hi, args.steps) for lo, hi in limits]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(
        -1, len(spec.joints))
    print(f"\n후보 {len(grid)}개에 대해 IK + 충돌 검사")
    ok = []
    for theta in grid:
        good = checker.is_reachable(theta)
        ok.append(good)
        mark = "가능" if good else "불가"
        print(f"  q={np.round(np.degrees(theta), 0)}  {mark}")
    print(f"\n통과 {sum(ok)}/{len(grid)}")


if __name__ == "__main__":
    main()
