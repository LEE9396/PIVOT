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

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "robot_learning" / "scripts"))
import visualize_drake_rb5_hammer_payload as rb5  # noqa: E402

MM = 1e-3
ARM_JOINT_NAMES = rb5.ARM_JOINT_NAMES

# AFT200 마운트: 브래킷 + 센서 퍽을 합친 원기둥 하나로 모형화한다.
AFT_TOTAL_H_M = 0.0523
AFT_DIAMETER_M = rb5.AFT200_DIAMETER_M
AFT_MASS_KG = rb5.AFT200_BRACKET_MASS_KG + rb5.AFT200_SENSOR_MASS_KG

PGC_TCP_Z_M = rb5.PGC_TCP_Z_M          # 0.125 — 그리퍼 베이스에서 파지점까지
PGC_FINGER_ORIGIN_Y_M = 0.0265         # 손가락 원점의 y (열림 최대)
PGC_STROKE_M = 0.025                   # 손가락당 행정

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

# 로봇은 테이블이 아니라 앞쪽 긴 변의 독립 받침대 위에 선다.
PEDESTAL_SIZE_M = (0.22, 0.22, ROBOT_BASE_XYZ_M[2])
PEDESTAL_CENTER_M = (ROBOT_BASE_XYZ_M[0], ROBOT_BASE_XYZ_M[1],
                     ROBOT_BASE_XYZ_M[2] / 2.0)

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
CAMERA = CAMERAS[0]
CAMERA_BODY_SIZE_M = (0.124, 0.026, 0.029)   # D456 외형 (124 x 26 x 29 mm)
CAMERA_MOUNT_RADIUS_M = 0.016                # 지지대 봉

MIN_DISTANCE_M = 0.006     # 이 값보다 가까워지면 충돌로 본다
ANGLE_TOL_RAD = np.deg2rad(1.0)

# 물체를 그리퍼에 어떻게 물릴지. roll 은 파지축 선택.
#  3-link 는 단면이 44x44 라 어느 축이든 같고, 2-link 는 46x40 이므로
#  README 권고대로 40 mm 축(z)을 죠 방향에 맞춘다 -> roll 90 deg.
GRASP_ROLL_DEG = {"2link": 90.0, "3link": 0.0}


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
                builder=None, include_visuals=True, gripper="robotiq2f85"):
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
    plant.WeldFrames(plant.GetFrameByName("tcp", arm), mount.body_frame(),
                     RigidTransform([0.0, 0.0, AFT_TOTAL_H_M / 2.0]))

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
    payload, parts, sensor_frame = _add_object(
        plant, spec, densities, joint_limits_rad)
    roll = np.deg2rad(GRASP_ROLL_DEG.get(spec.key, 0.0))
    X_pgc_sensor = RigidTransform(RotationMatrix.MakeXRotation(roll),
                                  [0.0, 0.0, tcp_z])
    X_sensor_base = RigidTransform(
        np.array(spec.base_bbox_center_in_sensor_mm) * MM)
    plant.WeldFrames(plant.GetFrameByName(gripper_spec.base_frame, gripper),
                     parts[spec.parts[0].name].body_frame(),
                     X_pgc_sensor @ X_sensor_base)

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

    add_fixture("table", Box(*TABLE_SIZE_M),
                RigidTransform(np.array(TABLE_CENTER_M)),
                [*_LAB["table"]["surface_rgb"], 1.0])
    add_fixture("pedestal", Box(*PEDESTAL_SIZE_M),
                RigidTransform(np.array(PEDESTAL_CENTER_M)),
                [0.30, 0.31, 0.33, 1.0])

    # 카메라 본체와 지지봉. 팔이 여기에 닿으면 안 되므로 충돌 대상이다.
    for camera in CAMERAS:
        position = np.array(camera["position_xyz_m"])
        target = np.array(camera["look_at_xyz_m"])
        pose = look_at_pose(position, target)
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
    # link0..link4 와 물체의 나머지 부위, 테이블은 그대로 검사한다.

    return dict(
        builder=builder, plant=plant, scene_graph=scene_graph,
        arm=arm, gripper=gripper, payload=payload, parts=parts,
        mount=mount, sensor_frame=sensor_frame, spec=spec,
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
    dims = part.bbox_mm
    roll = GRASP_ROLL_DEG.get(spec.key, 0.0)
    # roll=0 이면 물체 y 가 죠 방향, roll=90 이면 물체 z 가 죠 방향.
    return (dims[1] if abs(roll) < 45.0 else dims[2]) * MM


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
                 gripper="robotiq2f85"):
        scene = build_scene(spec, densities, joint_limits_rad,
                            include_visuals=False, gripper=gripper)
        self.spec = spec
        self.plant = scene["plant"]
        self.arm = scene["arm"]
        self.gripper = scene["gripper"]
        self.sensor_frame = scene["sensor_frame"]
        self.min_distance_m = min_distance_m
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
        ik.AddMinimumDistanceLowerBoundConstraint(self.min_distance_m, 0.01)

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
        self._last_solution = solution.copy()
        return solution

    # ------------------------------------------------------------------
    def solve_at(self, theta, g_hat, position, tol_m=0.01):
        """제시 위치를 지정해서 푸는 변형. 시작 자세를 찾을 때 쓴다."""
        box = (np.array(position) - tol_m, np.array(position) + tol_m)
        return self.solve(theta, g_hat, warm_start=False, workspace=box)

    def arm_pose_is_clear(self, arm_q, theta):
        """팔 자세를 고정한 채 물체 관절만 바꿨을 때 충돌이 없는가.

        작업자가 물체를 손으로 돌리는 동안 로봇은 멈춰 있으므로,
        이 검사가 시작 자세의 안전 조건이다.
        """
        q = np.asarray(arm_q).copy()
        for joint, value in zip(self.object_joints, np.atleast_1d(theta)):
            q[joint.position_start()] = value
        self.plant.SetPositions(self.context, q)
        query = self.plant.get_geometry_query_input_port().Eval(self.context)
        pairs = query.ComputeSignedDistancePairwiseClosestPoints(
            self.min_distance_m)
        return all(pair.distance >= self.min_distance_m for pair in pairs)

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

    def solutions_for(self, theta):
        """중력 3방향의 팔 자세. 각도만으로 결과가 정해지도록 푼다."""
        self._last_solution = None
        return {tuple(g): self.solve_robust(theta, g) for g in alg.G_DIRS}


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
    parser.add_argument("--object", choices=tuple(obj.OBJECTS), default="3link")
    parser.add_argument("--joint-range-deg", type=float, nargs="+", default=None,
                        help="관절 구동범위. 2개(전 관절 공통) 또는 관절당 2개")
    parser.add_argument("--steps", type=int, default=5,
                        help="관절당 격자 점 수")
    parser.add_argument("--min-distance-mm", type=float, default=6.0)
    parser.add_argument("--plan", type=Path, default=None,
                        help="탐색 계획을 JSON 으로 저장한다")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--hinge-torque", type=float,
                        default=obj.MG_PLASTIC_DEFAULT_NM)
    parser.add_argument("--safety", type=float, default=obj.DEFAULT_SAFETY)
    parser.add_argument("--auto-scale", action="store_true")
    args = parser.parse_args()

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
