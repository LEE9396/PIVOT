"""RB5 + AFT200 + 2F-85 + 물체 씬에서 측정 프레임 S 의 운동량을 뽑는다.

측정 프레임 S 는 AFT200 퍽(감지부) 중심에 둔다. 논문의 정적 파이프라인이
쓰는 `obj_sensor`(파지점) 와 다르다 — 동적 식별에서는 센서가 실제로 힘을
재는 자리가 기준이어야 하고, 센서 아래에 매달린 것(그리퍼 + 물체) 전부가
측정 대상이다.

robot_scene.build_scene 을 그대로 쓰므로 테이블·받침대·카메라·자기충돌
필터도 논문 파이프라인과 같다.
"""
import sys
from pathlib import Path

import numpy as np
from pydrake.math import RigidTransform

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

import robot_scene as rs                       # noqa: E402
import density_id_objects as obj               # noqa: E402
import visualize_drake_rb5_hammer_payload as rb5   # noqa: E402

from .inertial import transport_from_rigid_transform, from_spatial_inertia, N_PARAM  # noqa: E402

G_W = np.array([0.0, 0.0, -9.81])
# 퍽 중심: 마운트 원통(52.3 mm) 중심에서 위로 (H/2 − h_sensor/2)
SENSOR_CENTER_IN_MOUNT_M = np.array(
    [0.0, 0.0, rs.AFT_TOTAL_H_M / 2.0 - rb5.AFT200_SENSOR_HEIGHT_M / 2.0])


class SceneModel:
    """씬 하나 = 물체 하나 + 그리퍼 하나. 관절 형상 θ 는 호출마다 준다."""

    def __init__(self, spec, densities=None, gripper="robotiq2f85",
                 grasp_transform=None, min_distance_m=rs.MIN_DISTANCE_M):
        self.spec = spec
        scene = rs.build_scene(spec, densities, include_visuals=False,
                               gripper=gripper, grasp_transform=grasp_transform)
        self.scene = scene
        self.plant = scene["plant"]
        self.diagram = scene["builder"].Build()
        self.root_context = self.diagram.CreateDefaultContext()
        self.context = self.plant.GetMyContextFromRoot(self.root_context)
        self.arm_joints = [self.plant.GetJointByName(n, scene["arm"])
                           for n in rs.ARM_JOINT_NAMES]
        self.object_joints = [self.plant.GetJointByName(j.name, scene["payload"])
                              for j in spec.joints]
        self.q_idx = np.array([j.position_start() for j in self.arm_joints])
        self.v_idx = np.array([j.velocity_start() for j in self.arm_joints])
        self.theta_idx = np.array([j.position_start() for j in self.object_joints])
        self.mount = scene["mount"]
        self.X_MS = RigidTransform(SENSOR_CENTER_IN_MOUNT_M)
        self.grasp_frame = scene["sensor_frame"]          # obj_sensor = 파지점
        self.part_bodies = [self.plant.GetBodyByName(p.name, scene["payload"])
                            for p in spec.parts]
        self.min_distance_m = float(min_distance_m)
        self.n_parts = len(spec.parts)
        self.nq = self.plant.num_positions()
        self.nv = self.plant.num_velocities()
        lows, highs = self.plant.GetPositionLowerLimits(), self.plant.GetPositionUpperLimits()
        self.q_lower = lows[self.q_idx].copy()
        self.q_upper = highs[self.q_idx].copy()
        self.q_lower[~np.isfinite(self.q_lower)] = -np.pi
        self.q_upper[~np.isfinite(self.q_upper)] = np.pi
        self._downstream = self._downstream_sets()

    # ------------------------------------------------------------------
    def _downstream_sets(self):
        """관절 i 아래에 매달린 부위 인덱스 집합 (자식과 그 자손)."""
        name_to_index = {p.name: k for k, p in enumerate(self.spec.parts)}
        children = {}
        for joint in self.spec.joints:
            children.setdefault(joint.parent, []).append(joint.child)
        sets = []
        for joint in self.spec.joints:
            stack, found = [joint.child], []
            while stack:
                name = stack.pop()
                found.append(name_to_index[name])
                stack.extend(children.get(name, []))
            sets.append(sorted(found))
        return sets

    @property
    def downstream(self):
        return self._downstream

    # ------------------------------------------------------------------
    def _full_q(self, q_arm, theta):
        q = self.plant.GetPositions(self.context).copy()
        q[self.q_idx] = q_arm
        q[self.theta_idx] = np.atleast_1d(theta)
        return q

    def set_state(self, q_arm, theta, qd_arm=None):
        self.plant.SetPositions(self.context, self._full_q(q_arm, theta))
        v = np.zeros(self.nv)
        if qd_arm is not None:
            v[self.v_idx] = qd_arm
        self.plant.SetVelocities(self.context, v)

    def sensor_pose(self, q_arm, theta):
        self.set_state(q_arm, theta)
        return self.plant.EvalBodyPoseInWorld(self.context, self.mount) @ self.X_MS

    def evaluate(self, q_arm, qd_arm, qdd_arm, theta):
        """S 의 자세와, S 축으로 표현한 (a_o, ω, α, g)."""
        self.set_state(q_arm, theta, qd_arm)
        vdot = np.zeros(self.nv)
        vdot[self.v_idx] = qdd_arm
        X_WM = self.plant.EvalBodyPoseInWorld(self.context, self.mount)
        X_WS = X_WM @ self.X_MS
        V_WM = self.plant.EvalBodySpatialVelocityInWorld(self.context, self.mount)
        A_WM = self.plant.CalcSpatialAccelerationsFromVdot(
            self.context, vdot)[self.mount.index()]
        p_MS_W = X_WM.rotation().matrix() @ self.X_MS.translation()
        A_WS = A_WM.Shift(p_MS_W, V_WM.rotational())
        R = X_WS.rotation().matrix()
        return dict(
            X_WS=X_WS,
            a_o=R.T @ A_WS.translational(),
            omega=R.T @ V_WM.rotational(),
            alpha=R.T @ A_WS.rotational(),
            g=R.T @ G_W,
            p_WS=X_WS.translation(),
        )

    # ------------------------------------------------------------------
    def part_transports(self, theta, q_arm=None):
        """각 부위 몸체 프레임 → S 의 10x10 변환. 팔 자세와 무관 (물체는 S 에 고정)."""
        if q_arm is None:
            q_arm = np.zeros(6)
        self.set_state(q_arm, theta)
        X_WS = self.plant.EvalBodyPoseInWorld(self.context, self.mount) @ self.X_MS
        maps = []
        for body in self.part_bodies:
            X_SB = X_WS.inverse() @ self.plant.EvalBodyPoseInWorld(self.context, body)
            maps.append(transport_from_rigid_transform(X_SB))
        return maps

    def part_poses_in_S(self, theta, q_arm=None):
        if q_arm is None:
            q_arm = np.zeros(6)
        self.set_state(q_arm, theta)
        X_WS = self.plant.EvalBodyPoseInWorld(self.context, self.mount) @ self.X_MS
        return [X_WS.inverse() @ self.plant.EvalBodyPoseInWorld(self.context, b)
                for b in self.part_bodies]

    def hinge_geometry(self, theta, q_arm=None):
        """각 물체 관절의 (힌지 점 r_h, 축 â) 를 S 좌표로."""
        if q_arm is None:
            q_arm = np.zeros(6)
        self.set_state(q_arm, theta)
        X_WS = self.plant.EvalBodyPoseInWorld(self.context, self.mount) @ self.X_MS
        R_SW = X_WS.rotation().matrix().T
        out = []
        for joint in self.object_joints:
            X_WF = joint.frame_on_parent().CalcPoseInWorld(self.context)
            axis_W = X_WF.rotation().matrix() @ np.asarray(joint.revolute_axis(), float)
            r_h = R_SW @ (X_WF.translation() - X_WS.translation())
            out.append((r_h, R_SW @ axis_W))
        return out

    def grasp_point_in_S(self, theta, q_arm=None):
        if q_arm is None:
            q_arm = np.zeros(6)
        self.set_state(q_arm, theta)
        X_WS = self.plant.EvalBodyPoseInWorld(self.context, self.mount) @ self.X_MS
        X_WG = self.grasp_frame.CalcPoseInWorld(self.context)
        return X_WS.inverse().rotation().matrix() @ (X_WG.translation() - X_WS.translation())

    # ------------------------------------------------------------------
    def tool_phi(self):
        """센서가 함께 재는 툴(그리퍼 몸체+손가락)의 phi, S 기준. 참값 역할."""
        self.set_state(np.zeros(6), np.zeros(len(self.object_joints)))
        X_WS = self.plant.EvalBodyPoseInWorld(self.context, self.mount) @ self.X_MS
        total = np.zeros(N_PARAM)
        for name in self.scene["gripper_spec"].body_names:
            body = self.plant.GetBodyByName(name, self.scene["gripper"])
            X_SB = X_WS.inverse() @ self.plant.EvalBodyPoseInWorld(self.context, body)
            total += transport_from_rigid_transform(X_SB) @ from_spatial_inertia(
                body.default_spatial_inertia())
        return total

    # ------------------------------------------------------------------
    def min_distance(self, q_arm, theta, max_distance=None):
        """충돌 필터를 거친 최소 부호거리. 가까운 쌍이 없으면 max_distance."""
        cap = self.min_distance_m * 4.0 if max_distance is None else max_distance
        self.set_state(q_arm, theta)
        query = self.plant.get_geometry_query_input_port().Eval(self.context)
        pairs = query.ComputeSignedDistancePairwiseClosestPoints(cap)
        if not pairs:
            return cap
        return min(p.distance for p in pairs)

    def pose_checker(self, **kwargs):
        """논문 파이프라인의 IK/충돌 검사기. 시작 자세를 구할 때 쓴다."""
        return rs.PoseChecker(self.spec, **kwargs)
