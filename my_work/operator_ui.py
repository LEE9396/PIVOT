"""작업자 인터페이스 — 사람이 물체 관절을 직접 조정하는 측정 절차.

robot_scene.py 가 미리 계산해 둔 계획(JSON)을 받아 라운드마다 아래를 반복한다.

  1. 시작 자세로 이동            로봇 움직임 · 표시등 빨강 · 접근 금지
  2. 시작 자세 도달              로봇 정지 · 표시등 초록 · 접근 허용
                                 화면이 이번 라운드에 필요한 관절각을 알려준다
  3. 작업자가 물체 관절을 조정    슬라이더로 각도를 맞춘다
  4. 조정 완료 확인              각도가 목표와 맞는지 시스템이 검사한다
  5. 안전 확인 (2단)             "손을 뗐습니다" 를 눌러야 다음 단계로 간다
  6. 탐색 자세로 이동 후 측정    중력 3방향 · 표시등 빨강

로봇은 2~5 단계 동안 절대 움직이지 않는다. 6단계는 5단계 확인 없이는
시작되지 않는다.

실행:
    cd ~/Desktop/PIVOT/my_work
    # 먼저 계획을 만든다
    ../robot_learning/scripts/run_drake_env.sh python robot_scene.py \
        --object 3link --joint-range-deg 20 150 --hinge-torque 0.5 \
        --auto-scale --plan outputs/plan_3link.json
    # 그 계획을 실행한다
    ../robot_learning/scripts/run_drake_env.sh python operator_ui.py \
        --plan outputs/plan_3link.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from pydrake.geometry import Box, Rgba, StartMeshcat
from pydrake.math import RigidTransform
from pydrake.systems.framework import DiagramBuilder
from pydrake.visualization import AddDefaultVisualization

import density_id_drake as alg
import density_id_objects as obj
import robot_scene as rs

# 작업자가 맞춰야 하는 허용 오차.
#
# 이 값은 "작업자 손재주" 가 아니라 **각도를 읽는 쪽의 정확도** 로 정해진다.
# 판정은 FoundationPose 가 읽은 각도로 하기 때문이다(dual_view.adjust_manually).
# 그러니 이 값이 FoundationPose 자체 오차보다 작으면 작업자가 아무리 잘
# 맞춰도 통과할 수 없고, 라운드가 무한히 반복된다.
#
# 3 도로 두었다가 5 도로 올렸다. 실물 FoundationPose 오차가 3 도를 넘을 수
# 있다고 보기 때문이다. 대신 이 창을 넓히면 **실제 각도가 목표에서 그만큼
# 벗어난 채로 로봇이 움직인다**. 그래서 후보 검사도 같이 넓혀야 한다
# (dual_view.PlannerScreen.angle_probes 참고). 한쪽만 고치면, 검사는
# 안 해 본 각도에서 로봇이 움직이게 된다.
ANGLE_TOL_DEG = 5.0
STATUS_PATH = "/status/lamp"
MOVE_STEPS = 60              # 이동 애니메이션 분할 수
MOVE_DT_S = 0.02


class Console:
    """Meshcat 위의 상태 표시와 버튼. 안전 문구를 크게 보여주는 것이 목적."""

    def __init__(self, meshcat, auto=False):
        self.meshcat = meshcat
        self.auto = auto          # True 면 작업자 입력을 기다리지 않는다 (검사용)
        self._buttons = []
        self._sliders = []

    # --- 표시등 -------------------------------------------------------
    def lamp(self, color, label):
        self.meshcat.SetObject(STATUS_PATH, Box(0.6, 0.02, 0.22), color)
        self.meshcat.SetTransform(
            STATUS_PATH, RigidTransform([0.55, 0.42, 0.75]))
        print(f"\n  ┌{'─' * 62}")
        print(f"  │ {label}")
        print(f"  └{'─' * 62}")

    def moving(self, what):
        self.lamp(Rgba(0.80, 0.13, 0.10, 1.0),
                  f"[이동 중] {what} — 로봇에 접근하지 마십시오")

    def stopped(self, what):
        self.lamp(Rgba(0.10, 0.62, 0.30, 1.0),
                  f"[정지] {what} — 로봇이 멈췄습니다. 접근해도 안전합니다")

    def measuring(self):
        self.lamp(Rgba(0.85, 0.55, 0.05, 1.0),
                  "[측정 중] 중력 3방향 — 로봇에 접근하지 마십시오")

    # --- 버튼 ---------------------------------------------------------
    def clear(self):
        for name in self._buttons:
            self.meshcat.DeleteButton(name)
        self._buttons.clear()

    def clear_sliders(self):
        for name in self._sliders:
            self.meshcat.DeleteSlider(name)
        self._sliders.clear()

    def button(self, name):
        self.meshcat.AddButton(name)
        self._buttons.append(name)
        return name

    def slider(self, name, low, high, value):
        self.meshcat.AddSlider(name, min=low, max=high, step=0.5, value=value)
        self._sliders.append(name)
        return name

    def wait_for(self, name):
        """버튼이 눌릴 때까지 대기. 그동안 로봇은 정지 상태를 유지한다."""
        if self.auto:
            return
        start = self.meshcat.GetButtonClicks(name)
        while self.meshcat.GetButtonClicks(name) == start:
            time.sleep(0.05)


class Session:
    def __init__(self, plan, meshcat, auto=False):
        self.plan = plan
        self.spec = obj.OBJECTS[plan["object"]]
        self.meshcat = meshcat
        self.console = Console(meshcat, auto=auto)

        limits = [(np.deg2rad(lo), np.deg2rad(hi))
                  for lo, hi in plan["joint_range_deg"]]
        self.limits_deg = plan["joint_range_deg"]

        builder = DiagramBuilder()
        scene = rs.build_scene(self.spec, plan["density_gt"], limits,
                               builder=builder, include_visuals=True)
        self.plant = scene["plant"]
        self.arm_joints = [self.plant.GetJointByName(n, scene["arm"])
                           for n in rs.ARM_JOINT_NAMES]
        self.finger_joints = [self.plant.GetJointByName(n, scene["gripper"])
                              for n in ("finger1_joint", "finger2_joint")]
        self.object_joints = [self.plant.GetJointByName(j.name, scene["payload"])
                              for j in self.spec.joints]
        AddDefaultVisualization(builder, meshcat)
        self.diagram = builder.Build()
        self.context = self.diagram.CreateDefaultContext()
        self.plant_context = self.plant.GetMyMutableContextFromRoot(self.context)

        q = self.plant.GetPositions(self.plant_context).copy()
        for joint in self.finger_joints:
            q[joint.position_start()] = plan["finger_position_m"]
        self.q = q
        self.arm_q = np.array(plan["arm_q_start"])[
            [j.position_start() for j in self.arm_joints]]
        self.object_q_deg = np.array(plan["rounds"][0]["object_joint_deg"])

        # 추정기를 이 물체에 묶는다 (측정 생성용).
        self.rho_gt = np.array(plan["density_gt"])
        obj.bind_object(self.spec)
        alg.TRUE_RHO = self.rho_gt
        truth, _ = obj.build_plant(self.spec, self.rho_gt)
        alg.TRUTH_PLANT, alg.TRUTH_CTX = truth, truth.CreateDefaultContext()

    # ------------------------------------------------------------------
    def publish(self):
        self.plant.SetPositions(self.plant_context, self.q)
        self.diagram.ForcedPublish(self.context)

    def set_arm(self, arm_q):
        for joint, value in zip(self.arm_joints, arm_q):
            self.q[joint.position_start()] = value

    def set_object_deg(self, degrees):
        for joint, value in zip(self.object_joints, np.atleast_1d(degrees)):
            self.q[joint.position_start()] = np.deg2rad(value)

    def move_arm_to(self, target_full_q, what):
        """현재 팔 자세에서 목표까지 보간 이동. 물체 관절각은 유지한다."""
        self.console.moving(what)
        target_arm = np.array([target_full_q[j.position_start()]
                               for j in self.arm_joints])
        start_arm = np.array([self.q[j.position_start()]
                              for j in self.arm_joints])
        for s in np.linspace(0.0, 1.0, MOVE_STEPS):
            self.set_arm((1.0 - s) * start_arm + s * target_arm)
            self.publish()
            time.sleep(MOVE_DT_S)
        self.arm_q = target_arm

    # ------------------------------------------------------------------
    def adjust_phase(self, target_deg, round_index, total, entry=None):
        """작업자가 물체 관절을 목표 각도로 맞추는 단계. 로봇은 정지."""
        self.console.stopped(f"시작 자세 (round {round_index}/{total})")
        names = [j.name for j in self.spec.joints]
        print(f"  이번 라운드에 필요한 물체 관절각:")
        for name, value in zip(names, target_deg):
            print(f"      {name} = {value:6.1f} deg")
        print(f"  허용 오차 +/- {ANGLE_TOL_DEG} deg")
        print(f"  Meshcat 슬라이더로 각도를 맞춘 뒤 '조정 완료' 를 누르세요.")
        print(f"  * 여기서 고정한 각도는 이번 라운드 내내 유지되는 것으로")
        print(f"    가정합니다 (실시간 추적 없음). 추정식에는 목표값이 아니라")
        print(f"    실제로 맞춘 값이 들어갑니다.")
        if entry is not None:
            print(f"  * 측정 중 힌지가 견뎌야 할 최악 토크"
                  f" {max(entry['hinge_torque_nm']):.3f} N·m"
                  f" / 한계 {entry['hinge_limit_nm']:.3f} N·m"
                  f"  (여유 {entry['hinge_margin']:.1f}x)")

        self.console.clear()
        self.console.clear_sliders()
        slider_names = []
        for name, value, (lo, hi) in zip(names, self.object_q_deg,
                                         self.limits_deg):
            slider_names.append(
                self.console.slider(f"{name} [deg]", lo, hi, float(value)))
        for name, value in zip(names, target_deg):
            self.console.button(f"목표: {name} = {value:.1f} deg")
        done = self.console.button("① 조정 완료 — 각도 확인")

        if self.console.auto:      # 검사 모드: 작업자가 맞춘 것으로 친다
            for name, value in zip(slider_names, target_deg):
                self.meshcat.SetSliderValue(name, float(value))
        while True:
            # 슬라이더를 돌리는 동안 물체만 움직이고 로봇은 그대로다.
            current = np.array([self.meshcat.GetSliderValue(n)
                                for n in slider_names])
            self.set_object_deg(current)
            self.publish()
            if self.console.auto or self.meshcat.GetButtonClicks(done) > 0:
                error = np.abs(current - np.asarray(target_deg))
                if np.all(error <= ANGLE_TOL_DEG):
                    self.object_q_deg = current
                    print(f"  각도 확인됨: {np.round(current, 1)} deg"
                          f"  (오차 {np.round(error, 2)} deg)")
                    break
                print(f"  아직 목표와 다릅니다. 현재 {np.round(current, 1)},"
                      f" 목표 {np.round(target_deg, 1)},"
                      f" 오차 {np.round(error, 1)} deg — 다시 맞춰주세요.")
                self.console.clear()
                for name, value in zip(names, target_deg):
                    self.console.button(f"목표: {name} = {value:.1f} deg")
                done = self.console.button("① 조정 완료 — 각도 확인")
            time.sleep(0.05)

        # 2단 안전 확인 — 이걸 누르기 전에는 로봇이 절대 움직이지 않는다.
        self.console.clear()
        self.console.clear_sliders()
        confirm = self.console.button("② 물체에서 손을 뗐습니다 — 이동 시작")
        print(f"  물체에서 손을 뗀 뒤 '② 손을 뗐습니다' 를 누르세요."
              f" 누르기 전까지 로봇은 정지 상태를 유지합니다.")
        self.console.wait_for(confirm)
        self.console.clear()

    # ------------------------------------------------------------------
    def measure_phase(self, entry, theta_deg):
        """theta_deg 는 작업자가 실제로 맞춘 각도. 목표값이 아니다."""
        self.console.measuring()
        theta = np.deg2rad(theta_deg)
        for g_hat, arm_q in zip(entry["gravity_dirs"], entry["arm_q_measure"]):
            self.move_arm_to(np.array(arm_q),
                             f"측정 자세 (중력 {np.round(g_hat, 0)})")
            time.sleep(0.4)
        return alg.measure(theta)

    # ------------------------------------------------------------------
    def run(self):
        rounds = self.plan["rounds"]
        Sigma = alg.SIGMA0.copy()
        A_all = np.empty((0, alg.P))
        y_all = np.empty(0)

        print(f"\n{self.plan['label']}")
        print(f"힌지 {self.plan['hinge']['label']},"
              f" 유지토크 {self.plan['hinge']['torque_nm']} N·m")
        print(f"후보 {self.plan['n_candidates']}개 중"
              f" 도달·충돌 통과 {self.plan['n_reachable']}개")
        print(f"Meshcat: {self.meshcat.web_url()}")
        self.set_arm(np.array(self.plan["arm_q_start"])[
            [j.position_start() for j in self.arm_joints]])
        self.set_object_deg(rounds[0]["object_joint_deg"])
        self.publish()
        start = self.console.button("브라우저를 연 뒤 눌러 시작")
        self.console.wait_for(start)
        self.console.clear()

        for index, entry in enumerate(rounds, start=1):
            target = np.array(entry["object_joint_deg"])
            self.move_arm_to(np.array(self.plan["arm_q_start"]), "시작 자세")
            self.adjust_phase(target, index, len(rounds), entry=entry)

            # 고정된 실제 각도를 그대로 추정식에 넣는다.
            actual = self.object_q_deg.copy()
            y = self.measure_phase(entry, actual)
            A = alg.regressor(np.deg2rad(actual))
            A_all = np.vstack([A_all, A])
            y_all = np.concatenate([y_all, y])
            Sigma = alg.posterior_covariance(Sigma, A)
            rho_hat = alg.constrained_map(A_all, y_all)

            print(f"\n  [round {index} 측정 완료]")
            for part, gt, est in zip(self.spec.parts, self.rho_gt, rho_hat):
                print(f"      {part.name:<13} GT {gt:7.0f}   추정 {est:8.1f}"
                      f"   오차 {100 * abs(est - gt) / gt:5.2f}%")
            print(f"      RMSE {np.sqrt(np.mean((rho_hat - self.rho_gt) ** 2)):.1f}"
                  f" kg/m^3")

        self.move_arm_to(np.array(self.plan["arm_q_start"]), "시작 자세로 복귀")
        self.console.stopped("측정 종료")
        self.console.button("완료")
        print("\n모든 라운드가 끝났습니다.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--self-test", action="store_true",
                        help="작업자 입력을 자동으로 대신해 절차를 점검한다")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    meshcat = StartMeshcat()
    Session(plan, meshcat, auto=args.self_test).run()


if __name__ == "__main__":
    main()
