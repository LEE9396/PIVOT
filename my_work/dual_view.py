"""화면 두 개로 나눈 탐색 — 계획 화면(왼쪽)과 로봇 화면(오른쪽).

  왼쪽 (planner)  물체만 놓고 후보 자세를 훑으며 정보이득을 계산해
                  다음 exploration 자세를 고른다. 지금까지의 시뮬레이션.
  오른쪽 (robot)  고른 자세를 받아 로봇이 실제로 움직인다.
                  starting position <-> exploration position 왕복.

두 화면은 **버스(PoseBus)** 로만 이야기한다. 지금은 같은 프로세스 안의
큐지만, 메시지가 JSON 으로 직렬화 가능한 dict 이므로 TCP/IP 소켓이나
ROS1 토픽으로 바꿔 끼우면 오른쪽 화면이 그대로 실물 로봇을 몰 수 있다.

    planner --- exploration_target ---> robot        (관절각 목표)
    planner <--- measurement ---------- robot        (렌치 측정값)

실행:
    cd ~/Desktop/PIVOT/my_work
    ../robot_learning/scripts/run_drake_env.sh python dual_view.py \
        --object 3link --joint-range-deg 20 150 --hinge-torque 0.5 --auto-scale

    브라우저 탭 두 개를 엽니다. 주소는 실행 시 출력됩니다.
"""

import argparse
import time
from pathlib import Path

import numpy as np
from pydrake.geometry import Rgba, StartMeshcat
from pydrake.systems.framework import DiagramBuilder
from pydrake.visualization import AddDefaultVisualization

import density_id_drake as alg
import density_id_objects as obj
import angle_aware as aa
import design_core as dc
import explore_view as ev
import path_planning as pp
import robot_scene as rs
from operator_ui import ANGLE_TOL_DEG, Console
from pose_bus import LocalBus, TcpBus

# 준정적 이동. 관절이 흐르지 않도록 천천히 옮긴다. 힌지 유지토크를 모르므로
# 한계 기준을 세울 수 없어, 넉넉히 느린 시간을 기본으로 둔다.
DEFAULT_MOVE_DURATION_S = 8.0
DEFAULT_FRAME_DT_S = 0.03


# ---------------------------------------------------------------------------
# 준비 — 후보 자세와 팔 자세를 미리 계산해 둔다
# ---------------------------------------------------------------------------
# 전용 자세로 옮겨서 얻는 이득이 이 배수보다 작으면 그냥 파지 자세에서 읽는다.
MIN_VIEW_GAIN = 1.15


def prepare(spec, hinge, joint_limits_rad, safety, steps, min_distance_m,
            density_scale, prior="weight", gripper="robotiq2f85",
            view_poses=True):
    """hinge=None 이면 관절이 절대 움직이지 않는다고 보고 토크 필터를 건너뛴다.

    이 연구는 노트북·스탠드·폴더블처럼 사용자가 각도를 맞춰 두면 그대로
    고정되는 물체를 대상으로 한다. 힌지가 버티는지 따질 필요가 없으므로,
    남는 제약은 로봇 쪽 세 가지(도달·자세 충돌·경로)뿐이다.
    """
    rho_gt = obj.bind_object(spec, hinge=hinge, safety=safety,
                             density_scale=density_scale)
    alg.JOINT_LIMITS = list(joint_limits_rad)

    # 초기값. 정답은 여기에 절대 들어가지 않는다.
    #   weight : 물체를 저울에 올려 총무게만 잰 상태 -> 평균 밀도가 초기값
    #   mesh   : 메시 외형 부피만 아는 상태 -> 아주 넓은 사전분포
    if prior == "weight":
        # 저울에는 힌지까지 붙은 채로 올라간다.
        total = obj.assembled_mass_kg(spec, rho_gt)
        mu, _, mean_density = obj.apply_weight_prior(spec, total)
        print(f"  초기값: 저울 총무게 {1000*total:.1f} g"
              f" -> 모든 부위를 평균 밀도 {mean_density:.0f} kg/m^3 로 시작")
    else:
        mu, Sigma0, lows, highs = obj.apply_mesh_prior(spec)
        print(f"  초기값: 메시 외형 부피만 앎"
              f" -> {mu[0]:.0f} +/- {np.sqrt(Sigma0[0,0]):.0f} kg/m^3")
    if hinge is None:
        hinge_ok = lambda theta: True
        print("  관절 고정 가정 — 힌지 토크 필터를 쓰지 않습니다")
    else:
        hinge_ok = obj.make_is_feasible(spec, hinge, safety, rho_gt)
    checker = rs.PoseChecker(spec, densities=rho_gt,
                             joint_limits_rad=joint_limits_rad,
                             min_distance_m=min_distance_m, gripper=gripper)

    axes = [np.linspace(lo, hi, steps) for lo, hi in joint_limits_rad]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(
        -1, len(spec.joints))

    feasible, arm_solutions = [], {}
    for theta in grid:
        if not hinge_ok(theta):
            continue
        solutions = checker.solutions_for(theta)
        if any(v is None for v in solutions.values()):
            continue
        feasible.append(theta)
        arm_solutions[tuple(np.round(theta, 9))] = solutions
    if not feasible:
        raise RuntimeError("힌지·도달·충돌을 모두 통과하는 자세가 없다")

    start_q, presentation = rs.find_starting_pose(checker, feasible)
    if start_q is None:
        raise RuntimeError("전 구동범위에서 안전한 시작 자세를 찾지 못했다")

    # 자세가 안전해도 시작 자세에서 거기까지 가는 **경로**가 없을 수 있다.
    # 반복 횟수를 15배 늘려도 못 찾는 자세가 있는데, 그건 탐색 부족이 아니라
    # 실제로 연결 불가능한 것이다. 그런 자세는 후보에서 뺀다.
    indices = [joint.position_start() for joint in checker.arm_joints]
    planner = pp.ArmPathPlanner(checker.plant, checker.context,
                                checker.arm_joints, min_distance_m,
                                np.array(start_q))
    connected, dropped = [], []
    for theta in feasible:
        key = tuple(np.round(theta, 9))
        q = np.array(start_q).copy()
        for joint, value in zip(checker.object_joints, np.atleast_1d(theta)):
            q[joint.position_start()] = value
        planner.set_fixed(q)
        home = np.array(start_q)[indices]
        if all(planner.plan(home, np.array(arm)[indices]) is not None
               and planner.plan(np.array(arm)[indices], home) is not None
               for arm in arm_solutions[key].values()):
            connected.append(theta)
        else:
            dropped.append(np.degrees(theta))
    if not connected:
        raise RuntimeError("경로까지 연결되는 자세가 없다")
    if dropped:
        print(f"  경로 계획 실패로 제외한 자세 {len(dropped)}개: "
              + ", ".join(str(np.round(d, 0).astype(int)) for d in dropped))
    print(f"  최종 후보 {len(connected)}/{len(grid)}"
          f" (힌지·도달·충돌·경로 모두 통과)")

    feasible = connected
    keys = {tuple(np.round(t, 9)) for t in feasible}
    alg.is_feasible = lambda th: tuple(np.round(np.asarray(th), 9)) in keys

    # 관절각을 읽을 자세 — 관절 하나에 하나씩. 카메라가 그 관절의 회전을
    # 가장 잘 보는 방향으로 물체를 든다 (robot_scene.find_viewing_poses).
    scorer = rs.ViewScorer(spec, rho_gt) if view_poses else None

    return dict(rho_gt=rho_gt, feasible=feasible, arm_solutions=arm_solutions,
                start_q=start_q, presentation=presentation,
                finger=checker.finger_value, n_grid=len(grid),
                checker=checker, planner=planner, gripper=gripper,
                scorer=scorer, view_poses=view_poses)


def report_viewing_poses(spec, setup, theta=None):
    """관절별 각도 측정 자세를 표로 보여 준다.

    파지 자세 하나에서 다 읽을 때와 견줘 얼마나 좋아지는지가 요점이다.
    단위는 화소/도 — 관절을 1도 돌렸을 때 움직이는 부위가 화면에서 몇 화소
    움직이는가. 클수록 FoundationPose 가 각도를 또렷하게 읽는다.
    """
    if not setup.get("view_poses"):
        print("  각도 측정 자세: 쓰지 않음 (--no-view-poses) — 파지 자세에서 읽음")
        return None
    checker = setup["checker"]
    if theta is None:
        theta = np.asarray(setup["feasible"][len(setup["feasible"]) // 2],
                           dtype=float)
    poses = rs.find_viewing_poses(checker, theta, scorer=setup.get("scorer"))

    # 견줄 대상: 파지(시작) 자세에서 그 관절각 그대로 읽었을 때.
    baseline_q = np.array(setup["start_q"], dtype=float).copy()
    for object_joint, value in zip(checker.object_joints, np.atleast_1d(theta)):
        baseline_q[object_joint.position_start()] = float(value)
    checker.plant.SetPositions(checker.context, baseline_q)

    print(f"  관절별 각도 측정 자세  (관절각 {np.round(np.degrees(theta), 0)} deg 기준)")
    print(f"    {'관절':<10}{'파지자세':>10}{'전용자세':>10}{'배':>7}"
          f"   {'축-시선':>8}")
    for index, (joint, pose) in enumerate(zip(spec.joints, poses)):
        checker.plant.SetPositions(checker.context, baseline_q)
        base = rs.observability_px_per_deg(checker.plant, checker.context,
                                           spec, index, checker.payload)
        if pose["arm_q"] is None:
            print(f"    {joint.name:<10}{base:>9.2f}{'못 찾음':>11}")
            continue
        print(f"    {joint.name:<10}{base:>9.2f}{pose['observability']:>10.2f}"
              f"{pose['observability']/max(base, 1e-9):>6.1f}배"
              f"   {pose['axis_view_deg']:>7.0f}°")
    return poses


# ---------------------------------------------------------------------------
# 왼쪽 화면 — 후보를 훑고 다음 자세를 고른다
# ---------------------------------------------------------------------------
class PlannerScreen:
    """왼쪽 화면 — 후보를 훑고 다음 관절각을 추천한다.

    각도 오차(FoundationPose 상대오차)를 측정 공분산에 반영하므로,
    정보이득이 현재 밀도 추정값에 의존한다. 그래서 추천이 라운드마다
    실제로 달라진다. base 를 띄워 중력 방향 변화도 재생한다.
    """

    def __init__(self, spec, setup, meshcat, pace, target,
                 angle_rel_error=aa.DEFAULT_ANGLE_REL_ERROR,
                 angle_floor_deg=aa.DEFAULT_ANGLE_FLOOR_DEG,
                 select_mode="continuous", criterion="D", estimator="tls",
                 stop_rule="residual", systematic=0.3, search_attempts=12,
                 block_radius_deg=15.0, probe_side=5):
        self.spec = spec
        self.setup = setup
        self.meshcat = meshcat
        self.pace = pace
        self.target = target
        self.angle_rel_error = angle_rel_error
        # 각도 오차 모형:  sigma = max(rel_error * |theta|,  floor)
        #   rel_error>0, floor 작게  -> 각도에 비례 (예전 기본값)
        #   rel_error=0,  floor=2도  -> **고정 오차 2도**
        # FoundationPose 가 각도에 비례해 틀릴 이유는 별로 없다. 보통은
        # 텍스처·가림에 좌우되는 고정 오차에 가깝다.
        self.angle_floor_deg = angle_floor_deg
        # 검토 지점 ①③④⑤ 를 실행 시점에 갈아끼울 수 있게 열어 둔다.
        # 각각 study_continuous / study_tls / study_criterion / study_stopping
        # 으로 근거를 확인한 뒤 고르면 된다.
        self.select_mode = select_mode      # grid | continuous
        self.criterion = criterion          # D | A | E
        self.estimator = estimator          # wls | tls
        self.stop_rule = stop_rule          # variance | residual | bias
        self.systematic = systematic
        # 연속 탐색이 막힌 자리를 배워가며 다시 찾는 횟수와 차단 반경.
        self.max_search_attempts = search_attempts
        self.block_radius_rad = np.deg2rad(block_radius_deg)
        # 각도 오차 구간을 몇 칸 격자로 훑을지. 충돌 질의라 키워도 거의 공짜다.
        self.probe_side = probe_side
        self.rho_gt = setup["rho_gt"]

        builder = DiagramBuilder()
        plant, bodies = ev.build_floating(spec, self.rho_gt, builder)
        AddDefaultVisualization(builder, meshcat)
        self.plant = plant
        self.diagram = builder.Build()
        self.context = self.diagram.CreateDefaultContext()
        self.player = ev.Player(spec, meshcat, plant, bodies,
                                self.context, self.diagram)

        self.Sigma = alg.SIGMA0.copy()
        self.rho_hat = alg.MU0.copy()
        self.blocks = []
        self.rounds = []          # (measured_theta, y) — TLS 가 쓴다
        self.inflate = 1.0
        self.bias_cov = None

    # ------------------------------------------------------------------
    def show(self, theta, g_hat=(0.0, 0.0, -1.0)):
        self.player.show(theta, g_hat)

    def half_width(self):
        """정지 조건. GT 는 쓰지 않는다.

        variance : 사후 공분산만 (현행)
        residual : + 잔차가 모형보다 크면 그만큼 부풀린다
        bias     : + 계통 각도오차가 답을 얼마나 미는지 재적합으로 재서 더한다
        """
        inflate = 1.0 if self.stop_rule == "variance" else self.inflate
        bias = self.bias_cov if self.stop_rule == "bias" else None
        # 힌지처럼 이미 아는 양은 정지 판단에서 뺀다 (design_core.stopping_width)
        return dc.stopping_width(
            dc.half_width(self.Sigma, self.rho_hat, Cov_bias=bias,
                          inflate=inflate), len(self.spec.parts))

    def converged(self):
        return self.half_width() <= self.target

    # ------------------------------------------------------------------
    def score(self, theta):
        return dc.utility(theta, self.rho_hat, self.Sigma, dc.CANONICAL_TRIAD,
                          self.criterion, self.angle_rel_error,
                          self.angle_floor_deg)

    def angle_probes(self, theta):
        """각도 측정 오차 95 % 구간을 격자로 훑는 점들.

        모서리만 보면 안 된다. IK 도달 영역은 매끄러운 덩어리가 아니라
        구멍이 점점이 박힌 지형이라, 모서리는 우연히 구멍을 피하고 실제
        측정값이 구멍에 빠진다. 실제로 -87.45 도에서 통과한 후보가
        -86.90 도에서 실패해 라운드가 통째로 날아갔다.
        """
        bounds = [j.limits_rad for j in self.spec.joints]
        lows = np.array([lo for lo, _ in bounds])
        highs = np.array([hi for _, hi in bounds])
        theta = np.atleast_1d(theta)
        sigma = np.sqrt(np.diag(aa.angle_covariance(
            theta, self.angle_rel_error, self.angle_floor_deg)))
        axes = [np.linspace(t - 1.96 * s, t + 1.96 * s, self.probe_side)
                for t, s in zip(theta, sigma)]
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(
            -1, len(theta))
        return np.clip(grid, lows, highs)

    def check_candidate(self, theta):
        """후보 자세가 실제로 쓸 수 있는가. (사유, 팔자세) 를 돌려준다.

        핵심: **IK 는 한 번만 푼다.**

        IK 가 거는 제약은 세 가지인데,
          (a) 중력 방향  센서 프레임이 월드 -z 를 g_hat 으로 본다
          (b) 작업공간   센서 원점이 상자 안에 있다
          (c) 최소거리   아무 것도 안 부딪힌다

        센서 프레임은 잡힌 base part 에 용접돼 있고 base part 는 그리퍼에
        용접돼 있다. 물체 관절각 theta 는 센서 프레임보다 **아래**(자식
        방향)에 있으므로, theta 를 바꿔도 센서 프레임은 안 움직인다.
        측정으로 확인했다 — 팔을 고정한 채 theta 를 흔들었을 때 센서 프레임
        변화가 정확히 0 이었다.

        따라서 (a)(b) 는 theta 와 무관하고 (c) 만 theta 에 의존한다.
        각도가 흔들릴 때 다시 볼 것은 **충돌뿐**이다. 그래서

          1. 명목 각도에서 IK 한 번    -> 중력 3방향의 팔 자세
          2. 각도 오차 구간을 격자로 훑으며 그 팔 자세들의 충돌만 확인
          3. 시작 자세와의 왕복 경로 확인

        비용이 크게 준다. 후보 하나당 격자 25 점 기준으로
          IK 재풀이 3497 ms  ->  충돌 질의 170 ms   (21 배)
        격자를 4 점에서 25 점으로 키워도 169 -> 170 ms 라 사실상 공짜다.

        그리고 실행 단계도 **이 팔 자세를 그대로 쓴다**. 검사한 자세로
        실행하니 검사와 실행이 어긋날 수 없다.
        """
        checker = self.setup["checker"]
        theta = np.atleast_1d(theta)

        # 1) IK 는 여기서 한 번만.
        solutions = checker.solutions_for(theta)
        if any(v is None for v in solutions.values()):
            return "도달·충돌", None

        # 2) 각도가 흔들려도 그 팔 자세로 안 부딪히는가 (충돌 질의만).
        for probe in self.angle_probes(theta):
            for arm in solutions.values():
                if not checker.arm_pose_is_clear(arm, probe):
                    return "각도 오차 여유", None

        # 3) 거기까지 갈 수 있는가.
        if not self.path_exists(theta, solutions):
            return "경로 없음", None
        return None, solutions

    def path_exists(self, theta, solutions):
        """시작 자세 <-> 탐색 자세 왕복 경로가 있는가."""
        planner = self.setup.get("planner")
        if planner is None:
            return True
        checker = self.setup["checker"]
        indices = [j.position_start() for j in checker.arm_joints]
        q = np.array(self.setup["start_q"]).copy()
        for joint, value in zip(checker.object_joints, theta):
            q[joint.position_start()] = value
        planner.set_fixed(q)
        home = np.array(self.setup["start_q"])[indices]
        for arm in solutions.values():
            goal = np.array(arm)[indices]
            if planner.plan(home, goal) is None:
                return False
            if planner.plan(goal, home) is None:
                return False
        return True

    def select_continuous(self, round_index):
        """안전한 후보가 나올 때까지 연속 탐색을 반복한다.

            연속 최적화 -> 그 각도를 물체에 적용 -> IK/충돌/여유/경로 검사
              -> 통과하면 사용자에게 전달
              -> 막히면 그 근방을 목적함수에서 눌러 놓고 다시 최적화

        격자와 달리 미리 검증해 둔 목록이 없으므로, 막힌 자리를 하나씩
        배워가며 좁혀 들어간다. 차단 영역이 구동범위를 거의 덮을 때까지
        못 찾으면 그때만 미리 검증된 격자 후보로 후퇴한다.
        """
        bounds = [j.limits_rad for j in self.spec.joints]
        blocked = []

        def score(theta):
            value = self.score(theta)
            for center in blocked:      # 막힌 자리 주변을 눌러 둔다
                d = np.linalg.norm(np.asarray(theta) - center)
                value -= 10.0 * np.exp(-(d / self.block_radius_rad) ** 2)
            return value

        for attempt in range(self.max_search_attempts):
            theta, info = dc.continuous_best(bounds, score, n_starts=8,
                                             seed=1000 * round_index + attempt)
            theta = np.atleast_1d(theta)
            if any(np.linalg.norm(theta - c) < 0.25 * self.block_radius_rad
                   for c in blocked):
                break                   # 같은 자리만 맴돈다 = 더 볼 곳이 없다
            reason, solutions = self.check_candidate(theta)
            if reason is None:
                if attempt:
                    print(f"        {attempt}번 막힌 뒤 찾음")
                self.show(theta)
                return theta, self.score(theta), solutions
            print(f"        후보 {np.round(np.degrees(theta), 1)} 탈락"
                  f" ({reason}) — 그 근방을 빼고 다시 탐색")
            blocked.append(theta)

        # 끝내 못 찾으면 준비 단계에서 이미 검증해 둔 격자 후보로 후퇴한다.
        print(f"        {len(blocked)}곳이 막혀 연속 탐색을 접고 격자로 후퇴")
        best, best_score, best_sol = None, -np.inf, None
        for candidate in sorted(self.setup["feasible"],
                                key=self.score, reverse=True):
            reason, solutions = self.check_candidate(candidate)
            if reason is None:
                best, best_score, best_sol = (np.atleast_1d(candidate),
                                              self.score(candidate), solutions)
                break
        if best is None:                # 그래도 없으면 검증 없이 최고값
            candidates = self.setup["feasible"]
            scores = [self.score(t) for t in candidates]
            best = np.atleast_1d(candidates[int(np.argmax(scores))])
            best_score, best_sol = float(np.max(scores)), None
        self.show(best)
        return best, best_score, best_sol

    def select(self, round_index):
        print(f"\n[왼쪽] round {round_index} 탐색 — "
              f"{'연속 최적화' if self.select_mode == 'continuous' else '격자'}"
              f" / {self.criterion}-최적"
              f" (각도 오차 {100*self.angle_rel_error:.0f}% 또는"
              f" 최소 {self.angle_floor_deg:.1f}도 반영)")

        if self.select_mode == "continuous":
            # 왼쪽 화면에 지형을 한 번 훑어 보여준 뒤 최적화한다
            for theta in self.setup["feasible"]:
                self.show(theta)
                time.sleep(self.pace)
            theta, best_score, solutions = self.select_continuous(round_index)
        else:
            candidates = self.setup["feasible"]
            gains = []
            for candidate in candidates:
                gains.append(self.score(candidate))
                self.show(candidate)
                time.sleep(self.pace)
            gains = np.array(gains)
            order = np.argsort(gains)[::-1]
            for rank in range(min(3, len(order))):
                deg = np.round(np.degrees(candidates[order[rank]]), 0)
                print(f"        {rank + 1}위 q={deg}"
                      f" 기준값 {gains[order[rank]]:.3f}")
            theta = np.atleast_1d(candidates[int(np.argmax(gains))])
            best_score = float(gains.max())
            _, solutions = self.check_candidate(theta)
            self.show(theta)
        R_eff = aa.effective_covariance(self.spec, theta, self.rho_hat,
                                        self.angle_rel_error)
        share = 1.0 - float(np.mean(alg.R_STACK_DIAG)) / float(np.mean(np.diag(R_eff)))
        deg = np.degrees(theta)
        print(f"[왼쪽] 추천 q={np.round(deg, 1)} deg"
              f"  (측정 불확실성 중 각도 기여 {100*share:.0f}%)"
              f" -> 오른쪽 화면으로 전달")
        # 검사에서 쓴 팔 자세를 그대로 실어 보낸다. 실행 쪽이 다시 풀면
        # 검사와 실행이 어긋나고, 그게 라운드가 날아가는 직접 원인이었다.
        # 팔 자세는 float 목록이라 TCP/ROS 로 그대로 나간다.
        arm_solutions = None
        if solutions is not None:
            arm_solutions = [[float(v) for v in solutions[tuple(g)]]
                             for g in alg.G_DIRS]
        return dict(round=round_index,
                    object_joint_deg=[float(v) for v in deg],
                    arm_solutions=arm_solutions,
                    reason=f"{self.criterion}-optimal={best_score:.4f}")

    # ------------------------------------------------------------------
    def update(self, reply):
        """FoundationPose 가 알려준 각도로 갱신한다. 명령값이 아니다."""
        measured = np.deg2rad(reply["object_joint_deg_measured"])
        print(f"[왼쪽] 측정값 수신"
              f" (FoundationPose 각도 {np.round(np.degrees(measured), 2)} deg)"
              f" — 중력 3방향 재생")
        for g_hat in alg.G_DIRS:
            self.show(measured, g_hat)
            time.sleep(0.5)
        self.show(measured)

        wrench = np.array(reply["wrench"])
        A = alg.regressor(measured)
        R_eff = aa.effective_covariance(self.spec, measured, self.rho_hat,
                                        self.angle_rel_error)
        self.blocks.append((A, wrench, R_eff))
        self.rounds.append((measured, wrench))
        self.Sigma = aa.posterior_covariance(self.Sigma, A, R_eff)

        rho_wls = aa.constrained_map(self.blocks, alg.MU0, alg.SIGMA0,
                                     alg.RHO_BOUNDS)
        if self.estimator == "tls":
            # 각도 보정량을 밀도와 함께 푼다. 계수행렬이 틀렸다는 사실이
            # 모형 안에 있으므로 오차변수 치우침이 남지 않는다.
            self.rho_hat, _ = dc.tls_map(
                self.rounds, alg.MU0, alg.SIGMA0, alg.RHO_BOUNDS,
                dc.CANONICAL_TRIAD, rho_init=rho_wls,
                rel_error=self.angle_rel_error)
        else:
            self.rho_hat = rho_wls

        self.inflate = dc.residual_inflation(self.blocks, self.rho_hat)
        if self.stop_rule == "bias":
            self.bias_cov = dc.bias_by_refit(
                self.rounds, alg.MU0, alg.SIGMA0, alg.RHO_BOUNDS,
                dc.CANONICAL_TRIAD, self.rho_hat, self.systematic,
                self.estimator, self.angle_rel_error)

        half = dc.half_width(
            self.Sigma, self.rho_hat,
            Cov_bias=self.bias_cov if self.stop_rule == "bias" else None,
            inflate=1.0 if self.stop_rule == "variance" else self.inflate)
        print(f"[왼쪽] round {reply['round']} 추정 갱신"
              f"  (95% 상대반폭 {100*self.half_width():.2f}%"
              f" / 목표 {100*self.target:.2f}%"
              f"{f', 잔차팽창 x{self.inflate:.2f}' if self.inflate > 1.005 else ''})")
        for row, gt, est, hw in zip(obj.body_table(self.spec), self.rho_gt,
                                    self.rho_hat, half):
            mark = "" if row["kind"] == "part" else "  (힌지, 저울로 앎)"
            print(f"        {row['name']:<13} 추정 {est:8.1f} +/-{hw*est:6.1f}"
                  f"   [GT {gt:7.0f}  오차 {100*abs(est-gt)/gt:5.2f}%]{mark}")


# ---------------------------------------------------------------------------
# 오른쪽 화면 — 로봇이 실제로 움직인다
# ---------------------------------------------------------------------------
class RobotScreen:
    """로봇 쪽. 화면 두 개를 쓴다.

      robot_meshcat   Drake 로봇이 움직이는 화면. **시뮬레이션 모드에서만** 쓴다.
                      실물 배포 모드에서는 None 을 준다. 그래도 IK 와 경로
                      계획은 그대로 돈다 — 실물 로봇에 보낼 경유점을 만들어야
                      하기 때문이다. 그리지만 않을 뿐이다.
      console_meshcat 작업자 UI (신호등·슬라이더·버튼). 두 모드 모두 쓴다.

    driver 를 주면 이동이 실물로 나간다 (hardware.RobotDriver). 안 주면
    Drake 안에서 애니메이션으로 움직인다.
    """

    def __init__(self, spec, setup, meshcat, console_meshcat=None,
                 driver=None, wrench_sensor=None, pose_sensor=None,
                 tare=None, manual=True, autostart=False,
                 adjust_steps=45, adjust_dt=0.02, adjust_hold=0.8,
                 move_duration_s=DEFAULT_MOVE_DURATION_S,
                 frame_dt_s=DEFAULT_FRAME_DT_S,
                 angle_rel_error=aa.DEFAULT_ANGLE_REL_ERROR,
                 angle_floor_deg=aa.DEFAULT_ANGLE_FLOOR_DEG, seed=0,
                 min_distance_m=rs.MIN_DISTANCE_M, plan_iters=20000,
                 settle_s=0.3):
        self.spec = spec
        self.setup = setup
        self.meshcat = meshcat                      # None 이면 로봇 화면 없음
        # 버튼·슬라이더는 콘솔(작업자 UI 화면) 위에 만들어진다. 로봇 화면과
        # 다른 Meshcat 이므로, 값을 읽을 때도 반드시 이쪽을 봐야 한다.
        # 로봇 화면은 배포 모드에서 아예 없을 수도 있다.
        self.driver = driver                        # None 이면 Drake 안에서 이동
        self.wrench_sensor = wrench_sensor
        self.pose_sensor = pose_sensor
        self.tare = tare
        self.manual = manual
        # console.auto 는 "작업자 입력을 기다리지 않는다"는 뜻이다.
        self.ui = console_meshcat if console_meshcat is not None else meshcat
        if self.ui is None:
            raise ValueError("작업자 UI 화면(console_meshcat)이 필요합니다")
        self.console = Console(self.ui, auto=not manual)
        self.autostart = autostart
        self.adjust_steps = adjust_steps
        self.adjust_dt = adjust_dt
        self.adjust_hold = adjust_hold
        self.move_duration_s = move_duration_s
        self.frame_dt_s = frame_dt_s
        self.angle_rel_error = angle_rel_error
        self.angle_floor_deg = angle_floor_deg
        self.rng = np.random.default_rng(seed)

        builder = DiagramBuilder()
        scene = rs.build_scene(spec, setup["rho_gt"],
                               alg.JOINT_LIMITS, builder=builder,
                               include_visuals=True,
                               gripper=setup.get("gripper", "robotiq2f85"))
        self.plant = scene["plant"]
        self.arm_joints = [self.plant.GetJointByName(n, scene["arm"])
                           for n in rs.ARM_JOINT_NAMES]
        # 죠를 움직이는 관절 이름은 그리퍼마다 다르다. Robotiq 2F-85 는
        # 개구량을 URDF 에 구워 넣고 고정으로 붙이므로 여기서 셀 관절이 없다.
        self.finger_joints = [self.plant.GetJointByName(n, scene["gripper"])
                              for n in scene["gripper_spec"].finger_joint_names]
        self.object_joints = [self.plant.GetJointByName(j.name, scene["payload"])
                              for j in spec.joints]
        self.payload = scene["payload"]
        self.view_poses = bool(setup.get("view_poses", False))
        if meshcat is not None:
            AddDefaultVisualization(builder, meshcat)
        self.diagram = builder.Build()
        self.context = self.diagram.CreateDefaultContext()
        self.plant_context = self.plant.GetMyMutableContextFromRoot(self.context)

        self.q = self.plant.GetPositions(self.plant_context).copy()
        for joint in self.finger_joints:
            self.q[joint.position_start()] = setup["finger"]
        self.limits_deg = [(float(np.degrees(lo)), float(np.degrees(hi)))
                           for lo, hi in alg.JOINT_LIMITS]
        self.object_q_deg = np.array([lo for lo, _ in self.limits_deg])
        # 물체가 실제로 놓인 각도(object_q_deg)와, 로봇이 안다고 믿는 각도를
        # 구분한다. IK 와 경로 계획은 반드시 '믿는 각도' 로 일관되게 해야 한다.
        # 실물에서는 실제 각도를 알 방법이 없기 때문이다.
        self.believed_q_deg = self.object_q_deg.copy()

        # 경로 계획기. 화면 표시용 컨텍스트와 섞이지 않도록 별도 컨텍스트를 쓴다.
        self.plan_root = self.diagram.CreateDefaultContext()
        self.planner = pp.ArmPathPlanner(
            self.plant, self.plant.GetMyContextFromRoot(self.plan_root),
            self.arm_joints, min_distance_m, self.q, seed=seed)
        self.plan_iters = plan_iters
        self.plan_cache = {}
        self.samples_per_hold = obj.DEFAULT_SAMPLES_PER_HOLD
        self.settle_s = settle_s

    def arm_solutions_for(self, theta):
        """가장 가까운 후보 자세의 팔 자세 묶음. 어긋나면 예외를 낸다."""
        candidates = self.setup["feasible"]
        distances = [np.linalg.norm(np.asarray(c) - np.asarray(theta))
                     for c in candidates]
        index = int(np.argmin(distances))
        if distances[index] > np.deg2rad(0.5):
            raise KeyError(
                f"요청 자세 {np.round(np.degrees(theta), 2)} deg 에 해당하는"
                f" 후보가 없다 (가장 가까운 것과 "
                f"{np.degrees(distances[index]):.2f} deg 차이)")
        key = tuple(np.round(candidates[index], 9))
        return self.setup["arm_solutions"][key]

    # -- 표시 ----------------------------------------------------------
    def publish(self):
        self.plant.SetPositions(self.plant_context, self.q)
        if self.meshcat is not None:
            self.diagram.ForcedPublish(self.context)

    def set_arm_from(self, full_q):
        for joint in self.arm_joints:
            self.q[joint.position_start()] = full_q[joint.position_start()]

    def set_object_deg(self, degrees):
        for joint, value in zip(self.object_joints, np.atleast_1d(degrees)):
            self.q[joint.position_start()] = np.deg2rad(value)

    def move_to(self, full_q, what, duration_s=None):
        """준정적 이동. 정해진 시간에 걸쳐 부드럽게 움직인다.

        관절이 고정된다는 가정은 중력만 버티면 된다는 뜻이 아니다. 빨리
        움직이면 관성 반력이 힌지에 추가로 걸려 각도가 흐를 수 있다.
        시작·끝 속도와 가속도가 0 인 프로파일로 천천히 이동한다.
        """
        duration_s = duration_s or self.move_duration_s
        target = np.array([full_q[j.position_start()] for j in self.arm_joints])
        start = np.array([self.q[j.position_start()] for j in self.arm_joints])

        # RRT-Connect 로 충돌 없는 경로를 찾는다. 직선 보간은 두 끝점만
        # 안전할 뿐 사이를 보장하지 않는다 (물체가 테이블을 관통했다).
        path = self.plan_path(start, target)
        if path is None:
            self.console.moving(f"{what}  (경로 계획 실패 — 이동 중단)")
            print(f"[로봇] 경고: 충돌 없는 경로를 찾지 못했습니다. 이동을 건너뜁니다.")
            return False

        self.console.moving(f"{what}  ({duration_s:.0f}초, 경유점 {len(path)}개)")
        steps = max(int(duration_s / self.frame_dt_s), 2)
        waypoints = pp.ArmPathPlanner.resample(path, steps + 1)

        if self.driver is not None:
            # 실물. 경유점을 그대로 넘기고 **도착할 때까지 막힌다**.
            # 도착 전에 돌아오면 작업자가 움직이는 로봇에 접근할 수 있다.
            self.driver.follow(list(waypoints), duration_s)
            self.driver.stop()
            actual = np.asarray(self.driver.joint_positions(), dtype=float)
            for joint, value in zip(self.arm_joints, actual):
                self.q[joint.position_start()] = value
            self.publish()
            return True

        for step in range(steps + 1):
            fraction = step / steps
            # 사이클로이드 프로파일로 호길이를 따라간다 (시작·끝 속도 0)
            s = fraction - np.sin(2.0 * np.pi * fraction) / (2.0 * np.pi)
            index = min(int(round(s * steps)), steps)
            for joint, value in zip(self.arm_joints, waypoints[index]):
                self.q[joint.position_start()] = value
            self.publish()
            time.sleep(self.frame_dt_s)
        for joint, value in zip(self.arm_joints, path[-1]):
            self.q[joint.position_start()] = value
        self.publish()
        return True

    def plan_path(self, start, target):
        """로봇이 믿는 물체 형상으로 팔 경로를 계획한다.

        IK 도 같은 각도로 풀어야 시작·목표 자세가 이 형상에서 유효하다.
        실제 각도로 검사하면 IK 가 푼 목표가 여기서는 충돌로 판정되어
        경로를 못 찾는다.
        """
        key = (tuple(np.round(start, 6)), tuple(np.round(target, 6)),
               tuple(np.round(self.believed_q_deg, 3)))
        if key in self.plan_cache:
            return self.plan_cache[key]
        fixed = self.q.copy()
        for joint, value in zip(self.object_joints, self.believed_q_deg):
            fixed[joint.position_start()] = np.deg2rad(value)
        self.planner.set_fixed(fixed)
        path = self.planner.plan(start, target, max_iters=self.plan_iters)
        self.plan_cache[key] = path
        return path

    def inertial_share(self, duration_s, samples=25):
        """이동 중 관성 토크가 중력 토크에 비해 얼마나 되는가.

        힌지 유지토크를 몰라도 쓸 수 있는 기준이다. 준정적이라는 말은
        "관성 효과가 우리가 재려는 중력 효과에 비해 무시할 만하다"는 뜻이므로
        둘의 비율만 보면 된다.
        """
        from pydrake.multibody.tree import MultibodyForces

        start = np.array(self.setup["start_q"])
        entry = next(iter(self.setup["arm_solutions"].values()))
        goal = np.array(next(iter(entry.values())))
        delta = goal - start
        indices = [j.velocity_start() for j in self.object_joints]
        saved = self.q.copy()

        def payload_torque(vdot, v):
            forces = MultibodyForces(self.plant)
            self.plant.SetVelocities(self.plant_context, v)
            self.plant.CalcForceElementsContribution(self.plant_context, forces)
            tau = self.plant.CalcInverseDynamics(self.plant_context, vdot, forces)
            return np.abs(tau[indices])

        gravity_peak, total_peak = 0.0, 0.0
        zero = np.zeros(self.plant.num_velocities())
        for fraction in np.linspace(0.0, 1.0, samples):
            s = fraction - np.sin(2.0 * np.pi * fraction) / (2.0 * np.pi)
            ds = 1.0 - np.cos(2.0 * np.pi * fraction)
            dds = 2.0 * np.pi * np.sin(2.0 * np.pi * fraction)
            self.plant.SetPositions(self.plant_context, start + s * delta)
            gravity_peak = max(gravity_peak,
                               float(np.max(payload_torque(zero, zero))))
            v = delta * ds / duration_s
            vdot = delta * dds / duration_s ** 2
            total_peak = max(total_peak, float(np.max(payload_torque(vdot, v))))
        self.q = saved
        self.plant.SetPositions(self.plant_context, self.q)
        self.plant.SetVelocities(self.plant_context, zero)
        return 0.0 if gravity_peak < 1e-12 else abs(total_peak - gravity_peak) / gravity_peak

    # -- 조정 단계 -----------------------------------------------------
    def adjust(self, target_deg, round_index):
        self.console.stopped(f"시작 자세 (round {round_index})")
        names = [j.name for j in self.spec.joints]
        print(f"[로봇] 왼쪽 화면이 요청한 관절각:")
        for name, value in zip(names, target_deg):
            print(f"          {name} = {value:6.1f} deg")
        print(f"[로봇] 여기서 고정한 각도는 이번 라운드 내내 유지된다고"
              f" 가정합니다.")
        if not self.manual:
            self.adjust_automatically(target_deg)
            return
        print(f"[로봇] 허용 오차 +/- {ANGLE_TOL_DEG} deg."
              f" 슬라이더로 맞춘 뒤 ①, ② 를 누르세요.")
        self.adjust_manually(target_deg, names)

    def adjust_automatically(self, target_deg):
        """시뮬레이션만 돌릴 때. 관절이 목표 각도로 스스로 돌아간다.

        실물에서는 사람이 손으로 돌리는 자리다. 눈으로 보이도록 보간해서
        움직이고, 다 돌린 뒤 잠시 멈춰 로봇이 정지 상태임을 보여준다.
        """
        start = self.object_q_deg.copy()
        target = np.asarray(target_deg, dtype=float)
        print(f"[로봇] 관절각 자동 조정 {np.round(start, 1)}"
              f" -> {np.round(target, 1)} deg")
        for s in np.linspace(0.0, 1.0, self.adjust_steps):
            self.set_object_deg((1.0 - s) * start + s * target)
            self.publish()
            time.sleep(self.adjust_dt)
        self.object_q_deg = target.copy()
        time.sleep(self.adjust_hold)
        print(f"[로봇] 조정 완료 — 손을 뗀 것으로 간주하고 이동합니다")

    def adjust_manually(self, target_deg, names):
        self.console.clear()
        self.console.clear_sliders()
        sliders = []
        for name, value, (lo, hi) in zip(names, self.object_q_deg,
                                         self.limits_deg):
            sliders.append(self.console.slider(f"{name} [deg]", lo, hi,
                                               float(value)))
        for name, value in zip(names, target_deg):
            self.console.button(f"목표: {name} = {value:.1f} deg")
        done = self.console.button("① 조정 완료 — 각도 확인")

        if self.console.auto:
            for name, value in zip(sliders, target_deg):
                self.ui.SetSliderValue(name, float(value))
        while True:
            current = np.array([self.ui.GetSliderValue(n) for n in sliders])
            self.set_object_deg(current)
            self.publish()
            if self.console.auto or self.ui.GetButtonClicks(done) > 0:
                error = np.abs(current - np.asarray(target_deg))
                if np.all(error <= ANGLE_TOL_DEG):
                    self.object_q_deg = current
                    print(f"[로봇] 각도 확인 {np.round(current, 1)} deg"
                          f" (오차 {np.round(error, 2)})")
                    break
                print(f"[로봇] 목표와 다릅니다 — 오차 {np.round(error, 1)} deg")
                self.console.clear()
                done = self.console.button("① 조정 완료 — 각도 확인")
            time.sleep(0.05)

        self.console.clear()
        self.console.clear_sliders()
        confirm = self.console.button("② 물체에서 손을 뗐습니다 — 이동 시작")
        print("[로봇] 손을 뗀 뒤 ② 를 누르기 전까지 로봇은 움직이지 않습니다.")
        self.console.wait_for(confirm)
        self.console.clear()

    # -- 본체 ----------------------------------------------------------
    def begin(self):
        """두 화면이 같이 시작하도록 시작 버튼을 기다린다.

        조정이 자동이어도 이 버튼만은 기다린다. 그래야 탭 두 개를 연 뒤
        왼쪽 탐색을 처음부터 볼 수 있다. --autostart 로 건너뛴다.
        """
        self.set_arm_from(self.setup["start_q"])
        self.set_object_deg(self.object_q_deg)
        self.publish()
        if self.autostart:
            print("[로봇] --autostart 라 바로 시작합니다.")
            return
        button = self.console.button("두 화면을 모두 연 뒤 눌러 시작")
        print("[로봇] 왼쪽·오른쪽 화면을 모두 연 뒤 오른쪽 화면의"
              " 시작 버튼을 누르세요.")
        start = self.ui.GetButtonClicks(button)
        while self.ui.GetButtonClicks(button) == start:
            time.sleep(0.05)
        self.console.clear()

    def execute(self, target):
        """왼쪽이 고른 자세 하나를 실행하고 측정값을 만든다.

        순서가 중요하다.
          1) 시작 자세로 이동 (로봇 정지)
          2) 작업자가 물체 관절을 조정
          3) **FoundationPose 가 각도를 측정**
          4) 그 측정 각도로 팔 자세를 새로 풀고 (IK)
          5) 그 물체 형상 그대로 경로를 계획해 (RRT) 이동
          6) 렌치 측정

        3) 이 4~5) 앞에 와야 한다. 작업자가 맞춘 각도는 목표와 다를 수
        있고, 물체 형상이 달라지면 도달 가능성과 충돌이 함께 달라진다.
        미리 계산해 둔 팔 자세는 '명령한 각도' 기준이라 쓸 수 없다.
        """
        deg = np.array(target["object_joint_deg"])
        self.move_to(self.setup["start_q"], "시작 자세")
        self.adjust(deg, target["round"])

        actual = self.object_q_deg.copy()          # 물체가 실제로 놓인 각도

        # --- 3) FoundationPose 로 각도 측정 ---
        measured = self.measure_angles(actual, deg)
        # 관절은 물리적 한계를 넘을 수 없다. 측정값이 한계를 벗어나면
        # 그건 측정 오차이므로 잘라낸다. 우리가 아는 사전 지식이다.
        lows = np.array([lo for lo, _ in self.limits_deg])
        highs = np.array([hi for _, hi in self.limits_deg])
        clipped = np.clip(measured, lows, highs)
        if not np.allclose(clipped, measured):
            print(f"[로봇] 측정값 {np.round(measured, 2)} 가 구동범위를 벗어나"
                  f" {np.round(clipped, 2)} 로 보정")
        measured = clipped
        # 경로 계획도 계획 쪽이 검사한 그 형상으로 해야 검사와 일치한다.
        # 측정 각도는 밀도 추정에만 쓴다.
        self.believed_q_deg = (deg.copy() if target.get("arm_solutions")
                               else measured.copy())
        print(f"[로봇] FoundationPose 각도 측정 {np.round(measured, 2)} deg"
              f"  (명령 {np.round(deg, 2)}, 실제 {np.round(actual, 2)})")

        # --- 4) 팔 자세는 **계획 쪽이 검사한 그것을 그대로** 쓴다 ---
        #
        # 다시 풀지 않는다. 중력 방향과 작업공간 제약은 센서 프레임만 보는데
        # 센서 프레임은 물체 관절각과 무관하기 때문이다 (팔을 고정한 채
        # theta 를 흔들어도 센서 프레임 변화가 정확히 0 임을 확인했다).
        # theta 에 의존하는 것은 충돌뿐이고, 그건 계획 쪽이 각도 오차 구간
        # 전체에 대해 이미 확인했다.
        #
        # 다시 풀면 다른 해가 나오거나 실패해서 검사와 실행이 어긋난다.
        # 실제로 -87.45 도에서 통과한 후보가 -86.90 도에서 실패해 라운드가
        # 통째로 날아갔다.
        if target.get("arm_solutions"):
            solutions = {tuple(g): np.array(arm)
                         for g, arm in zip(alg.G_DIRS, target["arm_solutions"])}
        else:
            solutions = self.solve_exploration_poses(measured)
        if solutions is None:
            print("[로봇] 이 각도에서 도달 가능한 탐색 자세를 찾지 못했습니다."
                  " 이 라운드를 건너뜁니다.")
            return dict(round=target["round"], aborted=True)

        # --- 5~6) 자세마다 이동하고 **그 자리에서** 렌치를 읽는다 ---
        # 한 자세로 다 돌고 나서 한꺼번에 읽으면 안 된다. 실물 센서는 지금
        # 손목에 걸린 것만 읽으므로, 중력 방향마다 그 자리에서 재야 한다.
        self.console.measuring()
        readings = []
        for g_hat in alg.G_DIRS:
            if not self.move_to(solutions[tuple(g_hat)],
                                f"탐색 자세 (중력 {np.round(g_hat, 0)})"):
                return dict(round=target["round"], aborted=True)
            time.sleep(self.settle_s)         # 흔들림이 가라앉기를 기다린다
            readings.append(self.read_one(g_hat, actual))
        wrench = np.concatenate(readings)
        return dict(round=target["round"],
                    object_joint_deg=[float(v) for v in actual],
                    object_joint_deg_measured=[float(v) for v in measured],
                    wrench=[float(v) for v in wrench],
                    aborted=False)

    # -- 각도 측정 ------------------------------------------------------
    def observability(self, index):
        """지금 이 자세에서 관절 index 의 각도가 화면에서 얼마나 잘 보이나."""
        return rs.observability_px_per_deg(self.plant, self.plant_context,
                                           self.spec, index, self.payload)

    def measure_angles(self, actual, commanded):
        """관절마다 **자기 자세로 가서** 각도를 읽는다.

        왜 자세를 나누나
        ----------------
        관절 축 방향은 관절마다 다르다. 3-link 는 joint1 이 z, joint2 가 -y 라
        한 자세로 둘 다 잘 볼 수 없다. 축이 시선과 직각이면 회전이 화면 밖
        (깊이 방향)으로 나가 카메라가 거의 못 본다. 파지 자세 하나에서 다
        읽으면 joint2 는 0.37 px/deg 밖에 안 나오는데, 전용 자세에서는
        1.51 px/deg 다 (4.1 배, study_startpose.py 로 잰 값).

        시뮬레이션의 오차 모형
        ----------------------
        자세 추정기의 잔차를 화소 단위로 보면, 각도 오차는

            sigma_angle = (화소 잔차) / (도당 화소 이동)

        이다. 화소 잔차는 모르므로, **파지 자세에서 --angle-floor-deg 가
        나오도록** 맞춰 둔다. 그러면 전용 자세로 옮겨 얻는 이득만 남는다.
        """
        n = len(self.spec.joints)
        if (self.pose_sensor is None and self.angle_floor_deg <= 0.0
                and self.angle_rel_error <= 0.0):
            return np.asarray(actual, float).copy()

        home = self.q.copy()
        base_obs = [self.observability(k) for k in range(n)]
        measured = np.empty(n)
        used = []
        # 자세를 찾을 때 쓰는 각도는 **명령한 각도** 다. 실제 각도는 아직
        # 모르는 값이고 (그걸 재려고 가는 것이다), 실물에서는 알 방법도 없다.
        # 명령값과 실제값의 차이는 몇 도 수준이라 관측성이 거의 안 변한다.
        poses = (self.viewing_poses(commanded) if self.view_poses
                 else [None] * n)

        # 옮겨서 **더 나빠지면 옮기지 않는다.** 파지 자세가 이미 그 관절을 잘
        # 보는 경우가 있다 (램프는 두 축이 거의 나란하고 파지 자세가 이미
        # 4.4 px/deg 라, 전용 자세로 가면 오히려 4.2 로 떨어졌다). 이동은
        # 시간이 들고 파지가 흔들릴 기회도 주므로, 이득이 확실할 때만 간다.
        worth = {}
        for k in range(n):
            pose = poses[k]
            if pose is None or pose["arm_q"] is None:
                continue
            group = pose.get("group")
            ratio = pose["observability"] / max(base_obs[k], 1e-9)
            worth[group] = min(worth.get(group, np.inf), ratio)
        skipped = {g for g, ratio in worth.items() if ratio < MIN_VIEW_GAIN}
        for group in sorted(skipped):
            names = "+".join(self.spec.joints[j].name for j in range(n)
                             if poses[j] is not None
                             and poses[j].get("group") == group)
            print(f"[로봇] {names}: 파지 자세가 이미 충분해 그대로 읽습니다"
                  f" (전용 자세 이득 {worth[group]:.2f}배)")

        here = None          # 지금 서 있는 측정 자세의 무리 번호
        for k in range(n):
            pose = poses[k]
            if pose is not None and pose.get("group") in skipped:
                pose = None
            moved = here is not None and pose is not None \
                and pose.get("group") == here
            if pose is not None and pose["arm_q"] is not None and not moved:
                # 축이 나란한 관절끼리는 한 자세로 묶여 있다. 같은 무리면
                # 이미 그 자리에 서 있으므로 다시 가지 않는다.
                names = "+".join(self.spec.joints[j].name for j in range(n)
                                 if poses[j] is not None
                                 and poses[j].get("group") == pose.get("group"))
                moved = self.move_to(pose["arm_q"], f"{names} 각도 측정 자세")
                here = pose.get("group") if moved else None
                if not moved:
                    print(f"[로봇] {names} 측정 자세로 가는 경로가 없어"
                          f" 파지 자세에서 읽습니다")
            obs = self.observability(k) if moved else base_obs[k]
            used.append(obs)
            if self.pose_sensor is not None:
                reading, _ = self.pose_sensor.object_joint_deg()
                measured[k] = np.atleast_1d(np.asarray(reading, float))[k]
            else:
                sigma = max(self.angle_rel_error * abs(actual[k]),
                            self.angle_floor_deg) * base_obs[k] / max(obs, 1e-9)
                measured[k] = actual[k] + self.rng.normal(0.0, sigma)

        if self.view_poses:
            gain = " ".join(
                f"{j.name} {b:.2f}->{u:.2f} px/deg"
                for j, b, u in zip(self.spec.joints, base_obs, used))
            print(f"[로봇] 각도 관측성 {gain}")
        if here is not None:              # 어딘가로 갔으면 돌아온다
            self.move_to(home, "각도 측정 끝 — 파지 자세로 복귀")
        return measured

    def viewing_poses(self, commanded_deg):
        """이번 라운드의 관절각에서, 관절마다 가장 잘 보이는 자세를 찾는다.

        관절각이 바뀌면 축 방향도 바뀌므로 (joint2 의 축은 joint1 위에 있다)
        라운드마다 다시 찾아야 한다. 가벼운 plant 로 점수를 먼저 매기므로
        0.2 초쯤 걸린다.
        """
        try:
            return rs.find_viewing_poses(
                self.setup["checker"],
                np.radians(np.asarray(commanded_deg, float)),
                scorer=self.setup.get("scorer"))
        except Exception as exc:                       # noqa: BLE001
            print(f"[로봇] 각도 측정 자세를 못 찾았습니다 ({exc}) —"
                  f" 파지 자세에서 읽습니다")
            return [None] * len(self.spec.joints)

    def read_one(self, g_hat, actual_deg):
        """지금 이 자세에서 렌치 6개를 읽는다.

        실물이면 센서에서 읽고 **타어링 값을 뺀다**. 안 빼면 그리퍼·마운트·
        손가락 무게가 그대로 섞여 들어와 밀도가 통째로 틀어진다.
        시뮬레이션이면 진리 plant 가 이 방향 하나만 계산한다.
        """
        if self.wrench_sensor is None:
            return alg.measure(np.deg2rad(actual_deg),
                               g_dirs=[np.asarray(g_hat)])
        if self.tare is None:
            raise RuntimeError(
                "타어링 표가 없습니다. 물체를 잡기 전에 hardware.run_tare() 로"
                " 중력 방향별 공회전 렌치를 재 두어야 합니다.")
        # 모의 센서는 지금 물체가 어떤 자세인지 알아야 값을 만들 수 있다.
        # 실물 센서는 알 필요가 없다 — 손목에 걸린 것을 그냥 읽으면 된다.
        # 그래서 인터페이스에는 안 넣고, 있으면 부르는 식으로 둔다.
        if hasattr(self.wrench_sensor, "set_pose"):
            self.wrench_sensor.set_pose(
                np.deg2rad(np.atleast_1d(actual_deg)), g_hat)
        raw = self.wrench_sensor.read_raw(self.samples_per_hold)
        return self.tare.apply(g_hat, raw)

    def solve_exploration_poses(self, angle_deg):
        """측정된 관절각에서 중력 3방향의 팔 자세를 새로 푼다.

        미리 계산해 둔 표는 격자점(명령 각도) 기준이라, 작업자가 맞춘
        실제 각도와 다르다. 물체 형상이 바뀌면 도달 가능성도 충돌도
        달라지므로 그 자리에서 다시 풀어야 한다.
        """
        checker = self.setup["checker"]
        checker._last_solution = None
        theta = np.deg2rad(np.atleast_1d(angle_deg))
        solutions = {}
        for g_hat in alg.G_DIRS:
            arm_q = checker.solve(theta, g_hat)
            if arm_q is None:
                # 직전 해를 초기 추측으로 쓰다가 막힌 것일 수 있다.
                # 기본 시드에서 다시 한 번 풀어 본 뒤에야 포기한다.
                checker._last_solution = None
                arm_q = checker.solve(theta, g_hat, warm_start=False)
            if arm_q is None:
                return None
            solutions[tuple(g_hat)] = arm_q
        return solutions

    def finish(self):
        self.move_to(self.setup["start_q"], "시작 자세로 복귀")
        self.console.stopped("측정 종료")
        self.console.button("완료")

# ---------------------------------------------------------------------------
def simulated_hardware():
    """장비 없이 배포 경로를 끝까지 돌려보기 위한 모의 구현.

    실물 코드가 지나가는 길(자세마다 읽기, 타어링 빼기)을 그대로 밟는다.
    타어링 값은 0 이다 — 시뮬레이션에는 그리퍼 무게가 렌치에 안 섞이므로.
    """
    import hardware as hw

    class _Wrench(hw.WrenchSensor):
        """진리 plant 로 지금 자세의 렌치를 만든다.

        set_pose 는 모의 구현에만 있다. 실물 센서는 자세를 몰라도 되므로
        WrenchSensor 인터페이스에 넣지 않았다.
        """

        def __init__(self):
            self.theta = None
            self.g_hat = np.array([0.0, 0.0, -1.0])

        def set_pose(self, theta_rad, g_hat):
            self.theta = np.atleast_1d(np.asarray(theta_rad, dtype=float))
            self.g_hat = np.asarray(g_hat, dtype=float)

        def read_raw(self, n_samples):
            if self.theta is None:
                raise RuntimeError("set_pose 가 먼저 불려야 합니다")
            return alg.measure(self.theta, g_dirs=[self.g_hat])

    tare = hw.TareTable()
    for g_hat in alg.G_DIRS:
        tare.record(g_hat, np.zeros(6))
    return None, _Wrench(), None, tare


def connect_hardware(args):
    """실물 장비를 붙인다. 아직 구현이 없으면 무엇을 채워야 하는지 알려준다.

    반환: (RobotDriver, WrenchSensor, PoseSensor, TareTable)

    타어링은 **물체를 잡기 전에** 끝나 있어야 한다. 모형의 y 는 물체만
    만드는 렌치인데 센서는 그리퍼·마운트·손가락까지 다 읽기 때문이다.
    """
    import hardware as hw

    raise NotImplementedError(
        "\n실물 배포에 필요한 것 (hardware.py 참고):\n"
        "  1) RB5 드라이버        -> hw.Rb5Driver\n"
        "     follow() 는 도착할 때까지 막혀야 합니다.\n"
        "  2) AFT200 렌치 센서    -> hw.Aft200Sensor\n"
        "  3) FoundationPose      -> hw.FoundationPoseSensor\n"
        "  4) 타어링              -> hw.run_tare(...) 로 중력 3방향 공회전 렌치\n"
        "     물체를 잡기 전에 재야 합니다.\n"
        "이 함수 안에서 넷을 만들어 돌려주면 나머지 파이프라인은 그대로 돕니다.\n"
        "로봇 쪽을 다른 PC/ROS1 에서 돌리려면 --bus tcp 를 쓰세요.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("sim", "deploy"), default="sim",
                        help="sim=화면 3개(계획·로봇·UI, Drake 안에서 이동), "
                             "deploy=화면 2개(계획·UI, 실물 로봇이 이동)")
    parser.add_argument("--object",
                        choices=tuple(obj.OBJECTS) + ("desklamp",),
                        default="3link")
    parser.add_argument("--joint-range-deg", type=float, nargs="+", default=None)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument("--target", type=float, default=0.01,
                        help="정지 조건: 밀도의 95%% 상대 반폭")
    parser.add_argument("--angle-error", type=float,
                        default=aa.DEFAULT_ANGLE_REL_ERROR,
                        help="FoundationPose 각도 상대오차 (기본 0.05). "
                             "0 으로 두면 고정 오차 모형이 된다")
    parser.add_argument("--angle-floor-deg", type=float,
                        default=aa.DEFAULT_ANGLE_FLOOR_DEG,
                        help="각도 오차의 하한 [deg]. sigma = max(상대오차*|각도|,"
                             " 이 값). --angle-error 0 --angle-floor-deg 2 면"
                             " 각도와 무관한 고정 2도 오차가 된다")
    parser.add_argument("--prior", choices=("weight", "mesh"), default="weight",
                        help="초기값: weight=저울로 총무게만 잼, mesh=메시만 앎")
    parser.add_argument("--urdf-out", default=None)
    parser.add_argument("--plan-iters", type=int, default=20000,
                        help="RRT-Connect 최대 반복")
    parser.add_argument("--pace", type=float, default=0.08)
    parser.add_argument("--hinge-torque", type=float, default=None,
                        help="힌지 유지토크 [N·m]. 주지 않으면 관절이"
                             " 절대 움직이지 않는다고 가정한다")
    parser.add_argument("--safety", type=float, default=obj.DEFAULT_SAFETY)
    parser.add_argument("--auto-scale", action="store_true")
    parser.add_argument("--min-distance-mm", type=float, default=6.0)
    parser.add_argument("--samples-per-hold", type=int,
                        default=obj.DEFAULT_SAMPLES_PER_HOLD,
                        help="정지 자세에서 평균낼 F/T 샘플 수"
                             " (AFT200 은 1000 Hz)")
    parser.add_argument("--bias-fraction", type=float,
                        default=obj.DEFAULT_BIAS_FRACTION,
                        help="평균으로 줄지 않는 센서 바이어스 비율")
    parser.add_argument("--auto-adjust", action="store_true",
                        help="관절각을 사람 대신 자동으로 맞춘다."
                             " 기본은 수동 조작(슬라이더 + ①② 안전 확인)")
    parser.add_argument("--autostart", action="store_true",
                        help="시작 버튼도 기다리지 않는다 (검사용)")
    parser.add_argument("--move-duration", type=float,
                        default=DEFAULT_MOVE_DURATION_S,
                        help="이동 하나에 쓰는 시간 [s]. 클수록 준정적")
    # ---- 검토 지점 ①③④⑤ 를 여기서 갈아끼운다 (study_*.py 로 근거 확인) ----
    parser.add_argument("--select", choices=("continuous", "grid"),
                        default="continuous",
                        help="후보 자세: 연속 최적화 / 격자 top-1")
    parser.add_argument("--criterion", choices=dc.CRITERIA, default="D",
                        help="선택 기준: D=부피, A=분산합, E=최악방향")
    parser.add_argument("--estimator", choices=("tls", "wls"), default="tls",
                        help="추정기: tls=총최소제곱(각도 보정량도 같이 품),"
                             " wls=가중최소제곱(현행)")
    parser.add_argument("--stop-rule",
                        choices=("variance", "residual", "bias"),
                        default="residual",
                        help="정지 조건: variance=사후분산만,"
                             " residual=+잔차팽창, bias=+치우침몫")
    parser.add_argument("--systematic", type=float, default=0.3,
                        help="stop-rule=bias 일 때 가정하는 계통 각도오차 몫")
    parser.add_argument("--search-attempts", type=int, default=12,
                        help="연속 탐색에서 막힌 자리를 배워가며 다시 찾는 횟수")
    parser.add_argument("--block-radius-deg", type=float, default=15.0,
                        help="탈락한 후보 주변을 목적함수에서 눌러 두는 반경")
    parser.add_argument("--grasp", default="centroid",
                        choices=("centroid", "base_frame"),
                        help="desklamp 전용: 잡는 부위의 어디를 잡는가."
                             " centroid=그 부위 도심(기본)")
    parser.add_argument("--gripper", default="robotiq2f85",
                        choices=("pgc140", "robotiq2f85"),
                        help="그리퍼. pgc140=개구 53 mm, robotiq2f85=개구 78 mm."
                             " 램프처럼 단면이 굵은 물체는 robotiq2f85 가 필요")
    parser.add_argument("--grasp-part", default="link_3",
                        choices=("link_1", "link_2", "link_3"),
                        help="desklamp 전용: 어느 부위를 잡는가."
                             " link_3=연결부(Arm), link_2=베이스, link_1=Head")
    parser.add_argument("--seed", type=int, default=0,
                        help="측정 잡음·경로 계획의 난수 씨앗")
    parser.add_argument("--no-view-poses", action="store_true",
                        help="관절별 각도 측정 자세를 쓰지 않고 파지 자세에서"
                             " 모든 각도를 읽는다 (예전 방식)")
    parser.add_argument("--separate-ui", action="store_true",
                        help="작업자 UI 를 로봇 화면에서 떼어 따로 띄운다"
                             " (시뮬레이션 모드에서 화면 3개)")
    parser.add_argument("--probe-side", type=int, default=5,
                        help="각도 오차 구간을 훑는 격자 칸 수 (관절당). "
                             "충돌 질의라 키워도 비용이 거의 안 는다")
    parser.add_argument("--bus", choices=("local", "tcp"), default="local",
                        help="로봇 쪽과의 통신. tcp 면 로봇 쪽이 다른"
                             " 프로세스/PC 에서 붙는다")
    parser.add_argument("--bus-port", type=int, default=5555)
    parser.add_argument("--hardware", choices=("real", "sim"), default="real",
                        help="deploy 모드에서 쓸 장비. sim 이면 배선만 확인한다")
    parser.add_argument("--settle-s", type=float, default=0.3,
                        help="자세 도착 후 렌치를 읽기 전 대기 [s]")
    args = parser.parse_args()

    if args.object == "desklamp":
        import desk_lamp as lamp
        spec = lamp.build_spec(grasp_at=args.grasp,
                               grasp_part=args.grasp_part)
    else:
        spec = obj.OBJECTS[args.object]
    limits = rs.parse_joint_range(spec, args.joint_range_deg)
    if args.hinge_torque is None:
        hinge, scale = None, 1.0
    else:
        hinge = obj.Hinge(obj.HINGES["mg_plastic"].label, args.hinge_torque,
                          obj.HINGES["mg_plastic"].note)
        scale = (min(1.0, obj.max_feasible_density_scale(spec, hinge,
                                                         args.safety))
                 if args.auto_scale else 1.0)

    print(f"{spec.label}")
    for joint, (lo, hi) in zip(spec.joints, limits):
        print(f"  {joint.name} 구동범위 "
              f"{np.degrees(lo):.0f} ~ {np.degrees(hi):.0f} deg")
    noise = obj.set_measurement_averaging(args.samples_per_hold,
                                          args.bias_fraction)
    print(f"  F/T 정지 평균 {noise['n_samples']} 샘플"
          f" (바이어스 비율 {noise['bias_fraction']:.0%})"
          f" -> 유효 sigma_F {noise['sigma_f_n']*1000:.2f} mN,"
          f" sigma_T {noise['sigma_t_nm']*1000:.3f} mN·m")
    setup = prepare(spec, hinge, limits, args.safety, args.steps,
                    args.min_distance_mm * rs.MM, scale, prior=args.prior,
                    gripper=args.gripper, view_poses=not args.no_view_poses)
    print(f"  사용 가능한 자세 {len(setup['feasible'])}/{setup['n_grid']}")
    print(f"  시작 자세 제시 위치 {np.round(setup['presentation'], 3)} m")
    # 각도 측정 자세는 카메라가 어디서 보느냐로 정해진다. 어느 값을 쓰고
    # 있는지 반드시 사람이 보게 한다 — 명목값으로 실물을 돌리면 애써 고른
    # 자세가 실제로는 최적이 아니다.
    print(f"  카메라 {rs.CAMERA_ID}: {rs.CAMERA['source']}"
          f"  위치 {np.round(rs.camera_pose(rs.CAMERA).translation(), 3)} m")
    if args.hardware == "real" and "캘리브레이션" not in rs.CAMERA["source"]:
        print("  [주의] 실물인데 카메라 캘리브레이션 파일이 없습니다."
              " calibration/README.md 를 보세요.")
    report_viewing_poses(spec, setup)

    # ---- 화면 구성 -------------------------------------------------
    #   sim    : [1] 계획   [2] 로봇 + 작업자 UI        = 2개
    #   deploy : [1] 계획   [2] 작업자 UI               = 2개
    #
    # 작업자 UI(신호등·슬라이더·버튼)는 로봇 화면 위에 같이 띄운다. 작업자가
    # 로봇을 보면서 조작하는 것이 자연스럽고, 탭을 하나 덜 열어도 된다.
    # 실물 배포에서는 로봇 화면이 없으므로 UI 만 따로 띄운다.
    # --separate-ui 로 셋으로 나눌 수 있다.
    #
    # deploy 에서도 IK 와 경로 계획은 그대로 돈다. 실물에 보낼 경유점을
    # 만들어야 하기 때문이다. 그리지만 않을 뿐이다.
    planner_meshcat = StartMeshcat()
    robot_meshcat = StartMeshcat() if args.mode == "sim" else None
    if robot_meshcat is not None and not args.separate_ui:
        console_meshcat = robot_meshcat          # 로봇 화면에 UI 를 얹는다
    else:
        console_meshcat = StartMeshcat()
    print()
    print(f"  [1] 계획·탐색 화면   {planner_meshcat.web_url()}")
    if robot_meshcat is not None:
        label = ("로봇 화면" if console_meshcat is not robot_meshcat
                 else "로봇 + 작업자 UI")
        print(f"  [2] {label:<14} {robot_meshcat.web_url()}")
    if console_meshcat is not robot_meshcat:
        index = 3 if robot_meshcat is not None else 2
        print(f"  [{index}] 작업자 UI 화면   {console_meshcat.web_url()}")
    print(f"  위 주소들을 브라우저 탭으로 나란히 열어 주세요."
          f"  (모드 {args.mode})\n")

    planner = PlannerScreen(spec, setup, planner_meshcat, args.pace,
                            args.target, args.angle_error,
                            args.angle_floor_deg,
                            select_mode=args.select, criterion=args.criterion,
                            estimator=args.estimator, stop_rule=args.stop_rule,
                            systematic=args.systematic,
                            search_attempts=args.search_attempts,
                            block_radius_deg=args.block_radius_deg,
                            probe_side=args.probe_side)
    print(f"  설정: 후보={args.select}  기준={args.criterion}-최적"
          f"  추정기={args.estimator.upper()}  정지={args.stop_rule}")
    if args.bus == "tcp":
        print("  로봇 쪽은 robot_node.py 가 맡습니다."
              " 이 프로세스는 계획만 합니다.")
        robot = None
    driver = wrench_sensor = pose_sensor = tare = None
    if args.mode == "deploy" and args.bus != "tcp":
        if args.hardware == "real":
            print("  실물 배포 모드 — hardware.py 의 드라이버를 연결합니다.")
            driver, wrench_sensor, pose_sensor, tare = connect_hardware(args)
        else:
            print("  실물 배포 모드 (모의 장비) — 배선만 확인합니다.")
            driver, wrench_sensor, pose_sensor, tare = simulated_hardware()

    robot = RobotScreen(spec, setup, robot_meshcat,
                        console_meshcat=console_meshcat,
                        driver=driver, wrench_sensor=wrench_sensor,
                        pose_sensor=pose_sensor, tare=tare,
                        manual=not args.auto_adjust,
                        autostart=args.autostart,
                        move_duration_s=args.move_duration,
                        angle_rel_error=args.angle_error,
                        angle_floor_deg=args.angle_floor_deg,
                        min_distance_m=args.min_distance_mm * rs.MM,
                        plan_iters=args.plan_iters,
                        seed=args.seed,
                        settle_s=args.settle_s)
    share = robot.inertial_share(args.move_duration)
    print(f"  준정적 이동: 한 번에 {args.move_duration:.0f}초"
          f"  -> 관성 토크가 중력 토크의 {100*share:.3f}%"
          f"  ({'충분히 준정적' if share < 0.01 else '더 느리게 권장'})")
    bus = LocalBus() if args.bus == "local" else TcpBus.serve(args.bus_port)
    remote = args.bus == "tcp"      # 로봇 쪽은 robot_node.py 가 맡는다

    # 두 화면을 번갈아 돌린다. 스레드를 쓰지 않으므로 Meshcat 발행이
    # 항상 메인 스레드에서 일어나고, 왼쪽이 훑는 동안 오른쪽은 멈춰 있다.
    # 버스를 거치므로 나중에 오른쪽을 다른 프로세스로 떼어내도 그대로다.
    if not remote:
        robot.begin()
    for index in range(1, args.max_rounds + 1):
        bus.send_target(planner.select(index))
        if not remote:
            # 같은 프로세스: 여기서 바로 실행한다.
            bus.send_measurement(robot.execute(bus.recv_target()))
        reply = bus.recv_measurement()
        if reply.get("aborted"):
            print("[왼쪽] 이 라운드는 실행되지 못했습니다. 다음으로 넘어갑니다.")
            continue
        planner.update(reply)
        if planner.converged():
            print(f"\n[왼쪽] 목표 불확실성 도달 — {index} 라운드에서 정지")
            break
    else:
        print(f"\n[왼쪽] {args.max_rounds} 라운드까지 돌았으나 목표 미달"
              f" (현재 {100*planner.half_width():.2f}%)")
    if remote:
        bus.send_target(dict(finished=True))
    else:
        robot.finish()

    # ---- URDF 생성은 사용자 승인을 받는다 ----
    print("\n" + "=" * 66)
    print("탐색 결과")
    for row, gt, est, sd in zip(obj.body_table(spec), planner.rho_gt,
                                planner.rho_hat,
                                np.sqrt(np.diag(planner.Sigma))):
        mark = "" if row["kind"] == "part" else "  (힌지, 저울로 앎)"
        print(f"  {row['name']:<13} 밀도 {est:8.1f} +/-{1.96*sd:6.1f} kg/m^3"
              f"   [GT {gt:7.0f}  오차 {100*abs(est-gt)/gt:5.2f}%]{mark}")
    print("=" * 66)

    out = Path(args.urdf_out or f"estimated_{spec.key}.urdf")
    try:
        answer = input(f"\n이 값으로 sim-ready URDF 를 만들까요? [{out}] (y/N): ")
    except EOFError:            # 로그로 넘겨 돌릴 때 (키보드가 없다)
        answer = "y" if args.urdf_out else "n"
        print(f"\n입력이 없어 자동으로 '{answer}' 로 답합니다.")
    if answer.strip().lower() in ("y", "yes"):
        import export_urdf as eu
        eu.write_urdf(eu.build_urdf(spec, planner.rho_hat, planner.Sigma), out)
        print(f"  저장 -> {out}")
        for row in eu.verify_urdf(out, spec, planner.rho_hat):
            print(f"    {row['name']:<13} 되읽기 검증"
                  f"  질량오차 {row['mass_err']:.1e} kg"
                  f"  관성 상대오차 {row['inertia_err']:.1e}")
    else:
        print("  URDF 를 만들지 않았습니다.")
    print("\n종료합니다.")


if __name__ == "__main__":
    main()
