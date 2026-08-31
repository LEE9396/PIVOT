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
import sys
import time
from pathlib import Path

import numpy as np
from pydrake.geometry import Box, Mesh, Rgba, Sphere, StartMeshcat
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.systems.framework import DiagramBuilder
from pydrake.visualization import AddDefaultVisualization

import density_id_drake as alg
import density_id_objects as obj
import angle_aware as aa
import design_core as dc
import explore_view as ev
import gripper_hw as gh
import path_planning as pp
import robot_scene as rs
from operator_ui import ANGLE_TOL_DEG, Console
from pose_bus import LocalBus, TcpBus

# 준정적 이동. 관절이 흐르지 않도록 천천히 옮긴다. 힌지 유지토크를 모르므로
# 한계 기준을 세울 수 없어, 넉넉히 느린 시간을 기본으로 둔다.
DEFAULT_MOVE_DURATION_S = 8.0
# 관절 속도 상한 [deg/s]. 8초 이동에서 관성 토크가 중력 토크의 3.5% 였고,
# 그때 관절 속도가 12~16 deg/s 였다. 그 언저리로 묶어 둔다.
DEFAULT_MAX_JOINT_SPEED_DEG = 15.0
DEFAULT_FRAME_DT_S = 0.03

# 파지 안내 그림. 작업자 화면(콘솔) 위에 그리퍼 기준 좌표계로 그린다.
GRASP_GUIDE_PATH = "/grasp_guide"
GRASP_NUDGE_M = 0.002          # '조금 열기/닫기' 한 번의 크기
# 2F-85 패드를 표시용으로 근사한 크기 [m] (두께, 폭, 길이).
PAD_SIZE_M = (0.004, 0.022, 0.0375)


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
    if prior == "water":
        # 물체와 무관한 고정 기준에서 출발한다. 화면(창 4)에서 "물보다
        # 무겁다/가볍다" 로 바로 읽히고, 물체가 바뀌어도 출발점이 같다.
        mu, _ = obj.apply_water_prior(spec)
        print(f"  초기값: 모든 부위를 물 밀도 {mu[0]:.0f} kg/m^3 로 시작")
    elif prior == "weight":
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
                 block_radius_deg=15.0, probe_side=5, grasp_sigma_m=0.0):
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
        # 파지점이 얼마나 어긋날 수 있는지 [m]. 0 이면 파지점을 정확히 안다고
        # 본다 (시뮬레이션). 실물에서는 지그를 대도 몇 mm 는 어긋나므로 켠다.
        self.grasp_sigma_m = float(grasp_sigma_m)
        self.grasp_hat = np.zeros(3)
        self.total_mass_kg = float(obj.assembled_mass_kg(spec, setup["rho_gt"]))
        self.rho_gt = setup["rho_gt"]

        builder = DiagramBuilder()
        plant, bodies = ev.build_floating(spec, self.rho_gt, builder)
        AddDefaultVisualization(builder, meshcat)
        self.plant = plant
        self.diagram = builder.Build()
        self.context = self.diagram.CreateDefaultContext()
        self.player = ev.Player(spec, meshcat, plant, bodies,
                                self.context, self.diagram)

        self.draw_grasp_marker()
        self.Sigma = alg.SIGMA0.copy()
        self.rho_hat = alg.MU0.copy()
        # 탐색이 끝난 뒤 "얼마나 좋아졌나" 를 그리려면 출발점을 남겨야 한다.
        # MU0 는 물체를 바꿔 끼울 때 갈아엎히는 전역값이라 여기서 복사해 둔다.
        self.rho_prior = alg.MU0.copy()
        self.blocks = []
        self.rounds = []          # (measured_theta, y) — TLS 가 쓴다
        self.inflate = 1.0
        self.bias_cov = None

    # ------------------------------------------------------------------
    def show(self, theta, g_hat=(0.0, 0.0, -1.0)):
        self.player.show(theta, g_hat)

    def draw_grasp_marker(self):
        """계획 화면(창 1)에도 파지점을 찍는다.

        작업자는 이 화면에서 '어느 각도로 맞출까' 를 보고, 카메라 화면(창 2)
        에서 '어디를 잡을까' 를 본다. 두 화면이 같은 점을 가리키는지 눈으로
        맞춰 볼 수 있어야 해서 여기에도 같은 표시를 둔다.

        파지점은 **센서 프레임 원점**이다. 이 화면의 물체는 base 부위의 몸체
        프레임이 원점이므로, 거기서 base_bbox_center_in_sensor_mm 만큼
        되돌리면 파지점이 된다 (robot_scene._add_object 의 obj_sensor 와 같다).
        """
        origin = -np.array(self.spec.base_bbox_center_in_sensor_mm) * rs.MM
        jaw, _ = rs.grasp_axes(self.spec)
        jaw = np.asarray(jaw, dtype=float)
        jaw = jaw / np.linalg.norm(jaw)
        half = 0.5 * rs.jaw_dimension_m(self.spec) + 0.5 * PAD_SIZE_M[0]

        self.meshcat.SetObject(f"{GRASP_GUIDE_PATH}/point", Sphere(0.006),
                               Rgba(0.95, 0.10, 0.08, 1.0))
        self.meshcat.SetTransform(f"{GRASP_GUIDE_PATH}/point",
                                  RigidTransform(origin))
        for sign, name in ((1.0, "pad_a"), (-1.0, "pad_b")):
            node = f"{GRASP_GUIDE_PATH}/{name}"
            self.meshcat.SetObject(node, Box(*PAD_SIZE_M),
                                   Rgba(0.16, 0.17, 0.20, 0.85))
            self.meshcat.SetTransform(
                node, RigidTransform(origin + sign * half * jaw))
        print(f"  [1] 파지점 표시: 부위 {self.spec.parts[0].name},"
              f" 단면 {1000*rs.jaw_dimension_m(self.spec):.1f} mm,"
              f" 개구 {1000*self.setup['finger']:.1f} mm")

    def half_width(self, per_part=False):
        """정지 조건. GT 는 쓰지 않는다.

        per_part=True 면 부위별 상대 반폭 배열을 그대로 돌려준다 (창 4 가
        부위마다 막대를 그리는 데 쓴다). 기본값은 예전대로 스칼라 하나다.

        variance : 사후 공분산만 (현행)
        residual : + 잔차가 모형보다 크면 그만큼 부풀린다
        bias     : + 계통 각도오차가 답을 얼마나 미는지 재적합으로 재서 더한다
        """
        inflate = 1.0 if self.stop_rule == "variance" else self.inflate
        bias = self.bias_cov if self.stop_rule == "bias" else None
        # 힌지처럼 이미 아는 양은 정지 판단에서 뺀다 (design_core.stopping_width)
        half = dc.half_width(self.Sigma, self.rho_hat, Cov_bias=bias,
                             inflate=inflate)
        if per_part:
            return np.asarray(half)[:len(self.spec.parts)]
        return dc.stopping_width(half, len(self.spec.parts))

    def converged(self):
        return self.half_width() <= self.target

    # ------------------------------------------------------------------
    def score(self, theta):
        return dc.utility(theta, self.rho_hat, self.Sigma, dc.CANONICAL_TRIAD,
                          self.criterion, self.angle_rel_error,
                          self.angle_floor_deg)

    def angle_probes(self, theta):
        """실제 물체 각도가 놓일 수 있는 범위를 격자로 훑는 점들.

        모서리만 보면 안 된다. IK 도달 영역은 매끄러운 덩어리가 아니라
        구멍이 점점이 박힌 지형이라, 모서리는 우연히 구멍을 피하고 실제
        측정값이 구멍에 빠진다. 실제로 -87.45 도에서 통과한 후보가
        -86.90 도에서 실패해 라운드가 통째로 날아갔다.

        범위는 **두 몫의 합**이다. 예전에는 앞의 것만 봤다.

          1.96 sigma      FoundationPose 가 각도를 잘못 읽는 몫
          ANGLE_TOL_DEG   작업자 조정을 통과시키는 창

        둘째 몫이 왜 들어가야 하나. 조정 통과 판정은 **읽은 각도** 로 하고
        (adjust_manually), 그 판정은 목표에서 ANGLE_TOL_DEG 까지 벗어난
        각도를 통과시킨다. 즉 로봇은 목표가 아니라 '통과된 각도' 에서
        움직인다. 그 각도까지 충돌 검사를 해 두지 않으면, 검사한 적 없는
        형상으로 로봇이 이동하게 된다.

        시뮬레이션에서는 이 몫이 0 이라 안 보인다 — adjust_automatically 가
        목표에 정확히 맞춰 주기 때문이다. 실물에서만 드러난다.
        """
        bounds = [j.limits_rad for j in self.spec.joints]
        lows = np.array([lo for lo, _ in bounds])
        highs = np.array([hi for _, hi in bounds])
        theta = np.atleast_1d(theta)
        sigma = np.sqrt(np.diag(aa.angle_covariance(
            theta, self.angle_rel_error, self.angle_floor_deg)))
        reach = 1.96 * sigma + np.deg2rad(ANGLE_TOL_DEG)
        axes = [np.linspace(t - r, t + r, self.probe_side)
                for t, r in zip(theta, reach)]
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

        grasp_init = None
        if self.grasp_sigma_m > 0.0:
            # 파지점 어긋남을 미지수로 함께 푼다. 먼저 선형으로 풀어 두고
            # (안정적이다) 그 값을 TLS 의 시작점으로 넘긴다. 찬 시작으로
            # 넣으면 각도 보정과 파지점이 서로를 흉내내며 국소해에 빠진다
            # (어긋남이 없는데도 9 mm 를 지어내며 오차 110% 가 났다).
            rho_lin, grasp_init, Sigma_full = dc.grasp_map(
                self.blocks, alg.MU0, alg.SIGMA0, alg.RHO_BOUNDS,
                dc.CANONICAL_TRIAD, self.total_mass_kg,
                grasp_sigma_m=self.grasp_sigma_m)
            self.grasp_hat = grasp_init
            self.Sigma = Sigma_full[:len(alg.MU0), :len(alg.MU0)]

        rho_wls = (rho_lin if self.grasp_sigma_m > 0.0
                   else aa.constrained_map(self.blocks, alg.MU0, alg.SIGMA0,
                                           alg.RHO_BOUNDS))
        if self.estimator == "tls":
            # 각도 보정량을 밀도와 함께 푼다. 계수행렬이 틀렸다는 사실이
            # 모형 안에 있으므로 오차변수 치우침이 남지 않는다.
            self.rho_hat, tls_info = dc.tls_map(
                self.rounds, alg.MU0, alg.SIGMA0, alg.RHO_BOUNDS,
                dc.CANONICAL_TRIAD, rho_init=rho_wls,
                rel_error=self.angle_rel_error,
                grasp_sigma_m=self.grasp_sigma_m,
                total_mass_kg=self.total_mass_kg,
                grasp_init=grasp_init)
            if self.grasp_sigma_m > 0.0 and len(tls_info.get("grasp", ())) == 3:
                self.grasp_hat = np.asarray(tls_info["grasp"], dtype=float)
        else:
            self.rho_hat, tls_info = rho_wls, None

        # 미지수로 함께 푼 것은 **무엇이든** 잔차를 잴 때 반영해야 한다.
        # 파지점 몫은 예측에 더하고, 각도 보정량은 회귀행렬에 넣는다. 후자를
        # 빠뜨리면 TLS 가 찾아낸 보정량이 통째로 잔차로 잡혀 팽창이 80배까지
        # 뛰고, 추정이 이미 목표를 넘었는데도 정지 조건이 만족되지 않는다.
        grasp_offset = (dc.grasp_columns(dc.CANONICAL_TRIAD,
                                         self.total_mass_kg) @ self.grasp_hat
                        if self.grasp_sigma_m > 0.0 else None)
        self.inflate = dc.residual_scale(
            self.blocks, self.rounds, self.rho_hat, dc.CANONICAL_TRIAD,
            tls_info if self.estimator == "tls" else None,
            stop_rule=self.stop_rule, grasp_offset=grasp_offset,
            n_grasp=3 if self.grasp_sigma_m > 0.0 else 0)
        if self.stop_rule == "bias":
            self.bias_cov = dc.bias_by_refit(
                self.rounds, alg.MU0, alg.SIGMA0, alg.RHO_BOUNDS,
                dc.CANONICAL_TRIAD, self.rho_hat, self.systematic,
                self.estimator, self.angle_rel_error)

        half = dc.half_width(
            self.Sigma, self.rho_hat,
            Cov_bias=self.bias_cov if self.stop_rule == "bias" else None,
            inflate=1.0 if self.stop_rule == "variance" else self.inflate)
        if self.grasp_sigma_m > 0.0:
            print(f"[왼쪽] 파지점 어긋남 추정"
                  f" {np.round(1000 * self.grasp_hat, 2)} mm"
                  f"  (크기 {1000 * np.linalg.norm(self.grasp_hat):.2f} mm)")
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
                 settle_s=0.3, grasp_error_m=None,
                 max_joint_speed_deg=DEFAULT_MAX_JOINT_SPEED_DEG,
                 check_motion=True, gripper=None):
        self.spec = spec
        self.setup = setup
        self.meshcat = meshcat                      # None 이면 로봇 화면 없음
        # 실물 그리퍼. None 이면 사람이 손으로 여닫는다 (예전 동작).
        self.gripper = gripper
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
        # 관절 속도 상한 [rad/s]. 경로가 길어도 이보다 빨리 돌지 않는다.
        self.max_joint_speed_rad = np.deg2rad(max_joint_speed_deg)
        # 한 프레임에 이만큼 넘게 튀면 알린다. 8초 이동에서 관절이 최대
        # 15 deg/s 로 도니 30 ms 프레임에서는 0.5 deg 가 정상이다.
        self.jump_warn_deg = 5.0
        # 이동 중 실제로 부딪히는지 프레임마다 확인할지. 질의가 프레임마다
        # 들어가 조금 느려지지만, 계획과 실행이 어긋나는지 알려면 필요하다.
        self.check_motion = check_motion
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
        # 파지 안내 화면을 그리는 데 필요한 것들. 씬이 물체를 그리퍼에
        # **용접해** 붙였으므로, 그리퍼 기준 물체 자세가 곧 "이렇게 물려야
        # 한다" 는 답이다. 사람에게 보여줄 그림을 여기서 그대로 뽑아 쓴다.
        self.gripper_model = scene["gripper"]
        self.gripper_spec = scene["gripper_spec"]
        self.part_bodies = scene["parts"]
        self.sensor_frame = scene["sensor_frame"]
        self.jaw_opening_m = scene["jaw_opening_m"]
        self.tcp_z_m = scene["tcp_z_m"]
        # 실제로 물었을 때의 개구량. 파지 단계에서 채워지고 보고에 쓰인다.
        self.grasp_report = None
        # 파지력이 충분한지 가늠하는 데 쓴다 (실물에서는 저울로 잰 값).
        self.object_mass_kg = float(obj.assembled_mass_kg(spec,
                                                          setup["rho_gt"]))
        self.view_poses = bool(setup.get("view_poses", False))
        # 시뮬레이션에서 '파지점이 어긋난 척' 하기 위한 값. 실물에서는 None
        # (실제로 어긋나 있으므로 흉내낼 필요가 없다).
        self.grasp_error_m = (None if grasp_error_m is None
                              else np.asarray(grasp_error_m, dtype=float))
        if meshcat is not None:
            AddDefaultVisualization(builder, meshcat)
        self.diagram = builder.Build()
        self.context = self.diagram.CreateDefaultContext()
        self.plant_context = self.plant.GetMyMutableContextFromRoot(self.context)

        self.q = self.plant.GetPositions(self.plant_context).copy()
        self._last_published = None
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
        self.min_distance_m = min_distance_m
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
        # 순간이동 감시. 화면 프레임 사이에 관절이 크게 튀면 사람 눈에는
        # "순간이동" 으로 보이고, 실물이라면 위험한 명령이다. 계획된 경로를
        # 따라가는 동안에는 한 프레임에 몇 도씩만 움직여야 한다.
        if self._last_published is not None:
            jump = np.degrees(np.max(np.abs(
                [self.q[j.position_start()] - self._last_published[j.position_start()]
                 for j in self.arm_joints])))
            self._max_jump_deg = max(getattr(self, "_max_jump_deg", 0.0), jump)
            if jump > self.jump_warn_deg:
                print(f"[로봇] 경고: 화면이 {jump:.1f} deg 건너뜁니다"
                      f" (프레임 {getattr(self, '_frame_index', -1)},"
                      f" 한계 {self.jump_warn_deg:.0f} deg)"
                      f" — 순간이동처럼 보입니다")
        self._last_published = self.q.copy()
        self.plant.SetPositions(self.plant_context, self.q)
        if self.meshcat is not None:
            self.diagram.ForcedPublish(self.context)

    def clearance(self):
        """지금 자세에서 가장 가까운 두 형상 사이의 거리 [m]."""
        query = self.plant.get_geometry_query_input_port().Eval(self.plant_context)
        pairs = query.ComputeSignedDistancePairwiseClosestPoints(
            self.min_distance_m)
        if not pairs:
            return self.min_distance_m, None
        worst = min(pairs, key=lambda pair: pair.distance)
        inspector = query.inspector()
        names = tuple(
            self.plant.GetBodyFromFrameId(inspector.GetFrameId(gid)).name()
            for gid in (worst.id_A, worst.id_B))
        return worst.distance, names

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
        if self._last_published is not None:
            shown = np.array([self._last_published[j.position_start()]
                              for j in self.arm_joints])
            gap = np.degrees(np.abs(start - shown)).max()
            if gap > 1.0:
                print(f"[진단] 이동 시작점이 화면과 {gap:.1f} deg 다릅니다"
                      f" — 화면 {np.round(np.degrees(shown), 1)}"
                      f" / 시작 {np.round(np.degrees(start), 1)}")

        # RRT-Connect 로 충돌 없는 경로를 찾는다. 직선 보간은 두 끝점만
        # 안전할 뿐 사이를 보장하지 않는다 (물체가 테이블을 관통했다).
        path = self.plan_path(start, target)
        if path is None:
            self.console.moving(f"{what}  (경로 계획 실패 — 이동 중단)")
            print(f"[로봇] 경고: 충돌 없는 경로를 찾지 못했습니다. 이동을 건너뜁니다.")
            return False

        # 시간을 고정하면 경로가 길수록 빨라진다. 우회하는 경로가 잡히면
        # 같은 8초에 두 배를 도느라 관절 속도가 그만큼 뛴다. 준정적이라는
        # 가정은 '천천히'가 아니라 '관성 토크가 중력 토크에 비해 작게'라는
        # 뜻이고, 그건 속도로 정해진다. 그래서 **속도로 묶고** 필요하면
        # 시간을 늘린다 (줄이지는 않는다).
        length = float(np.sum(np.abs(np.diff(np.asarray(path), axis=0)).max(axis=1)))
        if self.max_joint_speed_rad > 0.0:
            # 사이클로이드 프로파일은 최고속도가 평균의 2배다.
            needed = 2.0 * length / self.max_joint_speed_rad
            if needed > duration_s:
                duration_s = float(needed)

        first = np.degrees(np.abs(np.asarray(path[0]) - start)).max()
        last = np.degrees(np.abs(np.asarray(path[-1]) - target)).max()
        if max(first, last) > 1.0:
            print(f"[진단] 경로가 어긋납니다: path[0]-start {first:.1f} deg,"
                  f" path[-1]-target {last:.1f} deg, 경유점 {len(path)}")
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

        worst, worst_pair = np.inf, None
        for step in range(steps + 1):
            self._frame_index = step
            fraction = step / steps
            # 사이클로이드 프로파일로 호길이를 따라간다 (시작·끝 속도 0)
            s = fraction - np.sin(2.0 * np.pi * fraction) / (2.0 * np.pi)
            index = min(int(round(s * steps)), steps)
            for joint, value in zip(self.arm_joints, waypoints[index]):
                self.q[joint.position_start()] = value
            self.publish()
            if self.check_motion:
                # 계획은 '믿는 형상' 으로 했지만 화면에 있는 것은 실제 형상이다.
                # 둘이 어긋나면 여기서 잡힌다.
                distance, pair = self.clearance()
                if distance < worst:
                    worst, worst_pair = distance, pair
            time.sleep(self.frame_dt_s)
        if self.check_motion:
            print(f"[로봇] {what}: {duration_s:.1f}초, 경유점 {len(path)}개,"
                  f" 프레임 최대 {getattr(self, '_max_jump_deg', 0.0):.2f} deg,"
                  f" 최소 간격 {1000*worst:.1f} mm")
            self._max_jump_deg = 0.0
        if self.check_motion and worst < 0.0:
            print(f"[로봇] 경고: 이동 중 {1000*abs(worst):.1f} mm 파고들었습니다"
                  f" {worst_pair}")
        elif self.check_motion and worst < self.min_distance_m:
            print(f"[로봇] 이동 중 최소 간격 {1000*worst:.1f} mm {worst_pair}"
                  f" (기준 {1000*self.min_distance_m:.0f} mm)")
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
        if self.pose_sensor is not None:
            # 실물: 각도는 사람이 아니라 FoundationPose 가 읽는다.
            self.adjust_by_pose(target_deg, names)
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
        # 모의 트래커에게도 '실물 관절이 여기로 갔다' 고 알려 준다. 안 그러면
        # 뒤이은 measure_angles 가 읽을 진실이 없다.
        rehearse = getattr(self.pose_sensor, "set_truth", None)
        if rehearse is not None:
            rehearse(self.object_q_deg)
        time.sleep(self.adjust_hold)
        print(f"[로봇] 조정 완료 — 손을 뗀 것으로 간주하고 이동합니다")

    def read_pose_deg(self, n_samples=1):
        """FoundationPose 가 읽은 지금 관절각 [deg]. 못 읽으면 None.

        판정에 쓸 때는 여러 번 읽어 **중앙값**을 쓴다. 한 샘플로 판정하면
        트래커 잡음 하나에 통과/불통과가 뒤집힌다.
        """
        readings = []
        for _ in range(max(1, int(n_samples))):
            try:
                angles, _ = self.pose_sensor.object_joint_deg()
            except Exception as exc:                   # noqa: BLE001
                self._pose_error = str(exc)
                return None
            readings.append(np.atleast_1d(np.asarray(angles, dtype=float)))
            if n_samples > 1:
                time.sleep(0.02)
        self._pose_error = None
        return np.median(np.vstack(readings), axis=0)

    def adjust_by_pose(self, target_deg, names):
        """실물용 조정 단계 — 각도를 **FoundationPose 로 읽는다.**

        슬라이더를 쓰면 안 되는 이유. 실물에서 슬라이더는 아무것도 안
        움직인다. 작업자가 손으로 관절을 돌린 뒤 "몇 도로 맞췄다" 를 **눈대중으로
        입력하는** 칸이 될 뿐이고, 그러면 통과 판정이 사람의 짐작 위에서
        일어난다. 정작 추정식에는 FoundationPose 값이 들어가므로, 통과한
        각도와 쓰이는 각도가 서로 다른 값이 된다.

        그래서 여기서는 트래커가 읽은 값으로 화면을 갱신하고, 그 값으로
        허용 오차를 판정한다. 사람이 하는 일은 물체를 돌리는 것과, 다 됐다고
        누르는 것뿐이다.
        """
        self.console.clear()
        self.console.clear_sliders()
        # 리허설(모의 트래커)에서는 슬라이더가 **실물 관절을 대신** 돌린다.
        # 실물에서는 이 슬라이더가 없다 — 사람이 물체를 직접 돌린다.
        rehearse = getattr(self.pose_sensor, "set_truth", None)
        sliders = []
        if rehearse is not None:
            rehearse(self.object_q_deg)
            for name, value, (lo, hi) in zip(names, self.object_q_deg,
                                             self.limits_deg):
                sliders.append(self.console.slider(
                    f"{name} [deg] (리허설: 실물 관절 대신)", lo, hi,
                    float(value)))
        for name, value in zip(names, target_deg):
            self.console.button(f"목표: {name} = {value:.1f} deg")
        done = self.console.button("① 조정 완료 — 각도 확인 (FoundationPose)")
        print(f"[로봇] 허용 오차 +/- {ANGLE_TOL_DEG} deg."
              f" **각도는 FoundationPose 가 읽습니다** — 물체를 손으로 돌리면"
              f" 화면의 물체가 따라 움직입니다. 다 맞으면 ① 을 누르세요.")
        if rehearse is not None:
            print("[로봇] (리허설) 슬라이더가 실물 관절 자리를 대신합니다."
                  " 실물에서는 이 슬라이더가 없습니다.")

        target = np.asarray(target_deg, dtype=float)
        self._pose_error = None
        last_print, last_error = 0.0, None
        while True:
            if rehearse is not None:
                rehearse([self.ui.GetSliderValue(n) for n in sliders])
            live = self.read_pose_deg()
            if live is None:
                # 트래커가 죽었는데 마지막 값으로 계속 그리면 사람은 잘
                # 맞춰지고 있다고 믿는다. 조용히 넘어가면 안 된다.
                if time.time() - last_print > 2.0:
                    print(f"[로봇] FoundationPose 를 못 읽습니다:"
                          f" {self._pose_error}")
                    last_print = time.time()
                time.sleep(0.2)
                continue
            self.set_object_deg(live)
            self.publish()
            error = np.abs(live - target)
            inside = bool(np.all(error <= ANGLE_TOL_DEG))
            # 신호등으로 '지금 범위 안인가' 를 보여준다. 터미널을 안 봐도
            # 물체를 돌리면서 알 수 있어야 한다.
            if inside != last_error:
                self.console.stopped(
                    f"각도 맞음 — ① 을 누르세요 {np.round(live, 1)} deg"
                    if inside else
                    f"각도를 맞추는 중 {np.round(live, 1)} deg"
                    f" (목표 {np.round(target, 1)})")
                last_error = inside
            if time.time() - last_print > 1.0:
                print(f"[로봇] FoundationPose {np.round(live, 1)} deg"
                      f"  목표 {np.round(target, 1)}"
                      f"  오차 {np.round(error, 1)}"
                      f"  {'범위 안' if inside else '아직'}")
                last_print = time.time()

            if self.ui.GetButtonClicks(done) > 0:
                # 판정만은 여러 샘플의 중앙값으로 한다.
                settled = self.read_pose_deg(n_samples=5)
                if settled is None:
                    print(f"[로봇] 판정할 각도를 못 읽었습니다:"
                          f" {self._pose_error}")
                else:
                    error = np.abs(settled - target)
                    if np.all(error <= ANGLE_TOL_DEG):
                        self.object_q_deg = settled
                        print(f"[로봇] 각도 확인 {np.round(settled, 1)} deg"
                              f" (오차 {np.round(error, 2)}) — FoundationPose")
                        break
                    print(f"[로봇] 목표와 다릅니다 — 오차"
                          f" {np.round(error, 1)} deg")
                self.console.clear()
                for name, value in zip(names, target_deg):
                    self.console.button(f"목표: {name} = {value:.1f} deg")
                done = self.console.button(
                    "① 조정 완료 — 각도 확인 (FoundationPose)")
                last_error = None       # 신호등을 다시 그리게 한다
            time.sleep(0.05)

        self.console.clear()
        self.console.clear_sliders()
        confirm = self.console.button("② 물체에서 손을 뗐습니다 — 이동 시작")
        print("[로봇] 손을 뗀 뒤 ② 를 누르기 전까지 로봇은 움직이지 않습니다.")
        self.console.wait_for(confirm)
        self.console.clear()

    def adjust_manually(self, target_deg, names):
        """시뮬레이션용 — 슬라이더가 물체를 **실제로** 돌린다.

        실물에서는 쓰이지 않는다 (adjust_by_pose 로 간다). 여기서 슬라이더는
        '작업자가 손으로 돌린 결과' 를 대신하는 것이지 입력칸이 아니다.
        """
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
    def sync_from_robot(self):
        """팔이 지금 실제로 어디 있는지 읽어 화면과 계획기의 출발점을 맞춘다.

        예전에는 시작 자세에 있다고 **가정** 했다. 시뮬레이션에서는 그게
        사실이지만 실물에서는 아니다. 전원을 켠 자리, 지난 세션이 끝난
        자리, 작업자가 티치펜던트로 옮겨 둔 자리 — 어디든 될 수 있다.

        가정이 틀리면 첫 이동이 조용히 위험해진다. move_to 는 '내가 아는
        자세' 를 출발점으로 경로를 계획하는데, 그게 start_q 로 박혀 있으면
        RRT 는 start_q -> start_q 를 풀어 경유점 하나를 내놓고, 드라이버는
        그 한 점으로 팔을 **계획에 없는 직선으로** 보낸다. 실제 자세가
        멀수록 크게 튄다.
        """
        if self.driver is None:
            self.set_arm_from(self.setup["start_q"])
            return
        actual = np.asarray(self.driver.joint_positions(), dtype=float)
        for joint, value in zip(self.arm_joints, actual):
            self.q[joint.position_start()] = value
        # 화면을 실제 자세로 '옮겨 놓는' 것이지 로봇이 움직인 게 아니다.
        # 기준선을 같이 옮겨 두지 않으면 순간이동 감시가 헛경고를 낸다.
        self._last_published = None
        start = np.array([self.setup["start_q"][j.position_start()]
                          for j in self.arm_joints])
        print(f"[로봇] 현재 팔 자세를 읽었습니다"
              f" {np.round(np.degrees(actual), 1)} deg"
              f"  (시작 자세와 최대 {np.degrees(np.abs(actual - start)).max():.1f} deg 차이)")

    # -- 파지 안내 그림 ------------------------------------------------
    def _draw_guide_part(self, path, part, X_GP, rgba=None):
        """부위 하나를 안내 화면에 그린다.

        우선순위(색 조각 -> 메시 하나 -> AABB 상자)와 메시 보정은
        density_id_objects.register_part_visual 과 같게 맞춘다. 안내 그림이
        실제 씬과 다른 모양이면 안내가 아니라 오해가 된다.
        """
        X_mesh = RigidTransform(np.array(part.mesh_offset_m))
        if str(part.visual_mesh or "").lower().endswith(".gltf"):
            X_mesh = RigidTransform(RotationMatrix.MakeXRotation(-np.pi / 2),
                                    np.array(part.mesh_offset_m))
        if part.visual_pieces and rgba is None:
            for index, (mesh_path, color) in enumerate(part.visual_pieces):
                node = f"{path}/piece_{index}"
                self.ui.SetObject(node, Mesh(str(mesh_path), 1.0), Rgba(*color))
                self.ui.SetTransform(node, X_GP @ X_mesh)
        elif part.visual_mesh:
            self.ui.SetObject(path, Mesh(str(part.visual_mesh), 1.0),
                              Rgba(*(rgba if rgba is not None else part.color)))
            self.ui.SetTransform(path, X_GP @ X_mesh)
        else:
            dims = tuple(d * rs.MM for d in part.bbox_mm)
            self.ui.SetObject(path, Box(*dims),
                              Rgba(*(rgba if rgba is not None else part.color)))
            self.ui.SetTransform(path, X_GP)

    def draw_grasp_guide(self, opening_m):
        """"물체가 죠 사이 어디에 앉아야 하는가" 를 그림으로 보여준다.

        실물 배포에서는 로봇 화면이 없어서(meshcat=None) 작업자가 볼 3D 가
        하나도 없었다. 그런데 파지 자세는 글로 설명하기 가장 어려운 것이다 —
        어느 쪽이 위인지, 죠가 어느 축을 무는지, 파지점이 부위의 어디인지.
        그래서 **그리퍼 기준 좌표계**를 원점으로 삼아, 씬이 용접해 둔 그대로의
        물체와 죠 두 장을 작업자 화면에 따로 그린다.
        """
        if self.ui is self.meshcat:
            return          # 로봇 화면과 같은 화면이면 실제 씬이 이미 보인다
        self.plant.SetPositions(self.plant_context, self.q)
        base = self.plant.GetFrameByName(self.gripper_spec.base_frame,
                                         self.gripper_model)
        X_GW = base.CalcPoseInWorld(self.plant_context).inverse()
        for part in self.spec.parts:
            X_GP = X_GW @ self.plant.EvalBodyPoseInWorld(
                self.plant_context, self.part_bodies[part.name])
            self._draw_guide_part(f"{GRASP_GUIDE_PATH}/{part.name}", part, X_GP)
        # 파지점 = 센서 프레임 원점. 모든 모멘트팔의 기준이라 눈에 띄게 찍는다.
        point = f"{GRASP_GUIDE_PATH}/point"
        self.ui.SetObject(point, Sphere(0.005), Rgba(0.95, 0.10, 0.08, 1.0))
        self.ui.SetTransform(point, X_GW @ self.sensor_frame.CalcPoseInWorld(
            self.plant_context))
        self.update_grasp_pads(opening_m)
        self.ui.SetCameraPose([0.22, -0.24, self.tcp_z_m + 0.12],
                              [0.0, 0.0, self.tcp_z_m])

    def update_grasp_pads(self, opening_m):
        """죠 두 장을 지금 개구량 자리에 다시 그린다.

        죠 축은 그리퍼 x (robot_scene.grasp_rotation), 패드 면의 높이는
        tcp_z 다. 그래서 이 두 상자 사이가 곧 실제 죠 사이다.
        """
        if self.ui is self.meshcat:
            return
        half = 0.5 * float(opening_m) + 0.5 * PAD_SIZE_M[0]
        for sign, name in ((1.0, "pad_a"), (-1.0, "pad_b")):
            node = f"{GRASP_GUIDE_PATH}/{name}"
            self.ui.SetObject(node, Box(*PAD_SIZE_M),
                              Rgba(0.16, 0.17, 0.20, 0.9))
            self.ui.SetTransform(node, RigidTransform(
                [sign * half, 0.0, self.tcp_z_m]))

    def clear_grasp_guide(self):
        if self.ui is not self.meshcat:
            self.ui.Delete(GRASP_GUIDE_PATH)

    # -- 파지 단계 -----------------------------------------------------
    def await_grasp(self):
        """작업자가 물체를 그리퍼에 물리는 단계. 이 동안 팔은 정지한다.

        왜 여기가 따로 있어야 하나. 시뮬레이션에서는 물체가 그리퍼에
        **용접돼** 있어 파지라는 단계가 아예 없다. 실물에서는 누군가
        물체를 물려야 하고, 그 사람이 로봇 작업영역 안에 들어간다.

        순서가 중요하다. 물체를 문 **뒤에** 시작 자세로 가야 한다.
          - 경로계획의 충돌 모형은 '물체를 든 팔' 이다. 빈 손으로 먼저
            움직이면 계획과 실제가 다르고, 물체를 든 뒤 그 자세가 여유를
            지킨다는 보장도 없다.
          - 작업자가 로봇 옆에 있는 동안 로봇이 움직이면 안 된다.

        그리퍼가 붙어 있으면(gripper != None) 여닫기까지 여기서 한다.
        눈대중으로 물면 개구량이 계획과 달라지고, 개구량이 다르면 물체가
        죠 안에서 다른 자리에 앉는다 — 그게 곧 파지점이 어긋나는 것이다
        (study_grasp.py: 2 mm 에 밀도 오차 113 %).
        """
        if not self.manual or (self.driver is None and self.gripper is None):
            return
        if self.driver is not None:
            self.driver.stop()
            self.driver.servo_off()
        self.console.stopped("로봇 정지 — 물체를 그리퍼에 물려 주세요")
        target_mm = 1000.0 * self.jaw_opening_m
        if self.driver is not None:
            print("[로봇] 팔 서보를 껐습니다. 팔은 움직이지 않습니다.")
        # 처음에는 **계획 개구량** 자리에 죠를 그린다. 활성화 전의 실물
        # 상태값은 아직 뜻이 없고, 사람이 맞춰야 할 것은 계획값이다.
        self.draw_grasp_guide(self.jaw_opening_m)
        if self.ui is not self.meshcat:
            print("[파지] 작업자 화면에 파지 목표를 그렸습니다 —"
                  " 빨간 점이 파지점, 검은 두 장이 죠입니다.")
        if self.gripper is None:
            self._await_grasp_by_hand(target_mm)
        else:
            self._await_grasp_with_gripper(target_mm)
        self.clear_grasp_guide()
        self.console.clear()
        self.console.clear_sliders()

        if self.driver is not None:
            self.driver.servo_on()
            # 서보가 꺼진 동안 중력으로 처졌을 수 있다. 다시 읽는다.
            self.sync_from_robot()
        distance, pair = self.clearance()
        if distance < self.min_distance_m:
            print(f"[로봇] 경고: 지금 자세의 최소 간격이"
                  f" {1000*distance:.1f} mm 로 기준"
                  f" {1000*self.min_distance_m:.0f} mm 에 못 미칩니다 {pair}."
                  f" 경로계획이 실패할 수 있습니다 —"
                  f" 팔을 조금 띄운 뒤 다시 시작하세요.")

    def set_grip_force_newton(self, newton):
        """다음 파지부터 쓸 힘을 [N] 으로 정한다. 바뀌었으면 알려 준다."""
        counts = gh.counts_for_force(newton)
        if counts == self.gripper.force:
            return
        was_holding = self.gripper.force
        self.gripper.force = counts
        self.report_grip_force()
        if counts > was_holding:
            print("          더 세게 물려면 '슬라이더 힘으로 다시 물기' 를"
                  " 누르세요 — 힘은 다음 파지 명령부터 걸립니다.")

    def report_grip_force(self):
        """지금 파지력이 얼마이고 무게 대비 몇 배인지 적는다.

        미끄럼 여유는 **하한**이다. 이 실험에서 진짜 문제는 미끄러지는 것이
        아니라 물체가 죠 안에서 **돌아가는 것**이다 — 무게중심이 파지점에서
        떨어진 만큼 모멘트가 걸리고, 라운드마다 조금씩 돌아가면 파지점
        어긋남 delta 가 상수가 아니게 되어 grasp_map 이 그걸 못 푼다.
        """
        counts = self.gripper.force
        newton = gh.force_newton(counts)
        margin = gh.slip_margin(counts, self.object_mass_kg)
        print(f"[그리퍼] 파지력 {newton:.0f} N (rFR {counts}/255)"
              f" — 물체 {1000*self.object_mass_kg:.0f} g 무게의"
              f" 약 {margin:.0f} 배 (마찰계수 0.5 가정)")
        if newton < 100.0:
            print("          [주의] 낮습니다. 손목을 돌리는 동안 물체가"
                  " 죠 안에서 돌아갈 수 있습니다.")

    def gripper_opening_m(self):
        """지금 그리퍼 개구량 [m]. 못 읽으면 마지막으로 명령한 값."""
        if self.gripper is None:
            return self.jaw_opening_m
        try:
            return self.gripper.status().opening_m
        except Exception as exc:                       # noqa: BLE001
            print(f"[그리퍼] 상태를 못 읽었습니다: {exc}")
            commanded = getattr(self.gripper, "commanded_opening_m", None)
            return self.jaw_opening_m if commanded is None else commanded

    def _await_grasp_by_hand(self, target_mm):
        """그리퍼가 안 붙어 있을 때 — 예전대로 사람이 손으로 여닫는다."""
        print("[파지] 그리퍼가 연결돼 있지 않습니다. 손으로 여닫으세요.")
        print("        1) 그리퍼를 열고 물체의 파지 부위를 죠 사이에 넣습니다")
        print(f"        2) 개구 {target_mm:.1f} mm 근처로 오므려 물립니다"
              f" (타어링 때와 같은 개구량이어야 합니다)")
        print("        3) 물체에서 손을 떼고 작업영역 밖으로 나옵니다")
        confirm = self.console.button("물체를 물렸고 손을 뗐습니다")
        self.console.wait_for(confirm)
        self.grasp_report = dict(opening_m=self.jaw_opening_m, verified=False,
                                 source="사람이 손으로")

    def _await_grasp_with_gripper(self, target_mm):
        """그리퍼를 화면에서 직접 여닫으며 파지점을 죠 가운데로 맞춘다."""
        print(f"[파지] 계획된 개구량은 {target_mm:.1f} mm 입니다"
              f" (물체 단면 {1000*rs.jaw_dimension_m(self.spec):.1f} mm)."
              f" 물 때는 {1000*gh.CLAMP_SQUEEZE_M:.0f} mm 더 좁게 명령해"
              f" 물체에 걸려 멈추게 합니다.")
        print("        타어링을 이 개구량에서 했다면, 여기서도 같은 값으로"
              " 물어야 렌치의 그리퍼 몫이 같아집니다.")
        try:
            # 활성화는 스트로크를 훑는 보정 동작이라 **물리기 전에** 한다.
            self.gripper.activate()
        except Exception as exc:                       # noqa: BLE001
            print(f"[그리퍼] 활성화 실패: {exc}")
            self._await_grasp_by_hand(target_mm)
            return

        self.console.clear()
        self.console.clear_sliders()
        slider = self.console.slider("개구 [mm]", 0.0,
                                     1000.0 * self.gripper.max_opening_m,
                                     float(target_mm))
        # 파지력은 세션 중에 계속 만질 수 있어야 한다. 물체마다 적정값이
        # 다르고 (얇은 플라스틱은 눌리고, 무거운 것은 돌아간다), 얼마나
        # 조여야 하는지는 실제로 물어 보고 흔들어 봐야 안다.
        force_slider = self.console.slider(
            "파지력 [N]", gh.FORCE_MIN_N, gh.FORCE_MAX_N,
            float(gh.force_newton(self.gripper.force)))
        self.report_grip_force()
        buttons = dict(
            wide=self.console.button("① 활짝 열기 — 물체를 넣으세요"),
            grip=self.console.button(f"② 계획 개구 {target_mm:.1f} mm 로 물기"),
            regrip=self.console.button("슬라이더 힘으로 다시 물기"),
            slide=self.console.button("슬라이더 개구로 이동"),
            more=self.console.button(f"조금 열기 (+{1000*GRASP_NUDGE_M:.0f} mm)"),
            less=self.console.button(f"조금 닫기 (-{1000*GRASP_NUDGE_M:.0f} mm)"),
            done=self.console.button("③ 파지 완료 — 손을 뗐습니다"),
        )
        print("[파지] ① 로 열고, 파지점(빨간 점)이 죠 가운데 오게 물체를 넣은 뒤"
              " ② 로 뭅니다. 다 되면 ③.")
        # 버튼만으로는 미세 조작이 답답하다 (한 번에 2 mm 씩 뛴다). 터미널에서
        # 키를 **누르고 있으면** 개구와 힘이 연속으로 바뀌게 같이 열어 둔다.
        # 로그로 넘겨 돌리면(tty 아님) 조용히 꺼지고 버튼만 남는다.
        keys = gh.KeyReader()
        if keys.usable:
            print("[파지] 이 터미널에서 키보드로도 조작할 수 있습니다:")
            print(gh.KEY_HELP)

        closed_once = False
        override = False
        state = None
        with keys:
            (closed_once, override, state) = self._grasp_loop(
                keys, slider, force_slider, buttons, target_mm)
        return self._finish_grasp(state)

    def _grasp_loop(self, keys, slider, force_slider, buttons, target_mm):
        """파지 단계의 조작 순환. 버튼과 키보드를 둘 다 받는다."""
        clicks = {key: self.ui.GetButtonClicks(name)
                  for key, name in buttons.items()}
        closed_once = False
        override = False
        state = None
        while True:
            pressed = set()
            for key, name in buttons.items():
                count = self.ui.GetButtonClicks(name)
                if count != clicks[key]:
                    clicks[key] = count
                    pressed.add(key)

            # 힘 슬라이더는 **버튼과 무관하게 매 순환** 읽는다. 버튼을 누를
            # 때만 읽으면 슬라이더를 돌려도 화면에 아무 반응이 없어서, 값이
            # 반영되고 있는지 사람이 알 길이 없다.
            self.set_grip_force_newton(self.ui.GetSliderValue(force_slider))

            # ---- 키보드 ----
            # 몰려 들어온 키를 하나의 목표로 접는다. 글자마다 명령하면
            # 도착을 기다리는 사이에 입력이 쌓여 조작이 끈적해진다.
            typed = keys.poll()
            command, clamp = None, False
            if typed:
                now = self.gripper_opening_m()
                target, force, key_clamp, special = gh.fold_keys(
                    typed, now, self.gripper.force,
                    self.gripper.max_opening_m, self.jaw_opening_m)
                force_changed = force != self.gripper.force
                if force_changed:
                    self.gripper.force = force
                    # 슬라이더도 같이 따라가야 한다. 안 그러면 다음 순환에서
                    # 슬라이더 값이 힘을 예전 값으로 되돌려 버린다.
                    self.ui.SetSliderValue(force_slider,
                                           gh.force_newton(force))
                    self.report_grip_force()
                if special == "accept":
                    pressed.add("done")
                if abs(target - now) > 1e-6 or key_clamp:
                    command, clamp = target, key_clamp
                elif force_changed:
                    # 힘은 위치 명령에 실려 나간다. 개구가 그대로여도 다시
                    # 눌러 줘야 새 힘이 실제로 걸린다.
                    command = now

            if command is not None:
                pass                       # 키보드가 이미 정했다
            elif "wide" in pressed:
                command = self.gripper.max_opening_m
            elif "grip" in pressed or "regrip" in pressed:
                command, clamp = self.jaw_opening_m, True
            elif "slide" in pressed:
                command = self.ui.GetSliderValue(slider) * 1e-3
            elif "more" in pressed:
                command = self.gripper_opening_m() + GRASP_NUDGE_M
            elif "less" in pressed:
                # 여기서는 clamp 를 쓰지 않는다. 이미 **지금 개구보다 좁게**
                # 명령하는 것이라 물체가 있으면 그대로 걸려 멈추고 힘이
                # 걸린다. clamp 까지 얹으면 한 번 누를 때마다 2 mm 가 아니라
                # 2+CLAMP_SQUEEZE = 6 mm 씩 닫혔다 (실제로 그랬다).
                command = self.gripper_opening_m() - GRASP_NUDGE_M

            if command is not None:
                command = float(np.clip(command, 0.0,
                                        self.gripper.max_opening_m))
                try:
                    # 물릴 때는 grip(). 물체 단면과 같은 개구를 명령하면 죠가
                    # 닿기만 하고 힘이 안 걸린다 (gripper_hw.CLAMP_SQUEEZE_M).
                    state = (self.gripper.grip(command) if clamp
                             else self.gripper.set_opening(command))
                except Exception as exc:               # noqa: BLE001
                    print(f"[그리퍼] 명령 실패: {exc}")
                    state = None
                else:
                    closed_once = closed_once or command < self.gripper.max_opening_m
                    print(f"[그리퍼] 개구 {1000*command:.1f} mm"
                          f"{' 물기' if clamp else ' 이동'}"
                          f" (힘 {gh.force_newton(self.gripper.force):.0f} N)"
                          f" -> {state.describe()}")
                    self.ui.SetSliderValue(slider, 1000.0 * state.opening_m)
                self.update_grasp_pads(self.gripper_opening_m())
                override = False                       # 명령을 바꿨으면 다시 확인

            if "done" in pressed:
                state = state if state is not None else self._safe_status()
                held = bool(state is not None and state.holding)
                miss = 1000.0 * abs(self.gripper_opening_m()
                                    - self.jaw_opening_m)
                if not override and not closed_once:
                    print("[파지] 아직 무는 명령을 준 적이 없습니다."
                          " ② 로 물린 뒤 ③ 을 누르세요.")
                elif not override and not held:
                    print("[파지] 그리퍼가 '물었다'(gOBJ) 를 보고하지 않습니다"
                          f" — {state.describe() if state else '상태 불명'}."
                          " 정말 물려 있다면 ③ 을 한 번 더 누르세요.")
                    override = True
                elif not override and miss > 2.0:
                    print(f"[파지] 개구량이 계획과 {miss:.1f} mm 다릅니다"
                          f" (지금 {1000*self.gripper_opening_m():.1f},"
                          f" 계획 {target_mm:.1f}). 파지점이 그만큼 어긋납니다"
                          " — 이대로 가려면 ③ 을 한 번 더 누르세요.")
                    override = True
                else:
                    break
            time.sleep(0.05)
        return closed_once, override, state

    def _finish_grasp(self, state):
        opening = self.gripper_opening_m()
        self.grasp_report = dict(
            opening_m=opening, planned_opening_m=self.jaw_opening_m,
            force_n=gh.force_newton(self.gripper.force),
            verified=bool(state is not None and state.holding),
            source=f"그리퍼 {getattr(self.gripper, 'port', '?')}")
        print(f"[파지] 확정 — 개구 {1000*opening:.1f} mm"
              f" (계획 {1000*self.jaw_opening_m:.1f} mm,"
              f" 어긋남 {1000*abs(opening - self.jaw_opening_m):.1f} mm),"
              f" 파지력 {gh.force_newton(self.gripper.force):.0f} N,"
              f" 물림 확인 {'예' if self.grasp_report['verified'] else '아니오'}")

    def _safe_status(self):
        try:
            return self.gripper.status()
        except Exception:                              # noqa: BLE001
            return None

    def begin(self):
        """두 화면이 같이 시작하도록 시작 버튼을 기다린다.

        조정이 자동이어도 이 버튼만은 기다린다. 그래야 탭 두 개를 연 뒤
        왼쪽 탐색을 처음부터 볼 수 있다. --autostart 로 건너뛴다.
        """
        self.sync_from_robot()
        self.set_object_deg(self.object_q_deg)
        self.publish()
        if self.autostart:
            print("[로봇] --autostart 라 바로 시작합니다.")
        else:
            button = self.console.button("두 화면을 모두 연 뒤 눌러 시작")
            print("[로봇] 왼쪽·오른쪽 화면을 모두 연 뒤 오른쪽 화면의"
                  " 시작 버튼을 누르세요.")
            start = self.ui.GetButtonClicks(button)
            while self.ui.GetButtonClicks(button) == start:
                time.sleep(0.05)
            self.console.clear()
        # 파지가 먼저, 이동이 나중. execute() 첫머리의 '시작 자세로 이동'이
        # 물체를 든 상태에서 일어나야 한다.
        self.await_grasp()

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
        # 각도를 재러 가는 이동도 **지금 물체 형상** 으로 계획해야 한다.
        # 예전에는 이 값이 지난 라운드 각도로 남아 있어서, 이번 라운드의
        # 물체 모양과 다른 형상으로 경로를 짰다. 그 경로는 실제로는 물체가
        # 다른 곳에 있으므로 부딪힐 수 있다.
        self.believed_q_deg = deg.copy()

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
            wrench = alg.measure(np.deg2rad(actual_deg),
                                 g_dirs=[np.asarray(g_hat)])
            if self.grasp_error_m is not None:
                # 실물처럼 파지점이 어긋난 척한다. 센서 원점이 delta 만큼
                # 옮겨지면 토크가 그만큼 달라진다 (힘은 그대로).
                #     tau_true = tau_model + M G (g x delta)
                mass = alg.TRUTH_PLANT.CalcTotalMass(alg.TRUTH_CTX)
                wrench = wrench.copy()
                wrench[3:6] += mass * alg.G_ACC * np.cross(
                    np.asarray(g_hat, float), self.grasp_error_m)
            return wrench
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

    class _Pose(hw.SimPose):
        """리허설용 트래커. '실물 관절' 자리를 슬라이더가 대신 채운다.

        예전에는 모의 장비에 자세 센서를 아예 안 붙였다. 그러면 리허설이
        adjust_manually(슬라이더로 판정) 로 가고 실물은 adjust_by_pose
        (트래커로 판정) 로 가서, **리허설이 실물과 다른 길을 밟았다.**
        여기에 센서를 붙여 두면 두 경우가 같은 코드를 지나간다.
        """

        def __init__(self):
            self._truth = None
            super().__init__(lambda: self._truth)

        def set_truth(self, degrees):
            self._truth = np.atleast_1d(np.asarray(degrees, dtype=float))

        def object_joint_deg(self):
            if self._truth is None:
                raise RuntimeError("set_truth 가 먼저 불려야 합니다")
            return super().object_joint_deg()

    tare = hw.TareTable()
    for g_hat in alg.G_DIRS:
        tare.record(g_hat, np.zeros(6))
    return None, _Wrench(), _Pose(), tare, None


def connect_gripper(args, spec):
    """실물 그리퍼를 붙인다. 못 붙이면 None (사람이 손으로 여닫는 옛 흐름).

    --gripper-port none 또는 --no-gripper 면 아예 안 붙인다. --hardware sim
    이면 모의 그리퍼를 붙여 UI 절차만 그대로 돌린다.
    """
    if getattr(args, "no_gripper", False):
        return None
    port = getattr(args, "gripper_port", gh.DEFAULT_PORT)
    if str(port).lower() in ("none", "off", ""):
        return None
    if getattr(args, "gripper", "robotiq2f85") != "robotiq2f85":
        print("  [주의] 실물 그리퍼 드라이버는 Robotiq 2F-85 만 있습니다"
              " — PGC 는 손으로 여닫으세요.")
        return None
    force = gh.counts_for_force(getattr(args, "gripper_force", None)
                                or gh.force_newton(gh.DEFAULT_FORCE))
    if getattr(args, "hardware", "real") != "real":
        print("  모의 그리퍼를 붙입니다 (절차만 확인).")
        return gh.SimGripper(hold_below_m=rs.jaw_dimension_m(spec),
                             force=force)
    try:
        return gh.Robotiq2F85(port=port, force=force)
    except gh.GripperError as exc:
        print(f"  [주의] {exc}")
        print("  그리퍼 없이 계속합니다 — 파지 단계는 손으로 하세요.")
        return None


def connect_hardware(args, spec=None):
    """실물 장비를 붙인다.

    반환: (RobotDriver, WrenchSensor, PoseSensor, TareTable, Gripper)

    안전 로직(속도 상한·준정적 시간·도착 검증·서보 상태기계·타어링 유효기간·
    센서 부호 규약)은 전부 hardware_real 에 구현돼 있고 장비 없이 검증된다.

        ../robot_learning/scripts/run_drake_env.sh python hardware_real.py --check

    벤더 API 가 들어오는 곳은 hardware_real.RbpodoBackend 의 네 함수뿐이다.

    --arm-backend fake 를 주면 **가짜 팔로 실물 경로 전체를 리허설**한다.
    타어링·안전검사·작업자 UI 가 실제와 같은 순서로 돌므로, 센서가 오기 전에
    배선과 절차를 확인할 수 있다.
    """
    import hardware_real as hr

    backend_kind = getattr(args, "arm_backend", "rbpodo")
    tare = _load_tare(args)

    # 그리퍼는 팔 백엔드와 따로 붙는다. 팔이 가짜여도 그리퍼는 실물일 수
    # 있고 (파지 절차만 실물로 리허설하는 경우), 그 반대도 된다.
    gripper = connect_gripper(args, spec) if spec is not None else None

    if backend_kind == "fake":
        print("  [리허설] 가짜 팔로 실물 경로를 돕니다. 팔 장비는 안 씁니다.")
        driver = hr.Rb5Driver(hr.FakeArm())
        wrench, pose = simulated_hardware()[1:3]
        return driver, wrench, pose, tare, gripper

    driver = hr.Rb5Driver(hr.RbpodoBackend(
        host=getattr(args, "robot_host", "192.168.0.10")))
    wrench = hr.Aft200Sensor(sample_fn=_ft_sample_fn(args))
    pose_fn, stamp_fn = _pose_fn(args)
    pose = hr.FoundationPoseSensor(
        pose_fn=pose_fn, stamp_fn=stamp_fn,
        n_joint=len(spec.joints) if spec is not None else 1,
        default_sigma_deg=getattr(args, "angle_error", 0.05) * 100.0,
        max_age_s=getattr(args, "pose_max_age_s", 2.0))
    return driver, wrench, pose, tare, gripper


# FoundationPose 가 내는 각도 키를 PIVOT 관절 순서에 맞춘다.
#
# 부호와 영점은 **아직 안 맞췄다** (my_work/NAMING.md). 여기서 잇는 것은
# '어느 키가 어느 관절인가' 뿐이다. 실물에서 한 자세를 두 방법으로 읽어
# 비교하기 전까지는 판정이 반대로 나올 수 있다.
POSE_KEYS = {
    # desklamp, --grasp-part link_3 일 때 spec.joints 는 [joint_2_3, joint_3_1].
    #   joint_2_3 : link_3(support) -> link_2(Head)   = support_head_deg
    #   joint_3_1 : link_3(support) -> link_1(베이스)  = base_support_deg
    "desklamp": ("support_head_deg", "base_support_deg"),
    "laptop": ("opening_angle_deg",),
}


def _load_tare(args):
    """3자세 타어를 파일에서 읽어 온다.

    예전에는 여기서 **빈** TimedTare 를 만들고 끝이었다. 그러면 첫 측정에서
    tare.apply 가 "missing tare for gravity direction" 으로 죽는다. 타어는
    별도 절차(MeshPCA pivot/tare_real.py)로 미리 재 두는 값이라, 세션은
    그걸 **읽어야** 한다.

    파일 형식은 팀원 aft_tare.TareTable.save 가 낸 그대로다:
        {"entries": [{"g_hat": [...], "wrench": [...]}, ...]}
    """
    import hardware_real as hr
    import json

    max_age = getattr(args, "tare_max_age_s", None)
    tare = hr.TimedTare(max_age_s=(hr.TARE_MAX_AGE_S if max_age is None
                                   else max_age))
    path = getattr(args, "tare_file", None)
    if not path:
        print("  [주의] --tare-file 이 없습니다. 첫 측정에서 그리퍼 무게를"
              " 못 빼고 멈춥니다 (MeshPCA pivot/tare_real.py 로 먼저 재세요).")
        return tare
    path = Path(path).expanduser()
    entries = json.loads(path.read_text())["entries"]
    for entry in entries:
        tare.record(entry["g_hat"], np.asarray(entry["wrench"], dtype=float))
    missing = [g for g in alg.G_DIRS if tare.key(g) not in tare.table]
    if missing:
        raise RuntimeError(
            f"{path} 에 중력 방향 {[np.round(g, 3).tolist() for g in missing]}"
            f" 의 타어가 없습니다. 3자세를 모두 재야 합니다.")
    print(f"  타어 {len(entries)} 방향을 읽었습니다: {path.name}")
    return tare


def _ft_sample_fn(args):
    """AFT200 한 샘플 [Fx Fy Fz Tx Ty Tz] 를 돌려주는 함수.

    팀원의 MeshPCA `pivot/aft_tare.Aft200Sensor` 를 쓴다. 그쪽 read_raw 는
    부를 때마다 TCP 를 새로 열고 n 개를 평균내는데, 여기서는 **한 샘플씩**
    필요하므로 stream() 생성기를 한 번 열어 두고 계속 뽑는다. 연결을 매번
    여닫으면 1000 샘플 평균 하나에 그만큼 시간이 더 든다.
    """
    sys.path.insert(0, str(Path(args.meshpca_root).expanduser() / "pivot"))
    from aft_tare import Aft200Sensor

    sensor = Aft200Sensor(args.aft_host, hz=args.aft_hz)
    stream = sensor.stream()
    return lambda: next(stream)


def _pose_fn(args):
    """FoundationPose latest.json 에서 (각도 deg, sigma deg) 를 읽는 함수.

    팀원 `run_desk_lamp_live.py` 가 그 파일을 **원자적으로**(tmp 에 쓰고
    replace) 계속 갱신한다. 그래서 읽는 쪽은 잠금이 필요 없다.

    같이 돌려주는 stamp_fn 이 중요하다. 트래커가 멈췄는데 마지막 값을 계속
    주면 알고리즘은 각도가 맞다고 믿고 측정을 진행한다 — 가장 위험한
    고장이다 (hardware_real.FoundationPoseSensor 주석).
    """
    import json

    path = Path(args.pose_file).expanduser()
    keys = POSE_KEYS.get(args.object)
    if args.pose_keys:
        keys = tuple(args.pose_keys)
    if not keys:
        raise RuntimeError(
            f"{args.object} 의 FoundationPose 각도 키를 모릅니다."
            f" --pose-keys 로 알려 주세요 (관절 순서대로).")
    state = {"stamp": 0.0}

    def read():
        data = json.loads(path.read_text())
        state["stamp"] = float(data.get("timestamp_s", 0.0))
        missing = [k for k in keys if k not in data]
        if missing:
            raise RuntimeError(f"{path} 에 {missing} 가 없습니다")
        angles = [float(data[k]) for k in keys]
        # **항상 2-튜플로 돌려준다.** hardware_real 은 길이 2 인 리스트를
        # (각도, sigma) 로 읽는데, 관절이 2개인 물체(램프)는 각도 리스트도
        # 길이가 2라서 그냥 리스트로 주면 각도 하나가 sigma 로 오해된다.
        # 실제로 그렇게 깨졌다.
        return angles, data.get("sigma_deg")

    return read, (lambda: state["stamp"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("sim", "deploy"), default="sim",
                        help="sim=화면 3개(계획·로봇·UI, Drake 안에서 이동), "
                             "deploy=화면 2개(계획·UI, 실물 로봇이 이동)")
    parser.add_argument("--object",
                        choices=tuple(obj.OBJECTS) + ("desklamp",),
                        default="3link")
    parser.add_argument("--arm-backend", choices=("rbpodo", "fake"),
                        default="rbpodo",
                        help="fake=장비 없이 실물 경로를 리허설")
    parser.add_argument("--robot-host", default="192.168.0.10")
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
    parser.add_argument("--prior", choices=("weight", "mesh", "water"),
                        default="weight",
                        help="초기값: weight=저울로 총무게만 잼, mesh=메시만 앎,"
                             " water=모든 부위를 물 밀도(1000)로 시작")
    parser.add_argument("--urdf-out", default=None)
    parser.add_argument("--plan-iters", type=int, default=20000,
                        help="RRT-Connect 최대 반복")
    parser.add_argument("--pace", type=float, default=0.08)
    parser.add_argument("--hinge-torque", type=float, default=None,
                        help="힌지 유지토크 [N·m]. 주지 않으면 관절이"
                             " 절대 움직이지 않는다고 가정한다")
    parser.add_argument("--safety", type=float, default=obj.DEFAULT_SAFETY)
    parser.add_argument("--auto-scale", action="store_true")
    parser.add_argument("--min-distance-mm", type=float,
                        default=rs.MIN_DISTANCE_M / rs.MM,
                        help="충돌로 보는 최소 간격. 자세와 경로 모두 지킨다.")
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
                        choices=("centroid", "pinch", "base_frame"),
                        help="desklamp 전용: 잡는 부위의 어디를 잡는가."
                             " centroid=그 부위 도심(기본),"
                             " pinch=볼록 조각에서 고른 '실제로 물리는 자리'"
                             " (desk_lamp.pinch_grasp — 굽은 팔은 도심이"
                             " 단면 중심이 아니라 패드가 허공을 문다)")
    parser.add_argument("--gripper", default="robotiq2f85",
                        choices=("pgc140", "robotiq2f85"),
                        help="그리퍼. pgc140=개구 53 mm, robotiq2f85=개구 78 mm."
                             " 램프처럼 단면이 굵은 물체는 robotiq2f85 가 필요")
    parser.add_argument("--gripper-port", default=gh.DEFAULT_PORT,
                        help="Robotiq 2F-85 의 USB(FTDI) 포트."
                             " none 이면 안 붙이고 사람이 손으로 여닫는다")
    parser.add_argument("--no-gripper", action="store_true",
                        help="그리퍼를 소프트웨어로 몰지 않는다")
    parser.add_argument("--gripper-force", type=float, default=None,
                        help=f"파지력 [N], {gh.FORCE_MIN_N:.0f}~"
                             f"{gh.FORCE_MAX_N:.0f} (기본"
                             f" {gh.force_newton(gh.DEFAULT_FORCE):.0f})."
                             " 세션 중에는 작업자 화면의 '파지력 [N]'"
                             " 슬라이더로 계속 바꿀 수 있다")
    parser.add_argument("--pose-file", default=None,
                        help="FoundationPose 가 갱신하는 latest.json 경로."
                             " 실물(--hardware real)에서 관절각을 여기서 읽는다")
    parser.add_argument("--pose-keys", nargs="+", default=None,
                        help="latest.json 에서 읽을 각도 키를 관절 순서대로."
                             " 기본값은 물체별 표(POSE_KEYS)")
    parser.add_argument("--pose-max-age-s", type=float, default=2.0,
                        help="이보다 오래된 트래커 값은 거부한다")
    parser.add_argument("--tare-file", default=None,
                        help="3자세 타어 JSON (MeshPCA pivot/tare_real.py 산출)."
                             " 실물에서는 반드시 필요하다")
    parser.add_argument("--tare-max-age-s", type=float, default=None,
                        help="이보다 오래된 타어는 거부한다 (기본 hardware_real)")
    parser.add_argument("--aft-host", default="192.168.50.51",
                        help="AFT200 컨트롤러 주소 (Modbus TCP 502)")
    parser.add_argument("--aft-hz", type=float, default=50.0)
    parser.add_argument("--meshpca-root", default="~/MeshPCA",
                        help="팀원 MeshPCA 체크아웃 (pivot/aft_tare.py 위치)")
    parser.add_argument("--no-density-view", action="store_true",
                        help="탐색이 끝난 뒤 밀도 비교 화면을 띄우지 않는다")
    parser.add_argument("--grasp-part", default="link_3",
                        choices=("link_1", "link_2", "link_3"),
                        help="desklamp 전용: 어느 부위를 잡는가."
                             " link_3=연결부(Arm), link_2=베이스, link_1=Head")
    parser.add_argument("--max-joint-speed-deg", type=float,
                        default=DEFAULT_MAX_JOINT_SPEED_DEG,
                        help="관절 속도 상한 [deg/s]. 경로가 길면 이동 시간을"
                             " 늘려서라도 이 속도를 넘지 않는다. 0 이면 끔.")
    parser.add_argument("--grasp-sigma-mm", type=float, default=None,
                        help="파지점이 어긋날 수 있는 크기 [mm]. 0 이면 정확히"
                             " 안다고 본다. 기본값은 sim 0, deploy 5.")
    parser.add_argument("--grasp-error-mm", type=float, default=0.0,
                        help="시뮬레이션에서 파지점을 일부러 이만큼 어긋뜨린다"
                             " (실물이 그렇기 때문). 방향은 --seed 로 정해진다.")
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

    grasp_sigma_m = (args.grasp_sigma_mm * 1e-3
                     if args.grasp_sigma_mm is not None
                     else (0.005 if args.mode == "deploy" else 0.0))
    grasp_error_m = None
    if args.grasp_error_mm > 0.0:
        direction = np.random.default_rng(args.seed).normal(size=3)
        grasp_error_m = direction / np.linalg.norm(direction) * (
            args.grasp_error_mm * 1e-3)
        print(f"  [시뮬레이션] 파지점을 일부러 {args.grasp_error_mm:.1f} mm"
              f" 어긋뜨립니다 {np.round(1000*grasp_error_m, 2)} mm")
    if grasp_sigma_m > 0.0:
        print(f"  파지점 어긋남도 함께 추정합니다"
              f" (사전분포 {1000*grasp_sigma_m:.1f} mm)")
    elif args.grasp_error_mm > 0.0:
        print(f"  [주의] 어긋뜨려 놓고 추정은 끄셨습니다 —"
              f" 치우침이 그대로 남습니다 (--grasp-sigma-mm 5 로 켜세요)")

    planner = PlannerScreen(spec, setup, planner_meshcat, args.pace,
                            args.target, args.angle_error,
                            args.angle_floor_deg,
                            select_mode=args.select, criterion=args.criterion,
                            estimator=args.estimator, stop_rule=args.stop_rule,
                            systematic=args.systematic,
                            search_attempts=args.search_attempts,
                            block_radius_deg=args.block_radius_deg,
                            probe_side=args.probe_side,
                            grasp_sigma_m=grasp_sigma_m)
    print(f"  설정: 후보={args.select}  기준={args.criterion}-최적"
          f"  추정기={args.estimator.upper()}  정지={args.stop_rule}")
    if args.bus == "tcp":
        print("  로봇 쪽은 robot_node.py 가 맡습니다."
              " 이 프로세스는 계획만 합니다.")
        robot = None
    driver = wrench_sensor = pose_sensor = tare = gripper = None
    if args.mode == "deploy" and args.bus != "tcp":
        if args.hardware == "real":
            print("  실물 배포 모드 — hardware.py 의 드라이버를 연결합니다.")
            (driver, wrench_sensor, pose_sensor, tare,
             gripper) = connect_hardware(args, spec)
        else:
            print("  실물 배포 모드 (모의 장비) — 배선만 확인합니다.")
            (driver, wrench_sensor, pose_sensor,
             tare, _) = simulated_hardware()
            gripper = connect_gripper(args, spec)

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
                        grasp_error_m=grasp_error_m,
                        max_joint_speed_deg=args.max_joint_speed_deg,
                        settle_s=args.settle_s,
                        gripper=gripper)
    share = robot.inertial_share(args.move_duration)
    print(f"  준정적 이동: 한 번에 {args.move_duration:.0f}초"
          f"  -> 관성 토크가 중력 토크의 {100*share:.3f}%"
          f"  ({'충분히 준정적' if share < 0.01 else '더 느리게 권장'})")
    # ---- 창 4: 라운드마다 갱신되는 밀도 화면 ----
    # 탐색이 끝난 뒤 한 번만 띄우면 작업자는 "지금 더 해야 하나" 를 알 수
    # 없다. 라운드마다 갱신해서 반폭이 목표 안으로 들어왔는지 보여준다.
    live_panel = None
    if not args.no_density_view:
        try:
            import density_view as dvw

            panel_meshcat = StartMeshcat()
            live_panel = dvw.DensityPanel(
                spec, panel_meshcat,
                theta_deg=setup["feasible"][0] * 180.0 / np.pi)
            live_panel.begin(planner.rho_prior, target_rel=args.target,
                             gt=(planner.rho_gt if args.mode == "sim"
                                 or args.hardware != "real" else None))
            print(f"  [4] 밀도 결과 화면   {panel_meshcat.web_url()}")
        except Exception as exc:                       # noqa: BLE001
            print(f"  [주의] 밀도 화면을 못 띄웠습니다: {exc}")

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
        if live_panel is not None:
            try:
                live_panel.update(planner.rho_hat,
                                  planner.half_width(per_part=True),
                                  index, planner.converged())
            except Exception as exc:                   # noqa: BLE001
                print(f"[주의] 밀도 화면 갱신 실패: {exc}")
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
    if robot is not None and robot.grasp_report is not None:
        report = robot.grasp_report
        print(f"  파지: 개구 {1000*report['opening_m']:.1f} mm"
              f" ({report['source']},"
              f" 물림 확인 {'예' if report.get('verified') else '아니오'})")

    # ---- 결과를 색으로 보여주는 화면 ----
    # 숫자표는 위에 이미 있다. 여기서는 "어느 부위가 무겁다고 나왔나" 를
    # 물체 그림 위에서 보게 한다. 정답 열은 시뮬레이션에서만 붙는다 —
    # 실물에서는 애초에 모르는 값이고, 어차피 채점용이다.
    density_panel = None
    if not args.no_density_view:
        # 이 화면 하나 때문에 세션 마지막에 죽으면 안 된다. 바로 뒤가
        # URDF 산출이고, 그게 이 실험의 결과물이다.
        try:
            import density_view as dvw

            show_gt = args.hardware != "real" or args.mode == "sim"
            panel_meshcat = StartMeshcat()
            density_panel = dvw.DensityPanel(
                spec, panel_meshcat,
                theta_deg=getattr(robot, "object_q_deg", None))
            density_panel.show(
                dvw.panel_columns(planner.rho_prior, planner.rho_hat,
                                  planner.rho_gt if show_gt else None),
                half_width=1.96 * np.sqrt(np.diag(planner.Sigma)))
            print(f"\n  [밀도 비교 화면] {panel_meshcat.web_url()}")
            print("    왼쪽=초기값(저울 총무게만 앎), 가운데=탐색 결과"
                  + (", 오른쪽=정답(채점용)" if show_gt else "")
                  + f".  색은 {dvw.COLORMAP} 무지개 컬러맵입니다.")
        except Exception as exc:                       # noqa: BLE001
            print(f"\n  [주의] 밀도 비교 화면을 못 띄웠습니다: {exc}")

    out = Path(args.urdf_out or Path("outputs") / f"estimated_{spec.key}.urdf")
    try:
        answer = input(f"\n이 값으로 sim-ready URDF 를 만들까요? [{out}] (y/N): ")
    except EOFError:            # 로그로 넘겨 돌릴 때 (키보드가 없다)
        answer = "y" if args.urdf_out else "n"
        print(f"\n입력이 없어 자동으로 '{answer}' 로 답합니다.")
    if answer.strip().lower() in ("y", "yes"):
        import export_urdf as eu
        eu.export(spec, planner.rho_hat, out, planner.Sigma)
        for row in eu.verify_urdf(out, spec, planner.rho_hat):
            print(f"    {row['name']:<13} 되읽기 검증"
                  f"  질량오차 {row['mass_err']:.1e} kg"
                  f"  관성 상대오차 {row['inertia_err']:.1e}")
    else:
        print("  URDF 를 만들지 않았습니다.")

    # 화면은 이 프로세스와 함께 죽는다. 다 보기 전에 닫히지 않게 붙잡는다.
    if density_panel is not None:
        try:
            input("\n밀도 비교 화면을 다 보셨으면 Enter 를 누르세요: ")
        except EOFError:
            pass
    print("\n종료합니다.")


if __name__ == "__main__":
    main()
