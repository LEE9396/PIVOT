#!/usr/bin/env python3
"""Plan safe RB5 paths and record empty Robotiq wrench in three orientations.

언제 재나 — 세팅 때 한 번, 그 뒤로는 확인만
------------------------------------------
    --setup   3자세를 다 잰다. **실험 환경을 세팅할 때 한 번.**
    --verify  기록된 자세 하나로만 돌아가 다시 읽고 그때와 비교한다.
              차이가 문턱 안이면 세팅 때 값을 그대로 쓴다. 10초면 된다.

왜 "예측"이 아니라 "확인"인가
-----------------------------
세팅 때 한 번 재고 그 뒤로는 계산으로 때우려면
    raw(g) = 공구무게 * g + 영점치우침
이 성립해야 한다. 실측 데이터로 맞춰 보면 **안 맞는다** — 라벨·순서를 어떻게
바꿔도 잔차가 4.35 N 아래로 안 내려간다. 램프 전체 무게 5.60 N 의 78 % 다.
반복오차가 0.02 N 인 것을 보면 잡음이 아니라 모형에 없는 계통 효과이고,
센서 케이블 당김이 유력하다.

자세별 표는 그 효과를 **모형화하지 않고 통째로 빼기** 때문에 문제가 안 된다.
영점 조정과 측정이 같은 자세면 그 4.35 N 이 양쪽에 똑같이 들어가 상쇄된다.
모형으로 갈아타는 순간 그것이 오차로 튀어나온다. 그래서 예측하지 않고
**같은 자세로 돌아가 다시 읽어** 드리프트만 확인한다.

두 가지 방식
------------
기본     로봇이 스스로 세 자세로 이동한다. IK 가 푼 자세를 그대로 가므로
         목표 중력방향과 정확히 맞는다.
--manual 로봇에 **명령을 보내지 않는다.** 작업자가 직접교시(freedrive)로
         팔을 세 자세로 옮기고, 이 프로그램은 관절각과 렌치를 **읽기만**
         한다. 준비 과정에서 로봇이 스스로 움직이는 구간을 없애고 싶을 때.

--manual 의 대가 — 자세 정확도
------------------------------
영점 조정값은 "그 중력방향에서 빈 그리퍼가 만드는 렌치" 다. 자세가 목표에서
어긋난 만큼 그 값이 틀리고, 나중에 그대로 빼진다. 센서 아래 무게는
그리퍼 0.414 kg + 센서 퍽 0.236 kg = 0.65 kg 이므로

    자세 오차 0.5 deg -> 0.06 N   (램프 전체 5.60 N 의  1.0 %)
    자세 오차 1.0 deg -> 0.11 N   (                     2.0 %)
    자세 오차 2.0 deg -> 0.22 N   (                     4.0 %)
    자세 오차 5.0 deg -> 0.56 N   (                     9.9 %)

목표 불확실성이 5 % 이므로 **1 deg 이내**로 맞춰야 한다. 그래서 이 모드는
지금 자세가 목표에서 몇 도 떨어져 있는지 실시간으로 찍어 주고, 문턱 안에
들어오기 전에는 안 읽는다. 사람이 6 관절을 다 맞출 필요는 없다 — 제약은
**손목이 향하는 방향** 하나뿐이라 손목만 돌리면 수렴한다.
"""

import argparse
import os
from pathlib import Path
import sys
import time

import numpy as np
from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.solvers import Solve

from aft_tare import Aft200Sensor, TareTable, stable_wrench

PIVOT_WORKDIR = Path(os.environ.get("PIVOT_WORKDIR", "")).expanduser()
if not (PIVOT_WORKDIR / "density_id_drake.py").is_file():
    raise RuntimeError("set PIVOT_WORKDIR to the PIVOT my_work directory")
sys.path.insert(0, str(PIVOT_WORKDIR.resolve()))

import density_id_drake as alg
import density_id_objects as obj
import hardware_real as hr
from path_planning import ArmPathPlanner
import robot_scene as scene


# 영점 조정에서 밟을 중력 방향들.
#
# 앞의 셋은 **반드시** 여기 있어야 한다 — dual_view 의 TareTable 이 이 키로
# 값을 찾는다 (density_id_drake.G_DIRS 와 같은 셋).
#
# 뒤의 다섯은 검산용이다. 미지수가 축(3) + 바이어스(3) + 무게(1) = 7 개인데
# 방향이 3 개면 식이 9 개뿐이라 거의 맞아떨어진다. 그러면 잔차가 0 에 가깝게
# 나와서 "값이 맞다" 와 "검산할 여유가 없다" 를 구별할 수 없다. 실제로 3
# 방향으로는 잔차 0.011 N, 같은 데이터를 8 방향으로 재면 0.102 N 이 나온다.
#
# 방향 하나당 손목 이동 몇 초. 전체가 5 분쯤 늘어난다.
TARE_DIRECTIONS = [
    (0.0, 0.0, -1.0),        # 필수 — 센서 +Z축을 천장으로
    (1.0, 0.0, 0.0),         # 필수 — 센서 +X축을 바닥으로
    (0.0, 1.0, 0.0),         # 필수 — 센서 +Y축을 바닥으로
    (-1.0, 0.0, 0.0),
    (0.0, -1.0, 0.0),
    (0.57735, 0.57735, -0.57735),
    (-0.57735, 0.57735, -0.57735),
    (0.57735, -0.57735, -0.57735),
]
REQUIRED_DIRECTIONS = TARE_DIRECTIONS[:3]
MIN_DIRECTIONS_FOR_AXES = 5


def direction_label(g_hat):
    return {
        (0.0, 0.0, -1.0): "센서 +Z축을 천장으로",
        (1.0, 0.0, 0.0): "센서 +X축을 바닥으로",
        (0.0, 1.0, 0.0): "센서 +Y축을 바닥으로",
    }.get(tuple(float(v) for v in g_hat),
          f"중력이 센서 좌표계에서 {np.round(g_hat, 2).tolist()}")


# 손에 무엇을 들고 영점을 잴 때의 그 물건 크기 [mm].
# None 이면 빈 그리퍼. --payload-mm 으로 설정한다.
#
# 왜 필요한가 — 아는 무게로 센서 배율을 검증하려면 물건을 들고 재야 한다.
# 그런데 충돌 검사는 여기서 만드는 형상만 본다. 빈 그리퍼로 두면 **손에
# 든 물건이 검사에 아예 안 보인다.** 손목이 크게 도는 동안 물병이 팔이나
# 책상에 닿아도 프로그램은 "여유 23 mm" 라고 말한다.
PAYLOAD_BOX_MM = None


def empty_tool_spec(payload_mm=None):
    """영점 조정 때 그리퍼 안에 있는 것의 형상.

    payload_mm 이 없으면 2 mm 짜리 얇은 판 = 사실상 빈 손이다.
    있으면 그 크기의 직육면체를 손에 쥔 것으로 본다.

    크기는 **물건을 감싸는 직육면체**로 준다. 물건이 어느 방향으로 놓이는지
    확신이 없으면 가장 긴 치수로 정육면체를 주는 것이 안전하다 — 검사가
    실제보다 빡빡해질 뿐, 놓치지는 않는다.
    """
    box = PAYLOAD_BOX_MM if payload_mm is None else payload_mm
    if box is None:
        bbox, width = (2.0, 78.0, 2.0), 78.0
    else:
        bbox = tuple(float(v) for v in box)
        width = min(bbox)              # 죠가 무는 쪽은 가장 좁은 단면
    part = obj.Part("tool", bbox, 0.01, 1000.0,
                    (bbox[0] / 2.0, 0.0, 0.0), (bbox[0] / 2.0, 0.0, 0.0),
                    (0.3, 0.3, 0.3, 1.0), grasp_width_mm=width)
    return obj.ObjectSpec("tare_tool", "empty Robotiq", [part], [],
                          (0.0, 0.0, 0.0))


def nearest_equivalent(q, reference, lower, upper):
    result = []
    for value, current, lo, hi in zip(q, reference, lower, upper):
        choices = [value + 2 * np.pi * k for k in range(-2, 3)
                   if lo <= value + 2 * np.pi * k <= hi]
        if not choices:
            raise RuntimeError("no equivalent joint angle inside the limits")
        result.append(min(choices, key=lambda item: abs(item - current)))
    return np.asarray(result)


def require_ready(data, require_real=False, allow_freedrive=False):
    state = data.request_data(2.0)
    if state is None:
        raise TimeoutError("no RB5 state response")
    status = state.sdata
    checks = {
        "activation": status.init_state_info == 6 and status.init_error == 0,
        "arm_power": ((status.information_chunk_1 >> 6) & 1) == 1,
        "idle": status.robot_state == 1 and status.task_state == 1,
        "collision_detection": status.collision_detect_onoff == 1,
        "freedrive_off": allow_freedrive or status.is_freedrive_mode == 0,
        "no_fault": not any((status.op_stat_collision_occur, status.op_stat_sos_flag,
                              status.op_stat_soft_estop_occur, status.op_stat_ems_flag)),
        "real_mode": not require_real or status.real_vs_simulation_mode == 0,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError("RB5 safety check failed: " + ", ".join(failed))
    return np.deg2rad(np.asarray(status.jnt_ang[:6], dtype=float))


def plan_paths(current, clearance_m, max_iters):
    checker = scene.PoseChecker(
        empty_tool_spec(), densities=[1000.0], joint_limits_rad=[],
        min_distance_m=clearance_m, gripper="robotiq2f85",
        ik_restarts=30, seed_q=current)
    indices = [joint.position_start() for joint in checker.arm_joints]
    fixed = checker.plant.GetPositions(checker.context).copy()
    planner = ArmPathPlanner(checker.plant, checker.context,
                             checker.arm_joints, clearance_m, fixed, seed=11)
    start = nearest_equivalent(current, np.clip(current, planner.lower, planner.upper),
                               planner.lower, planner.upper)
    for joint, value in zip(checker.arm_joints, start):
        fixed[joint.position_start()] = value
    planner.set_fixed(fixed)

    targets = []
    for g_hat in alg.G_DIRS:
        checker._last_solution = None
        full = checker.solve_robust(np.array([]), g_hat)
        if full is None:
            raise RuntimeError(f"IK failed for gravity direction {g_hat}")
        targets.append(np.asarray(full)[indices])

    paths, clearances, state = [], [], start
    for target in targets + [start]:
        target = nearest_equivalent(target, state, planner.lower, planner.upper)
        path = planner.plan(state, target, max_iters=max_iters)
        if path is None:
            raise RuntimeError("no collision-free path from the current pose")
        if not np.allclose(path[0], state) or not np.allclose(path[-1], target):
            raise RuntimeError("planned path endpoints are invalid")
        measured = planner.path_clearance(path, samples_per_edge=60)
        if measured < clearance_m * 0.9 - 1e-6:
            raise RuntimeError(f"path clearance is too small: {measured*1000:.2f} mm")
        paths.append(path)
        clearances.append(measured)
        state = target
    return start, paths, clearances


def wrist_window(joint, reference, limit_rad):
    """손목 관절 하나가 돌 수 있는 구간. 관절 한계와 +-limit 중 좁은 쪽.

    구간이 비면 (lower > upper) 그 관절은 한 발짝도 못 움직인다는 뜻이다.
    로봇이 손목 한계 근처(예: J4 = 333 deg)에 서 있으면 실제로 일어난다.
    """
    lower = max(joint.position_lower_limits()[0], reference - limit_rad)
    upper = min(joint.position_upper_limits()[0], reference + limit_rad)
    return lower, upper


def solve_wrist(checker, indices, fixed, base, reference, g_hat,
                max_wrist_deg, avoid_collision=True):
    """J1--J3 를 base 로 못박고 손목 J4--J6 만으로 g_hat 을 만든다.

    풀리면 팔 관절각 6개, 안 풀리면 None.
    """
    for joint, value in zip(checker.arm_joints[3:], reference[3:]):
        lower, upper = wrist_window(joint, value, np.deg2rad(max_wrist_deg))
        if lower > upper:
            return None                     # 창이 비었다 — 풀 것도 없다
    ik = InverseKinematics(checker.plant, checker.context,
                           with_joint_limits=True)
    prog, q = ik.prog(), ik.q()
    for joint, value in zip(checker.arm_joints[:3], base[:3]):
        index = joint.position_start()
        prog.AddBoundingBoxConstraint(value, value, q[index])
    for joint, value in zip(checker.arm_joints[3:], reference[3:]):
        index = joint.position_start()
        lower, upper = wrist_window(joint, value, np.deg2rad(max_wrist_deg))
        prog.AddBoundingBoxConstraint(lower, upper, q[index])
    for joint in checker.finger_joints:
        index = joint.position_start()
        prog.AddBoundingBoxConstraint(checker.finger_value,
                                      checker.finger_value, q[index])
    ik.AddAngleBetweenVectorsConstraint(
        checker.plant.world_frame(), np.array([0.0, 0.0, -1.0]),
        checker.sensor_frame, np.asarray(g_hat, dtype=float),
        0.0, scene.ANGLE_TOL_RAD)
    if avoid_collision:
        ik.AddMinimumDistanceLowerBoundConstraint(checker.ik_distance_m, 0.01)
    prog.AddQuadraticErrorCost(np.eye(3), reference[3:], q[indices[3:]])
    guess = fixed.copy()
    for index, value in zip(indices[:3], base[:3]):
        guess[index] = value
    for index, value in zip(indices[3:], reference[3:]):
        guess[index] = value
    prog.SetInitialGuess(q, guess)
    result = Solve(prog)
    if not result.is_success():
        return None
    solution = result.GetSolution(q)[indices]
    if np.max(np.abs(solution[:3] - np.asarray(base)[:3])) > 1e-9:
        return None                         # J1--J3 가 움직였으면 손목 전용이 아니다
    return solution


def explain_wrist_failure(checker, indices, fixed, base, reference, g_hat,
                          max_wrist_deg):
    """왜 손목만으로 안 되는지 좁힌다.

    제한을 하나씩 풀어 보고 **어느 것을 풀었을 때 풀리는지** 로 범인을
    가린다. 창을 넓히면 되면 손목 한계 문제이고, 충돌 검사를 꺼야 되면
    팔이 무언가에 닿는 것이고, 그래도 안 되면 이 J1--J3 배치 자체가
    그 방향을 만들 수 없는 것이다.
    """
    ladder = [
        (180.0, True, f"손목 창을 {max_wrist_deg:.0f} -> 180 deg 로 넓히면"),
        (360.0, True, "손목을 관절 한계까지 열면"),
        (360.0, False, "손목을 다 열고 충돌 검사까지 끄면"),
    ]
    for limit, avoid, label in ladder:
        if solve_wrist(checker, indices, fixed, base, reference, g_hat,
                       limit, avoid_collision=avoid) is not None:
            return label
    return "손목을 다 열고 충돌 검사를 꺼도 안 됩니다 — 이 J1--J3 배치로는 못 만드는 방향입니다"


def wrist_headroom_text(checker, reference, max_wrist_deg):
    """손목 세 관절이 지금 자리에서 좌우로 몇 도씩 남았는지."""
    parts = []
    for number, (joint, value) in enumerate(
            zip(checker.arm_joints[3:], reference[3:]), start=4):
        lower, upper = wrist_window(joint, value, np.deg2rad(max_wrist_deg))
        parts.append(f"J{number} {np.degrees(value):+7.1f} deg"
                     f" (-{np.degrees(max(value - lower, 0.0)):.0f}"
                     f" / +{np.degrees(max(upper - value, 0.0)):.0f})")
    return "   ".join(parts)


def pose_note(planner, checker, q):
    """자세 하나가 왜 못 쓰는지. 쓸 수 있으면 None."""
    q = np.asarray(q, dtype=float)
    outside = [f"J{i+1} {np.degrees(v):.1f} deg"
               for i, v in enumerate(q)
               if v < planner.lower[i] - 1e-9 or v > planner.upper[i] + 1e-9]
    if outside:
        return f"관절 한계 밖 ({', '.join(outside)})"
    if not planner.valid(q):
        planner.plant.SetPositions(planner.context, planner.full_q(q))
        query = planner.query_port.Eval(planner.context)
        return f"충돌 ({checker.closest_pair_text(query)})"
    return None


def wrist_base_candidates(checker, indices, start, required, tries=6, seed=17):
    """손목 전용 배치가 될 만한 팔 자세 후보들.

    첫째는 지금 자세다 — 되면 팔을 아예 안 움직인다. 그 다음은 필수 방향을
    팔 전체로 푼 자세들인데, IK 는 출발점에 따라 다른 답을 내므로 초기추측을
    흔들어 여러 개를 받는다. 하나만 받아 두면 그것이 못 가는 자리일 때
    시도할 것이 남지 않는다.
    """
    found = [("지금 자세", np.asarray(start, dtype=float))]
    rng = np.random.default_rng(seed)
    saved = np.array(checker.seed_q, dtype=float).copy()
    seen = []
    try:
        for g_hat in required:
            for attempt in range(tries):
                checker._last_solution = None
                checker.seed_q = (saved if attempt == 0 else
                                  saved + rng.uniform(-0.8, 0.8, size=saved.shape))
                solution = checker.solve(np.array([]),
                                         np.asarray(g_hat, dtype=float),
                                         warm_start=False)
                if solution is None:
                    continue
                q = np.asarray(solution)[indices]
                if any(np.max(np.abs(q - other)) < np.deg2rad(2.0)
                       for other in seen):
                    continue                    # 사실상 같은 자세는 한 번만
                seen.append(q)
                found.append((f"{direction_label(g_hat)} 를 팔 전체로 푼 자세",
                              q))
    finally:
        checker.seed_q = saved
    return found


def find_wrist_base(checker, planner, indices, fixed, start, max_wrist_deg,
                    required, max_iters, wanted=None, path_tries=8, log=print):
    """필수 방향을 **전부** 손목만으로 만들 수 있고, **거기까지 갈 수 있는**
    J1--J3 배치를 찾는다.

    두 조건을 따로 보면 안 된다. 손목만으로 세 방향이 되는 자리를 찾아 놓고
    거기까지 가는 길이 없으면 아무 쓸모가 없다. 그래서 후보마다 손목 검사와
    경로 검사를 **둘 다** 하고, 하나라도 걸리면 다음 후보로 넘어간다.

    돌려주는 것: (base, approach, note)
        approach 가 None 이면 지금 자리가 그대로 그 자리라 팔을 안 움직인다.
        base 가 None 이면 note 는 후보별 실패 이유 목록이다.
    """
    wanted = list(required if wanted is None else wanted)
    reasons, survivors = [], []
    for label, raw in wrist_base_candidates(checker, indices, start, required):
        candidate = nearest_equivalent(raw, start, planner.lower, planner.upper)
        move = np.degrees(np.abs(candidate[:3] - np.asarray(start)[:3])).max()
        head = f"  {label} (J1--J3 {move:.0f} deg)"

        blocked = None
        state = candidate
        for g_hat in required:
            target = solve_wrist(checker, indices, fixed, candidate, state,
                                 g_hat, max_wrist_deg)
            if target is None:
                blocked = (f"{direction_label(g_hat)} 를 손목만으로 못 만듭니다"
                           f" — {explain_wrist_failure(checker, indices, fixed, candidate, state, g_hat, max_wrist_deg)}")
                break
            state = target
        if blocked is not None:
            reasons.append(f"{head}: {blocked}")
            continue

        # 필수 셋만 보고 고르면 안 된다. 검산용 방향까지 몇 개나 닿는지가
        # **결과의 질을 좌우한다.** 지난 실측에서 g=(0,-1,0) 이 빠지는 바람에
        # 모형 y 방향이 한쪽으로만 측정됐고, 그 방향에 반응하는 센서 z 축의
        # 이득 추정이 12 % 나 흔들렸다. 축이 고장인지 아닌지를 가리는 바로
        # 그 숫자였다. 그래서 많이 닿는 후보를 먼저 본다.
        state, reach = candidate, 0
        for g_hat in wanted:
            target = solve_wrist(checker, indices, fixed, candidate, state,
                                 g_hat, max_wrist_deg)
            if target is not None:
                reach += 1
                state = target
        survivors.append((reach, move, label, candidate, head))

    # 많이 닿는 것 우선, 같으면 팔을 적게 움직이는 것 우선.
    survivors.sort(key=lambda item: (-item[0], item[1]))
    planned = 0
    for reach, move, label, candidate, head in survivors:
        head = f"{head[:-1]}, {reach}/{len(wanted)} 방향)"
        if move <= 1e-9:
            log(f"{head}: 손목만으로 됩니다. 팔은 안 움직입니다.")
            return candidate, None, label

        why = pose_note(planner, checker, candidate)
        if why is not None:
            reasons.append(f"{head}: 손목은 되는데 그 자세 자체가 못 씁니다 — {why}")
            continue
        if planned >= path_tries:
            reasons.append(f"{head}: 손목은 되지만 경로를 안 찾아봤습니다"
                           f" (경로 탐색 {path_tries} 번을 다 썼습니다)")
            continue

        planned += 1
        approach = planner.plan(start, candidate, max_iters=max_iters)
        if approach is None:
            reasons.append(f"{head}: 손목은 되는데 거기까지 갈 충돌 없는 경로를"
                           f" {max_iters} 번 안에 못 찾았습니다")
            continue
        if (not np.allclose(approach[0], start)
                or not np.allclose(approach[-1], candidate)):
            reasons.append(f"{head}: 계획된 경로의 끝점이 어긋났습니다")
            continue
        log(f"{head}: 손목만으로 세 방향 다 되고, 경로도 있습니다.")
        return candidate, approach, label

    return None, None, reasons


def plan_wrist_paths(current, clearance_m, max_wrist_deg=120.0,
                     directions=None, max_iters=10000, log=print):
    """세 중력방향을 J1--J3 한 배치에서 손목 J4--J6 만으로 만든다.

    왜 손목만인가 — 영점 조정은 "그 자세에서 빈 그리퍼가 만드는 렌치" 를
    통째로 빼는 방식이라, 영점을 잰 자세와 실제로 재는 자세가 가까울수록
    잘 상쇄된다. 지난 세션은 이 둘이 587--681 deg 나 떨어져 있었다.

    예전 코드의 구멍 — 로봇이 서 있던 자세의 J1--J3 를 그대로 쓰고, 그
    자세로 안 되면 그냥 멈췄다. 그 자세가 좋은 자리일 이유가 없는데도.
    이제는 되는 자리를 **찾아서**, 거기까지 한 번만 팔 전체로 (충돌 검사를
    거쳐) 옮기고, 그 다음부터 손목만 쓴다.
    """
    checker = scene.PoseChecker(
        empty_tool_spec(), densities=[1000.0], joint_limits_rad=[],
        min_distance_m=clearance_m, gripper="robotiq2f85",
        ik_restarts=30, seed_q=current)
    indices = [joint.position_start() for joint in checker.arm_joints]
    fixed = checker.plant.GetPositions(checker.context).copy()
    planner = ArmPathPlanner(checker.plant, checker.context,
                             checker.arm_joints, clearance_m, fixed, seed=11)
    # 로봇이 읽어 준 각도가 모형의 관절 한계 밖일 수 있다 (예: J4 = 333 deg).
    # 그대로 쓰면 IK 의 구간이 비어 무조건 실패한다. plan_paths 가 이미
    # 하던 정규화를 여기서도 한다.
    start = nearest_equivalent(current,
                               np.clip(current, planner.lower, planner.upper),
                               planner.lower, planner.upper)
    for joint, value in zip(checker.arm_joints, start):
        fixed[joint.position_start()] = value
    planner.set_fixed(fixed)

    wanted = [np.asarray(g, dtype=float)
              for g in (directions if directions is not None else alg.G_DIRS)]
    required = [np.asarray(g, dtype=float) for g in REQUIRED_DIRECTIONS]

    log(f"  지금 자세: {np.round(np.degrees(start), 1).tolist()} deg")
    # 지금 서 있는 자세부터 본다. 여기가 이미 안 되면 어느 후보를 골라도
    # 출발을 못 하므로, 다른 것을 다 해 보기 전에 먼저 말해 준다.
    why = pose_note(planner, checker, start)
    if why is not None:
        raise RuntimeError(
            f"지금 로봇이 서 있는 자세를 쓸 수 없습니다 — {why}\n"
            f"  안전 여유 {clearance_m*1000:.0f} mm 기준입니다."
            " 작업영역을 치우거나, 직접교시로 팔을 빼낸 뒤 --plan 을 다시 하세요.")
    log(f"  손목 여유 (창 {max_wrist_deg:.0f} deg):"
        f" {wrist_headroom_text(checker, start, max_wrist_deg)}")

    base, approach, note = find_wrist_base(
        checker, planner, indices, fixed, start, max_wrist_deg, required,
        max_iters, wanted=wanted, log=log)
    if base is None:
        raise RuntimeError(
            "필수 중력 방향 셋을 손목만으로 만들 수 있고 거기까지 갈 수도 있는"
            " J1--J3 배치를 못 찾았습니다. 후보별로:\n"
            + "\n".join(note)
            + "\n  손 쓸 수 있는 것:"
            "\n    - 팔을 다른 자세로 옮기고 --plan 을 다시 (직접교시로 충분합니다)"
            "\n    - 작업영역을 치운다 (위에 '충돌' 이 많으면 이것부터)"
            f"\n    - 손목 창을 넓힌다: --max-wrist-deg {max_wrist_deg + 60:.0f}"
            "\n      (케이블이 그만큼 꼬여도 되는지 눈으로 먼저 확인하세요)"
            "\n    - 경로를 더 오래 찾는다: --max-iters 40000"
            "\n    - 손목 전용을 포기하고 사람이 맞춘다: --manual")

    if approach is not None:
        measured = planner.path_clearance(approach, samples_per_edge=60)
        if measured < clearance_m * 0.9 - 1e-6:
            raise RuntimeError(f"접근 경로 여유가 너무 작습니다:"
                               f" {measured*1000:.2f} mm")
        log(f"  접근: {len(approach)} waypoints, 여유 {measured*1000:.2f} mm,"
            f" J1--J3 {np.degrees(np.abs(base[:3] - start[:3])).max():.1f} deg 이동")

    paths, clearances, reached, state = [], [], [], base
    for g_hat in wanted:
        is_required = any(np.allclose(g_hat, r) for r in required)
        target = solve_wrist(checker, indices, fixed, base, state, g_hat,
                             max_wrist_deg)
        why = None
        if target is None:
            why = explain_wrist_failure(checker, indices, fixed, base, state,
                                        g_hat, max_wrist_deg)
        elif not planner.edge_valid(state, target):
            why = "손목 직선 경로가 충돌합니다"
        if why is not None:
            # 필수 셋은 위에서 이미 되는 것을 확인하고 고른 배치다. 여기서
            # 걸리면 경로(edge) 쪽 문제이므로 멈춘다. 검산용 방향은 못 가도
            # 그냥 건너뛴다 — 손목 창 밖일 수 있고, 그것 때문에 준비가
            # 막히면 안 된다.
            if is_required:
                raise RuntimeError(
                    f"필수 중력 방향 {g_hat.tolist()}"
                    f" ({direction_label(g_hat)}): {why}")
            log(f"  건너뜀 {np.round(g_hat, 2).tolist()}: {why}")
            continue
        path = [state, target]
        if approach is not None and not paths:
            path = list(approach) + [target]
        paths.append(path)
        clearances.append(planner.path_clearance(path, samples_per_edge=60))
        reached.append(g_hat)
        state = target

    return start, paths, clearances, reached


def gravity_in_sensor(checker, q):
    """팔 관절각 q 일 때, 센서 프레임에서 본 월드 아래 방향.

    IK 가 거는 제약(AddAngleBetweenVectorsConstraint)과 같은 양이다. 자동
    모드에서는 IK 가 이것을 g_hat 에 맞춰 풀고, --manual 에서는 사람이 맞춘
    결과를 여기서 확인한다.
    """
    for joint, value in zip(checker.arm_joints, q):
        joint.set_angle(checker.context, float(value))
    R_WS = checker.sensor_frame.CalcPoseInWorld(checker.context).rotation()
    return R_WS.matrix().T @ np.array([0.0, 0.0, -1.0])


def orientation_error_deg(checker, q, g_hat):
    achieved = gravity_in_sensor(checker, q)
    cosine = float(np.clip(np.dot(achieved, np.asarray(g_hat, dtype=float)),
                           -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine))), achieved


TOOL_MASS_KG = 0.650          # 센서 아래: Robotiq 0.414 + AFT200 퍽 0.236


def tare_error_n(error_deg):
    """자세 오차가 영점 조정에 만드는 힘 오차 [N]."""
    return TOOL_MASS_KG * 9.81 * float(np.sin(np.deg2rad(error_deg)))


def hold_and_read(data, checker, sensor, g_hat, label, tol_deg, samples,
                  poll_s=0.5):
    """작업자가 자세를 만들 때까지 기다렸다가 읽는다. 로봇에 명령하지 않는다."""
    print(f"\n[{label}] 목표 중력방향 {np.asarray(g_hat).tolist()}")
    direction = {
        (0.0, 0.0, -1.0): "센서 +Z축을 천장으로",
        (1.0, 0.0, 0.0): "센서 +X축을 바닥으로",
        (0.0, 1.0, 0.0): "센서 +Y축을 바닥으로",
    }[tuple(np.asarray(g_hat, dtype=float))]
    print(f"  방향: {direction} 향하게 하세요. TCP 위치는 중요하지 않습니다.")
    print("  직접교시로 팔을 옮긴 뒤 freedrive 를 끄세요 (hold).")
    print(f"  목표에서 {tol_deg:.1f} deg 안에 들어오면 자동으로 읽습니다."
          f"  (Ctrl-C 로 중단)")
    last = None
    while True:
        try:
            q = require_ready(data, allow_freedrive=True)
        except RuntimeError as exc:
            if last != str(exc):
                print(f"  대기: {exc}")
                last = str(exc)
            time.sleep(poll_s)
            continue
        error_deg, achieved = orientation_error_deg(checker, q, g_hat)
        try:
            require_ready(data)
            holding = True
        except RuntimeError as exc:
            if "freedrive_off" not in str(exc):
                raise
            holding = False
        state = "hold" if holding else "freedrive"
        print(f"\r  오차 {error_deg:6.2f} deg"
              f"  현재 g={np.round(achieved, 2).tolist()}  [{state}]"
              + ("  방향 OK -> freedrive를 끄세요" if error_deg <= tol_deg
                 and not holding else "") + "   ", end="")
        if error_deg <= tol_deg and holding:
            print("\n  문턱 안. 렌치를 읽습니다 (손을 떼고 기다리세요).")
            time.sleep(1.0)
            raw = stable_wrench(sensor, samples)
            return raw, q, error_deg, achieved
        last = None
        time.sleep(poll_s)


def path_duration(path, speed_deg_s):
    max_delta = max(float(np.max(np.abs(np.degrees(b - a))))
                    for a, b in zip(path[:-1], path[1:]))
    return len(path) * max(10.0, 2.0 * max_delta / speed_deg_s)


def nearest_entry(entries, q):
    """지금 자세에 가장 가까운 기록 자세를 고른다."""
    best, best_gap = None, None
    for entry in entries:
        recorded = np.deg2rad(np.asarray(entry["joint_deg"], dtype=float))
        recorded = np.arctan2(np.sin(recorded), np.cos(recorded))
        folded = np.arctan2(np.sin(q), np.cos(q))
        gap = float(np.degrees(np.abs(np.arctan2(
            np.sin(folded - recorded), np.cos(folded - recorded)))).max())
        if best_gap is None or gap < best_gap:
            best, best_gap = entry, gap
    return best, best_gap


def run_verify(args, data, current):
    """드리프트 확인. 세팅 때 잰 값을 그대로 써도 되는지 10초 만에 본다.

    자세를 **다시 만들어** 읽는다. 예측하지 않는다 (위 문서 참고).
    로봇이 이동하는 것은 기록된 자세 하나까지, 한 번뿐이다.
    --manual 이면 그것도 사람이 직접교시로 옮긴다.
    """
    import json

    if not args.output.is_file():
        raise SystemExit(f"확인할 영점 조정 파일이 없습니다: {args.output}\n"
                         f"  먼저 세팅 때 한 번 재세요 (--setup)")
    payload = json.loads(args.output.read_text())
    entries = [e for e in payload["entries"] if "joint_deg" in e]
    if not entries:
        raise SystemExit("영점 조정 파일에 joint_deg 가 없어 자세를 되짚을 수 없습니다.\n"
                         "  --setup 으로 다시 재세요 (지금 코드는 자세를 함께 남깁니다)")

    entry, gap_deg = nearest_entry(entries, current)
    target = np.deg2rad(np.asarray(entry["joint_deg"], dtype=float))
    target = np.arctan2(np.sin(target), np.cos(target))
    print(f"확인 대상 자세  g={entry['g_hat']}")
    print(f"  지금 자세와의 차이 {gap_deg:.2f} deg")

    if gap_deg > args.pose_tol_deg:
        if args.manual:
            print(f"  직접교시로 아래 관절각까지 옮기세요"
                  f" (허용 {args.pose_tol_deg:.1f} deg):")
            print(f"    {np.round(np.degrees(target), 2).tolist()}")
            while True:
                q = require_ready(data)
                _, gap_deg = nearest_entry([entry], q)
                print(f"\r  차이 {gap_deg:6.2f} deg   ", end="")
                if gap_deg <= args.pose_tol_deg:
                    print("\n  도착.")
                    break
                time.sleep(0.5)
        else:
            print(f"  그 자세로 이동합니다 (한 번, 경로 계획됨)")
            start, paths, _ = plan_paths(current, args.clearance_mm / 1000.0,
                                         args.max_iters)
            checker_planner = None
            robot = hr.Rb5Driver(
                hr.RbpodoBackend(host=args.robot_ip),
                max_speed=np.deg2rad(args.speed_deg_s))
            try:
                index = [tuple(np.round(e["g_hat"], 6))
                         for e in entries].index(tuple(np.round(entry["g_hat"], 6)))
                robot.follow(paths[index], path_duration(paths[index],
                                                         args.speed_deg_s))
                robot.stop()
            finally:
                robot.stop()

    time.sleep(1.0)
    sensor = Aft200Sensor(args.robot_ip, args.aft_hz)
    raw = stable_wrench(sensor, args.samples)
    recorded = np.asarray(entry["wrench"], dtype=float)
    drift = raw - recorded
    force_drift = float(np.linalg.norm(drift[:3]))

    print(f"\n  세팅 때  {np.round(recorded, 3).tolist()}")
    print(f"  지금     {np.round(raw, 3).tolist()}")
    print(f"  차이     {np.round(drift, 3).tolist()}")
    print(f"\n  힘 드리프트 {force_drift:.3f} N"
          f"   (문턱 {args.drift_tol_n:.2f} N,"
          f" 램프 5.60 N 의 {100*force_drift/5.60:.1f} %)")

    payload.setdefault("verifications", []).append(dict(
        time=time.time(), g_hat=entry["g_hat"],
        pose_gap_deg=gap_deg, drift=drift.tolist(),
        force_drift_n=force_drift, tolerance_n=args.drift_tol_n,
        passed=bool(force_drift <= args.drift_tol_n)))
    args.output.write_text(json.dumps(payload, indent=2) + "\n")

    if force_drift <= args.drift_tol_n:
        print("  통과 — 세팅 때 값을 그대로 쓰세요.")
        return 0
    print("  **초과 — 다시 재세요** (--setup)")
    print("  온도가 변했거나 그리퍼 구성이 바뀐 것입니다.")
    return 1


def run_manual(args, data, current):
    """직접교시 모드. 로봇에 **명령을 보내지 않는다.**

    경로 계획도 안 한다 — 움직일 경로가 없으니 계획할 것이 없다. 대신
    작업자가 만든 자세가 목표 중력방향과 얼마나 맞는지만 계속 확인한다.
    IK 목표 관절각은 참고용으로만 알려 준다 (그대로 맞출 필요 없다 —
    제약은 손목이 향하는 방향 하나다).
    """
    checker = scene.PoseChecker(
        empty_tool_spec(), densities=[1000.0], joint_limits_rad=[],
        min_distance_m=args.clearance_mm / 1000.0, gripper="robotiq2f85",
        ik_restarts=30, seed_q=current)
    indices = [joint.position_start() for joint in checker.arm_joints]

    print("직접교시 모드 — 이 프로그램은 로봇에 명령하지 않습니다.")
    print(f"  허용 오차 {args.angle_tol_deg:.1f} deg"
          f"  (영점 조정 오차 {tare_error_n(args.angle_tol_deg):.2f} N,"
          f" 램프 5.60 N 의 {100*tare_error_n(args.angle_tol_deg)/5.60:.1f} %)")

    # 참고용 목표 자세. 못 풀어도 진행한다 — 사람은 손목만 맞추면 된다.
    hints = []
    for g_hat in alg.G_DIRS:
        checker._last_solution = None
        try:
            full = checker.solve_robust(np.array([]), g_hat)
        except Exception:                                       # noqa: BLE001
            full = None
        hints.append(None if full is None else np.asarray(full)[indices])
    for g_hat, hint in zip(alg.G_DIRS, hints):
        if hint is None:
            print(f"  g={g_hat.tolist()}: 참고 자세를 못 풀었습니다"
                  f" (손목 방향만 맞추면 됩니다)")
        else:
            print(f"  g={g_hat.tolist()}: 참고 관절각"
                  f" {np.round(np.degrees(hint), 1).tolist()} deg")

    if args.plan_only:
        print("\nPLAN ONLY: 로봇 명령 없음 (직접교시 모드는 원래 명령이 없습니다)")
        return

    sensor = Aft200Sensor(args.robot_ip, args.aft_hz)
    tare = TareTable()
    records = []
    for g_hat, label in zip(alg.G_DIRS, ("g-down", "g-x", "g-y")):
        raw, q, error_deg, achieved = hold_and_read(
            data, checker, sensor, g_hat, label,
            args.angle_tol_deg, args.samples)
        tare.record(g_hat, raw)
        records.append(dict(label=label, g_hat=np.asarray(g_hat).tolist(),
                            wrench=np.asarray(raw).tolist(),
                            joints_rad=np.asarray(q).tolist(),
                            achieved_g_hat=np.asarray(achieved).tolist(),
                            orientation_error_deg=error_deg,
                            tare_error_n=tare_error_n(error_deg)))
        print(f"  기록 {label}: {np.round(raw, 4).tolist()}")

    tare.save(args.output)
    # 실제로 맞춘 자세와 그 오차를 함께 남긴다. TareTable.load 는 entries 만
    # 읽으므로 다른 키를 더해도 안전하다. 나중에 결과가 이상할 때 여기부터 본다.
    #
    # joint_deg 는 **실물 장비가 내는 파일과 같은 키 이름**이다. --verify 가
    # 이 값으로 자세를 되짚으므로 이름이 갈리면 안 된다.
    import json
    payload = json.loads(args.output.read_text())
    for entry, record in zip(payload["entries"], records):
        entry["joint_deg"] = np.degrees(
            np.asarray(record["joints_rad"], dtype=float)).tolist()
        entry["direction_error_deg"] = record["orientation_error_deg"]
    worst = max(r["orientation_error_deg"] for r in records)
    payload["manual"] = dict(
        method="freedrive hold, no robot command",
        angle_tol_deg=args.angle_tol_deg,
        worst_orientation_error_deg=worst,
        worst_tare_error_n=tare_error_n(worst),
        tool_mass_kg=TOOL_MASS_KG,
        records=records)
    payload["created_at_s"] = time.time()
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nsaved: {args.output.resolve()}")
    print(f"  최악 자세 오차 {worst:.2f} deg"
          f" -> 영점 조정 오차 {tare_error_n(worst):.2f} N"
          f" (램프 5.60 N 의 {100*tare_error_n(worst)/5.60:.1f} %)")
    if worst > args.angle_tol_deg:
        print("  [주의] 문턱을 넘겼습니다 — 다시 재세요")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-ip", default="192.168.50.51")
    parser.add_argument("--output", type=Path,
                        default=Path("calibration/aft_tare_current.json"))
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--aft-hz", type=float, default=50.0)
    parser.add_argument("--speed-deg-s", type=float, default=3.0)
    parser.add_argument("--clearance-mm", type=float, default=10.0)
    parser.add_argument("--max-iters", type=int, default=10000)
    parser.add_argument("--payload-mm", type=float, nargs=3,
                        metavar=("X", "Y", "Z"),
                        help="손에 든 물건을 감싸는 직육면체 크기 [mm]."
                             " 아는 무게로 배율을 검증할 때 **반드시** 준다."
                             " 안 주면 충돌 검사가 그 물건을 못 본다."
                             " 방향이 헷갈리면 가장 긴 치수로 정육면체를"
                             " 주는 것이 안전하다.")
    parser.add_argument("--max-wrist-deg", type=float, default=120.0,
                        help="손목 J4-J6 를 한 자리에서 몇 도까지 돌릴지."
                             " 넓힐수록 도달할 수 있는 방향은 늘지만"
                             " 센서 케이블이 그만큼 꼬인다.")
    parser.add_argument("--dirs", type=int, default=len(TARE_DIRECTIONS),
                        help="밟을 중력 방향 개수 (앞의 3개는 필수). 5개 이상이면"
                             " 센서 축까지 풀리고 잔차로 검산할 여유가 생긴다")
    parser.add_argument("--tool-kg", type=float, default=1.0,
                        help="센서 아래 매달린 무게의 카탈로그 값"
                             " (Robotiq 2F-85 0.925 + 커플러)")
    parser.add_argument("--force", action="store_true",
                        help="검산에 불합격해도 저장한다")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--manual", action="store_true",
                        help="로봇에 명령하지 않는다. 사람이 직접교시로 자세를 "
                             "만들고 이 프로그램은 읽기만 한다")
    parser.add_argument("--setup", action="store_true",
                        help="3자세를 다 잰다 (실험 환경 세팅 때 한 번). "
                             "아무 모드도 안 주면 이것이 기본이다")
    parser.add_argument("--verify", action="store_true",
                        help="기록된 자세 하나로 돌아가 드리프트만 확인한다 "
                             "(실험 직전, 10초)")
    parser.add_argument("--drift-tol-n", type=float, default=0.10,
                        help="--verify 통과 문턱 [N]. 반복오차 0.03 N 보다 크고 "
                             "램프 5.60 N 의 1.8 %% 다")
    parser.add_argument("--pose-tol-deg", type=float, default=0.5,
                        help="--verify 에서 기록 자세로 돌아왔다고 볼 관절각 오차")
    parser.add_argument("--angle-tol-deg", type=float, default=1.0,
                        help="--manual 에서 허용하는 중력방향 오차 [deg]. "
                             "1 deg 가 램프 무게의 2 %% 에 해당한다")
    args = parser.parse_args()
    if (args.output.exists() and not args.overwrite and not args.plan_only
            and not args.verify):
        parser.error(f"refusing to overwrite {args.output}")

    global PAYLOAD_BOX_MM
    if args.payload_mm is not None:
        PAYLOAD_BOX_MM = tuple(args.payload_mm)
        print(f"  손에 든 물건: {PAYLOAD_BOX_MM} mm 직육면체로 충돌 검사합니다")
    else:
        print("  손에 든 물건: 없음 (빈 그리퍼로 충돌 검사합니다)")

    rbpodo = __import__("rbpodo")
    data = rbpodo.CobotData(args.robot_ip)
    current = require_ready(data)

    if args.verify:
        return run_verify(args, data, current)
    if args.manual:
        return run_manual(args, data, current)

    directions = TARE_DIRECTIONS[:max(3, args.dirs)]
    start, paths, clearances, reached = plan_wrist_paths(
        current, args.clearance_mm / 1000.0,
        max_wrist_deg=args.max_wrist_deg, directions=directions,
        max_iters=args.max_iters)
    print("start joints [deg]:", np.round(np.degrees(start), 2).tolist())
    for g_hat, path, clearance in zip(reached, paths, clearances):
        print(f"g={np.round(g_hat, 2).tolist()}: {len(path)} waypoints,"
              f" clearance {clearance*1000:.2f} mm,"
              f" goal {np.round(np.degrees(path[-1]), 1).tolist()} deg")
    if len(reached) < MIN_DIRECTIONS_FOR_AXES:
        print(f"  [주의] 갈 수 있는 방향이 {len(reached)}개뿐입니다."
              f" {MIN_DIRECTIONS_FOR_AXES}개 미만이면 센서 축을 못 풀고,"
              f" 잔차도 실제보다 작게 나옵니다.")
    if args.plan_only:
        print("PLAN ONLY: no robot command")
        return

    unchanged = require_ready(data)
    error = np.degrees(np.abs(np.arctan2(np.sin(unchanged - current),
                                         np.cos(unchanged - current))))
    if np.max(error) > 0.05:
        raise RuntimeError("robot pose changed while planning")

    robot = hr.Rb5Driver(
                hr.RbpodoBackend(host=args.robot_ip),
                max_speed=np.deg2rad(args.speed_deg_s))
    sensor = Aft200Sensor(args.robot_ip, args.aft_hz)
    tare = TareTable()
    records = []
    try:
        require_ready(data)
        for g_hat, path in zip(reached, paths):
            robot.follow(path, path_duration(path, args.speed_deg_s))
            robot.stop()
            time.sleep(1.0)
            raw = stable_wrench(sensor, args.samples)
            tare.record(g_hat, raw)
            q = robot.joint_positions()
            checker = scene.PoseChecker(
                empty_tool_spec(), densities=[1000.0], joint_limits_rad=[],
                min_distance_m=args.clearance_mm / 1000.0,
                gripper="robotiq2f85", seed_q=q)
            error_deg, _ = orientation_error_deg(checker, q, g_hat)
            records.append((q, error_deg))
            print(f"tare g={g_hat.tolist()}: {np.round(raw, 4).tolist()}")
    finally:
        robot.stop()

    import json
    import tempfile

    # 검산을 통과하기 전에는 args.output 을 건드리지 않는다. 예전에는 여기서
    # 바로 덮어써 놓고 아래에서 "쓰지 않았습니다" 라고 찍었다 — 불합격한
    # 원시값이 그대로 영점 파일이 되어 있었고, 로그는 아니라고 말했다.
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as raw:
        raw_path = Path(raw.name)
    tare.save(raw_path)
    payload = json.loads(raw_path.read_text())
    raw_path.unlink(missing_ok=True)
    for entry, (q, error_deg) in zip(payload["entries"], records):
        entry["joint_deg"] = np.degrees(q).tolist()
        entry["direction_error_deg"] = error_deg
    payload["automatic"] = {"method": "J1-J3 fixed, J4-J6 only",
                            "n_directions": len(reached)}
    payload["created_at_s"] = time.time()

    # 저장하기 전에 **물리적으로 말이 되는지 검산한다.**
    #
    # 지금까지는 재서 저장하고 끝이었다. 그런데 그리퍼 질량은 이미 아는 값
    # (2F-85 = 0.925 kg + 커플러) 이므로, 이 3~8 자세는 그 자체로 "알려진 추를
    # 여러 방향에서 재는" 검증 실험이다. 추가 장비가 필요 없다.
    #
    # 검산이 이 넷을 한 번에 판정한다.
    #   힘의 부호      density_id_drake.FORCE_SIGN 과 맞나
    #   힘의 크기·단위 되찾은 공구 질량이 카탈로그와 맞나
    #   토크 기준점    되찾은 지렛대가 그리퍼 무게중심 거리인가
    #   배선 상태      맞춤 잔차. 케이블이 센서를 당기면 여기로 나온다
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "my_work"))
    import tare_check

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
        tmp.write(json.dumps(payload))
        tmp_path = tmp.name
    print()
    ok, info = tare_check.check(tmp_path, tool_kg=args.tool_kg)
    Path(tmp_path).unlink(missing_ok=True)

    # 검산에서 나온 값을 파일에 남긴다. 코드에 박아둔 상수(TOOL_MASS_KG,
    # robot_scene.GRIPPER_MOUNT_YAW_RAD)와 대조하라는 뜻이다.
    payload["measured"] = {
        "tool_mass_kg": float(info["tool_mass_kg"]),
        "force_sign_vs_gravity": int(info["force_sign"]),
        "com_lever_m": float(info["lever_m"]),
        "residual_force_n": float(info["residual_force_n"]),
        "residual_torque_nm": float(info["residual_torque_nm"]),
        "bias_force_n": [float(v) for v in info["bias_force"]],
        "passed": bool(ok),
    }
    if info.get("axes") is not None:
        payload["measured"]["sensor_axis_rotation_deg"] = float(
            info["axes"]["angle_deg"])
        payload["measured"]["sensor_axis_R"] = [
            [float(v) for v in row] for row in info["axes"]["R"]]
    if abs(info["tool_mass_kg"] - TOOL_MASS_KG) > 0.1:
        print(f"\n  [주의] 코드의 TOOL_MASS_KG = {TOOL_MASS_KG:.3f} kg 인데"
              f" 실측은 {info['tool_mass_kg']:.3f} kg 입니다.")
        print( "         tare_error_n() 의 자세오차 -> 힘오차 환산이 그만큼"
               " 틀립니다.")

    if not ok and not args.force:
        fail = args.output.with_suffix(".failed.json")
        fail.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\n  검산 불합격 — {args.output.name} 를 쓰지 않았습니다.")
        print(f"  잰 값은 확인용으로 {fail} 에 남겼습니다.")
        print( "  고치고 다시 재세요. 그래도 이 값을 쓰려면 --force 입니다.")
        raise SystemExit(1)

    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nsaved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
