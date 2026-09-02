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
타어와 측정이 같은 자세면 그 4.35 N 이 양쪽에 똑같이 들어가 상쇄된다.
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
타어는 "그 중력방향에서 빈 그리퍼가 만드는 렌치" 다. 자세가 목표에서
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

from aft_tare import Aft200Sensor, TareTable, stable_wrench

PIVOT_WORKDIR = Path(os.environ.get("PIVOT_WORKDIR", "")).expanduser()
if not (PIVOT_WORKDIR / "density_id_drake.py").is_file():
    raise RuntimeError("set PIVOT_WORKDIR to the PIVOT my_work directory")
sys.path.insert(0, str(PIVOT_WORKDIR.resolve()))

import density_id_drake as alg
import density_id_objects as obj
import hardware as hw
from path_planning import ArmPathPlanner
import robot_scene as scene


def empty_tool_spec():
    part = obj.Part("tool", (2.0, 78.0, 2.0), 0.01, 1000.0,
                    (1.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                    (0.3, 0.3, 0.3, 1.0), grasp_width_mm=78.0)
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


def require_ready(data, require_real=False):
    state = data.request_data(2.0)
    if state is None:
        raise TimeoutError("no RB5 state response")
    status = state.sdata
    checks = {
        "activation": status.init_state_info == 6 and status.init_error == 0,
        "arm_power": ((status.information_chunk_1 >> 6) & 1) == 1,
        "idle": status.robot_state == 1 and status.task_state == 1,
        "collision_detection": status.collision_detect_onoff == 1,
        "freedrive_off": status.is_freedrive_mode == 0,
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
    """자세 오차가 타어에 만드는 힘 오차 [N]."""
    return TOOL_MASS_KG * 9.81 * float(np.sin(np.deg2rad(error_deg)))


def hold_and_read(data, checker, sensor, g_hat, label, tol_deg, samples,
                  poll_s=0.5):
    """작업자가 자세를 만들 때까지 기다렸다가 읽는다. 로봇에 명령하지 않는다."""
    print(f"\n[{label}] 목표 중력방향 {np.asarray(g_hat).tolist()}")
    print("  직접교시로 팔을 옮긴 뒤 freedrive 를 끄세요 (hold).")
    print(f"  목표에서 {tol_deg:.1f} deg 안에 들어오면 자동으로 읽습니다."
          f"  (Ctrl-C 로 중단)")
    last = None
    while True:
        try:
            q = require_ready(data)              # freedrive 가 켜져 있으면 여기서 걸린다
        except RuntimeError as exc:
            if last != str(exc):
                print(f"  대기: {exc}")
                last = str(exc)
            time.sleep(poll_s)
            continue
        error_deg, achieved = orientation_error_deg(checker, q, g_hat)
        print(f"\r  오차 {error_deg:6.2f} deg"
              f"  (타어 오차 {tare_error_n(error_deg):5.2f} N,"
              f" 램프 5.60 N 의 {100*tare_error_n(error_deg)/5.60:4.1f} %)"
              f"   관절 {np.round(np.degrees(q), 1).tolist()}   ", end="")
        if error_deg <= tol_deg:
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
        raise SystemExit(f"확인할 타어 파일이 없습니다: {args.output}\n"
                         f"  먼저 세팅 때 한 번 재세요 (--setup)")
    payload = json.loads(args.output.read_text())
    entries = [e for e in payload["entries"] if "joint_deg" in e]
    if not entries:
        raise SystemExit("타어 파일에 joint_deg 가 없어 자세를 되짚을 수 없습니다.\n"
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
            robot = hw.Rb5Driver(args.robot_ip, enable_motion=True,
                                 max_speed_deg_s=args.speed_deg_s,
                                 max_accel_deg_s2=2 * args.speed_deg_s)
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
          f"  (타어 오차 {tare_error_n(args.angle_tol_deg):.2f} N,"
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
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nsaved: {args.output.resolve()}")
    print(f"  최악 자세 오차 {worst:.2f} deg"
          f" -> 타어 오차 {tare_error_n(worst):.2f} N"
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

    rbpodo = __import__("rbpodo")
    data = rbpodo.CobotData(args.robot_ip)
    current = require_ready(data)

    if args.verify:
        return run_verify(args, data, current)
    if args.manual:
        return run_manual(args, data, current)

    start, paths, clearances = plan_paths(
        current, args.clearance_mm / 1000.0, args.max_iters)
    print("start joints [deg]:", np.round(np.degrees(start), 2).tolist())
    for label, path, clearance in zip(("g-down", "g-x", "g-y", "return"),
                                      paths, clearances):
        print(f"{label}: {len(path)} waypoints, clearance {clearance*1000:.2f} mm, "
              f"goal {np.round(np.degrees(path[-1]), 1).tolist()} deg")
    if args.plan_only:
        print("PLAN ONLY: no robot command")
        return

    unchanged = require_ready(data)
    error = np.degrees(np.abs(np.arctan2(np.sin(unchanged - current),
                                         np.cos(unchanged - current))))
    if np.max(error) > 0.05:
        raise RuntimeError("robot pose changed while planning")

    robot = hw.Rb5Driver(args.robot_ip, enable_motion=True,
                         max_speed_deg_s=args.speed_deg_s,
                         max_accel_deg_s2=2 * args.speed_deg_s)
    sensor = Aft200Sensor(args.robot_ip, args.aft_hz)
    tare = TareTable()
    try:
        deadline = time.monotonic() + 3.0
        while data.request_data(2.0).sdata.real_vs_simulation_mode != 0:
            if time.monotonic() >= deadline:
                raise RuntimeError("RB5 did not enter Real mode within 3 seconds")
            time.sleep(0.1)
        require_ready(data, require_real=True)
        for g_hat, path in zip(alg.G_DIRS, paths[:3]):
            robot.follow(path, path_duration(path, args.speed_deg_s))
            robot.stop()
            time.sleep(1.0)
            raw = stable_wrench(sensor, args.samples)
            tare.record(g_hat, raw)
            print(f"tare g={g_hat.tolist()}: {np.round(raw, 4).tolist()}")
        robot.follow(paths[3], path_duration(paths[3], args.speed_deg_s))
        robot.stop()
    finally:
        robot.stop()

    tare.save(args.output)
    print(f"saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
