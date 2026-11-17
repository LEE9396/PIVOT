"""시작점 정지 F/T 기록 또는 손목 한 관절의 작은 왕복 검증.

기본은 계획만 하며 로봇 명령을 보내지 않는다. --record는 현재 자리에서 읽기만,
--run은 빈 그리퍼를 최대 2° 이동해 기록한 뒤 검증된 직선으로 복귀한다.
작은 각도 자료는 전체 3축 캘리브레이션을 대체하지 않는다.
"""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "my_work"))
import density_id_objects as obj
import hardware_real as hr
from path_planning import ArmPathPlanner
import robot_scene as rs


def empty_scene():
    # 빈 손의 형상을 명시한다. 이전 램프 세션의 파지 변환을 가져오지 않는다.
    part = obj.Part("empty", (2., 2., 2.), 0.001, 1000., (0., 0., 0.),
                    (0., 0., 0.), (0.3, 0.3, 0.3, 1.), grasp_width_mm=78.)
    spec = obj.ObjectSpec("local_ft_empty", "empty tool", [part], [], (0., 0., 0.))
    return rs.build_scene(spec, joint_limits_rad=[], gripper="robotiq2f85",
                           include_visuals=False, include_aft_cable=True,
                           grasp_transform=np.eye(4))


def local_plan(start, joint=6, delta_deg=0.0, clearance_m=0.02):
    start = np.asarray(start, dtype=float)
    if (start.shape != (6,) or not np.all(np.isfinite(start)) or joint not in (4, 5, 6)
            or not np.isfinite(delta_deg) or abs(delta_deg) > 2
            or not np.isfinite(clearance_m) or clearance_m < rs.MIN_DISTANCE_M):
        raise ValueError("손목 J4–J6 한 관절 ±2° 이내, 기존 충돌 여유 이상이어야 합니다")
    scene = empty_scene()
    diagram = scene["builder"].Build()
    root_context = diagram.CreateDefaultContext()
    plant = scene["plant"]
    context = plant.GetMyContextFromRoot(root_context)
    joints = [plant.GetJointByName(n, scene["arm"]) for n in rs.ARM_JOINT_NAMES]
    planner = ArmPathPlanner(plant, context, joints, clearance_m,
                             plant.GetPositions(context).copy())
    # 명령은 한 관절만 움직인다. 복귀 검사에서는 실측 도착 오차 0.1°까지 허용한다.
    radius = np.full(6, np.deg2rad(0.1))
    radius[joint - 1] += np.deg2rad(abs(delta_deg))
    planner.limit_to_start(start, radius)
    target = start.copy()
    target[joint - 1] += np.deg2rad(delta_deg)
    path = planner.plan(start, target)
    if path is None or not planner.edge_valid(target, start):
        raise RuntimeError("현재 자세 또는 작은 왕복 경로가 충돌/관절 제한을 통과하지 못했습니다. 이동 안 함")
    route = [start, target, start] if delta_deg else [start, start]
    minimum, a, b = planner.path_closest_pair(route, samples_per_edge=101)
    external, ea, eb = planner.path_closest_pair(route, samples_per_edge=101,
                                                external_only=True)
    report = dict(start_joint_deg=np.degrees(start).tolist(),
                  route_joint_deg=np.degrees(route).tolist(), joint=joint,
                  delta_deg=delta_deg, clearance_m=clearance_m,
                  closest_pair=[a, b], closest_distance_m=minimum if np.isfinite(minimum) else None,
                  external_closest_pair=[ea, eb],
                  external_distance_m=external if np.isfinite(external) else None,
                  distance_query_radius_m=0.05, motion_radius_deg=np.degrees(radius).tolist(),
                  direct_only=True, empty_tool=True, wrench_frame="ft_mount",
                  full_axis_calibration=False,
                  workspace_obstacles=scene["workspace_obstacles"],
                  collision_policy=dict(
                      self_penetration_tolerance_m=0.0,
                      external_clearance_m=clearance_m,
                      cable_radius_m=rs.AFT_CABLE_RADIUS_M,
                      cable_length_m=rs.AFT_CABLE_LENGTH_M,
                      obstacle_bodies=[plant.get_body(i).name() for i in
                          plant.GetBodyIndices(plant.GetModelInstanceByName("lab"))],
                      check_interpolated_edges=True, check_live_feedback=True))
    return report, planner, scene, diagram, root_context


def select_stable_tail(report, count=5, force_span_n=0.1):
    """마지막 연속 구간만 판정한다. 원자료를 보존하고 유리한 구간을 검색하지 않는다."""
    if count < 3 or not np.isfinite(force_span_n) or force_span_n <= 0:
        raise ValueError("안정화 검사는 3구간 이상과 양의 유한한 허용 변화량이 필요합니다")
    blocks = report["blocks"]
    assessment = dict(block_count=count, force_span_limit_n=force_span_n,
                      passed=False, channel_force_spans_n={})
    report["stability"] = assessment
    if len(blocks) < count:
        raise ValueError("안정화 검사에 필요한 구간 수가 부족합니다")
    for field in ("wrench_mean", "controller_wrench_mean"):
        values = np.asarray([b[field] for b in blocks[-count:]], dtype=float)
        if values.shape != (count, 6) or not np.all(np.isfinite(values)):
            raise ValueError("안정화 검사에 유효한 두 채널 평균이 필요합니다")
        force = values[:, :3]
        span = float(np.max(np.linalg.norm(force[:, None] - force[None, :], axis=2)))
        assessment["channel_force_spans_n"][field] = span
    assessment["passed"] = all(v <= force_span_n for v in
                                assessment["channel_force_spans_n"].values())
    if not assessment["passed"]:
        raise ValueError(f"힘 평균이 안정화 기준을 통과하지 못했습니다: {assessment}")
    report["all_recorded_blocks"] = blocks
    report["blocks"] = blocks[-count:]


def compare_load(current, baseline, mass_kg, mean_field="wrench_mean"):
    """같은 정지 자세의 차분을 검산한다. 아직 보정 전이므로 합격을 선언하지 않는다."""
    if not np.isfinite(mass_kg) or mass_kg <= 0:
        raise ValueError("알려진 질량은 양의 유한수여야 합니다")
    if baseline.get("status") != "recorded" or not baseline.get("blocks"):
        raise ValueError("완료된 무부하 기록이 필요합니다")
    if (not current.get("sensor_sha256")
            or baseline.get("sensor_sha256") != current.get("sensor_sha256")):
        raise ValueError("영점과 하중 측정의 센서 코드가 다릅니다")
    anchor = np.asarray(baseline["blocks"][0]["joint_before_deg"])
    for block in baseline["blocks"] + current["blocks"]:
        poses = [block["joint_before_deg"], block["joint_after_deg"]]
        for q in poses + block.get("controller_joint_deg", []):
            q = np.asarray(q)
            if (q.shape != (6,) or not np.all(np.isfinite(q))
                    or np.max(np.abs((q - anchor + 180.) % 360. - 180.)) > 0.1):
                raise ValueError("무부하와 하중의 실제 자세가 다릅니다. 같은 자리에서 다시 측정하세요")
    unloaded = np.mean([b[mean_field] for b in baseline["blocks"]], axis=0)
    loaded = np.mean([b[mean_field] for b in current["blocks"]], axis=0)
    delta = loaded - unloaded
    expected = mass_kg * 9.81
    return dict(known_mass_kg=mass_kg, expected_force_n=expected,
                delta_wrench=delta.tolist(), observed_force_norm_n=float(np.linalg.norm(delta[:3])),
                force_norm_relative_error=float(abs(np.linalg.norm(delta[:3]) - expected) / expected),
                provisional=True, calibration_valid=False)


def read_state(data):
    reply = data.request_data(2.0)
    if reply is None:
        raise TimeoutError("RB5 상태 응답 없음")
    status = reply.sdata
    if status.robot_state != 1 or status.task_state != 1:
        raise hr.SafetyViolation("정지 상태가 아닙니다")
    if any((status.op_stat_collision_occur, status.op_stat_sos_flag,
            status.op_stat_soft_estop_occur, status.op_stat_ems_flag)):
        raise hr.SafetyViolation("로봇 fault/정지 상태를 확인하세요")
    q = hr.principal_angles(np.deg2rad(np.asarray(status.jnt_ang[:6], dtype=float)))
    if q.shape != (6,) or not np.all(np.isfinite(q)):
        raise ValueError("유효하지 않은 관절각")
    return q, status


def read_q(data):
    return read_state(data)[0]


gripper_snapshot = hr.gripper_snapshot


def record(sensor, data, samples, blocks, report, gripper_status_file=None,
           duration_s=None, checkpoint=None):
    # stream의 개별 출력과 정지구간 평균을 둘 다 저장한다. 독립 표본 수로 단정하지 않는다.
    stream = sensor.stream()
    report["blocks"] = []
    # 실제 팔 자세에서 센서 축의 중력을 기록한다. 물체 자세로 대신하지 않는다.
    scene = empty_scene() if gripper_status_file else None
    if scene:
        plant = scene["plant"]
        context = plant.CreateDefaultContext()
    if duration_s is not None and (not np.isfinite(duration_s) or duration_s <= 0):
        raise ValueError("기록 시간은 양의 유한수여야 합니다")
    started = time.monotonic()
    anchor = read_q(data)
    grip_anchor = gripper_snapshot(gripper_status_file) if gripper_status_file else None
    report["record_started_at_s"] = time.time()
    try:
        while (time.monotonic() - started < duration_s if duration_s is not None
               else len(report["blocks"]) < blocks):
            before = read_q(data)
            grip_before = gripper_snapshot(gripper_status_file) if gripper_status_file else None
            values, registers, stamps = [], [], []
            controller_values, controller_times, controller_q = [], [], []
            for _ in range(samples):
                values.append(np.asarray(next(stream), dtype=float))
                registers.append(sensor.last_registers.copy())
                stamps.append(sensor.last_sample_time_s)
                # 두 채널은 순차 수신이다. 시각을 따로 남겨 동시 샘플로 오인하지 않는다.
                requested = time.time()
                q, status = read_state(data)
                if np.max(np.abs(hr.principal_angles(q - anchor))) > np.deg2rad(0.1):
                    raise hr.SafetyViolation("전체 기록 중 시작 자세에서 벗어났습니다")
                if gripper_status_file:
                    grip = gripper_snapshot(gripper_status_file)
                    if (abs(grip["position"] - grip_anchor["position"]) > 1
                            or grip.get("requested_position") != grip_anchor.get("requested_position")):
                        raise ValueError("기록 중 그리퍼 개구 또는 명령이 변했습니다")
                controller_times.append([requested, time.time()])
                if np.max(np.abs(hr.principal_angles(q - before))) > np.deg2rad(0.1):
                    raise hr.SafetyViolation("평균 구간에서 로봇 자세가 변했습니다")
                wrench = [float(getattr(status, "eft_" + axis))
                          for axis in ("fx", "fy", "fz", "mx", "my", "mz")]
                if not np.all(np.isfinite(wrench)):
                    raise ValueError("제어기 F/T 샘플이 유효하지 않습니다")
                controller_values.append(wrench)
                controller_q.append(np.degrees(q).tolist())
            after = read_q(data)
            if np.max(np.abs(hr.principal_angles(after - before))) > np.deg2rad(0.1):
                raise hr.SafetyViolation("평균 구간에서 로봇 자세가 변했습니다")
            values = np.asarray(values)
            if values.shape != (samples, 6) or not np.all(np.isfinite(values)):
                raise ValueError("F/T 샘플이 유효하지 않습니다")
            report["blocks"].append(dict(
                joint_before_deg=np.degrees(before).tolist(),
                joint_after_deg=np.degrees(after).tolist(), sample_time_s=stamps,
                registers=registers, wrench_samples=values.tolist(),
                wrench_mean=values.mean(axis=0).tolist(),
                controller_wrench_samples=controller_values,
                controller_wrench_mean=np.mean(controller_values, axis=0).tolist(),
                controller_request_receive_time_s=controller_times,
                controller_joint_deg=controller_q,
                sample_covariance=np.cov(values, rowvar=False).tolist()))
            if gripper_status_file:
                grip_after = gripper_snapshot(gripper_status_file)
                if abs(grip_after["position"] - grip_before["position"]) > 1:
                    raise ValueError("F/T 측정 중 그리퍼 개구가 변했습니다")
                for name, value in zip(rs.ARM_JOINT_NAMES, after):
                    plant.GetJointByName(name, scene["arm"]).set_angle(context, value)
                rotation = plant.GetBodyByName("ft_mount").body_frame().CalcPoseInWorld(
                    context).rotation().matrix()
                report["blocks"][-1].update(
                    achieved_g_hat=(rotation.T @ np.array([0., 0., -1.])).tolist(),
                    gripper_before=grip_before, gripper_after=grip_after)
            print("정지 구간 평균:", np.round(values.mean(axis=0), 5).tolist(), flush=True)
            report["record_elapsed_s"] = time.monotonic() - started
            if checkpoint:
                checkpoint(report)
    finally:
        report["record_elapsed_s"] = time.monotonic() - started
        stream.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--record", action="store_true", help="현재 자리에서 센서만 읽기")
    mode.add_argument("--run", action="store_true", help="검증된 작은 왕복 경로 실행 (빈 그리퍼)")
    ap.add_argument("--start-deg", nargs=6, type=float, help="오프라인 계획용 실제 시작 관절각")
    ap.add_argument("--joint", type=int, choices=(4, 5, 6), default=6)
    ap.add_argument("--delta-deg", type=float, default=0.)
    ap.add_argument("--empty-tool", action="store_true", help="이동/계획 시 물체 없는 상태 명시")
    ap.add_argument("--clearance-mm", type=float, default=1000 * rs.MIN_DISTANCE_M)
    ap.add_argument("--host", default="192.168.50.51")
    ap.add_argument("--meshpca-root", type=Path, default=Path("/home/cheon/MeshPCA-real-setup"))
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--duration-s", type=float, help="정지 기록 최소 경과시간. 마지막 구간까지 저장")
    ap.add_argument("--hz", type=float, default=50.)
    ap.add_argument("--settle-s", type=float, default=2.)
    ap.add_argument("--stable-force-span-n", type=float,
                    help="마지막 연속 구간의 3축 힘 평균 최대 거리 허용값 [N]")
    ap.add_argument("--stable-blocks", type=int, default=5)
    ap.add_argument("--label", default="unloaded", help="추/부착점 등 이번 측정 조건")
    ap.add_argument("--baseline", type=Path, help="같은 자세에서 기록한 무부하 JSON")
    ap.add_argument("--known-mass-kg", type=float, help="차분 검산용 질량. 물병은 0.541")
    ap.add_argument("--output", type=Path)
    ap.add_argument("--gripper-status-file", type=Path,
                    help="분리 실험용 창3 hardware_status.json 읽기 전용 경로")
    ap.add_argument("--ft-session-id", help="센서 영점·장착·연결이 유지되는 동안 같은 식별자")
    ap.add_argument("--load-state", choices=("empty", "loaded"))
    ap.add_argument("--configuration-id", help="같은 파지점·물체 관절각을 유지하는 기록 그룹")
    args = ap.parse_args()
    if args.duration_s is not None and (not args.record or not np.isfinite(args.duration_s)
                                       or args.duration_s <= 0):
        ap.error("--duration-s는 --record에서 양의 유한수만 허용합니다")
    metadata = [args.gripper_status_file, args.ft_session_id, args.load_state, args.configuration_id]
    if any(metadata) and (not all(metadata) or not args.record):
        ap.error("분리 실험은 --record와 gripper-status-file, ft-session-id, load-state, configuration-id 모두 필요합니다")
    if args.stable_force_span_n is not None and (
            not np.isfinite(args.stable_force_span_n) or args.stable_force_span_n <= 0
            or args.stable_blocks < 3 or args.blocks < args.stable_blocks):
        ap.error("안정화 허용값은 양의 유한수, 측정 구간 수는 안정화 구간 수(3 이상) 이상이어야 합니다")
    if (not np.isfinite(args.delta_deg) or abs(args.delta_deg) > 2
            or not np.isfinite(args.settle_s) or args.settle_s < 0
            or not np.isfinite(args.hz) or args.hz <= 0
            or args.samples < 2 or args.blocks < 1):
        ap.error("이동 ±2° 이내, 유한한 양의 측정 설정이 필요합니다")
    if args.record and args.delta_deg:
        ap.error("--record는 이동하지 않습니다. --delta-deg를 주지 마세요")
    if bool(args.baseline) != (args.known_mass_kg is not None) or (args.baseline and not args.record):
        ap.error("하중 비교는 --record --baseline 파일 --known-mass-kg 질량을 함께 사용하세요")
    baseline = json.loads(args.baseline.read_text()) if args.baseline else None
    if not args.record and not args.empty_tool:
        ap.error("빈 그리퍼의 충돌 모형입니다. --empty-tool로 상태를 명시하세요")
    if args.start_deg is not None and (args.run or args.record):
        ap.error("실물 실행에서는 --start-deg 대신 실제 관절각을 읽습니다")
    destination = args.output or ROOT / "my_work/outputs" / (
        "local_ft_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".json")
    if destination.exists() or destination.name.startswith("aft_tare_current"):
        ap.error("기존 결과나 정상 영점 파일을 덮어쓰지 않습니다")
    data = None
    if args.start_deg is None:
        import rbpodo
        data = rbpodo.CobotData(args.host)     # 읽기 전용 상태 채널
        start = read_q(data)
    else:
        start = np.deg2rad(args.start_deg)
    planner = None
    if args.record:
        report = dict(start_joint_deg=np.degrees(start).tolist(), delta_deg=0.)
    else:
        report, planner, scene, diagram, context = local_plan(
            start, args.joint, args.delta_deg, args.clearance_mm / 1000.)
    report.update(label=args.label, created_at_s=time.time(), status="planned",
                  calibration_valid=False, source="local_ft_check", robot_commands_sent=False)
    if args.gripper_status_file:
        report.update(wrench_frame="ft_mount", ft_session_id=args.ft_session_id,
                      load_state=args.load_state, configuration_id=args.configuration_id,
                      kinematics_sha256=hashlib.sha256(Path(rs.__file__).read_bytes()).hexdigest(),
                      gripper_status_file=str(args.gripper_status_file.resolve()))
    driver = None
    destination.parent.mkdir(parents=True, exist_ok=True)
    # 먼저 파일을 예약하여 읽기 실패/중단 시에도 이미 모은 자료를 보존한다.
    with destination.open("x") as output:
        try:
            if args.run or args.record:
                sys.path.insert(0, str(args.meshpca_root / "pivot"))
                import aft_tare
                sensor = aft_tare.Aft200Sensor(args.host, hz=args.hz)
                source = Path(aft_tare.__file__)
                report["sensor_module"] = str(source)
                report["sensor_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
                if args.run and args.delta_deg:
                    state = data.request_data(2.0)
                    if state is None:
                        raise TimeoutError("실행 전 RB5 상태 응답 없음")
                    status = state.sdata
                    if (status.collision_detect_onoff != 1 or status.is_freedrive_mode != 0
                            or status.real_vs_simulation_mode != 0 or status.init_state_info != 6):
                        raise hr.SafetyViolation("실물 구동/충돌감지/서보 상태가 준비되지 않았습니다")
                    current = read_q(data)
                    if np.max(np.abs(current - start)) > np.deg2rad(0.05):
                        raise hr.SafetyViolation("계획 후 시작 자세가 변했습니다")
                    target = np.deg2rad(report["route_joint_deg"][1])
                    if not planner.edge_valid(current, target):
                        raise hr.SafetyViolation("실행 직전 경로 검사 실패")
                    driver = hr.Rb5Driver(hr.RbpodoBackend(args.host), max_speed=np.deg2rad(1.))
                    driver.limit_to_start(start, np.deg2rad(report["motion_radius_deg"]))
                    driver.set_collision_planner(planner)
                    report["robot_commands_sent"] = True
                    driver.follow([target], 4.)
                time.sleep(args.settle_s)
                def checkpoint(value):
                    temporary = destination.with_suffix(".partial.json.tmp")
                    temporary.write_text(json.dumps(value, allow_nan=False))
                    temporary.replace(destination.with_suffix(".partial.json"))
                record(sensor, data, args.samples, args.blocks, report, args.gripper_status_file,
                       args.duration_s, checkpoint if args.duration_s else None)
                if args.stable_force_span_n is not None:
                    select_stable_tail(report, args.stable_blocks, args.stable_force_span_n)
                if baseline is not None:
                    report["load_difference"] = compare_load(report, baseline, args.known_mass_kg)
                    if all("controller_wrench_mean" in b for b in baseline["blocks"]):
                        report["controller_load_difference"] = compare_load(
                            report, baseline, args.known_mass_kg, "controller_wrench_mean")
                    report["baseline_file"] = str(args.baseline.resolve())
                if driver is not None:
                    current = read_q(data)
                    if not planner.edge_valid(current, start):
                        raise hr.SafetyViolation("현재 자세의 복귀 경로 검사 실패. 자동 복귀 안 함")
                    driver.follow([start], 4.)
                report["status"] = "recorded"
            print(json.dumps({k: v for k, v in report.items()
                              if k not in ("blocks", "all_recorded_blocks")}, indent=2))
        except BaseException as exc:
            report.update(status="failed", error=str(exc))
            raise
        finally:
            try:
                if driver is not None:
                    driver.stop()
            finally:
                json.dump(report, output, indent=2, allow_nan=False)
                output.write("\n")
                print("기록:", destination)


if __name__ == "__main__":
    main()
