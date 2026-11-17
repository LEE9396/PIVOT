#!/usr/bin/env python3
"""직접교시로 자세를 많이 만들어 (관절각, 센서 기준 중력 방향, 렌치) 를 계속 모은다.

    PIVOT_WORKDIR=$PWD/my_work robot_learning/scripts/run_drake_env.sh \\
      env PYTHONPATH=$HOME/Research/1.PIVOT/code/MeshPCA/pivot python -u tools/ft_collect_poses.py \\
      --robot-ip 192.168.50.51 --out calibration/ft_poses_empty.jsonl --load-kg 0

로봇에 **명령을 보내지 않는다.** 작업자가 freedrive 로 팔을 옮기고 브레이크를 걸면
(freedrive OFF, 정지) 프로그램이 관절각과 렌치 N 표본을 읽어 한 줄(JSON)로 붙인다.
목표 방향은 없다 — 아무 자세나 되고, **골고루 다른 방향**일수록 회귀가 좋아진다.
빈 그리퍼면 --load-kg 0, 알려진 추를 물렸으면 그 질량(예: 1.332) 을 적는다.
같은 파일에 계속 이어 붙이므로 여러 번에 나눠 모아도 된다.
기본은 자세당 300표본(50 Hz에서 약 6초). 채팅 입력 없이 다음 정지 자세를 자동 기록한다.

한 줄 형식:
  {"t": 초, "load_kg": 0, "joint_deg": [6], "g_hat": [3] (ft_mount 기준 FK),
   "wrench": [6] (레지스터x0.02, **보정 전**), "wrench_std": [6], "n": N, "freedrive_seen": bool}

새 기록은 wrench_samples[N][6], sample_elapsed_s[N], joint_samples_deg[N][6],
state_elapsed_s[N], wrench_max_abs_deviation[6]도 저장한다. 표준편차와 최대 편차는
평균과 동일한 N표본 기준이다. F/T 직후 상태를 순차 확인하므로 두 읽기는 동시가 아니며,
실제 측정 시간은 record_elapsed_s에 기록한다. 상태/관절 검사 실패 시 저장하지 않고 멈춘다.

회귀는 tools/tool_param_regression.py 가 한다.  Ctrl-C 로 끝낸다.
"""

import argparse
import json
from contextlib import closing
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "my_work"))
sys.path.insert(0, str(ROOT / "integration" / "meshpca"))


def read_pose(sensor, read_ready, reference_q, samples, max_drift_deg):
    """같은 표본에서 평균·편차를 구하고, 상태 검사가 실패하면 기록하지 않는다."""
    values, times, joints, state_times = [], [], [], []
    started = time.monotonic()
    with closing(sensor.stream()) as stream:
        for _ in range(samples):
            value = np.asarray(next(stream), dtype=float)
            times.append(time.monotonic() - started)
            if value.shape != (6,) or not np.isfinite(value).all():
                raise ValueError("invalid F/T sample")
            q = np.asarray(read_ready(), dtype=float)
            state_times.append(time.monotonic() - started)
            if q.shape != (6,) or not np.isfinite(q).all():
                raise ValueError("invalid joint state")
            if np.max(np.abs(np.degrees(q - reference_q))) > max_drift_deg:
                raise RuntimeError("측정 중 관절 변화: 이번 자세는 저장하지 않습니다")
            values.append(value)
            joints.append(np.degrees(q).tolist())
    s = np.asarray(values)
    mean = s.mean(axis=0)
    return dict(wrench=mean.tolist(), wrench_std=s.std(axis=0).tolist(),
                wrench_max_abs_deviation=np.max(np.abs(s - mean), axis=0).tolist(),
                wrench_samples=s.tolist(), sample_elapsed_s=times,
                joint_samples_deg=joints, state_elapsed_s=state_times,
                record_elapsed_s=time.monotonic() - started,
                max_joint_drift_deg=max_drift_deg, n=len(values))


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot-ip", default="192.168.50.51")
    ap.add_argument("--aft-hz", type=float, default=50.0)
    ap.add_argument("--samples", type=int, default=300,
                    help="자세당 표본 수 (기본 300: 50 Hz에서 약 6초)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--load-kg", type=float, default=0.0, help="그리퍼에 물린 알려진 질량 [kg], 빈 손이면 0")
    ap.add_argument("--min-move-deg", type=float, default=5.0,
                    help="직전 기록과 관절각 차이가 이보다 작으면 같은 자세로 보고 안 잰다")
    ap.add_argument("--poll-s", type=float, default=0.5)
    ap.add_argument("--max-drift-deg", type=float, default=0.1)
    args = ap.parse_args(argv[1:])
    if args.samples < 2 or not all(np.isfinite(v) and v > 0 for v in
            (args.aft_hz, args.poll_s, args.min_move_deg, args.max_drift_deg)):
        ap.error("samples >= 2, frequency/interval/movement limits > 0 required")
    if not np.isfinite(args.load_kg) or args.load_kg < 0:
        ap.error("load-kg must be finite and nonnegative")

    import tare_real as tr                       # require_ready / gravity_in_sensor / PoseChecker
    import robot_scene as scene
    from aft_tare import Aft200Sensor

    rbpodo = __import__("rbpodo")
    data = rbpodo.CobotData(args.robot_ip)
    sensor = Aft200Sensor(args.robot_ip, args.aft_hz)      # 보정 전 원시 판독을 저장한다
    checker = scene.PoseChecker(tr.empty_tool_spec(), densities=[1000.0], joint_limits_rad=[],
                                min_distance_m=0.02, gripper="robotiq2f85", ik_restarts=1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_prev = sum(1 for _ in open(args.out)) if args.out.is_file() else 0
    print(f"기록 파일 {args.out} (기존 {n_prev}줄)   하중 {args.load_kg} kg")
    print("팔을 freedrive 로 옮기고 브레이크를 걸면 잽니다. 골고루 다른 방향으로. Ctrl-C 로 종료.")

    last_q, seen_freedrive, count = None, False, 0
    try:
        while True:
            try:
                q = tr.require_ready(data, require_real=True)
            except RuntimeError as exc:
                if str(exc) == "RB5 safety check failed: freedrive_off":
                    seen_freedrive = True
                else:
                    raise
                time.sleep(args.poll_s)
                continue
            if last_q is not None and np.max(np.abs(np.degrees(q - last_q))) < args.min_move_deg:
                time.sleep(args.poll_s)
                continue
            time.sleep(1.0)                                   # 정지 후 잔진동
            q2 = tr.require_ready(data, require_real=True)
            if np.max(np.abs(np.degrees(q2 - q))) > args.max_drift_deg:
                continue                                      # 아직 움직이는 중
            g_hat = tr.gravity_in_sensor(checker, q2)
            print("  측정 중: 기록 완료 표시까지 자세를 유지하세요.", flush=True)
            measurement = read_pose(sensor, lambda: tr.require_ready(data, require_real=True),
                                    q2, args.samples, args.max_drift_deg)
            raw = np.asarray(measurement["wrench"])
            row = dict(t=time.time(), load_kg=args.load_kg,
                       joint_deg=np.degrees(q2).tolist(), g_hat=np.asarray(g_hat).tolist(),
                       **measurement, freedrive_seen=seen_freedrive)
            with open(args.out, "a") as f:
                f.write(json.dumps(row, allow_nan=False) + "\n")
            count += 1; last_q, seen_freedrive = q2, False
            print(f"  [{n_prev + count:3d}] g={np.round(g_hat, 2).tolist()}  F={np.round(raw[:3], 2).tolist()} N"
                  f"  T={np.round(raw[3:], 3).tolist()}  관절 {np.round(np.degrees(q2), 1).tolist()}")
            print(f"    최대 |표본-평균|: {np.round(measurement['wrench_max_abs_deviation'], 4).tolist()}"
                  f"  기록 완료 ({measurement['record_elapsed_s']:.1f}초)", flush=True)
    except KeyboardInterrupt:
        print(f"\n종료. 이번에 {count}자세, 파일 총 {n_prev + count}줄.")


if __name__ == "__main__":
    main(sys.argv)
