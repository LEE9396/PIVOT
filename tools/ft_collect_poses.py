#!/usr/bin/env python3
"""직접교시로 자세를 많이 만들어 (관절각, 센서 기준 중력 방향, 렌치) 를 계속 모은다.

    PIVOT_WORKDIR=$PWD/my_work robot_learning/scripts/run_drake_env.sh \\
      env PYTHONPATH=$HOME/MeshPCA-real-setup/pivot python -u tools/ft_collect_poses.py \\
      --robot-ip 192.168.50.51 --out calibration/ft_poses_empty.jsonl --load-kg 0

로봇에 **명령을 보내지 않는다.** 작업자가 freedrive 로 팔을 옮기고 브레이크를 걸면
(freedrive OFF, 정지) 프로그램이 관절각과 렌치 N 표본을 읽어 한 줄(JSON)로 붙인다.
목표 방향은 없다 — 아무 자세나 되고, **골고루 다른 방향**일수록 회귀가 좋아진다.
빈 그리퍼면 --load-kg 0, 알려진 추를 물렸으면 그 질량(예: 1.332) 을 적는다.
같은 파일에 계속 이어 붙이므로 여러 번에 나눠 모아도 된다.

한 줄 형식:
  {"t": 초, "load_kg": 0, "joint_deg": [6], "g_hat": [3] (ft_mount 기준 FK),
   "wrench": [6] (레지스터x0.02, **보정 전**), "wrench_std": [6], "n": N, "freedrive_seen": bool}

회귀는 tools/tool_param_regression.py 가 한다.  Ctrl-C 로 끝낸다.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "my_work"))
sys.path.insert(0, str(ROOT / "integration" / "meshpca"))


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot-ip", default="192.168.50.51")
    ap.add_argument("--aft-hz", type=float, default=50.0)
    ap.add_argument("--samples", type=int, default=300)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--load-kg", type=float, default=0.0, help="그리퍼에 물린 알려진 질량 [kg], 빈 손이면 0")
    ap.add_argument("--min-move-deg", type=float, default=5.0,
                    help="직전 기록과 관절각 차이가 이보다 작으면 같은 자세로 보고 안 잰다")
    ap.add_argument("--poll-s", type=float, default=0.5)
    args = ap.parse_args(argv[1:])

    import tare_real as tr                       # require_ready / gravity_in_sensor / PoseChecker
    import robot_scene as scene
    from aft_tare import Aft200Sensor, stable_wrench

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
                q = tr.require_ready(data)                    # freedrive ON 이면 예외
            except RuntimeError as exc:
                if "freedrive" in str(exc):
                    seen_freedrive = True
                time.sleep(args.poll_s)
                continue
            if last_q is not None and np.max(np.abs(np.degrees(q - last_q))) < args.min_move_deg:
                time.sleep(args.poll_s)
                continue
            time.sleep(1.0)                                   # 정지 후 잔진동
            q2 = tr.require_ready(data)
            if np.max(np.abs(np.degrees(q2 - q))) > 0.1:
                continue                                      # 아직 움직이는 중
            g_hat = tr.gravity_in_sensor(checker, q2)
            raw = np.asarray(sensor.read_raw(args.samples), dtype=float)
            # read_raw 가 평균만 주면 표준편차는 못 구한다 — stream 이 있으면 표본으로 계산
            std = [None] * 6
            try:
                s = np.array([next(sensor.stream()) for _ in range(min(50, args.samples))])
                std = s.std(axis=0).tolist()
            except Exception:                                  # noqa: BLE001
                pass
            row = dict(t=time.time(), load_kg=args.load_kg,
                       joint_deg=np.degrees(q2).tolist(), g_hat=np.asarray(g_hat).tolist(),
                       wrench=raw.tolist(), wrench_std=std, n=args.samples,
                       freedrive_seen=seen_freedrive)
            with open(args.out, "a") as f:
                f.write(json.dumps(row) + "\n")
            count += 1; last_q, seen_freedrive = q2, False
            print(f"  [{n_prev + count:3d}] g={np.round(g_hat, 2).tolist()}  F={np.round(raw[:3], 2).tolist()} N"
                  f"  T={np.round(raw[3:], 3).tolist()}  관절 {np.round(np.degrees(q2), 1).tolist()}")
    except KeyboardInterrupt:
        print(f"\n종료. 이번에 {count}자세, 파일 총 {n_prev + count}줄.")


if __name__ == "__main__":
    main(sys.argv)
