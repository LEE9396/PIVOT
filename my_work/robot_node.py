"""실물 로봇 쪽 프로세스. 계획 쪽과는 TCP/IP 로만 이야기한다.

왜 프로세스를 나누나
--------------------
Drake 가 있는 파이썬과 로봇 드라이버(ROS1 등)가 있는 파이썬은 보통 다르다.
한 프로세스에 억지로 넣으면 환경이 섞여 예고 없이 깨진다. 나눠 두면
양쪽 다 깨끗하고, 로봇을 다른 PC 에 두는 것도 그대로 된다.

    [작업 PC]  dual_view.py --mode deploy --bus tcp
                  화면 [1] 계획·탐색
                  포트 5555 에서 로봇 쪽 연결을 기다린다
                        |
                        |  JSON 한 줄씩 (pose_bus.py 규약)
                        v
    [로봇 PC]  robot_node.py --host <작업PC> --port 5555
                  화면 [2] 작업자 UI
                  RB5 + AFT200 + FoundationPose 를 몬다

무엇을 주고받나
---------------
    받는다: {"round", "object_joint_deg", "arm_solutions", "reason"}
    보낸다: {"round", "object_joint_deg", "object_joint_deg_measured",
             "wrench", "aborted"}

arm_solutions 는 계획 쪽이 **이미 충돌까지 검사한** 팔 자세다. 여기서
다시 풀지 않는다. 다시 풀면 검사와 실행이 어긋난다.

실행 (모의 장비로 배선만 확인):
  ../robot_learning/scripts/run_drake_env.sh python robot_node.py \
      --host 127.0.0.1 --port 5555 --hardware sim
"""

import argparse

import numpy as np
from pydrake.geometry import StartMeshcat

import density_id_drake as alg
import density_id_objects as obj
import dual_view as dv
import robot_scene as rs
from pose_bus import TcpBus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True, help="계획 쪽 PC 의 주소")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--object", default="3link",
                    choices=tuple(obj.OBJECTS) + ("desklamp",))
    ap.add_argument("--joint-range-deg", type=float, nargs="+", default=None)
    ap.add_argument("--grasp", default="centroid",
                    choices=("centroid", "base_frame"))
    ap.add_argument("--gripper", default="robotiq2f85",
                    choices=("pgc140", "robotiq2f85"))
    ap.add_argument("--grasp-part", default="link_3",
                    choices=("link_1", "link_2", "link_3"))
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--hardware", choices=("real", "sim"), default="real")
    ap.add_argument("--move-duration", type=float,
                    default=dv.DEFAULT_MOVE_DURATION_S)
    ap.add_argument("--settle-s", type=float, default=0.5)
    ap.add_argument("--min-distance-mm", type=float, default=6.0)
    ap.add_argument("--plan-iters", type=int, default=20000)
    ap.add_argument("--samples-per-hold", type=int,
                    default=obj.DEFAULT_SAMPLES_PER_HOLD)
    ap.add_argument("--auto-adjust", action="store_true",
                    help="작업자 대신 각도를 자동으로 맞춘다 (검사용)")
    ap.add_argument("--autostart", action="store_true")
    args = ap.parse_args()

    if args.object == "desklamp":
        import desk_lamp as lamp
        spec = lamp.build_spec(grasp_at=args.grasp,
                               grasp_part=args.grasp_part)
    else:
        spec = obj.OBJECTS[args.object]
    limits = rs.parse_joint_range(spec, args.joint_range_deg)
    obj.set_measurement_averaging(args.samples_per_hold,
                                  obj.DEFAULT_BIAS_FRACTION)

    print(f"{spec.label}  (로봇 쪽 노드)")
    setup = dv.prepare(spec, None, limits, obj.DEFAULT_SAFETY, args.steps,
                       args.min_distance_mm * rs.MM, 1.0, prior="weight",
                       gripper=args.gripper)

    if args.hardware == "real":
        driver, wrench, pose, tare = dv.connect_hardware(args)
    else:
        print("  모의 장비 — 배선만 확인합니다.")
        driver, wrench, pose, tare = dv.simulated_hardware()

    console_meshcat = StartMeshcat()
    print(f"\n  작업자 UI 화면   {console_meshcat.web_url()}\n")

    robot = dv.RobotScreen(spec, setup, None,            # 로봇 화면 없음
                           console_meshcat=console_meshcat,
                           driver=driver, wrench_sensor=wrench,
                           pose_sensor=pose, tare=tare,
                           manual=not args.auto_adjust,
                           autostart=args.autostart,
                           move_duration_s=args.move_duration,
                           min_distance_m=args.min_distance_mm * rs.MM,
                           plan_iters=args.plan_iters,
                           settle_s=args.settle_s)
    robot.samples_per_hold = args.samples_per_hold

    bus = TcpBus.connect(args.host, args.port)
    robot.begin()
    try:
        while True:
            target = bus.recv_target()
            if target.get("finished"):
                break
            reply = robot.execute(target)
            bus.send_measurement(reply)
    except (ConnectionError, KeyboardInterrupt) as exc:
        print(f"\n[로봇] 연결 종료: {exc}")
    finally:
        robot.finish()
        bus.close()
    print("\n종료합니다.")


if __name__ == "__main__":
    main()
