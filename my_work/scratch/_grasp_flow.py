"""수동 파지 흐름을 검증한다 — 실제 팔 자세를 읽는가, 파지가 이동보다 먼저인가.

--hardware sim 은 driver 를 주지 않아서(dual_view.simulated_hardware) 이 경로를
안 밟는다. 그래서 가짜 driver 를 직접 물려 확인한다.
"""

import sys as _sys, pathlib as _pathlib
# 이 폴더는 my_work 밖이라 형제 모듈이 안 보인다. run_drake_env.sh 가
# PYTHONPATH 를 지우므로 (ROS 오염 제거) 환경변수로는 못 넣는다.
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import density_id_objects as obj
import dual_view as dv
import robot_scene as rs
import hardware as hw

MM = 1e-3


class FakeDriver(hw.RobotDriver):
    """전원을 켠 자리가 시작 자세와 **다른** 실물을 흉내낸다."""

    def __init__(self, actual_q):
        self.actual = np.asarray(actual_q, float)
        self.log = []

    def joint_positions(self):
        self.log.append("joint_positions")
        return self.actual.copy()

    def follow(self, waypoints, duration_s):
        self.log.append(f"follow({len(waypoints)}점)")
        self.actual = np.asarray(waypoints[-1], float)

    def stop(self):
        self.log.append("stop")

    def servo_off(self):
        self.log.append("servo_off")
        # 서보가 꺼지면 중력으로 조금 처진다.
        self.actual = self.actual + np.deg2rad(0.4)

    def servo_on(self):
        self.log.append("servo_on")


def main():
    spec = obj.OBJECTS["3link"]
    limits = rs.parse_joint_range(spec, [0.0, 180.0])
    setup = dv.prepare(spec, None, limits, obj.DEFAULT_SAFETY, 3,
                       rs.MIN_DISTANCE_M, 1.0, prior="weight",
                       gripper="robotiq2f85", view_poses=False)

    start_q = np.array(setup["start_q"])
    from pydrake.geometry import StartMeshcat
    console = StartMeshcat()

    # 실물은 시작 자세에서 30도 떨어진 곳에 서 있다.
    robot = dv.RobotScreen(spec, setup, None, console_meshcat=console,
                           driver=None, manual=False, autostart=True,
                           move_duration_s=0.3, check_motion=False)
    arm_index = [j.position_start() for j in robot.arm_joints]
    # 시작 자세에서 떨어져 있되 **충돌하지는 않는** 자리를 고른다.
    # 아무 방향으로나 30도 밀면 그리퍼가 테이블에 박힌다 (실제로 그랬다).
    away = None
    for delta in np.deg2rad([25.0, 15.0, 10.0, 6.0, 3.0]):
        for axis in range(6):
            trial = start_q[arm_index].copy()
            trial[axis] += delta
            q = np.array(setup["start_q"]).copy()
            for i, v in zip(arm_index, trial):
                q[i] = v
            if robot.setup["checker"].arm_pose_is_clear(q, np.zeros(len(spec.joints))):
                away = trial
                break
        if away is not None:
            print(f"  (전원 켠 자리로 관절 {axis} 를 "
                  f"{np.degrees(delta):.0f} deg 민 자세를 쓴다)")
            break
    driver = FakeDriver(away)
    robot.driver = driver
    robot.manual = True
    robot.console.auto = True          # 버튼 대기 없이

    print("=== 1) sync_from_robot 이 실제 자세를 읽는가 ===")
    believed_before = np.array([robot.q[i] for i in arm_index])
    robot.begin()
    believed_after = np.array([robot.q[i] for i in arm_index])
    gap_before = np.degrees(np.abs(believed_before - away)).max()
    gap_after = np.degrees(np.abs(believed_after - driver.actual)).max()
    print(f"  begin() 전  : 내부 믿음과 실제 팔이 {gap_before:.1f} deg 다름")
    print(f"  begin() 후  : {gap_after:.3f} deg 다름  "
          f"-> {'통과' if gap_after < 0.01 else '실패'}")

    print("\n=== 2) 파지 단계가 순서대로 불렸는가 ===")
    print(f"  driver 호출 순서: {driver.log}")
    order = [c for c in driver.log if c in ("stop", "servo_off", "servo_on")]
    expect = ["stop", "servo_off", "servo_on"]
    print(f"  기대 {expect}\n  실제 {order}  "
          f"-> {'통과' if order == expect else '실패'}")

    print("\n=== 3) 서보 오프 중 처진 것을 다시 읽었는가 ===")
    # servo_off 가 0.4도 처지게 만들었다. begin() 후 믿음이 그것까지 반영해야 한다.
    print(f"  실제 팔 {np.round(np.degrees(driver.actual), 2)}")
    print(f"  믿는 값 {np.round(np.degrees(believed_after), 2)}")
    print(f"  -> {'통과' if gap_after < 0.01 else '실패'}")

    print("\n=== 4) 파지가 이동보다 먼저인가 ===")
    first_follow = next((i for i, c in enumerate(driver.log)
                         if c.startswith("follow")), None)
    first_servo_on = driver.log.index("servo_on")
    print(f"  servo_on 위치 {first_servo_on}, 첫 follow 위치 {first_follow}")
    ok = first_follow is None or first_follow > first_servo_on
    print(f"  -> {'통과 (begin 동안 팔을 안 움직였다)' if ok else '실패'}")

    print("\n=== 5) 그 다음 execute 가 시작 자세로 계획해 가는가 ===")
    before = len(driver.log)
    moved = robot.move_to(setup["start_q"], "시작 자세")
    print(f"  move_to 반환 {moved}, driver 호출 {driver.log[before:]}")
    reached = np.degrees(np.abs(driver.actual - start_q[arm_index])).max()
    print(f"  도착 오차 {reached:.3f} deg -> {'통과' if reached < 0.5 else '실패'}")


if __name__ == "__main__":
    main()
