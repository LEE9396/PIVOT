"""로봇과 카메라를 맞춘 결과를 파이프라인에 물린다.

왜 필요한가
-----------
각도를 읽는 자세(각 관절이 가장 잘 보이는 자세)는 **카메라가 어디서
보느냐**로 정해진다. 도면상의 명목 위치로 계산해 두고 실제 카메라가 20 cm
옆에 있으면, 애써 고른 자세가 실제로는 최적이 아니고 심하면 물체가 화면
밖으로 나간다. 카메라는 팔이 부딪힐 수 있는 **장애물**이기도 해서 충돌
판정도 같이 어긋난다.

그래서 캘리브레이션을 한 뒤에는 그 값을 파일로 남기고, 파이프라인이 그
파일을 읽게 한다. 파일이 있으면 자동으로 이긴다 (robot_scene.load_camera).

    calibration/camera_cam_d456_front.json

무엇을 넣나
-----------
X_WC — 월드(로봇 베이스) 기준 카메라 자세 4x4. Drake 카메라 규약은
       z 가 전방, y 가 아래다. OpenCV 규약과 같다.

손-눈 캘리브레이션 결과가 보통 이 형태로 나온다. ROS 의
easy_handeye, MoveIt 의 hand-eye, 또는 체커보드를 로봇에 들려 여러 자세에서
찍고 푸는 방식 모두 마찬가지다.

쓰는 법
-------
  # 4x4 행렬을 그대로 (행 우선 16개 숫자)
  python calibrate_camera.py --matrix 0.42 -0.90 ... --rms-px 0.31

  # 위치와 바라보는 점으로 (대충 맞출 때)
  python calibrate_camera.py --position 0.35 0.5 1.4 --look-at 0 -0.15 1.1

  # ROS tf 에서 받은 xyz + 쿼터니언 (x y z w)
  python calibrate_camera.py --position 0.35 0.5 1.4 --quat 0.1 0.2 0.3 0.9

  # 지금 무엇을 쓰고 있는지, 명목값과 얼마나 다른지 확인
  python calibrate_camera.py --show

확인할 것
---------
저장한 뒤 --show 로 명목값과의 차이를 본다. 차이가 크면 (10 cm 넘게) 실험실
설정 파일 자체가 낡았을 수 있으니 사람이 한 번 봐야 한다. 캘리브레이션이
틀리면 각도 측정이 통째로 틀어지는데, 그 오차는 라운드를 늘려도 안 없어진다.

실행:
  ../robot_learning/scripts/run_drake_env.sh python calibrate_camera.py --show
"""

import argparse
from datetime import date

import numpy as np
from pydrake.math import RigidTransform, RotationMatrix

import robot_scene as rs


def pose_from_args(args):
    if args.matrix:
        if len(args.matrix) != 16:
            raise SystemExit("--matrix 는 4x4 = 숫자 16개여야 한다")
        matrix = np.asarray(args.matrix, float).reshape(4, 4)
        return RigidTransform(RotationMatrix(matrix[:3, :3]), matrix[:3, 3])
    if args.position is None:
        raise SystemExit("--matrix 또는 --position 이 필요하다")
    position = np.asarray(args.position, float)
    if args.quat:
        from pydrake.common.eigen_geometry import Quaternion
        x, y, z, w = args.quat
        rotation = RotationMatrix(Quaternion(w=w, x=x, y=y, z=z))
        return RigidTransform(rotation, position)
    if args.look_at:
        return rs.look_at_pose(position, np.asarray(args.look_at, float))
    raise SystemExit("--quat 또는 --look-at 중 하나가 필요하다")


def show():
    camera = rs.load_camera(announce=False)
    pose = rs.camera_pose(camera)
    nominal = dict(next(c for c in rs._LAB["cameras"]
                        if c["id"] == rs.CAMERA_ID))
    nominal_pose = rs.look_at_pose(nominal["position_xyz_m"],
                                   nominal["look_at_xyz_m"])

    print(f"카메라 {camera['id']}")
    print(f"  출처       {camera['source']}")
    print(f"  파일 자리  {rs.calibration_path()}")
    print(f"  위치       {np.round(pose.translation(), 4)} m")
    print(f"  명목 위치  {np.round(nominal_pose.translation(), 4)} m")
    delta = pose.translation() - nominal_pose.translation()
    angle = (pose.rotation().inverse() @ nominal_pose.rotation()).ToAngleAxis()
    print(f"  차이       {1000 * np.linalg.norm(delta):.1f} mm,"
          f" {np.degrees(abs(angle.angle())):.2f} deg")
    if np.linalg.norm(delta) > 0.10:
        print("  [주의] 명목값과 10 cm 넘게 다릅니다. 실험실 설정 파일이"
              " 낡았는지 사람이 한 번 보세요.")
    intrinsics = camera["depth_intrinsics"]
    print(f"  내부파라미터 fx {intrinsics['fx']:.1f} fy {intrinsics['fy']:.1f}"
          f" cx {intrinsics['cx']:.1f} cy {intrinsics['cy']:.1f}")
    return camera


def compare_viewing_poses(object_key="3link"):
    """캘리브레이션 전후로 각도 측정 자세가 얼마나 달라지는지 보여 준다."""
    import density_id_objects as obj

    spec = obj.OBJECTS[object_key]
    obj.set_measurement_averaging()
    rho = obj.bind_object(spec)
    limits = rs.parse_joint_range(spec, None)
    checker = rs.PoseChecker(spec, densities=rho, joint_limits_rad=limits)
    scorer = rs.ViewScorer(spec, rho)
    theta = np.radians([90.0] * len(spec.joints))
    print(f"\n{spec.label} 의 각도 측정 자세 (관절각 90도 기준)")
    rs.find_viewing_poses(checker, theta, scorer=scorer, verbose=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true",
                        help="지금 쓰는 카메라 자세와 명목값의 차이")
    parser.add_argument("--matrix", type=float, nargs="*",
                        help="X_WC 4x4, 행 우선 숫자 16개")
    parser.add_argument("--position", type=float, nargs=3)
    parser.add_argument("--look-at", type=float, nargs=3)
    parser.add_argument("--quat", type=float, nargs=4,
                        help="쿼터니언 x y z w")
    parser.add_argument("--rms-px", type=float, default=None,
                        help="캘리브레이션 잔차 [px] — 기록용")
    parser.add_argument("--method", default="hand-eye",
                        help="어떻게 쟀는지 — 기록용")
    parser.add_argument("--check-poses", action="store_true",
                        help="이 카메라로 각도 측정 자세를 다시 계산해 본다")
    args = parser.parse_args()

    if args.show or not (args.matrix or args.position):
        show()
        if args.check_poses:
            compare_viewing_poses()
        return

    pose = pose_from_args(args)
    path = rs.save_calibration(
        pose, calibrated_at=date.today().isoformat(),
        method=args.method,
        **({"rms_px": args.rms_px} if args.rms_px is not None else {}))
    print(f"저장했습니다 -> {path}")
    print("이제 파이프라인이 이 값을 씁니다 (다시 실행하면 반영됩니다).\n")
    import importlib
    importlib.reload(rs)
    show()
    if args.check_poses:
        compare_viewing_poses()


if __name__ == "__main__":
    main()
