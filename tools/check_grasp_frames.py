"""두 가상환경이 물체를 같은 자리에 두고 있는지 확인한다.

왜 이게 필요한가
----------------
이 저장소에는 물체를 붙여 놓은 가상환경이 두 벌 있다.

    robot_scene            충돌·IK·경로·카메라 시야를 본다.
                           tools/grasp_measure.py 가 **실측한** 파지를 쓴다.
    density_id_objects     회귀행렬의 모멘트팔을 만든다.
                           자산에 적어둔 **짐작값**을 쓰고 회전이 없다.

둘이 어긋나면 로봇은 맞게 움직이는데 밀도만 틀린다. 그리고 시뮬레이션에서는
렌치도 같은 plant 에서 나오므로 **절대 안 드러난다.**

여기서 두 가지를 잰다.

  1) 토크 기준점    회귀행렬이 토크를 재는 점과 AFT200 이 실제로 재는 점이
                    얼마나 떨어져 있나. 이 거리 r 은 렌치를 옮기지 않고 쓰면
                    |r x F| 만큼의 토크 오차가 된다.
  2) 파지 불일치    실측 파지와 자산의 짐작이 얼마나 다른가 (위치, 회전).

실행
    R=../robot_learning/scripts/run_drake_env.sh
    $R python tools/check_grasp_frames.py --object desklamp
    $R python tools/check_grasp_frames.py --object desklamp --grasp <grasp.json>
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "my_work"))

import density_id_objects as obj                      # noqa: E402
import robot_scene as rs                              # noqa: E402


def pose_text(matrix):
    """4x4 를 "위치 mm / 회전 deg" 로 적는다."""
    from pydrake.math import RigidTransform, RotationMatrix
    X = RigidTransform(RotationMatrix(matrix[:3, :3]), matrix[:3, 3])
    rpy = np.degrees(X.rotation().ToRollPitchYaw().vector())
    return (f"위치 {np.round(1000 * X.translation(), 1)} mm"
            f"   회전 {np.round(rpy, 1)} deg")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", default="desklamp")
    ap.add_argument("--grasp", default=None,
                    help="grasp.json 경로. 없으면 세션 기본 위치에서 찾는다")
    ap.add_argument("--gripper", default="robotiq2f85")
    ap.add_argument("--grasp-part", default="link_3",
                    help="desklamp 전용: 어느 부위를 잡는가")
    ap.add_argument("--mass-kg", type=float, default=None,
                    help="물체 총질량. 안 주면 자산 GT 를 쓴다")
    args = ap.parse_args(argv)

    if args.object == "desklamp":
        import desk_lamp
        spec = desk_lamp.build_spec(grasp_at=args.grasp_part)
    else:
        spec = obj.OBJECTS[args.object]
    rho_gt = obj.bind_object(spec)
    mass = (args.mass_kg if args.mass_kg is not None
            else float(obj.assembled_mass_kg(spec, rho_gt)))

    measured = rs.load_measured_grasp(
        np.array(json.loads(Path(args.grasp).read_text())["X_G_O"])
        if args.grasp else None)

    print(f"물체 {args.object}   총질량 {1000*mass:.1f} g"
          f"   파지 부위 {spec.parts[0].name}")
    print(f"실측 파지 X_G_O: {'있음' if measured is not None else '없음 (짐작으로 감)'}\n")

    # 1) 토크 기준점 -----------------------------------------------------
    X_model = rs.sensor_object_transform(spec, args.gripper, measured, "model")
    X_aft = rs.sensor_object_transform(spec, args.gripper, measured, "aft")
    # 두 기준점 사이의 거리 = 같은 물체점을 양쪽에서 봤을 때의 차이
    lever = float(np.linalg.norm(X_aft[:3, 3] - X_model[:3, 3]))
    torque_err = lever * mass * 9.81

    print("1) 토크를 어느 점 기준으로 재는가")
    print(f"   회귀행렬 (obj_sensor) : {pose_text(X_model)}")
    print(f"   AFT200 몸체 원점      : {pose_text(X_aft)}")
    print(f"   두 점 사이 거리       : {1000*lever:.1f} mm")
    print(f"   -> 렌치를 옮기지 않으면 토크 오차 최대 {torque_err:.3f} N·m"
          f"  (센서 잡음 0.003 의 {torque_err/0.003:.0f} 배)")
    if 1000 * lever > 50.0:
        print(f"   [경고] 파지 오프셋 미지수의 허용 범위는 +-50 mm 뿐입니다."
              f" {1000*lever:.0f} mm 는 흡수되지 않고 경계에 붙습니다.")

    # 2) 실측 파지와 짐작의 차이 ----------------------------------------
    print("\n2) 실측 파지와 자산의 짐작이 얼마나 다른가")
    if measured is None:
        print("   실측값이 없어 비교할 수 없습니다."
              " tools/grasp_measure.py 를 먼저 돌리세요.")
    else:
        guess = rs.sensor_object_transform(spec, args.gripper, None, "aft")
        d = np.linalg.inv(guess) @ X_aft
        from pydrake.math import RotationMatrix
        angle = np.degrees(RotationMatrix(d[:3, :3]).ToAngleAxis().angle())
        print(f"   위치 차이 {np.round(1000*d[:3, 3], 1)} mm"
              f"  (크기 {1000*np.linalg.norm(d[:3, 3]):.1f} mm)")
        print(f"   회전 차이 {angle:.2f} deg")
        print(f"   -> 회귀행렬은 이 차이를 모릅니다."
              f" build_plant(X_sensor_object=...) 로 넘겨야 반영됩니다.")
        if angle > 30.0:
            print(f"   [경고] 회전 차이 {angle:.0f} deg 는 사람이 손으로 만들 수"
                  f" 있는 오차가 아닙니다.\n"
                  "          FoundationPose 가 쓰는 메시 좌표계와 자산(spec)의"
                  " 링크 좌표계가\n"
                  "          서로 다를 가능성이 큽니다. **이 상태로 실측값을"
                  " 회귀행렬에 넣으면\n"
                  "          지금보다 나빠집니다.** 두 좌표계 규약을 먼저"
                  " 맞추세요:\n"
                  "            - grasp_measure.py 의 X_C_O 가 어느 메시"
                  " 원점 기준인지\n"
                  "            - density_id_objects 의 part body frame 이"
                  " 어디인지 (bbox 중심)\n"
                  "          알려진 자세로 놓고 두 값이 같게 나오는지"
                  " 대조하는 것이 확인 방법입니다.")

    print("\n3) 지금 회귀행렬이 실제로 쓰는 값")
    print(f"   spec.base_bbox_center_in_sensor_mm ="
          f" {spec.base_bbox_center_in_sensor_mm}  (회전 없음)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
