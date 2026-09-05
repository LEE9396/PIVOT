"""영점 조정 자세와 실제 측정 자세가 같은지 확인한다.

왜 이게 중요한가
----------------
integration/meshpca/tare_real.py 의 첫머리가 이렇게 적고 있다.

    자세별 표는 그 효과를 모형화하지 않고 통째로 빼기 때문에 문제가 안 된다.
    **영점 조정과 측정이 같은 자세면** 그 4.35 N 이 양쪽에 똑같이 들어가
    상쇄된다.

맞는 말이다. 그런데 **전제가 성립하는지 아무도 확인하지 않는다.**

영점 조정 자세는 로봇이 그때 서 있던 자리에서 손목만 돌려 만든다.
탐색 자세는 물체를 잡은 채 IK 가 새로 푼 자세다. 중력 방향은 1 deg 안에서
같지만, **팔 관절각은 전혀 다른 값이 될 수 있다.**

케이블 장력·마운트 응력은 중력 방향이 아니라 **팔이 어떤 모양인가**에 따라
달라진다. 자세가 다르면 상쇄되지 않고 그대로 오차로 남는다.

이 도구는 두 자세를 나란히 놓고 얼마나 다른지 숫자로 보여 준다.

실행
    R=../robot_learning/scripts/run_drake_env.sh
    $R python tools/check_tare_poses.py --tare <aft_tare.json> --object desklamp
    $R python tools/check_tare_poses.py --tare <...> --theta-deg 24.4 71.4
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "my_work"))

import density_id_drake as alg                        # noqa: E402
import density_id_objects as obj                      # noqa: E402
import robot_scene as rs                              # noqa: E402

# 이보다 벌어지면 "같은 자세" 라고 볼 수 없다. 케이블이 굽는 정도가 달라진다.
JOINT_GAP_WARN_DEG = 60.0


def wrap_deg(a):
    return (np.asarray(a, dtype=float) + 180.0) % 360.0 - 180.0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tare", required=True, help="영점 조정 JSON")
    ap.add_argument("--object", default="desklamp")
    ap.add_argument("--grasp-part", default="link_3")
    ap.add_argument("--gripper", default="robotiq2f85")
    ap.add_argument("--theta-deg", type=float, nargs="+", default=None,
                    help="비교할 물체 관절각. 없으면 구동범위 가운데")
    args = ap.parse_args(argv)

    payload = json.loads(Path(args.tare).read_text())
    tare_q = {tuple(np.round(e["g_hat"], 6)): np.asarray(e["joint_deg"], float)
              for e in payload["entries"] if "joint_deg" in e}
    if not tare_q:
        print("영점 파일에 joint_deg 가 없습니다. 비교할 수 없습니다.")
        return 1

    if args.object == "desklamp":
        import desk_lamp
        spec = desk_lamp.build_spec(grasp_at=args.grasp_part)
    else:
        spec = obj.OBJECTS[args.object]
    rho = obj.bind_object(spec)
    limits = [j.limits_rad for j in spec.joints]
    theta = (np.deg2rad(args.theta_deg) if args.theta_deg
             else np.array([0.5 * (lo + hi) for lo, hi in limits]))

    checker = rs.PoseChecker(spec, densities=rho, joint_limits_rad=limits,
                             gripper=args.gripper)
    checker._last_solution = None
    index = [j.position_start() for j in checker.arm_joints]

    print(f"물체 {args.object}   관절각 {np.round(np.degrees(theta), 1)} deg")
    print("같은 중력 방향에서 영점 조정 자세와 탐색 자세가 얼마나 다른가\n")
    print(f"{'중력방향':>16}{'관절 차이 합':>14}{'최대 한 관절':>14}")
    print("-" * 46)

    worst = 0.0
    for g_hat in alg.G_DIRS:
        key = tuple(np.round(g_hat, 6))
        if key not in tare_q:
            print(f"{str(np.round(g_hat, 0)):>16}   영점 기록 없음")
            continue
        arm = checker.solve_robust(theta, g_hat)
        if arm is None:
            print(f"{str(np.round(g_hat, 0)):>16}   탐색 자세 IK 실패")
            continue
        gap = wrap_deg(np.degrees(np.asarray(arm)[index]) - tare_q[key])
        worst = max(worst, float(np.abs(gap).max()))
        print(f"{str(np.round(g_hat, 0)):>16}{np.abs(gap).sum():>13.0f}°"
              f"{np.abs(gap).max():>13.0f}°")

    print()
    if worst > JOINT_GAP_WARN_DEG:
        print(f"[경고] 최대 {worst:.0f} deg 벌어져 있습니다.")
        print("  영점을 잰 자세와 렌치를 읽는 자세가 전혀 다릅니다.")
        print("  케이블 장력은 팔 모양에 따라 달라지므로 **상쇄되지 않습니다.**")
        print("  tare_real.py 가 전제한 '같은 자세면 상쇄된다' 가 깨져 있습니다.\n")
        print("  할 수 있는 것 두 가지:")
        print("    1) 원인을 없앤다 — 케이블을 센서보다 **안쪽**(로봇 쪽)에서")
        print("       고정하고 바깥쪽에 여유를 줘서, 당기는 힘이 센서에 안 실리게.")
        print("       이게 진짜 해결이다.")
        print("    2) 크기를 잰다 — 물체를 안 잡은 채 탐색 자세들을 밟으며")
        print("       읽어 보고, 영점 값과 얼마나 다른지 확인한다. 그 차이가")
        print("       라운드마다 그대로 실리는 오차의 크기다.")
    else:
        print(f"[통과] 최대 {worst:.0f} deg. 자세가 충분히 비슷합니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
