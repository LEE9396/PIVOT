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
import json
import socket
from datetime import date

import numpy as np
from pydrake.math import RigidTransform, RollPitchYaw, RotationMatrix

import handeye as he
import robot_scene as rs


# ---------------------------------------------------------------------------
# 자동 캘리브레이션 (--run)
#
# 보정판을 그리퍼에 볼트로 붙이고, 로봇이 자세를 순회하며 두 가지를 기록한다.
#
#     X_WE   팔 tcp 의 월드 자세          <- 관절각 + Drake FK. 정확하다.
#     X_CT   카메라가 본 보정판 자세      <- 별도 프로세스의 검출기 (calib_detect.py)
#
# 그리고 handeye.solve_eye_to_hand 가 AX = XB 를 푼다. 자세한 것은 handeye.py.
#
# 왜 자세를 '아무거나' 뽑으면 안 되나
# ------------------------------------
# 세 조건을 동시에 만족해야 한다.
#   1. 부딪히지 않는다        (기존 최소거리 기준 + 보정판 몫)
#   2. 보정판이 카메라에 보인다 (화면 안 + 너무 비스듬하지 않게)
#   3. 회전축이 서로 많이 다르다
#
# 3번이 특히 중요한데 눈에 안 보인다. 축이 다 나란하면 AX = XB 가 특이해져
# 답이 안 나온다 (handeye.py 의 자기 검사가 그 경우를 재현해 둔다).
# ---------------------------------------------------------------------------
class DetectorClient:
    """calib_detect.py 에 "지금 찍어" 라고 물어보는 쪽."""

    def __init__(self, host, port, timeout_s=30.0):
        self.sock = socket.create_connection((host, port), timeout=timeout_s)
        self.reader = self.sock.makefile("r", encoding="utf-8")

    def ask(self, command="detect"):
        self.sock.sendall((json.dumps({"cmd": command}) + "\n").encode())
        line = self.reader.readline()
        if not line:
            raise ConnectionError("검출기가 연결을 끊었습니다")
        return json.loads(line)

    def close(self):
        try:
            self.ask("quit")
        except Exception:
            pass
        self.reader.close()
        self.sock.close()


class SimulatedDetector:
    """장비 없이 전 과정을 돌려보기 위한 가짜 검출기.

    '진짜' 카메라 자세와 판 부착 위치를 정해 놓고 X_CT 를 만들어 준다.
    즉 정답을 아는 상태라, 이 모드로 돌리면 **자세 몇 개면 충분한지**와
    **검출 잡음이 얼마면 몇 mm 로 떨어지는지**를 미리 알 수 있다.
    """

    def __init__(self, X_WC_true, X_ET_true, noise_mm=0.5, noise_deg=0.05,
                 seed=0):
        self.X_WC = X_WC_true
        self.X_ET = X_ET_true
        self.noise_mm = noise_mm
        self.noise_deg = noise_deg
        self.rng = np.random.default_rng(seed)
        self.X_WE = None

    def set_pose(self, X_WE):
        self.X_WE = X_WE

    def ask(self, command="detect"):
        X_CT = he.invert(self.X_WC) @ self.X_WE @ self.X_ET
        X_CT = X_CT @ he.transform(
            he.exp_so3(self.rng.normal(0.0, np.deg2rad(self.noise_deg), 3)),
            self.rng.normal(0.0, self.noise_mm * 1e-3, 3))
        return dict(ok=True, X_CT=X_CT.tolist(), corners=24,
                    rms_px=0.3, frames=10)

    def close(self):
        pass


def board_spec(args):
    """캘리브레이션 때 그리퍼에 물려 있는 것 = **보정판**. 물체가 아니다.

    처음에는 탐색에 쓰는 물체 사양을 그대로 썼는데, 무작위로 뽑은 자세
    3000개 중 부딪히지 않는 것이 **0개**였다. 당연하다 — 353 mm 짜리
    3-link 를 물린 채 관절공간을 헤집으면 거의 다 테이블을 뚫는다.

    캘리브레이션 장면의 페이로드는 보정판 하나짜리 판때기다. 그렇게 모형을
    실제와 맞추면 충돌 판정이 '보수적' 이 아니라 **맞는** 것이 된다.
    (물체로 대신 보면 판보다 클 때는 지나치게 막고, 작을 때는 위험하다)

    관절이 없는 페이로드다. 판은 볼트로 고정돼 있으므로 실제로 안 움직인다.
    """
    import density_id_objects as obj

    thickness_mm = args.board_thickness_mm
    size = (args.board_width_mm, args.board_height_mm, thickness_mm)
    part = obj.Part(
        name="calib_board",
        bbox_mm=size,
        volume_cm3=1e-3 * size[0] * size[1] * size[2],
        rho_gt=700.0,                     # 알루미늄판 + 종이. 값은 안 쓰인다.
        bbox_center_in_link_mm=(0.0, 0.0, 0.0),
        shell_centroid_in_link_mm=(0.0, 0.0, 0.0),
        color=(0.9, 0.9, 0.9, 1.0),
        grasp_width_mm=min(size[0], size[1], thickness_mm),
    )
    return obj.ObjectSpec(
        key="calib_board", label="캘리브레이션 보정판",
        parts=[part], joints=[],
        base_bbox_center_in_sensor_mm=(0.0, 0.0, 0.5 * thickness_mm),
        notes="손-눈 캘리브레이션용. 그리퍼에 볼트로 고정된 판.")


def _calibration_scene(args):
    """보정판을 물린 장면. 여유는 평소 기준 그대로 지킨다."""
    import density_id_objects as obj

    spec = board_spec(args)
    obj.set_measurement_averaging()
    rho = obj.bind_object(spec)
    checker = rs.PoseChecker(spec, densities=rho, joint_limits_rad=[],
                             min_distance_m=rs.MIN_DISTANCE_M,
                             gripper=args.gripper)
    return spec, checker


def _full_q(checker, arm_q, theta):
    """팔 6축 + 물체 관절각을 plant 전체 위치벡터로 만든다.

    checker 의 충돌 검사는 **전체 q** 를 받는다 (팔·그리퍼·물체가 한 plant 에
    같이 있다). 팔 6개만 넘기면 조용히 엉뚱한 관절을 쓴다.
    """
    q = checker.plant.GetPositions(checker.context).copy()
    for joint, value in zip(checker.arm_joints, np.asarray(arm_q).ravel()):
        q[joint.position_start()] = value
    for joint, value in zip(checker.object_joints, np.atleast_1d(theta)):
        q[joint.position_start()] = value
    for joint in checker.finger_joints:
        q[joint.position_start()] = checker.finger_value
    return q


def _tcp_pose(checker, arm_q, theta):
    """팔 관절각에서 tcp 의 월드 자세 4x4 (= X_WE)."""
    checker.plant.SetPositions(checker.context, _full_q(checker, arm_q, theta))
    X = checker.plant.GetFrameByName("tcp", checker.arm).CalcPoseInWorld(
        checker.context)
    return he.transform(X.rotation().matrix(), X.translation())


def _board_is_visible(X_WE, X_EB, camera, min_cos=0.5, margin_px=60):
    """보정판이 화면 안에 있고, 너무 비스듬하지 않은가.

    비스듬하면 코너가 뭉개져 자세가 흔들린다. 판의 법선과 시선이 이루는
    각이 60도(min_cos=0.5) 안쪽이어야 쓸 만하다.
    """
    X_WB = X_WE @ X_EB
    X_WC = rs.camera_pose(camera)
    R_CW = X_WC.rotation().matrix().T
    p_C = R_CW @ (X_WB[:3, 3] - X_WC.translation())
    if p_C[2] < 0.25 or p_C[2] > 2.5:            # 너무 가깝거나 멀다
        return False
    intr = camera["depth_intrinsics"]
    width, height = camera.get("resolution", (1280, 720))
    u = intr["fx"] * p_C[0] / p_C[2] + intr["cx"]
    v = intr["fy"] * p_C[1] / p_C[2] + intr["cy"]
    if not (margin_px <= u <= width - margin_px
            and margin_px <= v <= height - margin_px):
        return False
    # 판의 z 축(법선)이 카메라를 향하는가.
    normal_C = R_CW @ (X_WB[:3, :3] @ np.array([0.0, 0.0, 1.0]))
    return float(-normal_C @ (p_C / np.linalg.norm(p_C))) > min_cos


def generate_poses(checker, X_EB, n_want, camera, seed=0, n_try=4000):
    """조건 셋을 다 만족하는 자세를 고른다.

    회전축 다양성은 **탐욕적으로** 채운다 — 이미 고른 자세들과 가장 다른
    방향을 보는 것을 하나씩 더한다. 무작위로 뽑으면 팔이 비슷한 자리에
    몰려 축이 나란해지기 쉽다.
    """
    rng = np.random.default_rng(seed)
    theta0 = np.zeros(len(checker.object_joints))
    lower = np.array([j.position_lower_limits()[0] for j in checker.arm_joints])
    upper = np.array([j.position_upper_limits()[0] for j in checker.arm_joints])

    pool = []
    for _ in range(n_try):
        arm_q = lower + (upper - lower) * rng.random(len(lower))
        X_WE = _tcp_pose(checker, arm_q, theta0)
        if not _board_is_visible(X_WE, X_EB, camera):
            continue
        if not checker.arm_pose_is_clear(_full_q(checker, arm_q, theta0),
                                         theta0):
            continue
        pool.append((arm_q, X_WE))
        if len(pool) >= 25 * n_want:
            break
    if len(pool) < 2:
        raise RuntimeError(
            f"조건을 만족하는 자세를 {len(pool)}개밖에 못 찾았다. 보정판 부착 위치"
            " (--board-xyz/--board-rpy) 나 카메라 설정을 확인하라.")

    chosen = [pool[0]]
    while len(chosen) < min(n_want, len(pool)):
        def novelty(item):
            axes = [he.log_so3(item[1][:3, :3].T @ c[1][:3, :3])
                    for c in chosen]
            return min(float(np.linalg.norm(a)) for a in axes)
        best = max((p for p in pool if not any(p is c for c in chosen)),
                   key=novelty)
        if novelty(best) < np.deg2rad(8.0):
            break
        chosen.append(best)
    return chosen, len(pool)


def board_offset(args):
    """tcp 프레임에서 본 보정판 자세 (대략값이면 된다).

    정확할 필요가 없다 — 판이 정확히 어디 붙었는지는 X_ET 로 **함께 풀린다**.
    여기 값은 '그 자세에서 판이 카메라에 보이나' 를 가늠하는 데만 쓴다.
    """
    rotation = RotationMatrix(RollPitchYaw(np.deg2rad(args.board_rpy)))
    return he.transform(rotation.matrix(), np.array(args.board_xyz))


def run(args):
    spec, checker = _calibration_scene(args)
    camera = rs.load_camera(announce=True)
    X_EB = board_offset(args)

    print(f"\n자세를 고릅니다 (부딪히지 않고 + 보정판이 보이고 + 축이 다양하게)")
    chosen, n_pool = generate_poses(checker, X_EB, args.poses, camera,
                                    seed=args.seed)
    print(f"  후보 {n_pool}개 중 {len(chosen)}개 선정")
    spread = np.degrees(max(
        np.linalg.norm(he.log_so3(a[1][:3, :3].T @ b[1][:3, :3]))
        for a in chosen for b in chosen))
    print(f"  자세 사이 최대 회전차 {spread:.0f} deg"
          f"  ({'충분' if spread > 60 else '부족 — 답이 흔들릴 수 있음'})")
    if len(chosen) < args.poses:
        print(f"  [주의] {args.poses}개를 요청했는데 {len(chosen)}개뿐입니다.")

    if args.simulate:
        rng = np.random.default_rng(args.seed + 1)
        detector = SimulatedDetector(
            he.transform(rs.camera_pose(camera).rotation().matrix(),
                         rs.camera_pose(camera).translation()),
            he.transform(X_EB[:3, :3], X_EB[:3, 3] + rng.normal(0, 0.01, 3)),
            noise_mm=args.sim_noise_mm, noise_deg=args.sim_noise_deg,
            seed=args.seed)
        print(f"\n  --simulate: 장비 없이 돌립니다."
              f" 검출 잡음 {args.sim_noise_mm} mm / {args.sim_noise_deg} deg")
    else:
        detector = DetectorClient(args.detector_host, args.detector_port)
        print(f"\n  검출기 {args.detector_host}:{args.detector_port} 연결됨")
        print(f"  [안전] 로봇이 {len(chosen)}개 자세를 순회합니다."
              f" 작업영역에서 나오세요.")
        input("  준비되면 Enter: ")

    driver = None
    if not args.simulate and not args.dry_run:
        import dual_view as dv
        driver, *_ = dv.connect_hardware(args)

    X_WE_list, X_CT_list, rows = [], [], []
    for index, (arm_q, X_WE) in enumerate(chosen, start=1):
        if driver is not None:
            driver.follow([arm_q], args.move_duration)
            driver.stop()
            import time
            time.sleep(args.settle_s)
            # 실제로 간 자세로 FK 를 다시 한다. 명령값이 아니라 실측값이어야 한다.
            X_WE = _tcp_pose(checker, driver.joint_positions(),
                             np.zeros(len(checker.object_joints)))
        if isinstance(detector, SimulatedDetector):
            detector.set_pose(X_WE)
        reply = detector.ask()
        if not reply.get("ok"):
            print(f"  {index:>3}/{len(chosen)}  건너뜀 — {reply.get('reason')}")
            continue
        X_WE_list.append(X_WE)
        X_CT_list.append(np.array(reply["X_CT"], dtype=float))
        rows.append(reply)
        print(f"  {index:>3}/{len(chosen)}  코너 {reply.get('corners')}개"
              f"  재투영 {reply.get('rms_px', float('nan')):.3f} px")
    detector.close()

    if len(X_WE_list) < 5:
        raise SystemExit(f"쓸 수 있는 자세가 {len(X_WE_list)}개뿐이라 못 풉니다. "
                         "보정판이 카메라에 잘 보이는지 --preview 로 확인하세요.")

    out = he.solve_eye_to_hand(X_WE_list, X_CT_list)
    print(f"\n푼 결과 ({len(X_WE_list)}자세, 자세쌍 {out['n_pairs']}개)")
    print(f"  자기 일관성  {out['residual_mm']:.2f} mm rms"
          f" / {out['residual_deg']:.3f} deg rms"
          f"   (최악 {out['worst_mm']:.2f} mm)")
    print(f"  보정판 부착   tcp 에서 {1000*out['X_ET'][:3, 3]} mm")
    if out["residual_mm"] > 5.0:
        print("  [주의] 자기 일관성이 5 mm 를 넘습니다. 판이 흔들렸거나,"
              " 자세가 너무 비슷하거나, --square-mm 이 실제와 다릅니다.")

    X_WC = out["X_WC"]
    pose = RigidTransform(RotationMatrix(X_WC[:3, :3]), X_WC[:3, 3])
    if args.simulate or args.dry_run:
        print(f"\n  (--{'simulate' if args.simulate else 'dry-run'} 이라 "
              f"저장하지 않습니다)  위치 {np.round(X_WC[:3, 3], 4)} m")
        if args.simulate:
            truth = rs.camera_pose(camera).translation()
            print(f"  진짜 위치와의 차이 "
                  f"{1000*np.linalg.norm(X_WC[:3, 3] - truth):.2f} mm")
        return
    path = rs.save_calibration(
        pose, calibrated_at=date.today().isoformat(),
        method=f"ChArUco eye-to-hand, {len(X_WE_list)}자세",
        rms_px=float(np.mean([r.get("rms_px", 0.0) for r in rows])),
        residual_mm=out["residual_mm"])
    print(f"\n저장했습니다 -> {path}")
    import importlib
    importlib.reload(rs)
    show()


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
    # --- 자동 캘리브레이션 ---
    parser.add_argument("--run", action="store_true",
                        help="보정판을 붙이고 자세를 순회하며 자동으로 잰다")
    parser.add_argument("--simulate", action="store_true",
                        help="장비 없이 전 과정 확인 (정답을 알고 돌린다)")
    parser.add_argument("--dry-run", action="store_true",
                        help="자세만 고르고 로봇을 안 움직인다")
    parser.add_argument("--poses", type=int, default=20)
    parser.add_argument("--board-width-mm", type=float, default=245.0,
                        help="보정판 바깥 치수 (7칸 x 35 mm = 245)")
    parser.add_argument("--board-height-mm", type=float, default=175.0)
    parser.add_argument("--board-thickness-mm", type=float, default=6.0)
    parser.add_argument("--gripper", default="robotiq2f85",
                        choices=("pgc140", "robotiq2f85"))
    parser.add_argument("--detector-host", default="127.0.0.1")
    parser.add_argument("--detector-port", type=int, default=5566)
    parser.add_argument("--board-xyz", type=float, nargs=3,
                        default=(0.0, 0.0, 0.12),
                        help="tcp 에서 본 보정판 중심 [m]. 대략값이면 된다")
    parser.add_argument("--board-rpy", type=float, nargs=3,
                        default=(0.0, 0.0, 0.0),
                        help="tcp 에서 본 보정판 방향 [deg]. 대략값이면 된다")
    parser.add_argument("--board-margin-mm", type=float, default=30.0,
                        help="보정판이 물체보다 클 수 있으므로 더 두는 여유")
    parser.add_argument("--move-duration", type=float, default=8.0)
    parser.add_argument("--settle-s", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sim-noise-mm", type=float, default=0.5)
    parser.add_argument("--sim-noise-deg", type=float, default=0.05)
    parser.add_argument("--hardware", default="real", choices=("real", "sim"))
    args = parser.parse_args()

    if args.run:
        run(args)
        return

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
