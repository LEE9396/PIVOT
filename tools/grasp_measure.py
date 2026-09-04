#!/usr/bin/env python
"""파지 완료 시점에 **그리퍼 기준 물체 자세** 를 잰다 (용접 변환 X_G_O).

왜 필요한가
-----------
지금 시뮬레이터는 물체가 그리퍼의 **어디에** 물렸는지 모른다.
`desk_lamp.pinch_grasp()` 가 볼록 조각에서 추측한 자리에 용접해 놓고,
사람이 그 추측에 맞춰 물어 주기를 바란다. 그런데

  - 시뮬에서는 그 추측이 곧 정답이라 **절대 안 틀린다** (렌치도 같은
    용접에서 계산되므로 자기 일관적이다). 그래서 시뮬로는 못 잡는다.
  - 실물에서는 물체가 사람이 놓은 자리에 있다. 어긋나면 조용히 틀린다 —
    study_grasp.py 기준 **2 mm 에 밀도 오차 113 %**.
  - 2026-09-02 사고에서는 그 추측이 통째로 뒤집혀 head 가 F/T 마운트를
    41 mm 파고들었다.

측정하면 이 문제가 통째로 사라진다. 창 2 가 물체 자세를 주고, 핸드아이가
카메라 자세를 주고, 로봇이 자기 관절각을 준다. 세 개면 충분하다.

    X_G_O = X_W_G^-1 · X_W_C · X_C_O

      X_C_O  카메라 기준 물체   <- FoundationPose (창 2)
      X_W_C  월드 기준 카메라   <- calibration/camera_*.json (핸드아이)
      X_W_G  월드 기준 그리퍼   <- 로봇 q 로 정기구학

이 값을 `build_scene` 의 `WeldFrames` 에 그대로 넣으면
`GRASP_LONG_AXIS`, `grasp_rotation`, 볼록 조각 정점평균 같은 **추측이 전부
필요 없어진다.** 물체별 상수가 코드에 필요해진다면 파지를 측정하지 않고
있다는 신호다.

실행
    # 자기검사 (장비 불필요) — 아는 X_G_O 를 넣고 되찾아지는지 본다
    $R python tools/grasp_measure.py --self-test

    # 실물: 파지 완료 직후
    $R python tools/grasp_measure.py \
        --pose-file /tmp/lamp_foundationpose_live/latest.json \
        --robot-host 192.168.50.51 \
        --session my_work/sessions/session_20260902_1258
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "my_work"))


def to_matrix(value):
    """여러 표기를 4x4 동차행렬로. (행렬 / 위치+쿼터니언 / 위치+행렬)"""
    if isinstance(value, dict):
        if "matrix" in value:
            return np.asarray(value["matrix"], dtype=float).reshape(4, 4)
        position = np.asarray(value.get("position") or value.get("translation")
                              or value.get("xyz"), dtype=float)
        matrix = np.eye(4)
        matrix[:3, 3] = position
        if "quaternion" in value or "quat" in value:
            q = np.asarray(value.get("quaternion") or value["quat"], dtype=float)
            if len(q) != 4:
                raise ValueError("쿼터니언은 4개여야 합니다")
            # (w, x, y, z) 규약. FoundationPose 도 이 순서로 낸다.
            w, x, y, z = q / np.linalg.norm(q)
            matrix[:3, :3] = np.array([
                [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
                [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
                [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)]])
        elif "rotation" in value:
            matrix[:3, :3] = np.asarray(value["rotation"], dtype=float).reshape(3, 3)
        return matrix
    array = np.asarray(value, dtype=float)
    if array.shape == (4, 4):
        return array
    if array.size == 16:
        return array.reshape(4, 4)
    raise ValueError(f"자세를 못 읽었습니다: {value!r}")


def find_pose(data, key=None):
    """FoundationPose latest.json 에서 물체 자세를 찾는다."""
    if key:
        if key not in data:
            raise KeyError(f"{key} 가 없습니다. 있는 것: {list(data)}")
        return to_matrix(data[key])
    for name in ("X_CO", "pose", "object_pose", "pose_in_camera",
                 "T_cam_obj", "cam_T_obj"):
        if name in data:
            return to_matrix(data[name])
    raise KeyError(f"자세 키를 못 찾았습니다. --pose-key 로 알려 주세요."
                   f" 있는 것: {list(data)}")


def desk_lamp_body_pose(X_C_mesh, part):
    """FoundationPose 원본 mesh 자세를 PIVOT root body 자세로 바꾼다."""
    import desk_lamp as lamp

    link = next((name for name, label in lamp.FINAL_PART.items()
                 if label == part), None)
    if link is None:
        return X_C_mesh
    root = lamp.build_spec(grasp_part=link).parts[0]
    X_mesh_body = np.eye(4)
    X_mesh_body[:3, 3] = -np.asarray(root.mesh_offset_m, dtype=float)
    return X_C_mesh @ X_mesh_body


def invert(matrix):
    out = np.eye(4)
    out[:3, :3] = matrix[:3, :3].T
    out[:3, 3] = -matrix[:3, :3].T @ matrix[:3, 3]
    return out


def gripper_pose_in_world(q_rad, gripper="robotiq2f85"):
    """로봇 관절각 -> 월드 기준 그리퍼 베이스 자세 X_W_G.

    씬을 그대로 쓴다. 마운트 두께, 그리퍼 베이스 프레임 이름 같은 것을
    여기서 다시 적으면 robot_scene 과 어긋날 수 있다.
    """
    import robot_scene as rs
    import density_id_objects as obj

    spec = obj.OBJECTS["2link"]             # 팔 기구학만 쓰므로 물체는 아무거나
    scene = rs.build_scene(spec, None, None, include_visuals=False,
                           gripper=gripper)
    plant = scene["plant"]
    context = plant.CreateDefaultContext()
    joints = [plant.GetJointByName(n, scene["arm"]) for n in rs.ARM_JOINT_NAMES]
    for joint, value in zip(joints, np.atleast_1d(q_rad)):
        joint.set_angle(context, float(value))
    frame = plant.GetFrameByName(scene["gripper_spec"].base_frame,
                                 scene["gripper"])
    return frame.CalcPoseInWorld(context).GetAsMatrix4()


def measure(X_C_O, X_W_C, q_rad, gripper="robotiq2f85"):
    """X_G_O = X_W_G^-1 · X_W_C · X_C_O."""
    X_W_G = gripper_pose_in_world(q_rad, gripper)
    return invert(X_W_G) @ np.asarray(X_W_C, float) @ np.asarray(X_C_O, float), X_W_G


def fit_lamp_pose(data):
    """세 부품 중심을 함께 맞춘 카메라 기준 전체 램프 자세 X_C_O."""
    if not all(name in data.get("X_C_parts", {})
               for name in ("base", "support", "head")):
        return None
    import desk_lamp as lamp
    import density_id_objects as obj
    from dual_view import observed_to_model_deg

    spec = lamp.build_spec(grasp_at="pinch", grasp_part="link_3")
    plant, bodies = obj.build_plant(spec, np.ones(len(spec.parts)))
    context = plant.CreateDefaultContext()
    theta = np.deg2rad(observed_to_model_deg("desklamp", [
        data["base_support_deg"], data["support_head_deg"]]))
    plant.SetPositions(context, theta)
    model, observed = [], []
    for part in spec.parts:
        model.append(plant.EvalBodyPoseInWorld(
            context, bodies[part.name]).translation())
        X_C_M = np.asarray(data["X_C_parts"][lamp.FINAL_PART[part.name]],
                           dtype=float)
        center_M = -np.asarray(part.mesh_offset_m, dtype=float)
        observed.append(X_C_M[:3, :3] @ center_M + X_C_M[:3, 3])
    model, observed = np.asarray(model), np.asarray(observed)
    model_mean, observed_mean = model.mean(axis=0), observed.mean(axis=0)
    U, _, Vt = np.linalg.svd((model - model_mean).T
                             @ (observed - observed_mean))
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0.0:
        Vt[-1] *= -1.0
        R = Vt.T @ U.T
    X_C_O = np.eye(4)
    X_C_O[:3, :3] = R
    X_C_O[:3, 3] = observed_mean - R @ model_mean
    residual_mm = 1000.0 * np.sqrt(np.mean(np.sum(
        (model @ R.T + X_C_O[:3, 3] - observed) ** 2, axis=1)))
    return X_C_O, residual_mm


def sanity(X_G_O, tcp_z_m=0.147, max_offset_m=0.35):
    """말이 되는 값인가. 물체가 그리퍼에서 너무 멀면 뭔가 틀린 것이다."""
    offset = np.linalg.norm(X_G_O[:3, 3])
    notes = []
    if offset > max_offset_m:
        notes.append(f"물체 원점이 그리퍼에서 {1000*offset:.0f} mm 떨어져 있습니다"
                     f" — 핸드아이나 자세 키를 의심하세요")
    z = X_G_O[2, 3]
    if z < 0.0:
        notes.append(f"물체가 그리퍼 **뒤쪽**(z={1000*z:.0f} mm)에 있습니다"
                     f" — 마운트·손목 쪽입니다. 부호를 확인하세요")
    elif z < 0.5 * tcp_z_m:
        notes.append(f"물체가 죠 안쪽 깊숙이(z={1000*z:.0f} mm) 있습니다")
    return notes


def self_test(seed=0):
    """아는 X_G_O 를 넣어 만든 관측에서 그대로 되찾아지는지."""
    rng = np.random.default_rng(seed)
    q = np.deg2rad([10.0, -40.0, 95.0, -55.0, -90.0, 15.0])
    X_W_G = gripper_pose_in_world(q)
    true = np.eye(4)
    true[:3, 3] = [0.004, -0.002, 0.150]
    angle = np.deg2rad(12.0)
    true[:3, :3] = np.array([[np.cos(angle), -np.sin(angle), 0],
                             [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    X_W_C = np.eye(4)
    X_W_C[:3, 3] = [0.9, -0.3, 1.2]
    X_W_C[:3, :3] = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    if np.linalg.det(X_W_C[:3, :3]) < 0:
        X_W_C[:, 0] *= -1
    X_C_O = invert(X_W_C) @ X_W_G @ true             # 관측을 만든다
    got, _ = measure(X_C_O, X_W_C, q)
    position_mm = 1000 * np.linalg.norm(got[:3, 3] - true[:3, 3])
    rotation_deg = np.degrees(np.arccos(np.clip(
        (np.trace(got[:3, :3].T @ true[:3, :3]) - 1) / 2, -1, 1)))
    import desk_lamp as lamp
    import density_id_objects as obj
    from dual_view import observed_to_model_deg
    angles = [90.0, 100.0]
    spec = lamp.build_spec()
    plant, bodies = obj.build_plant(spec, np.ones(len(spec.parts)))
    context = plant.CreateDefaultContext()
    plant.SetPositions(context, np.deg2rad(
        observed_to_model_deg("desklamp", angles)))
    data = {"base_support_deg": angles[0], "support_head_deg": angles[1],
            "X_C_parts": {}}
    for part in spec.parts:
        X_B_M = np.eye(4)
        X_B_M[:3, 3] = part.mesh_offset_m
        data["X_C_parts"][lamp.FINAL_PART[part.name]] = (
            true @ plant.EvalBodyPoseInWorld(
                context, bodies[part.name]).GetAsMatrix4() @ X_B_M).tolist()
    fitted, fit_rms_mm = fit_lamp_pose(data)
    fit_position_mm = 1000 * np.linalg.norm(fitted[:3, 3] - true[:3, 3])
    fit_rotation_deg = np.degrees(np.arccos(np.clip(
        (np.trace(fitted[:3, :3].T @ true[:3, :3]) - 1) / 2, -1, 1)))
    # 부동소수점 잡음만 남아야 한다. arccos 는 1 근처에서 sqrt 만큼 부풀므로
    # 각도 문턱을 위치보다 느슨하게 둔다 (그래도 물리적으로는 무한히 엄격하다).
    ok = (position_mm < 1e-6 and rotation_deg < 1e-3
          and fit_position_mm < 1e-6 and fit_rotation_deg < 1e-3
          and fit_rms_mm < 1e-6)
    print(f"자기검사  위치 오차 {position_mm:.3e} mm,"
          f" 자세 오차 {rotation_deg:.3e} deg")
    print(f"  전체 정합 위치 {fit_position_mm:.3e} mm,"
          f" 자세 {fit_rotation_deg:.3e} deg, rms {fit_rms_mm:.3e} mm")
    print("  " + ("통과 — 변환 순서가 맞습니다" if ok
                  else "**실패** — 변환 순서를 확인하세요"))
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--pose-file", type=Path)
    ap.add_argument("--pose-key", default=None)
    ap.add_argument("--camera-calibration", type=Path, default=None)
    ap.add_argument("--robot-host", default=None)
    ap.add_argument("--joints-deg", default=None,
                    help="로봇 대신 직접 관절각을 준다 (쉼표 6개)")
    ap.add_argument("--gripper", default="robotiq2f85")
    ap.add_argument("--session", type=Path, default=None)
    ap.add_argument("--part", default="support", help="잡은 부위 이름")
    ap.add_argument("--opening-mm", type=float, default=None)
    args = ap.parse_args()

    if args.self_test:
        sys.exit(0 if self_test() else 1)
    if args.pose_file is None:
        sys.exit("--pose-file 이 필요합니다 (또는 --self-test)")

    import robot_scene as rs

    camera = rs.load_camera(announce=True)
    if "X_WC" not in camera and args.camera_calibration is None:
        print("[주의] 핸드아이 캘리브레이션이 없습니다 — **명목 카메라 위치**로"
              " 계산합니다. 이 값은 실험에 쓰면 안 됩니다.")
    X_W_C = rs.camera_pose(camera).GetAsMatrix4()

    data = json.loads(args.pose_file.read_text())
    fitted = fit_lamp_pose(data) if args.part == "support" else None
    if fitted is None:
        X_C_O = desk_lamp_body_pose(find_pose(data, args.pose_key), args.part)
        orientation_correction = None
    else:
        X_C_O, residual_mm = fitted
        orientation_correction = "base/support/head rigid fit"
        print(f"  전체 램프 정합 rms {residual_mm:.1f} mm")

    if args.joints_deg:
        q = np.deg2rad([float(v) for v in args.joints_deg.split(",")])
    elif args.robot_host:
        import hardware_real as hr
        backend = hr.RbpodoBackend(args.robot_host)
        q = np.asarray(backend.joint_positions(), dtype=float)
    else:
        sys.exit("--robot-host 또는 --joints-deg 가 필요합니다")
    print(f"  로봇 관절각 {np.round(np.degrees(q), 1)}")

    X_G_O, X_W_G = measure(X_C_O, X_W_C, q, args.gripper)
    print("\n그리퍼 기준 물체 자세 X_G_O")
    print(f"  위치 [mm] {np.round(1000 * X_G_O[:3, 3], 1)}")
    for row in X_G_O[:3, :3]:
        print(f"  회전     {np.round(row, 4)}")
    notes = sanity(X_G_O)
    for note in notes:
        print(f"  [주의] {note}")
    if notes:
        sys.exit("파지 변환 검산 실패 — 저장하지 않습니다. FoundationPose를"
                 " 다시 초기화하고 핸드아이·로봇 FK 정합을 확인하세요.")

    if args.session:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from pivot_session import Session

        session = Session(args.session, create=False)
        session.write("grasp.json", dict(
            X_G_O=X_G_O.tolist(), X_W_G=X_W_G.tolist(), X_W_C=X_W_C.tolist(),
            X_C_O=X_C_O.tolist(), joints_rad=q.tolist(), part=args.part,
            opening_mm=args.opening_mm, gripper=args.gripper,
            source="measured", camera_source=camera.get("source"),
            orientation_correction=orientation_correction))
        print(f"\n{session.path('grasp.json')} 에 저장했습니다.")
        print("  build_scene 이 이 값을 WeldFrames 에 그대로 씁니다 —")
        print("  GRASP_LONG_AXIS / grasp_rotation 추측은 더 안 씁니다.")


if __name__ == "__main__":
    main()
