#!/usr/bin/env python
"""EasyHeC 의 Tc_c2b.txt 를 PIVOT 캘리브레이션으로 옮긴다.

규약 대조 (문서 docs/rb5_xhand.md 와 실제 행렬로 확인함)
--------------------------------------------------------
EasyHeC 가 내는 것   `models/rb5_xhand/<name>/Tc_c2b.txt`
    p_cam = Tc_c2b · p_base        즉 Tc_c2b = X_CB (카메라 <- 베이스)
    base   = URDF link0 (베이스 플랜지 바닥 중심, +z 위)
    camera = D456 color 광학 프레임, **OpenCV** (x 오른쪽, y 아래, z 전방)

PIVOT 이 원하는 것   `calibration/camera_<id>.json` 의 `X_WC`
    월드 <- 카메라.  PIVOT 의 월드는 **로봇 베이스**다
    (robot_scene 이 link0 을 월드에 용접한다 — import_calibration.py 주석).
    Drake 카메라 규약도 z 전방·y 아래라 **OpenCV 와 같다** (look_at_pose 참고).

따라서 축을 뒤집을 것이 없다.  X_WC = inv(Tc_c2b)  하나면 끝난다.

검산 (2026-08-29 cap_20260829_1321 결과로 확인)
    inv(Tc_c2b) 의 위치  (0.262, -0.357, 0.531) m   문서와 0.3 mm 안에서 일치
    베이스에서 0.691 m,  광축 하향 37.7 deg          문서와 일치

주의 — 이 값이 아직 유효한가
----------------------------
핸드아이는 **카메라와 로봇 베이스의 상대 위치**다. 손끝에 무엇이 달렸는지와
무관하므로 XHand 로 잰 값을 Robotiq+AFT200 으로 바꾼 뒤에도 쓸 수 있다.
**단 카메라가 그 사이에 안 움직였어야 한다.** 그 저장소 문서에도 예전
ChArUco 값이 "카메라 이동으로 무효(렌더 0 px)" 가 된 기록이 있다.
그리퍼를 교체하면 로봇 근처에서 작업하게 되므로, 교체 뒤 반드시 확인하라
(--check-hint 참고).

실행
    $R python tools/import_easyhec.py \
        --input <EasyHeC>/models/rb5_xhand/cap_20260829_1321/Tc_c2b.txt \
        --intrinsics /tmp/lamp_live_intrinsics.json \
        --output calibration/camera_cam_d456_front.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "my_work"))

CHECK_HINT = """
카메라가 안 움직였는지 확인하는 법
  1) 그리퍼를 바꾼 뒤 EasyHeC 캡처 UI 로 3~4 장만 새로 찍는다
  2) render_robot.py 로 그 사진 위에 로봇을 렌더한다
  3) **link0~link6(팔)** 이 사진과 겹치는지 눈으로 본다
     - 팔이 맞으면 카메라는 안 움직인 것 -> 이 값을 그대로 쓴다
     - 손끝(XHand 자리)이 안 맞는 것은 당연하다. 그리퍼가 바뀌었으니까
  4) 팔이 어긋나면 캘리브레이션을 처음부터 다시 해야 한다
"""


def read_matrix(path):
    """Tc_c2b.txt 를 읽는다. 공백 구분 4x4, 또는 JSON."""
    path = Path(path)
    text = path.read_text()
    if path.suffix.lower() == ".json":
        data = json.loads(text)
        for key in ("Tc_c2b", "camera_from_base", "matrix"):
            if key in data:
                return np.asarray(data[key], dtype=float).reshape(4, 4), data
        raise KeyError(f"4x4 를 못 찾았습니다: {list(data)}")
    values = [float(t) for t in text.replace(",", " ")
              .replace("[", " ").replace("]", " ").split()]
    if len(values) != 16:
        raise ValueError(f"숫자가 16개가 아닙니다 ({len(values)}개): {path}")
    return np.asarray(values, dtype=float).reshape(4, 4), {}


def invert(matrix):
    out = np.eye(4)
    out[:3, :3] = matrix[:3, :3].T
    out[:3, 3] = -matrix[:3, :3].T @ matrix[:3, 3]
    return out


def check(matrix, name):
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError(f"{name} 의 마지막 행이 [0 0 0 1] 이 아닙니다")
    rotation = matrix[:3, :3]
    error = float(np.abs(rotation.T @ rotation - np.eye(3)).max())
    if error > 2e-3:
        raise ValueError(f"{name} 의 회전이 직교하지 않습니다 (오차 {error:.2e})")
    if abs(np.linalg.det(rotation) - 1.0) > 2e-3:
        raise ValueError(f"{name} 의 회전 행렬식이 1 이 아닙니다 — 좌우가"
                         f" 뒤집힌 값일 수 있습니다")
    return error


def read_intrinsics(path):
    if path is None:
        return None
    data = json.loads(Path(path).read_text())
    for key in ("depth_intrinsics", "intrinsics", "color_intrinsics"):
        if key in data:
            data = data[key]
            break
    return {k: float(data[k]) for k in ("fx", "fy", "cx", "cy") if k in data}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True, help="Tc_c2b.txt")
    ap.add_argument("--intrinsics", type=Path, default=None,
                    help="실제로 쓰는 스트림의 내부파라미터 JSON "
                         "(FoundationPose 가 쓰는 /tmp/..._intrinsics.json)")
    ap.add_argument("--camera-id", default=None)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--json-only", action="store_true",
                    help="PIVOT 에 바로 쓰지 않고 MeshPCA 형식 JSON 만 낸다")
    ap.add_argument("--check-hint", action="store_true")
    args = ap.parse_args()

    if args.check_hint:
        print(CHECK_HINT)
        return 0

    X_CB, extra = read_matrix(args.input)
    check(X_CB, "Tc_c2b")
    X_WC = invert(X_CB)                  # PIVOT 의 월드 = 로봇 베이스

    position = X_WC[:3, 3]
    optical = X_WC[:3, 2]                # OpenCV z = 전방
    print(f"입력  {args.input}")
    print(f"  Tc_c2b (카메라 <- 베이스) 직교성 오차 {check(X_CB, 'Tc_c2b'):.2e}")
    print(f"\n변환  X_WC = inv(Tc_c2b)   (월드=로봇 베이스 <- 카메라)")
    print(f"  카메라 위치   {np.round(position, 4)} m"
          f"   베이스에서 {np.linalg.norm(position):.3f} m")
    print(f"  광축          {np.round(optical, 3)}"
          f"   하향 {np.degrees(np.arcsin(-optical[2])):.1f} deg")
    if position[2] < 0.0:
        print("  [주의] 카메라가 베이스보다 **아래**에 있습니다 — 값을 의심하세요")

    intrinsics = read_intrinsics(args.intrinsics)
    if intrinsics:
        print(f"  내부파라미터  {intrinsics}")
    else:
        print("  [주의] 내부파라미터가 없습니다. EasyHeC 캘리브레이션 때와"
              " **다른 해상도**로 카메라를 돌리면 반드시 다시 읽어야 합니다"
              " (외부파라미터는 그대로 유효합니다).")

    payload = {
        "status": "valid",
        "camera_from_base": X_CB.tolist(),
        "base_from_camera": X_WC.tolist(),
        "intrinsics": intrinsics or {},
        "method": "EasyHeC differentiable rendering (rb5_xhand)",
        "source_file": str(args.input),
        "note": "XHand 로 잰 값. 그리퍼를 바꾼 뒤에도 카메라가 안 움직였다면 유효",
    }
    payload.update({k: v for k, v in extra.items()
                    if k not in payload and k != "Tc_c2b"})

    if args.json_only or args.output is None:
        target = args.output or Path("handeye_easyhec.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\n{target} 에 저장했습니다 (MeshPCA 형식).")
        print("PIVOT 에 넣으려면:")
        print(f"  $R python import_calibration.py --input {target}")
        return 0

    import import_calibration as ic
    converted = ic.convert(payload, args.camera_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(converted, indent=2) + "\n")
    print(f"\n{args.output} 에 저장했습니다 (PIVOT 형식).")
    print("  preflight.py 의 '핸드아이' 항목이 이제 통과합니다.")
    print("  이제 calibrate_table_rgbd.py 도 돌릴 수 있습니다 (핸드아이가"
          " 필요했습니다).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
