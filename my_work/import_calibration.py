"""팀원(MeshPCA) 캘리브레이션 JSON 을 PIVOT 형식으로 옮긴다.

왜 어댑터가 필요한가
--------------------
두 저장소가 같은 것을 다른 이름으로 적는다.

    MeshPCA calibration/import_easyhec.py 가 내는 것
        {"camera_from_base": 4x4,      카메라 <- 베이스
         "base_from_camera": 4x4,      베이스 <- 카메라   <- 우리가 쓸 것
         "intrinsics": {fx, fy, cx, cy}}

    PIVOT calibration/camera_<id>.json 이 기대하는 것
        {"id": ..., "X_WC": 4x4,       월드 <- 카메라
         "depth_intrinsics": {fx, fy, cx, cy}}

PIVOT 의 월드는 **로봇 베이스**다 (robot_scene 이 link0 을 월드에 용접한다).
그래서 `base_from_camera` 가 곧 `X_WC` 다 — 행렬을 옮기고 이름만 바꾸면 된다.

이걸 안 하면 PIVOT 은 캘리브레이션 파일이 없다고 보고 **설정값(명목 위치)**
으로 돕는다. 그러면 각도 측정 자세를 실제와 다른 카메라 위치 기준으로 고르고,
카메라를 장애물로 보는 충돌 판정도 어긋난다. 그리고 그 오차는 라운드를 늘려도
안 없어지는 치우침이다 (calibration/README.md).

실행:
    cd ~/Desktop/PIVOT/my_work
    ../robot_learning/scripts/run_drake_env.sh python import_calibration.py \\
        --input ~/MeshPCA/calibration/handeye_d456.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

import robot_scene as rs


def convert(data, camera_id=None):
    """팀원 JSON dict 을 PIVOT 캘리브레이션 dict 으로."""
    if data.get("status") not in (None, "valid"):
        raise RuntimeError(f"캘리브레이션 status 가 '{data['status']}' 입니다"
                           f" — valid 가 아니면 쓰면 안 됩니다")
    if "base_from_camera" not in data:
        raise KeyError("base_from_camera 가 없습니다. import_easyhec.py 의"
                       " 출력이 맞는지 확인하세요")
    X_WC = np.asarray(data["base_from_camera"], dtype=float)
    if X_WC.shape != (4, 4):
        raise ValueError(f"base_from_camera 가 4x4 가 아닙니다: {X_WC.shape}")
    if not np.allclose(X_WC[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError("base_from_camera 가 동차변환이 아닙니다")
    rotation = X_WC[:3, :3]
    error = float(np.abs(rotation.T @ rotation - np.eye(3)).max())
    if error > 2e-3:
        raise ValueError(f"회전이 직교하지 않습니다 (오차 {error:.2e})")

    camera = data.get("camera", {})
    payload = {
        "id": camera_id or rs.CAMERA_ID,
        "X_WC": X_WC.tolist(),
        "method": data.get("method", "EasyHeC (MeshPCA)"),
        "source_file": data.get("source"),
    }
    if "intrinsics" in data:
        payload["depth_intrinsics"] = dict(data["intrinsics"])
    if camera.get("serial"):
        payload["serial"] = camera["serial"]
    if camera.get("model"):
        payload["model"] = camera["model"]
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True,
                        help="MeshPCA import_easyhec.py 가 낸 JSON")
    parser.add_argument("--camera-id", default=None,
                        help=f"기본 {rs.CAMERA_ID}")
    parser.add_argument("--out", type=Path, default=None,
                        help="기본은 PIVOT calibration/camera_<id>.json")
    parser.add_argument("--calibrated-at", default=None)
    args = parser.parse_args()

    payload = convert(json.loads(args.input.read_text()), args.camera_id)
    if args.calibrated_at:
        payload["calibrated_at"] = args.calibrated_at
    out = args.out or rs.calibration_path(payload["id"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    position = np.asarray(payload["X_WC"], dtype=float)[:3, 3]
    print(f"저장 -> {out}")
    print(f"  카메라 위치 (로봇 베이스 기준) {np.round(position, 3)} m")
    # 되읽어서 PIVOT 이 실제로 쓰는지 확인한다. 파일만 쓰고 끝내면
    # 이름이 틀려도 알 수 없다.
    camera = rs.load_camera(payload["id"], announce=True)
    used = rs.camera_pose(camera).translation()
    if not np.allclose(used, position, atol=1e-9):
        raise RuntimeError("되읽은 카메라 위치가 다릅니다 — 파일 이름을 확인하세요")
    print("  되읽기 확인 — PIVOT 이 이 값을 씁니다")


if __name__ == "__main__":
    main()
