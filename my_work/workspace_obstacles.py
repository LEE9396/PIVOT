"""추가 고정 장애물의 실측 상자와 미확인 항목을 함께 관리한다."""

import hashlib
import json
from pathlib import Path
import re

import numpy as np

PATH = Path(__file__).resolve().parents[1] / "calibration/workspace_obstacles_current.json"


def load(path=None):
    path = PATH if path is None else Path(path)
    if not path.is_file():
        return dict(status="missing", boxes=[], unresolved=["workspace_obstacles"], sha256=None)
    raw = path.read_bytes()
    data = json.loads(raw)
    if (data.get("frame") != "robot_base" or data.get("units") != "m"
            or data.get("status") not in ("pending", "validated")
            or not isinstance(data.get("boxes"), list)
            or not isinstance(data.get("unresolved"), list)
            or not all(isinstance(s, str) and s for s in data["unresolved"])):
        raise ValueError("추가 장애물의 좌표계·단위·검증 상태가 올바르지 않습니다")
    names = set()
    for box in data["boxes"]:
        name = box.get("name", "")
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name) or name in names:
            raise ValueError("추가 장애물 이름은 중복 없는 소문자 식별자여야 합니다")
        names.add(name)
        for field in ("center_m", "size_m", "rpy_deg"):
            value = np.asarray(box[field], dtype=float)
            if value.shape != (3,) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name}: 유한한 {field} xyz가 필요합니다")
            box[field] = value.tolist()
        padding = float(box["padding_m"])
        if np.any(np.asarray(box["size_m"]) <= 0) or not np.isfinite(padding) or padding < 0:
            raise ValueError(f"{name}: 양의 크기와 음이 아닌 형상 여유가 필요합니다")
        box["padding_m"] = padding
    if data["status"] == "validated" and data["unresolved"]:
        raise ValueError("미확인 장애물이 남은 장면은 검증 완료일 수 없습니다")
    return dict(data, sha256=hashlib.sha256(raw).hexdigest())


def require_current_scene(plant):
    """미확인 형상 또는 이전 형상으로 만든 계획기로 실제 로봇을 움직이지 않는다."""
    data = load()
    if data["status"] != "validated":
        raise ValueError("추가 장애물 확인 미완료: " + ", ".join(data["unresolved"]))
    if plant is None or getattr(plant, "workspace_obstacles_sha256", None) != data["sha256"]:
        raise ValueError("추가 장애물 형상이 바뀌었습니다. 충돌 장면/경로를 다시 만드세요")
