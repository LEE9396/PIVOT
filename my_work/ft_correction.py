"""AFT200 판독값에 거는 6x6 보정 (센서·컨트롤러 축 정렬과 배율).

    F_true = C_F @ F_read,   T_true = C_T @ T_read      (C = blockdiag(C_F, C_T))

왜 필요한가 (2026-09-09, 로봇 PC)
  - 컨트롤러 공구 보상을 끈 뒤 8자세 수동 타어의 자유 선형 적합: 힘 응답이
    센서축마다 9.76 / 9.94 / 11.04 N, 축이 ft_mount 프레임에서 ~3도 돌아 있음.
  - 같은 자세에서 1.332 kg 병을 달면 20.7 N (기대 13.07) -> 절대 배율 1.483.
  - C_F = (평균gain · M^-1) / 1.483 을 걸면 tare_check 힘 잔차 2.08 -> 0.35 N (통과).
  - 토크는 같은 회전으로 정렬하고 배율은 힘과 같다고 **가정**한다 (미검증 —
    병을 수평 자세에서 물려 |Δτ| = 13.07 N x 레버 로 확인할 것).

파일: calibration/ft_correction.json  (없으면 단위행렬 = 보정 없음)
"""

import json
from pathlib import Path

import numpy as np

PATH = Path(__file__).resolve().parents[1] / "calibration" / "ft_correction.json"


def load(path=None):
    """6x6 보정 행렬과 메모. 파일이 없으면 단위행렬."""
    path = PATH if path is None else Path(path)
    if not path.is_file():
        return np.eye(6), dict(source=None, note="보정 파일 없음 — 원시 판독 그대로")
    data = json.loads(path.read_text())
    C = np.asarray(data["matrix"], dtype=float)
    if C.shape != (6, 6) or not np.all(np.isfinite(C)):
        raise ValueError(f"{path}: 6x6 유한 행렬이어야 합니다")
    return C, dict(source=str(path), **{k: v for k, v in data.items() if k != "matrix"})


def apply(C, wrench):
    return C @ np.asarray(wrench, dtype=float)


class CorrectedSensor:
    """read_raw / stream 을 가진 센서를 감싸 모든 표본에 C 를 건다."""

    def __init__(self, sensor, C):
        self._s, self.C = sensor, np.asarray(C, dtype=float)

    def read_raw(self, n_samples):
        return apply(self.C, self._s.read_raw(n_samples))

    def stream(self, *a, **k):
        for w in self._s.stream(*a, **k):
            yield apply(self.C, w)

    def __getattr__(self, name):          # last_registers 등은 그대로 노출
        return getattr(self._s, name)


def describe(C, info):
    sf = np.linalg.svd(C[:3, :3], compute_uv=False)
    st = np.linalg.svd(C[3:, 3:], compute_uv=False)
    src = info.get("source") or "(없음)"
    return (f"F/T 보정 {src}: 힘 배율 1/{1/sf.mean():.3f} (축별 {np.round(1/sf, 3).tolist()}),"
            f" 토크 배율 1/{1/st.mean():.3f}")
