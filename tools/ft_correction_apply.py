#!/usr/bin/env python3
"""타어 파일의 렌치에 F/T 보정(6x6)을 걸어 새 파일을 만들고 tare_check 를 다시 돌린다.

    python tools/ft_correction_apply.py calibration/aft_tare_current.failed.json \
        --correction calibration/ft_correction.json --out calibration/aft_tare_current.json

판독 경로(dual_view, tare_real)가 같은 보정을 걸기 시작하면 타어 파일도 보정된 단위여야
한다. 그래서 원시 단위로 잰 타어를 이 도구로 한 번 변환한다 (재측정 불필요).
"""

import argparse
import copy
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "my_work"))


def corrected_payload(payload, C, info):
    """원시 렌치를 보존해 같은 타어에 보정을 반복해도 배율이 누적되지 않게 한다."""
    import ft_correction as fc

    C = np.asarray(C, dtype=float)
    if C.shape != (6, 6) or not np.all(np.isfinite(C)):
        raise ValueError("보정에는 유한한 6x6 행렬이 필요합니다")
    result = copy.deepcopy(payload)
    previous = payload.get("ft_correction")
    records = result["entries"] + result.get("manual", {}).get("records", [])
    for record in records:
        raw = record.get("wrench_raw_uncorrected")
        if raw is None and ("ft_correction" in payload or "regression" in payload):
            # 옛 파일은 공회전 부가 기록의 원시값을 안 남겼다. 같은 보정이라는
            # 증거가 있을 때만 값을 유지하며, 다른 보정으로 원시값을 추측하지 않는다.
            old_C = np.asarray(previous.get("matrix") if isinstance(previous, dict)
                               else None, dtype=float)
            if (old_C.shape != (6, 6) or not np.all(np.isfinite(old_C))
                    or not np.array_equal(old_C, C)):
                raise ValueError("이미 보정된 기록의 원시 렌치 또는 동일한 보정 행렬 이력이 필요합니다")
            current = np.asarray(record["wrench"], dtype=float)
            if current.shape != (6,) or not np.all(np.isfinite(current)):
                raise ValueError("유한한 6축 렌치가 필요합니다")
            continue
        raw = np.asarray(record["wrench"] if raw is None else raw, dtype=float)
        if raw.shape != (6,) or not np.all(np.isfinite(raw)):
            raise ValueError("유한한 6축 원시 렌치가 필요합니다")
        record["wrench_raw_uncorrected"] = raw.tolist()
        record["wrench"] = fc.apply(C, raw).tolist()
    result["ft_correction"] = dict(info, matrix=C.tolist(), applied_at_s=time.time())
    return result


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tare", type=Path)
    ap.add_argument("--correction", type=Path, default=HERE.parent / "calibration" / "ft_correction.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tool-kg", type=float, default=None)
    args = ap.parse_args(argv[1:])

    import ft_correction as fc
    import tare_check as tc
    C, info = fc.load(args.correction)
    print(" ", fc.describe(C, info))
    payload = corrected_payload(json.loads(args.tare.read_text()), C, info)
    payload["created_at_s"] = payload.get("created_at_s", time.time())
    with tempfile.NamedTemporaryFile("w", suffix=".json") as tmp:
        json.dump(payload, tmp); tmp.flush()
        kw = {} if args.tool_kg is None else dict(tool_kg=args.tool_kg)
        passed, report = tc.check(tmp.name, **kw)
    payload["measured"] = dict(passed=bool(passed), forced=False,
                               override=tc.current_overrides())
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"  -> {args.out}  passed={passed}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main(sys.argv)
