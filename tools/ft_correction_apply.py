#!/usr/bin/env python3
"""타어 파일의 렌치에 F/T 보정(6x6)을 걸어 새 파일을 만들고 tare_check 를 다시 돌린다.

    python tools/ft_correction_apply.py calibration/aft_tare_current.failed.json \
        --correction calibration/ft_correction.json --out calibration/aft_tare_current.json

판독 경로(dual_view, tare_real)가 같은 보정을 걸기 시작하면 타어 파일도 보정된 단위여야
한다. 그래서 원시 단위로 잰 타어를 이 도구로 한 번 변환한다 (재측정 불필요).
"""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "my_work"))


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
    payload = json.loads(args.tare.read_text())
    for e in payload["entries"]:
        e["wrench_raw_uncorrected"] = list(e["wrench"])
        e["wrench"] = fc.apply(C, e["wrench"]).tolist()
    if "manual" in payload:
        for r in payload["manual"].get("records", []):
            r["wrench"] = fc.apply(C, r["wrench"]).tolist()
    payload["ft_correction"] = dict(info, matrix=C.tolist(), applied_at_s=time.time())
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
