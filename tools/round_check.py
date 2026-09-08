#!/usr/bin/env python3
"""탐색 한 라운드 뒤, 센서가 모형이 말하는 것을 재고 있는지 숫자 세 개로 판정한다.

    python tools/round_check.py <세션 폴더> [--conf setup/experiment.conf] [--round 1]
    python tools/round_check.py --self-test

세션의 exploration_round_N.json 하나로 판정한다. 통과 기준은 session_20260904_1736
의 실패 지문에서 나왔다 (SESSION_20260904_ROOT_CAUSE.md).

  1) 영점 뺀 힘의 크기      ≈ 저울 총질량 × g      (지난 세션: 58.7 N 오프셋)
     -> 틀리면 파지 실패 · 타어 무효 · 부호 규약
  2) residual_inflation     ≤ 100                  (지난 세션: 5 527)
     -> 틀리면 토크 기준점(GRASP_FRAME) · AFT200 기준면
  3) grasp_offset_mm        상자(±50) 안, 대개 ±σ   (지난 세션: [37, -50, 50] 에 붙음)
     -> 상자에 붙으면 기준점 불일치
  4) wrench_raw 18개 저장   있음                   (지난 세션: 없음 -> 재분석 불가)

왜 팽창이 1 이 아니라 100 인가: 코드가 가정하는 잡음(1000 샘플 평균 후
σ_T = 0.177 mN·m)보다 타어 게이트(0.02 N·m)가 113 배 크다. 게이트 턱걸이면
팽창 ~100, 타어가 좋으면 10~30. 5 527 은 다른 종류의 숫자다.
"""

import argparse
import json
import sys
from pathlib import Path

G_ACC = 9.81
FORCE_TOL_N = 0.5          # tare_check 의 힘 잔차 게이트와 같다
INFLATION_MAX = 100.0
GRASP_BOX_MM = 50.0
GRASP_RAIL_MM = 48.0       # 이 안쪽이면 '붙었다'고 본다


def wrench_blocks(values):
    """18개(3방향×6축) 또는 6의 배수 길이 렌치를 (n, 6) 으로."""
    if values is None:
        return None
    flat = [float(v) for v in values]
    if len(flat) % 6:
        return None
    return [flat[i:i + 6] for i in range(0, len(flat), 6)]


def check(payload, total_mass_kg, grasp_sigma_mm):
    rows = []
    m = payload.get("measurement", {})
    blocks = wrench_blocks(m.get("wrench"))
    raw = wrench_blocks(m.get("wrench_raw"))

    # 1) 힘 크기
    if blocks and total_mass_kg:
        expect = total_mass_kg * G_ACC
        mags = [sum(f * f for f in b[:3]) ** 0.5 for b in blocks]
        worst = max(abs(x - expect) for x in mags)
        ok = worst <= FORCE_TOL_N
        rows.append((ok, "영점 뺀 힘 크기",
                     f"{' / '.join(f'{x:.2f}' for x in mags)} N  (기대 {expect:.2f} ± {FORCE_TOL_N})",
                     "파지 실패 · 타어 무효 · 힘 부호 규약"))
    else:
        rows.append((False, "영점 뺀 힘 크기",
                     "wrench 가 없거나 TOTAL_MASS_KG 가 없어 판정 못 함", "conf 와 세션 파일 확인"))

    # 2) 잔차팽창
    infl = payload.get("residual_inflation")
    if infl is None:
        rows.append((False, "잔차팽창", "residual_inflation 없음", "세션 파일 확인"))
    else:
        rows.append((float(infl) <= INFLATION_MAX, "잔차팽창",
                     f"{float(infl):.1f}  (≤ {INFLATION_MAX:.0f}; 지난 세션 5 527)",
                     "토크 기준점 GRASP_FRAME=measured · AFT200 기준면"))

    # 3) 파지 오프셋
    off = payload.get("grasp_offset_mm")
    if off is None:
        rows.append((False, "파지 오프셋", "grasp_offset_mm 없음", "세션 파일 확인"))
    else:
        off = [float(v) for v in off]
        railed = any(abs(v) >= GRASP_RAIL_MM for v in off)
        big = any(abs(v) > 2.0 * grasp_sigma_mm for v in off) if grasp_sigma_mm else False
        txt = "[" + ", ".join(f"{v:+.1f}" for v in off) + f"] mm  (상자 ±{GRASP_BOX_MM:.0f}, σ {grasp_sigma_mm:g})"
        if railed:
            rows.append((False, "파지 오프셋", txt + "  ← 상자에 붙음", "토크 기준점 불일치 (173.9 mm 지문)"))
        elif big:
            rows.append((None, "파지 오프셋", txt + "  ← 2σ 밖", "실측 파지·손-눈 확인"))
        else:
            rows.append((True, "파지 오프셋", txt, ""))

    # 4) 원시 렌치
    rows.append((bool(raw), "원시 렌치 저장",
                 f"wrench_raw {len(raw) * 6 if raw else 0} 개",
                 "없으면 오프라인 재분석 불가 — 즉시 중단"))
    return rows


def render(rows, title):
    mark = {True: "  OK ", False: "*실패", None: " 주의"}
    print(title)
    width = max(len(r[1]) for r in rows)
    for ok, name, detail, fix in rows:
        print(f"{mark[ok]}  {name:<{width}}  {detail}")
        if ok is not True and fix:
            print(f"        -> {fix}")
    worst = "FAIL" if any(r[0] is False for r in rows) else ("WARN" if any(r[0] is None for r in rows) else "OK")
    print(f"  판정: {worst}")
    return worst


def self_test():
    good = dict(measurement=dict(wrench=[0, 0, 5.60, .1, -.2, .3] * 3,
                                 wrench_raw=[1, 2, 60, .1, .1, .1] * 3),
                residual_inflation=23.0, grasp_offset_mm=[3.1, -8.4, 5.0])
    bad = dict(measurement=dict(wrench=[0, 0, 58.7, .1, -.2, .3] * 3, wrench_raw=None),
               residual_inflation=5527.3, grasp_offset_mm=[37.1, -50.0, 50.0])
    a = render(check(good, 0.571, 15.0), "self-test: 정상 라운드")
    b = render(check(bad, 0.571, 15.0), "self-test: session_20260904_1736 지문")
    return a == "OK" and b == "FAIL"


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", nargs="?", type=Path)
    ap.add_argument("--conf", type=Path, default=Path("setup/experiment.conf"))
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv[1:])
    if args.self_test:
        sys.exit(0 if self_test() else 1)
    if args.session is None:
        sys.exit("세션 폴더를 주세요 (또는 --self-test)")

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from preflight import read_conf
    conf = read_conf(args.conf.expanduser())
    try:
        mass = float(conf.get("TOTAL_MASS_KG", ""))
    except ValueError:
        mass = None
    try:
        sigma = float(conf.get("GRASP_SIGMA_MM", "15"))
    except ValueError:
        sigma = 15.0

    path = args.session / f"exploration_round_{args.round}.json"
    if not path.is_file():
        sys.exit(f"{path} 가 없습니다 — 탐색이 아직 한 라운드도 안 끝났거나 세션 폴더가 다릅니다")
    payload = json.loads(path.read_text())
    worst = render(check(payload, mass, sigma), f"라운드 {args.round} 검산  ({path})")
    sys.exit({"OK": 0, "WARN": 0, "FAIL": 1}[worst])


if __name__ == "__main__":
    main(sys.argv)
