"""각도 정밀도가 벽을 옮기는가 — 그림 A 와 B 를 한 번에 뽑는다.

왜 한 번인가
------------
study_scaling 은 목표에 닿으면 멈춰서 **라운드 수만** 남긴다. 그래서
"고정 예산에서 반폭이 얼마인가" 는 답을 못 한다.

여기서는 target=0.0 으로 두어 조기 종료를 끄고 **매 라운드 반폭을 전부**
기록한다. 그 궤적 하나에서 둘 다 나온다.

  그림 A   목표 2 % 에 처음 닿은 라운드   =  worst <= 0.02 인 첫 index
  그림 B   고정 예산에서의 반폭            =  history[BUDGET-1]["worst"]

닿지 못하면 '—' 로 남는다. rel 을 줄일 때 '—' 가 물러나면, 병목은
추정기가 아니라 센싱이라는 뜻이다 (study_scaling 의 원래 질문과 같다).

물체와 seed 규약은 study_scaling.run_cell 과 맞춘다 — 링크 길이는 seed 마다
70~150 mm 무작위, closed_loop 의 잡음 seed 는 100*s. 그래야 주 결과표와
같은 물체를 본다.

실행:
  ../robot_learning/scripts/run_drake_env.sh python study_precision.py
"""
import argparse
import json
import time

import numpy as np

import density_id_objects as obj
import design_core as dc
import nlink


def run_cell(n_part, rel, seeds, max_rounds, n_starts):
    """한 (P, rel) 칸. 조기 종료 없이 max_rounds 까지 돌고 궤적을 남긴다."""
    tracks, errors = [], []
    for s in range(seeds):
        spec = nlink.make_spec(n_part, seed=1000 + s)
        obj.set_measurement_averaging()
        rho_gt = obj.bind_object(spec)
        obj.apply_weight_prior(spec, obj.assembled_mass_kg(spec))
        out = dc.closed_loop(spec, target=0.0, max_rounds=max_rounds,
                             seed=100 * s, rel_error=rel, n_starts=n_starts)
        tracks.append([float(h["worst"]) for h in out["history"]])
        part = slice(0, n_part)                   # 부위만 채점, 힌지는 아는 값
        errors.append([
            float(np.max(np.abs(h["rho"][part] - rho_gt[part]) / rho_gt[part]))
            for h in out["history"]])
    return dict(worst=tracks, err=errors)


def first_hit(track, target):
    """목표에 처음 닿은 라운드 번호. 못 닿으면 None."""
    for i, w in enumerate(track, start=1):
        if w <= target:
            return i
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    ap.add_argument("--rel", type=float, nargs="+",
                    default=[0.01, 0.02, 0.05, 0.10])
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--max-rounds", type=int, default=14)
    ap.add_argument("--budget", type=int, default=8,
                    help="그림 B 가 반폭을 읽는 라운드")
    ap.add_argument("--target", type=float, default=0.02)
    ap.add_argument("--starts", type=int, default=6)
    ap.add_argument("--json", default="figures/precision.json")
    args = ap.parse_args()

    print(f"예산 {args.max_rounds}라운드 (조기 종료 없음)   seed {args.seeds}개   "
          f"목표 {100*args.target:g}%   그림B 기준 R={args.budget}", flush=True)
    print("물체: nlink, 링크 길이 seed 마다 70~150 mm 무작위\n", flush=True)

    head = "".join(f"{'rel=' + str(round(100*r, 3)) + '%':>17}" for r in args.rel)
    print(f"{'P':>3}{'하한':>5}" + head, flush=True)
    print(f"{'':>8}" + "".join(f"{'라운드 / R=%d반폭' % args.budget:>17}"
                               for _ in args.rel), flush=True)
    print("-" * (8 + 17 * len(args.rel)), flush=True)

    table, t0 = {}, time.time()
    for p in args.parts:
        cells, line = [], f"{p:>3}{nlink.round_lower_bound(p):>5}"
        for rel in args.rel:
            cell = run_cell(p, rel, args.seeds, args.max_rounds, args.starts)
            table[f"{p}|{rel}"] = cell
            hits = [first_hit(t, args.target) for t in cell["worst"]]
            done = [h for h in hits if h is not None]
            at_b = [t[min(args.budget, len(t)) - 1] for t in cell["worst"]]
            r_txt = (f"{int(np.median(done))}({len(done)}/{args.seeds})"
                     if done else f"—(0/{args.seeds})")
            line += f"{r_txt + ' / ' + format(100*np.median(at_b), '.2f'):>17}"
            cells.append(cell)
        print(line, flush=True)
        # 부분 결과라도 쓸 수 있게 p 하나 끝날 때마다 저장한다
        json.dump(dict(parts=args.parts, rel=args.rel, seeds=args.seeds,
                       max_rounds=args.max_rounds, budget=args.budget,
                       target=args.target,
                       bound={str(q): nlink.round_lower_bound(q)
                              for q in args.parts},
                       done_parts=args.parts[:args.parts.index(p) + 1],
                       cells=table),
                  open(args.json, "w"), indent=1)

    print(f"\n총 {time.time()-t0:.0f}초", flush=True)
    print("\n읽는 법")
    print("  왼쪽 값 = 목표에 닿은 라운드(중앙값), '—' 는 예산 안에 못 닿음")
    print("  오른쪽 값 = 고정 예산 R 에서의 부위 최악 반폭 [%]")
    print("  rel 이 작아질 때 '—' 가 물러나고 반폭이 내려가면,")
    print("  병목은 추정기가 아니라 센싱이다.")
    print(f"\n수치 -> {args.json}")


if __name__ == "__main__":
    main()
