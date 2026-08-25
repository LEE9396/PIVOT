"""검토 ⑩ (T6): 부위 수 P 와 각도 정밀도를 함께 흔들어 '벽'의 위치를 잰다.

왜 이 실험인가
--------------
THEORY.md 정리 3은 전체 식별에 R >= ceil((n-1)/3) 라운드가 필요하다고 말한다
(n = 미지수 = 2P-1). 그런데 이 하한은 **rank** 만 보는 필요조건이라, 실제
라운드 수는 그보다 훨씬 크다. 예를 들어 P=8 은 하한이 5인데 실제로는 12
라운드에도 안 끝난다.

그 간극은 조건수가 만들고, 조건수는 **각도 측정 정밀도**가 만든다. 각도를
1 도 잘못 읽으면 회귀행렬 A 자체가 틀어지므로(EIV), 미지수가 많아질수록
방향을 가르는 데 필요한 정밀도가 급격히 올라간다.

그래서 rel 을 1 / 2 / 5 / 10 % 로 바꿔가며 벽이 어디로 옮겨가는지 잰다.
벽이 rel 을 줄일 때 오른쪽으로 밀린다면, 병목은 추정기가 아니라 **센싱**이다.
이건 주장이 아니라 측정으로 갈리는 문제다.

무엇을 쓰나
-----------
정식 폐루프 design_core.closed_loop 하나만 쓴다. 정지 판단에서 힌지를 빼는
것(stopping_width)도 거기 들어 있으므로 dual_view 와 판정이 같다.

물체는 nlink.make_spec(P) — 실물 3-link 치수 규칙에 **관절마다 실측 힌지
41 g** 을 붙인 사슬이다. 그래서 P=2, 3 은 실물 2-link / 3-link 와 미지수
개수도 라운드 수도 일치한다 (검증: study_theory.py).

실행:
  ../robot_learning/scripts/run_drake_env.sh python study_scaling.py
  ../robot_learning/scripts/run_drake_env.sh python study_scaling.py --parts 2 3 4 --seeds 8
"""

import argparse
import json
import time

import numpy as np

import density_id_objects as obj
import design_core as dc
import nlink

DEFAULT_RELS = (0.01, 0.02, 0.05, 0.10)


def run_cell(n_part, rho_gt, rel, seeds, target, max_rounds, n_starts,
             fixed_spec=None):
    """한 (P, rel) 칸을 seeds 번 돌린다. GT 는 채점에만 쓴다.

    fixed_spec 이 없으면 **seed 마다 링크 길이를 새로 뽑는다.** 그래야 결과가
    물체 하나가 아니라 '전형적인 사슬' 에 대한 것이 되고, seed 간 편차가 곧
    물체 간 편차가 되어 오차 막대가 생긴다.

    고정 규칙은 어느 쪽으로든 결과를 정한다 — 단조 감소는 말단 링크가 짧아져
    마지막 관절의 지렛대가 사라지고(p=4 에서 이미 link2/link3 반폭이
    1.27/1.29 % 로 붙는다), 등길이는 반대로 유리한 모양을 고른 셈이 된다.
    """
    rounds, converged, errors = [], [], []
    for s in range(seeds):
        if fixed_spec is None:
            spec = nlink.make_spec(n_part, seed=1000 + s)
            obj.set_measurement_averaging()
            rho_gt = obj.bind_object(spec)
            obj.apply_weight_prior(spec, obj.assembled_mass_kg(spec))
        else:
            spec = fixed_spec
        out = dc.closed_loop(spec, target=target, max_rounds=max_rounds,
                             seed=100 * s, rel_error=rel, n_starts=n_starts)
        rounds.append(out["rounds"])
        converged.append(bool(out["converged"]))
        part = slice(0, n_part)                       # 부위만 채점 (힌지는 아는 값)
        errors.append(float(np.max(np.abs(out["rho_hat"][part] - rho_gt[part])
                                   / rho_gt[part])))
    return dict(rounds=rounds, converged=converged, errors=errors)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", type=int, nargs="+", default=[2, 3, 4, 5, 6, 7, 8])
    ap.add_argument("--rel", type=float, nargs="+", default=list(DEFAULT_RELS))
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--target", type=float, default=0.01)
    ap.add_argument("--max-rounds", type=int, default=12)
    ap.add_argument("--starts", type=int, default=6)
    ap.add_argument("--fixed-lengths", action="store_true",
                    help="링크 길이를 예전 단조 감소 규칙으로 고정 (재현용)")
    ap.add_argument("--json", default="figures/study_scaling.json")
    ap.add_argument("--plot", default="figures/study_scaling.png")
    args = ap.parse_args()

    print(f"목표 반폭 {100*args.target:.1f}%   최대 {args.max_rounds}라운드   "
          f"seed {args.seeds}개   연속최적화 시작점 {args.starts}개")
    print("물체: nlink (관절마다 실측 힌지 41 g). 미지수 = 2P-1.")
    print("링크 길이: " + ("고정 (단조 감소)" if args.fixed_lengths else
          f"seed 마다 {nlink.LEN_MIN_MM:.0f}~{nlink.LEN_MAX_MM:.0f} mm 무작위") + "\n")
    print(f"  {'P':>2}{'미지수':>7}{'하한R':>6}"
          + "".join(f"{'rel=' + str(int(100*r)) + '%':>16}" for r in args.rel))
    print(f"  {'':>2}{'':>7}{'':>6}"
          + "".join(f"{'라운드(수렴/n)':>16}" for _ in args.rel))

    table, t_start = {}, time.time()
    for n_part in args.parts:
        spec = nlink.make_spec(n_part)
        obj.set_measurement_averaging()
        rho_gt = obj.bind_object(spec)
        obj.apply_weight_prior(spec, obj.assembled_mass_kg(spec))

        cells = []
        for rel in args.rel:
            cell = run_cell(n_part, rho_gt, rel, args.seeds, args.target,
                            args.max_rounds, args.starts,
                            fixed_spec=spec if args.fixed_lengths else None)
            table[f"{n_part}|{rel}"] = cell
            n_ok = sum(cell["converged"])
            ok_rounds = [r for r, c in zip(cell["rounds"], cell["converged"]) if c]
            shown = f"{np.median(ok_rounds):.0f}" if ok_rounds else "—"
            cells.append(f"{shown:>7} ({n_ok}/{args.seeds})".rjust(16))
        print(f"  {n_part:>2}{nlink.n_unknowns(n_part):>7}"
              f"{nlink.round_lower_bound(n_part):>6}" + "".join(cells))

    print(f"\n총 {time.time() - t_start:.0f}초")
    print("\n읽는 법")
    print("  '—' 는 그 정밀도로는 목표에 못 닿는다는 뜻 (수렴 0/n).")
    print("  하한R 은 rank 만 보는 필요조건이다. 실제 라운드가 그보다 크면,")
    print("  그 차이는 전부 조건수이고 조건수는 각도 정밀도가 만든다.")
    print("  rel 을 줄일 때 '—' 가 오른쪽으로 밀리면, 병목은 추정기가 아니라")
    print("  센싱이라는 뜻이다. 이것이 이 실험이 가르려는 단 하나의 질문이다.")

    payload = dict(parts=args.parts, rel=args.rel, seeds=args.seeds,
                   target=args.target, max_rounds=args.max_rounds,
                   unknowns={p: nlink.n_unknowns(p) for p in args.parts},
                   bound={p: nlink.round_lower_bound(p) for p in args.parts},
                   cells=table)
    with open(args.json, "w") as fh:
        json.dump(payload, fh, indent=1)
    print(f"\n수치 -> {args.json}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib 이 없어 그림은 건너뜁니다)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    ax = axes[0]
    for rel, style in zip(args.rel, ("o-", "s-", "^-", "v-", "d-")):
        ys = []
        for p in args.parts:
            cell = table[f"{p}|{rel}"]
            ok = [r for r, c in zip(cell["rounds"], cell["converged"]) if c]
            ys.append(np.median(ok) if ok else np.nan)
        ax.plot(args.parts, ys, style, label=f"angle err {100*rel:g}%")
    ax.step(args.parts, [nlink.round_lower_bound(p) for p in args.parts],
            "k--", where="mid", label="Prop.3 bound ceil((2P-2)/3)")
    ax.set_xlabel("number of parts P"); ax.set_ylabel(f"rounds to {100*args.target:g}% half-width")
    ax.set_title("where the wall is, and does precision move it")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    ax = axes[1]
    for rel, style in zip(args.rel, ("o-", "s-", "^-", "v-", "d-")):
        ys = [100.0 * np.mean(table[f"{p}|{rel}"]["converged"]) for p in args.parts]
        ax.plot(args.parts, ys, style, label=f"{100*rel:g}%")
    ax.set_xlabel("number of parts P"); ax.set_ylabel("converged [%]")
    ax.set_title("convergence rate vs angle precision")
    ax.set_ylim(-5, 105); ax.grid(alpha=0.3); ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(args.plot, dpi=150)
    print(f"그림 -> {args.plot}")


if __name__ == "__main__":
    main()
