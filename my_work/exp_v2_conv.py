"""Fig. 5 — 라운드별 불확실성 수렴 곡선 + 이론적 하한.

main_real.json 은 최종 반폭만 담고 라운드별 이력을 담지 않는다. 곡선을 그리려면
closed_loop 의 history 가 필요하므로 같은 설정으로 짧게 다시 돌린다 (수십 초).
GT 는 채점에도 안 쓴다 — 여기서 그리는 것은 알고리즘이 스스로 말하는 반폭뿐이다.
"""
import json, math
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import density_id_objects as obj
import design_core as dc

SEEDS = 8
REL = 0.05
MAX_ROUNDS = 14
PER_LINK = 0.005
OBJECTS = ["2link", "3link", "desklamp"]


def spec_for(key):
    # desklamp 는 실제 파이프라인과 같은 centroid 파지를 쓴다 (pinch 는 IK 가 안 풀린다).
    return obj.get_spec(key, volume_source="gt")


def run(key):
    spec = spec_for(key)
    obj.set_measurement_averaging()
    obj.bind_object(spec)
    obj.apply_weight_prior(spec, obj.assembled_mass_kg(spec))
    n_part = len(spec.parts)
    target = PER_LINK * n_part
    # 미지수 P 는 힌지를 포함한 몸체 수. 하한은 ceil((P-1)/3).
    P = len(obj.body_table(spec))
    rmin = max(1, math.ceil((P - 1) / 3))
    curves = []
    for s in range(SEEDS):
        out = dc.closed_loop(spec, target=target, max_rounds=MAX_ROUNDS,
                             seed=100 * s, rel_error=REL, n_starts=6,
                             estimator="tls")
        curves.append([h["worst"] for h in out["history"]])
    return dict(key=key, label=spec.label, n_part=n_part, P=P,
                target=target, rmin=rmin, curves=curves)


def main():
    res = [run(k) for k in OBJECTS]
    json.dump(res, open("figures/v2/convergence.json", "w"), indent=1)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.0), dpi=200)
    for ax, r in zip(axes, res):
        L = max(len(c) for c in r["curves"])
        grid = np.full((len(r["curves"]), L), np.nan)
        for i, c in enumerate(r["curves"]):
            grid[i, :len(c)] = c
            # 수렴 후에는 마지막 값을 유지해 곡선이 끊기지 않게 한다
            grid[i, len(c):] = c[-1]
        x = np.arange(1, L + 1)
        med = np.nanmedian(grid, axis=0) * 100
        lo = np.nanpercentile(grid, 25, axis=0) * 100
        hi = np.nanpercentile(grid, 75, axis=0) * 100
        ax.fill_between(x, lo, hi, color="0.75", alpha=.6, lw=0)
        ax.plot(x, med, "k-o", ms=3.5, lw=1.4, label="worst-part half-width")
        ax.axhline(100 * r["target"], ls=":", c="0.3", lw=1.2,
                   label=r"target $\tau(P)$")
        ax.axvline(r["rmin"], ls="--", c="0.15", lw=1.4,
                   label=r"bound $R_{\min}$")
        ax.set_yscale("log")
        ax.set_xlabel("round", fontsize=9)
        ax.set_xticks(x)
        ax.set_title(f"{r['key']}  ($P{{=}}{r['P']}$, $R_{{\\min}}{{=}}{r['rmin']}$)",
                     fontsize=9)
        ax.tick_params(labelsize=7.5)
        ax.grid(alpha=.25, lw=.5)
    axes[0].set_ylabel("half-width [%]", fontsize=9)
    axes[0].legend(fontsize=7.5, loc="upper right", framealpha=.9)
    fig.tight_layout()
    fig.savefig("figures/v2/fig_conv.png", bbox_inches="tight")
    print("wrote figures/v2/fig_conv.png")
    for r in res:
        rounds = [len(c) for c in r["curves"]]
        print(f"{r['key']:10} P={r['P']} Rmin={r['rmin']} "
              f"rounds median={int(np.median(rounds))} range={min(rounds)}-{max(rounds)} "
              f"target={100*r['target']:.2f}%")


if __name__ == "__main__":
    main()
