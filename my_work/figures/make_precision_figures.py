"""각도 정밀도가 벽을 옮기는가 — 그림 A 와 B.

study_precision.py 가 남긴 궤적 하나에서 둘 다 나온다.

  A  목표 2 % 에 닿은 라운드 vs 각도 오차
     각 p 의 이론 하한을 같은 색 점선으로 겹친다.
     하한은 rank 조건이라 각도 오차와 무관하다 — 평평하다.
     실선이 그 평평한 바닥으로 내려오는 것이 이 그림의 전부다.

  B  고정 예산(R=8)에서의 반폭 vs 각도 오차
     목표선과 만나는 지점이 '그 물체에 필요한 각도 정밀도' 다.
     설계 차트로 그대로 쓸 수 있다.

p 는 순서가 있는 양이므로 범주형 색이 아니라 **단일 색상 순차 램프**를 쓴다.
가장 밝은 칸은 250 스텝(#86b6ef) 아래로 내리지 않는다 — 이산 마크라
바탕에서 2:1 을 지켜야 한다. 선 구분은 색이 아니라 **끝에 붙인 이름**이
담당하므로 색만으로 식별하지 않는다.

실행:
    cd ~/Desktop/PIVOT/my_work
    ../robot_learning/scripts/run_drake_env.sh python figures/make_precision_figures.py
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
D = json.load(open(OUT / "precision.json"))

# p = 2..6 에 대한 순차 램프 (250 -> 700 스텝). 밝을수록 부위가 적다.
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
ACC = "#eb6834"                      # 목표선·경고. 데이터 색과 겹치지 않는다
SURF, INK, INK2, INK3 = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8a84"
GRID = "#e4e4e0"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 10.5,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "grid.color": GRID, "grid.linewidth": 0.7,
    "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2.0, "lines.markersize": 6.5,
})

PARTS = D["parts"]
RELS = np.array(D["rel"]) * 100.0
BUDGET, TARGET, RMAX = D["budget"], D["target"], D["max_rounds"]


def first_hit(track):
    for i, w in enumerate(track, start=1):
        if w <= TARGET:
            return i
    return None


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> figures/{name}.png / .pdf")


# ═════════════════════════════════════════════════════════════════════
# A. 벽이 옮겨가는가 — 라운드 vs 각도 오차
# ═════════════════════════════════════════════════════════════════════
# 계열끼리 y 값이 충돌한다 (p=4 의 실제 라운드 3 과 p=5 의 하한 3 이 같은 높이).
# 한 축에 겹쳐 그리면 어느 선이 무엇인지 읽을 수 없으므로 p 마다 칸을 나눈다.
fig, axs = plt.subplots(1, len(PARTS), figsize=(10.4, 3.3), sharey=True)

for ax, p, col in zip(axs, PARTS, RAMP):
    bound = D["bound"][str(p)]
    ax.plot([0.7, 14.5], [bound] * 2, ls=(0, (5, 3)), lw=1.4, color=INK,
            alpha=0.55, zorder=2)

    xs, ys, miss = [], [], []
    for rel, x in zip(D["rel"], RELS):
        hits = [first_hit(t) for t in D["cells"][f"{p}|{rel}"]["worst"]]
        done = [h for h in hits if h is not None]
        if len(done) >= len(hits) / 2:
            xs.append(x); ys.append(float(np.median(done)))
        else:
            miss.append(x)
    ax.plot(xs, ys, marker="o", color=col, zorder=4)
    for x in miss:
        ax.plot([x], [RMAX + 0.8], marker="^", ms=8, color=ACC,
                markerfacecolor=SURF, markeredgewidth=1.8, zorder=4)

    ax.axhspan(RMAX + 0.2, RMAX + 1.7, color=ACC, alpha=0.09, zorder=0)
    ax.set_xscale("log")
    ax.set_xticks(RELS); ax.set_xticklabels([f"{r:g}" for r in RELS], fontsize=8)
    ax.set_xlim(0.68, 14.8)
    ax.set_ylim(0, RMAX + 2.3)
    ax.grid(axis="y", zorder=0)
    ax.set_title(f"p = {p}", color=col, fontsize=11, pad=7)
    gap = (ys[0] - bound) if ys else None
    ax.text(np.sqrt(RELS[0] * RELS[-1]), max(bound - 1.0, 0.75),
            "on the bound" if gap == 0 else f"+{gap:g} at 1 %",
            ha="center", va="top", fontsize=8.2,
            color=(col if gap == 0 else INK2), fontweight="bold")

axs[0].set_yticks([0, 2, 4, 6, 8, 10, 12, 14])
axs[0].set_ylabel("rounds to reach\nthe 2 % target")
for ax in axs[1:]:
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

axs[0].plot([], [], ls=(0, (5, 3)), color=INK, alpha=0.55,
            label="Prop. 3 bound  ⌈(2p−2)/3⌉")
axs[0].plot([], [], marker="^", ls="none", color=ACC, markerfacecolor=SURF,
            markeredgewidth=1.8, label=f"not solved within {RMAX}")
fig.legend(*axs[0].get_legend_handles_labels(), loc="upper left", ncol=2,
           fontsize=8.6, labelcolor=INK2, bbox_to_anchor=(0.055, 1.045))

fig.text(0.5, -0.02, "joint-angle measurement error  [%]",
         ha="center", fontsize=9.5, color=INK2)
fig.suptitle("A.  The bound never moves with angle precision — the actual count does",
             x=0.055, ha="left", fontsize=11.5, color=INK, y=1.135)
fig.text(0.5, -0.155,
         "The dashed floor is a rank condition and is identical in all four "
         "columns of each panel.  p = 2 and 3 sit on it at every precision;\n"
         "p = 5 and 6 sit far above and close toward it as the angle is read "
         f"more precisely.  That gap is conditioning.  4 objects per cell, "
         f"{RMAX}-round budget.",
         ha="center", fontsize=8.6, color=INK2, linespacing=1.5)
save(fig, "fig_precision_rounds")


# ═════════════════════════════════════════════════════════════════════
# B. 얼마나 정확해야 하는가 — 반폭 vs 각도 오차
# ═════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6.8, 4.6))

ax.axhspan(0.1, 100 * TARGET, color="#1baf7a", alpha=0.07, zorder=0)
ax.axhline(100 * TARGET, color=ACC, lw=1.6, zorder=3)
ax.annotate(f"target {100*TARGET:g} %", (0.76, 100 * TARGET),
            textcoords="offset points", xytext=(0, 6), fontsize=9,
            color=ACC, fontweight="bold")

for p, col in zip(PARTS, RAMP):
    ys = [float(np.median([t[min(BUDGET, len(t)) - 1]
                           for t in D["cells"][f"{p}|{rel}"]["worst"]])) * 100
          for rel in D["rel"]]
    ax.plot(RELS, ys, marker="o", color=col, zorder=4)
    ax.annotate(f"p={p}", (RELS[-1], ys[-1]), textcoords="offset points",
                xytext=(9, 0), va="center", fontsize=9.5, color=col,
                fontweight="bold")

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xticks(RELS); ax.set_xticklabels([f"{r:g} %" for r in RELS])
ax.set_xlim(0.72, 16.5)
ax.set_yticks([0.1, 0.2, 0.5, 1, 2, 5, 10, 20])
ax.set_yticklabels(["0.1", "0.2", "0.5", "1", "2", "5", "10", "20"])
ax.set_ylim(0.1, 22)
ax.set_xlabel("joint-angle measurement error")
ax.set_ylabel(f"worst part half-width after {BUDGET} configurations [%]")
ax.grid(axis="y", zorder=0)
ax.set_title("B.  How precisely must the angle be read for a given object and budget",
             color=INK, loc="left", pad=10)
fig.text(0.5, -0.06,
         "Where a curve crosses the target line is the angle precision that "
         "object needs.  p≤4 clears it even at 10 %; p=5 needs about 5 %;\n"
         f"p=6 needs about 2 %.  4 objects per cell, fixed budget of "
         f"{BUDGET} configurations.",
         ha="center", fontsize=8.6, color=INK2, linespacing=1.5)
save(fig, "fig_precision_halfwidth")


# ═════════════════════════════════════════════════════════════════════
print("\n표로 옮기면")
print(f"{'p':>3}{'하한':>5}" + "".join(f"{f'{r:g}%':>18}" for r in RELS))
for p in PARTS:
    line = f"{p:>3}{D['bound'][str(p)]:>5}"
    for rel in D["rel"]:
        c = D["cells"][f"{p}|{rel}"]
        hits = [first_hit(t) for t in c["worst"]]
        done = [h for h in hits if h is not None]
        w = np.median([t[min(BUDGET, len(t)) - 1] for t in c["worst"]]) * 100
        r = f"{int(np.median(done))}" if done else "—"
        line += f"{f'{r}({len(done)}/{len(hits)}) {w:.2f}%':>18}"
    print(line)
