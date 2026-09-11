"""논문 마지막 Fig. 6(데모) 자리표시 이미지.

Screenshot from 2026-09-08 09-39-40.png 형태:
  왼쪽 열 = 우리 asset 으로 돌린 시뮬레이션, 오른쪽 열 = 같은 장면의 실물.
  가운데 점선으로 두 열을 가른다. 행 = 태스크 3개.
one column 폭.
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

TASKS = ["Pick-and-place", "Knocking over", "Sliding down a ramp"]
fig, ax = plt.subplots(figsize=(3.45, 3.5))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

HEAD, GAP = 0.075, 0.012
ch = (1 - HEAD) / len(TASKS)
cw = (1 - 3*GAP) / 2
ax.text(GAP + cw/2, 1 - HEAD/2, "Simulation (ours)", ha="center", va="center",
        fontsize=7, weight="bold")
ax.text(2*GAP + cw*1.5, 1 - HEAD/2, "Real world", ha="center", va="center",
        fontsize=7, weight="bold")
for i, t in enumerate(TASKS):
    yb = 1 - HEAD - (i+1)*ch + GAP/2
    for j, tint in enumerate(["#eef1f5", "#f4efe9"]):
        x = GAP + j*(cw + GAP)
        ax.add_patch(Rectangle((x, yb), cw, ch - GAP, fc=tint, ec="0.45", lw=0.8))
        ax.text(x + cw/2, yb + (ch-GAP)*0.62,
                "overlaid frames\nof the motion", ha="center", va="center",
                fontsize=5.2, color="0.42", style="italic")
        if j == 0:      # 태스크 이름은 행마다 한 번만
            ax.text(x + 0.012, yb + (ch-GAP)*0.10, t, ha="left", va="center",
                    fontsize=5.6, color="0.25", weight="bold")
# 가운데 점선 구분자
ax.add_line(Line2D([0.5, 0.5], [0.012, 1 - HEAD], color="0.2", lw=1.1,
                   ls=(0, (5, 4))))
ax.text(0.5, 0.5, "PLACEHOLDER", ha="center", va="center", fontsize=17,
        color="#b3261e", alpha=0.17, weight="bold", rotation=18)
fig.subplots_adjust(0, 0, 1, 1)
fig.savefig("figures/fig_demo.png", dpi=400)
print("saved")
