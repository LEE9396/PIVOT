"""Fig. 5(밀도 시각 비교) 자리표시 이미지.

실물 렌더가 오기 전까지 지면 크기와 배치를 확정해 두기 위한 격자 틀만 그린다.
행 = 물체, 열 = GT / baseline 3개 / Ours. 칸마다 자세 2개.
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

ROWS = ["Stand lamp", "Laptop", "3-link custom"]
COLS = ["GT", "PUGS", "SiPhy", "PhysX-Omni", "Ours"]
HEAD = 0.055
fig, ax = plt.subplots(figsize=(7.1, 2.9))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

x0, y0 = 0.115, 0.02
cw = (1 - x0) / len(COLS)
ch = (1 - y0 - HEAD) / len(ROWS)
for j, c in enumerate(COLS):
    face = "#fdecec" if c == "GT" else ("#e8effb" if c == "Ours" else "#f2f2f2")
    ax.add_patch(Rectangle((x0+j*cw, 1-HEAD), cw, HEAD, fc=face, ec="0.35", lw=0.8))
    ax.text(x0+(j+.5)*cw, 1-HEAD/2, c, ha="center", va="center",
            fontsize=9, weight="bold")
ax.add_patch(Rectangle((0, 1-HEAD), x0, HEAD, fc="#f2f2f2", ec="0.35", lw=0.8))
ax.text(x0/2, 1-HEAD/2, "OBJECT", ha="center", va="center", fontsize=8, weight="bold")

for i, r in enumerate(ROWS):
    yb = 1 - HEAD - (i+1)*ch
    ax.add_patch(Rectangle((0, yb), x0, ch, fc="#f7f7f7", ec="0.35", lw=0.8))
    ax.text(x0/2, yb+ch/2, r, ha="center", va="center", fontsize=8)
    for j in range(len(COLS)):
        ax.add_patch(Rectangle((x0+j*cw, yb), cw, ch, fc="white", ec="0.35", lw=0.8))
        for k in range(2):                       # 칸 안에 자세 2개
            ax.add_patch(Rectangle((x0+j*cw+0.006+k*(cw-0.014)/2, yb+0.02),
                                   (cw-0.018)/2, ch-0.04,
                                   fc="#eceff2", ec="0.75", lw=0.6, ls=":"))
        ax.text(x0+(j+.5)*cw, yb+ch/2, "density-coloured\nrender", ha="center",
                va="center", fontsize=5.6, color="0.45", style="italic")
ax.text(0.5, 0.5, "PLACEHOLDER", ha="center", va="center", fontsize=26,
        color="#b3261e", alpha=0.16, weight="bold", rotation=12)
fig.subplots_adjust(0, 0, 1, 1)
fig.savefig("figures/fig_density_grid.png", dpi=300)
print("saved")
