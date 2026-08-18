"""부위별 밀도 비교 패널을 독립 그림 파일로 만든다.

desklamp_density_render.py 가 만든 램프 3장을 받아서, 라벨 / 부위별 표 /
컬러바까지 붙인 한 장짜리 figure 로 합친다. 논문·슬라이드에 그대로 넣는 용도라
matplotlib 기본 폰트(DejaVu)만 쓰고 PNG 와 PDF 를 같이 뽑는다.

    python desklamp_density_panel.py

주의: figure 레벨 아티스트는 기본 zorder 가 axes(0) 보다 높다. 판 사각형을
zorder=0 으로 깔지 않으면 렌더 이미지와 컬러바가 그 밑으로 숨는다.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.image import imread
from matplotlib import colormaps

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- 색: render 스크립트와 같은 turbo / 같은 정의역 -------------------------
DMIN, DMAX = 300.0, 1050.0
TURBO = colormaps["turbo"]
SW = {"head": "#1fc9dd", "arm": "#4456c7", "base": "#f05b12", "prior": "#af1801"}

PLATE, RULE = "#eef0f4", "#d3d8e0"
INK, INK2 = "#1b2029", "#5a6474"
BAD, GOOD = "#b03410", "#0a7a0a"
SANS, MONO = "DejaVu Sans", "DejaVu Sans Mono"

COLS = [
    dict(key="before", name="Before exploration", src="scan + uniform 1000 kg/m³ prior",
         rows=[("head", "1000", "0.1657", "+97.3%", BAD, SW["prior"]),
               ("arm",  "1000", "0.2196", "+167.8%", BAD, SW["prior"]),
               ("base", "1000", "0.4410", "+11.4%", BAD, SW["prior"])]),
    dict(key="after", name="After exploration", src="grasp + wrist F/T identification",
         rows=[("head", "507.4", "0.0841", "+0.09%", GOOD, SW["head"]),
               ("arm",  "372.3", "0.0817", "−0.31%", GOOD, SW["arm"]),
               ("base", "897.8", "0.3960", "−0.01%", GOOD, SW["base"])]),
    dict(key="gt", name="Ground truth", src="measured mass ÷ same collision volume",
         rows=[("head", "507.0", "0.0840", "—", INK2, SW["head"]),
               ("arm",  "373.4", "0.0820", "—", INK2, SW["arm"]),
               ("base", "897.9", "0.3960", "—", INK2, SW["base"])]),
]
TICKS = [(300, "300", False), (372.3, "372 arm", True), (507.4, "507 head", True),
         (700, "700", False), (897.8, "898 base", True), (1000, "1000 prior", True),
         (1050, "1050", False)]

W, H = 12.6, 8.8
fig = plt.figure(figsize=(W, H), dpi=100)
fig.patch.set_facecolor("white")


def X(v): return v / W
def Y(v): return v / H


# ---- 머리말 ----------------------------------------------------------------
fig.text(X(0.35), Y(8.36), "Per-link density, side by side", ha="left", va="center",
         fontfamily=SANS, fontsize=17, fontweight="bold", color="#0f131a")
fig.text(X(12.25), Y(8.36), "Colour encodes effective density on the simulated collision volume.",
         ha="right", va="center", fontfamily=SANS, fontsize=11, color="#4c5566")
fig.add_artist(plt.Line2D([X(0.35), X(12.25)], [Y(8.08)] * 2, color="#dee3ec", lw=1))

# ---- 판 (zorder 0: 반드시 제일 아래) ---------------------------------------
plate = FancyBboxPatch((X(0.35), Y(0.30)), X(12.25) - X(0.35), Y(7.78) - Y(0.30),
                       boxstyle="round,pad=0,rounding_size=0.004",
                       facecolor=PLATE, edgecolor=RULE, lw=1, zorder=0)
fig.add_artist(plate)

COL_W, X0, PAD = 3.30, 0.70, 0.20
IMG_TOP, IMG_BOT = 6.88, 2.62
HDR_Y = 2.30
ROW_Y = [1.88, 1.46, 1.04]

def on_plate(path):
    """투명 PNG 를 판 색 위에 미리 합성한다.

    RGBA 그대로 imshow 하면 보간기가 RGB 와 A 를 따로 섞어서, 알파가 0 인
    픽셀의 RGB 가 윤곽에 번져 회색 할로가 생긴다. 먼저 합성하면 알파 경계가
    아예 사라진다."""
    im = imread(path)
    if im.shape[2] < 4:
        return im
    rgb, a = im[..., :3], im[..., 3:4]
    bg = np.array(matplotlib.colors.to_rgb(PLATE)).reshape(1, 1, 3)
    return np.clip(rgb * a + bg * (1 - a), 0, 1)


imgs = {c["key"]: on_plate(f"{HERE}/desklamp_density_{c['key']}.png") for c in COLS}
ih, iw = imgs["after"].shape[:2]
img_h = IMG_TOP - IMG_BOT
img_w = img_h * iw / ih

for i, c in enumerate(COLS):
    x0 = X0 + i * COL_W
    if i:
        fig.add_artist(plt.Line2D([X(x0 - 0.05)] * 2, [Y(0.65), Y(7.40)],
                                  color=RULE, lw=1, zorder=1))

    fig.text(X(x0 + PAD), Y(7.22), c["name"], ha="left", va="center",
             fontfamily=SANS, fontsize=13.5, fontweight="bold", color=INK, zorder=3)
    fig.text(X(x0 + PAD), Y(6.98), c["src"], ha="left", va="center",
             fontfamily=MONO, fontsize=8.6, color=INK2, zorder=3)

    ax = fig.add_axes([X(x0 + (COL_W - img_w) / 2), Y(IMG_BOT), X(img_w), Y(img_h)],
                      zorder=2)
    ax.imshow(imgs[c["key"]], interpolation="lanczos")
    ax.set_axis_off()
    ax.patch.set_alpha(0)

    cx = {"rho": x0 + 1.76, "m": x0 + 2.42, "d": x0 + COL_W - PAD}
    for lbl, key in [("PART", None), ("ρ KG/M³", "rho"), ("M KG", "m"), ("ΔM", "d")]:
        fig.text(X(x0 + PAD if key is None else cx[key]), Y(HDR_Y), lbl,
                 ha="left" if key is None else "right", va="center",
                 fontfamily=MONO, fontsize=7.4, color=INK2, zorder=3)
    fig.add_artist(plt.Line2D([X(x0 + PAD), X(x0 + COL_W - PAD)], [Y(HDR_Y - 0.13)] * 2,
                              color=RULE, lw=1, zorder=1))

    for (part, rho, m, dm, dcol, swc), yy in zip(c["rows"], ROW_Y):
        fig.add_artist(Rectangle((X(x0 + PAD), Y(yy - 0.05)), X(0.105), Y(0.105),
                                 facecolor=swc, edgecolor="#00000038", lw=.6, zorder=3))
        fig.text(X(x0 + PAD + 0.18), Y(yy), part, ha="left", va="center",
                 fontfamily=MONO, fontsize=9.2, color=INK2, zorder=3)
        for val, key, col in [(rho, "rho", INK), (m, "m", INK), (dm, "d", dcol)]:
            fig.text(X(cx[key]), Y(yy), val, ha="right", va="center",
                     fontfamily=MONO, fontsize=9.2, color=col, zorder=3)
        fig.add_artist(plt.Line2D([X(x0 + PAD), X(x0 + COL_W - PAD)], [Y(yy - 0.21)] * 2,
                                  color=RULE, lw=.8, zorder=1))

# ---- 컬러바 ----------------------------------------------------------------
CB_X, CB_W, CB_B, CB_H = 10.78, 0.30, 1.05, 5.60
cax = fig.add_axes([X(CB_X), Y(CB_B), X(CB_W), Y(CB_H)], zorder=2)
cax.imshow(np.linspace(1, 0, 512).reshape(-1, 1), cmap=TURBO, aspect="auto",
           extent=[0, 1, DMIN, DMAX])
cax.set_xticks([]); cax.set_yticks([])
for s in cax.spines.values():
    s.set_color(RULE); s.set_linewidth(1)

fig.text(X(CB_X), Y(7.22), "EFFECTIVE\nDENSITY\nKG/M³", ha="left", va="top",
         fontfamily=MONO, fontsize=7.6, color=INK2, linespacing=1.5, zorder=3)
for v, lbl, strong in TICKS:
    yy = CB_B + (v - DMIN) / (DMAX - DMIN) * CB_H
    fig.add_artist(plt.Line2D([X(CB_X + CB_W), X(CB_X + CB_W + 0.07)], [Y(yy)] * 2,
                              color=INK2 if strong else RULE, lw=1, zorder=3))
    fig.text(X(CB_X + CB_W + 0.12), Y(yy), lbl, ha="left", va="center", fontfamily=MONO,
             fontsize=8.4 if strong else 7.8, color=INK if strong else INK2,
             fontweight="bold" if strong else "normal", zorder=3)

for ext in ("png", "pdf"):
    p = f"{HERE}/desklamp_density_panel.{ext}"
    fig.savefig(p, dpi=220, facecolor="white")
    print(f"wrote {p}  ({os.path.getsize(p)/1024:.0f} KB)")
