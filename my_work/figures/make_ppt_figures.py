"""발표·논문용 그림 3장을 만든다.

기존 study_*.py 가 남기는 진단 그림과 달리, 이 셋은 **설명을 위해** 만든다.

  fig_nullspace       정리 2 의 영공간 — 왜 형상을 바꿔야 하는가 (논문 Fig.2)
  fig_column_angle    부위 열이 나란해진다 (47도 -> 0.5도)
  fig_spread_source   라운드 편차는 잡음이 만들지 물체가 만들지 않는다

색은 turbo 를 쓰지 않는다. 밀도는 크기(magnitude)라 **단일 색상 순차 램프**가
맞고, 가설 A/B 는 정체성(identity)이라 범주형 두 칸을 쓴다. 둘을 같은 그림에서
섞지 않도록 패널을 나눴다.

실행:
    cd ~/Desktop/PIVOT/my_work
    ../robot_learning/scripts/run_drake_env.sh python figures/make_ppt_figures.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from pathlib import Path

OUT = Path(__file__).resolve().parent

# ── 팔레트 ───────────────────────────────────────────────────────────────
# 범주형 1·2 번 칸. 인접 쌍 CVD dE 9.1 / 정상시야 dE 19.6 으로 검증된 조합.
C1, C2 = "#2a78d6", "#eb6834"
# 순차(단일 색상) 램프. 이산 마크라 가장 밝은 칸은 250 스텝 아래로 내리지 않는다.
SEQ = ["#86b6ef", "#2a78d6", "#0d366b"]
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


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> figures/{name}.png / .pdf")


# ═════════════════════════════════════════════════════════════════════════
# 1. 영공간 — 왜 형상을 바꿔야 하는가
# ═════════════════════════════════════════════════════════════════════════
# 실물 3-link 의 링크 길이를 그대로 쓴다 (mm). 단면은 40x40.
L = np.array([150.0, 110.0, 97.5])
SEC = 40.0
V = L * SEC * SEC * 1e-9          # m^3
RHO_A = np.array([5200.0, 1400.0, 700.0])   # 실제 제작한 재질

def chain(angles_deg):
    """관절각을 받아 각 링크의 (시작점, 방향, 도심) 을 돌려준다. 2D 평면."""
    p, th = np.zeros(2), 0.0
    starts, dirs, cs = [], [], []
    for i, li in enumerate(L):
        th += np.deg2rad(angles_deg[i])
        d = np.array([np.cos(th), np.sin(th)])
        starts.append(p.copy()); dirs.append(d); cs.append(p + d * li / 2)
        p = p + d * li
    return np.array(starts), np.array(dirs), np.array(cs)


EXT = [0.0, 0.0, 0.0]             # 펼침
FOLD = [0.0, -90.0, -90.0]        # 접음

_, _, c_ext = chain(EXT)

# 정리 2 의 영공간을 실제로 푼다:  sum u_i V_i = 0  이고  sum u_i V_i c_i = 0
# 펼친 자세에서는 도심이 x 축 위에 일렬이라 제약이 2개 -> 미지수 3개에 해가 1차원.
A_ns = np.vstack([V, V * c_ext[:, 0]])
_, _, Vt = np.linalg.svd(A_ns)
# 제약이 u 에 직접 걸린다: A_ns @ u = 0  (V 로 나누면 안 된다)
u = Vt[-1] / np.abs(Vt[-1]).max()   # 밀도 섭동 방향
RHO_B = RHO_A + 320.0 * u

m_A, m_B = RHO_A * V, RHO_B * V


def com(rho, angles):
    _, _, cs = chain(angles)
    m = rho * V
    return (m[:, None] * cs).sum(0) / m.sum()


com_ext_A, com_ext_B = com(RHO_A, EXT), com(RHO_B, EXT)
com_fold_A, com_fold_B = com(RHO_A, FOLD), com(RHO_B, FOLD)
gap = np.linalg.norm(com_fold_A - com_fold_B)

# 밀도 -> 순차 램프. 순위가 아니라 **절대값**에 매핑해야 A/B 의 차이가 보인다.
# 정의역은 로그다. 700~5200 은 7.4 배라 선형으로 깔면 5200 하나가 눈금을 먹는다.
from matplotlib.colors import LinearSegmentedColormap
CMAP = LinearSegmentedColormap.from_list("pivot_blue", SEQ)
LO, HI = np.log(620.0), np.log(5800.0)


def shades(rho):
    return [CMAP((np.log(r) - LO) / (HI - LO)) for r in rho]


def draw_chain(ax, angles, rho):
    starts, dirs, _ = chain(angles)
    for s_, d, li, col in zip(starts, dirs, L, shades(rho)):
        n = np.array([-d[1], d[0]])
        corner = s_ - n * SEC / 2
        ang = np.rad2deg(np.arctan2(d[1], d[0]))
        ax.add_patch(Rectangle(corner, li, SEC, angle=ang,
                               rotation_point=tuple(corner),
                               facecolor=col, edgecolor=SURF, linewidth=2.0))
    ax.plot(0, 0, marker="s", ms=7, color=INK, zorder=5)
    ax.annotate("wrist F/T", (0, 0), textcoords="offset points", xytext=(0, 15),
                ha="center", fontsize=8, color=INK2)


def frame(ax):
    ax.set_xlim(-45, 375); ax.set_ylim(-158, 46)
    ax.set_aspect("equal"); ax.axis("off")


fig = plt.figure(figsize=(11.6, 3.05))
gs = fig.add_gridspec(1, 3, wspace=0.06)
axs = [fig.add_subplot(gs[0, i]) for i in range(3)]

for ax, rho, com_pt, ttl, sub in [
    (axs[0], RHO_A, com_ext_A, "Density A", "the object we built"),
    (axs[1], RHO_B, com_ext_B, "Density B", "link1 −23 %,  link2 +29 %"),
]:
    draw_chain(ax, EXT, rho)
    frame(ax)
    ax.axvline(com_pt[0], color=INK3, lw=0.9, ls=(0, (4, 3)), ymax=0.62, zorder=0)
    ax.plot(*com_pt, marker="v", ms=12, color=INK, markeredgecolor=SURF,
            markeredgewidth=1.6, zorder=6)
    ax.set_title(ttl, color=INK, pad=16)
    ax.text(0.5, 1.005, sub, transform=ax.transAxes, ha="center",
            fontsize=8.8, color=INK3)
    ax.text(0.5, 0.105, "ρ = " + "  ".join(f"{r:,.0f}" for r in rho) + "  kg/m³",
            transform=ax.transAxes, ha="center", fontsize=8.6, color=INK2,
            family="monospace")
    ax.text(0.5, 0.008, f"total {rho @ V:.3f} kg     CoM x = {com_pt[0]:.1f} mm",
            transform=ax.transAxes, ha="center", fontsize=9.2, color=INK,
            family="monospace")

# ── 3번 패널 — 접으면 갈라진다 (간격이 2 mm 라 확대 인셋이 필요하다) ──────────
ax = axs[2]
draw_chain(ax, FOLD, RHO_A)
frame(ax)
ax.set_title("fold one joint", color=INK, pad=16)
ax.text(0.5, 1.005, "the same two hypotheses", transform=ax.transAxes,
        ha="center", fontsize=8.8, color=INK3)
for pt, col in [(com_fold_A, C1), (com_fold_B, C2)]:
    ax.plot(*pt, marker="v", ms=9, color=col, markeredgecolor=SURF,
            markeredgewidth=1.2, zorder=6)

mid = (com_fold_A + com_fold_B) / 2
ZW = 4.5                                   # 확대창 반폭 [mm]
ins = ax.inset_axes([0.60, 0.30, 0.38, 0.46])
ins.set_xlim(mid[0] - ZW, mid[0] + ZW)
ins.set_ylim(mid[1] - ZW * 0.85, mid[1] + ZW * 0.72)
ins.set_facecolor("#f1f3f6")
for sp in ins.spines.values():
    sp.set_color(INK3); sp.set_linewidth(0.9)
ins.set_xticks([]); ins.set_yticks([])
for pt, col, lab, dy in [(com_fold_A, C1, "A", 14), (com_fold_B, C2, "B", 14)]:
    ins.plot(*pt, marker="v", ms=13, color=col, markeredgecolor=SURF,
             markeredgewidth=1.6, zorder=6)
    ins.annotate(lab, pt, textcoords="offset points", xytext=(0, dy),
                 ha="center", fontsize=10, color=col, fontweight="bold")
ins.text(0.5, 0.10, f"{gap:.1f} mm", transform=ins.transAxes,
         ha="center", fontsize=10.5, color=INK, fontweight="bold")
from matplotlib.patches import Circle, ConnectionPatch
ax.add_patch(Circle(tuple(mid), 8.5, facecolor="none", edgecolor=INK3, lw=0.9,
                    zorder=7))
ax.add_artist(ConnectionPatch(xyA=(mid[0] + 6.0, mid[1] - 6.0), coordsA=ax.transData,
                              xyB=(0, 1), coordsB=ins.transAxes,
                              color=INK3, lw=0.9, zorder=7))
ins.text(0.045, 0.94, "×15", transform=ins.transAxes, ha="left", va="top",
         fontsize=8, color=INK2)
ax.text(0.5, 0.008, "the two hypotheses separate", transform=ax.transAxes,
        ha="center", fontsize=9.2, color=INK, family="monospace")

# ── 패널 1·2 를 묶는 판정 문구 ────────────────────────────────────────────
p0, p1 = axs[0].get_position(), axs[1].get_position()
xc = (p0.x0 + p1.x1) / 2
y_rule = p0.y0 - 0.055
fig.add_artist(plt.Line2D([p0.x0 + 0.015, p1.x1 - 0.015], [y_rule, y_rule],
                          transform=fig.transFigure, color=C2, lw=1.2))
fig.text(xc, y_rule - 0.115, "identical to the sensor  —  same mass, same CoM,"
                             "  same wrench in every gravity direction",
         ha="center", fontsize=9.4, color=C2, fontweight="bold")
fig.text(0.5, y_rule - 0.30,
         "Prop. 2   null M(θ) = { u : Σ uᵢVᵢ = 0  and  Σ uᵢVᵢcᵢ(θ) = 0 }"
         "      —  a redistribution that preserves total mass and first moment "
         "is invisible.  Changing θ is the only way to see it.",
         ha="center", fontsize=8.8, color=INK2)
save(fig, "fig_nullspace")


# ═════════════════════════════════════════════════════════════════════════
# 2. 부위 열이 나란해진다
# ═════════════════════════════════════════════════════════════════════════
# 백색화한 회귀행렬에서 두 부위 열이 이루는 각도. 무작위 자세 40개 평균.
p_ang = np.array([2, 3, 4, 5, 6])
adj = np.array([47.0, 36.8, 10.9, 1.6, 0.5])
non = np.array([np.nan, 51.8, 18.5, 3.4, 1.1])

fig, ax = plt.subplots(figsize=(6.6, 4.3))
ax.axhspan(0.25, 2.0, color=C2, alpha=0.09, zorder=0)
ax.text(1.88, 0.62, "numerically\nindistinguishable", ha="left", va="center",
        fontsize=8.5, color=C2, linespacing=1.35)

ax.plot(p_ang, non, marker="s", color=C2, label="non-adjacent parts", zorder=3)
ax.plot(p_ang, adj, marker="o", color=C1, label="adjacent parts", zorder=4)

for x, y, col, dx, dy, ha in [(p_ang[0], adj[0], C1, 0, -19, "center"),
                              (p_ang[-1], adj[-1], C1, 0, -18, "center"),
                              (p_ang[1], non[1], C2, 0, 12, "center"),
                              (p_ang[-1], non[-1], C2, 0, 12, "center")]:
    ax.annotate(f"{y:.1f}°", (x, y), textcoords="offset points", xytext=(dx, dy),
                ha=ha, fontsize=9, color=col, fontweight="bold")

ax.set_yscale("log")
ax.set_yticks([0.5, 1, 2, 5, 10, 20, 50])
ax.set_yticklabels(["0.5°", "1°", "2°", "5°", "10°", "20°", "50°"])
ax.set_xticks(p_ang)
ax.set_xlabel("number of links  p")
ax.set_ylabel("angle between part columns")
ax.set_xlim(1.78, 6.30); ax.set_ylim(0.25, 95)
ax.grid(axis="y", zorder=0)
ax.legend(loc="upper right", fontsize=9, labelcolor=INK2,
          bbox_to_anchor=(1.005, 1.02))
ax.set_title("Part columns become parallel as the chain grows",
             color=INK, loc="left", pad=10)
fig.text(0.5, -0.115,
         "Whitened regressor, mean over 40 random configurations.  Parts differ only in "
         "the six torque rows that carry signal (Cor. 1),\nand that is exactly where angle "
         "error dominates the sensor noise — 22× at p=3, 41,690× at p=4.  Whitening\n"
         "suppresses those rows, leaving the force rows, which were parallel to begin with.",
         ha="center", fontsize=8.6, color=INK2, linespacing=1.5)
save(fig, "fig_column_angle")


# ═════════════════════════════════════════════════════════════════════════
# 3. 편차의 원인 — 잡음이지 물체가 아니다
# ═════════════════════════════════════════════════════════════════════════
# p=5, 목표 반폭 2%, 예산 30라운드. 물체와 잡음을 분리해 각각 8회.
runs = {
    "object fixed,\nnoise varied":  np.array([11, 8, 10, 5, 7, 17, 13, 19]),
    "object varied,\nnoise fixed":  np.array([11, 15, 9, 8, 8, 8, 13, 17]),
}
cols = [C1, C2]

fig, ax = plt.subplots(figsize=(7.2, 3.35))
rng = np.random.default_rng(0)
for i, ((lab, vals), col) in enumerate(zip(runs.items(), cols)):
    y = len(runs) - 1 - i
    ax.plot([vals.min(), vals.max()], [y, y], color=col, lw=3.5, alpha=0.22,
            solid_capstyle="round", zorder=1)
    ax.scatter(vals, y + rng.uniform(-0.085, 0.085, vals.size), s=62, color=col,
               edgecolor=SURF, linewidth=1.6, zorder=3)
    ax.plot([vals.mean()], [y], marker="|", ms=22, mew=2.2, color=INK, zorder=4)
    ax.annotate(f"{vals.min()}–{vals.max()} rounds     σ = {vals.std():.1f}",
                (20.6, y), fontsize=9.5, color=col, va="center", fontweight="bold")

ax.set_yticks([1, 0]); ax.set_yticklabels(list(runs), fontsize=9.5, color=INK)
ax.set_ylim(-0.6, 1.6)
ax.set_xlim(3, 27.5)
ax.set_xticks([5, 10, 15, 20])
ax.set_xlabel("rounds to reach the 2 % target")
ax.grid(axis="x", zorder=0)
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.set_title("The spread comes from measurement noise, not from the object",
             color=INK, loc="left", pad=10)
fig.text(0.5, -0.155,
         "p = 5, 8 seeds each.  Holding the object fixed and varying only the noise "
         "spreads wider (σ = 4.5) than varying the object\nwith the noise held fixed "
         "(σ = 3.3)  —  so a median alone would mislead; we report the range and the "
         "fraction that converged.",
         ha="center", fontsize=8.6, color=INK2, linespacing=1.5)
save(fig, "fig_spread_source")

print("\n확인용 수치")
print(f"  섭동 방향 u          {np.round(u, 3)}")
print(f"  밀도 A               {np.round(RHO_A, 1)}")
print(f"  밀도 B               {np.round(RHO_B, 1)}")
print(f"  총질량 A / B         {m_A.sum():.6f} / {m_B.sum():.6f} kg")
print(f"  펼침 도심 A / B [mm] {np.round(com_ext_A, 4)} / {np.round(com_ext_B, 4)}")
print(f"  접음 도심 A / B [mm] {np.round(com_fold_A, 3)} / {np.round(com_fold_B, 3)}")
print(f"  접었을 때 간격       {gap:.3f} mm")
