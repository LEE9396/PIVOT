"""2-link / 3-link 를 '힌지 무시 / 힌지 모형화 / 실제값' 세 패널로 비교한다.

desklamp_density_panel.py 와 같은 판형이다. 다른 점은 두 가지.

  - 램프는 스캔 메시라 open3d 로 렌더했지만, 이 둘은 **직육면체** 라
    Drake FK 로 각 부위 자세를 받아 matplotlib 로 직접 그린다.
    (의존성이 늘지 않고, 상자는 볼록이라 면 정렬만으로 정확히 그려진다)
  - 비교축이 '탐색 전/후' 가 아니라 **힌지를 세느냐 마느냐** 다.

색은 turbo. 명도만이 아니라 색상으로 차이가 보이게 하려는 것이고,
램프 그림과 같은 규약이다. 값 자체는 표와 컬러바 눈금이 정확히 알려준다.

먼저 linkage_density_run.py 로 수치를 만들어 두어야 한다.

    ../robot_learning/scripts/run_drake_env.sh python figures/linkage_density_run.py
    ../robot_learning/scripts/run_drake_env.sh python figures/linkage_density_panel.py
"""
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import json

import numpy as np
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Polygon
from matplotlib import colormaps

import density_id_objects as obj

HERE = _pathlib.Path(__file__).resolve().parent
DATA = json.loads((HERE / "linkage_density_data.json").read_text())

TURBO = colormaps["turbo"]
PLATE, RULE = "#eef0f4", "#d3d8e0"
INK, INK2 = "#1b2029", "#5a6474"
BAD, WARN, GOOD = "#b03410", "#9a6a00", "#0a7a0a"
SANS, MONO = "DejaVu Sans", "DejaVu Sans Mono"

# 물체마다 보여 줄 자세. 관절이 접혀 있어야 부위 구분이 눈에 든다.
POSE_DEG = {"2link": [55.0], "3link": [50.0, 70.0]}
VIEW = {"2link": (34.0, 20.0), "3link": (38.0, 22.0)}
LIGHT = np.array([-0.35, -0.78, 0.52]); LIGHT /= np.linalg.norm(LIGHT)


# ---------------------------------------------------------------------------
# 상자 렌더 — Drake FK 로 자세를 받아 면 단위로 그린다
# ---------------------------------------------------------------------------
CORNERS = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1)
                    for sz in (-1, 1)], dtype=float)
FACES = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1),
         (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]


def box_faces(key):
    """부위마다 (면 꼭짓점 8x3, 면 목록) 을 월드 좌표로 돌려준다."""
    spec = obj.OBJECTS[key]
    rho = obj.bind_object(spec)
    plant, bodies = obj.build_plant(spec, rho)
    ctx = plant.CreateDefaultContext()
    plant.SetPositions(ctx, np.deg2rad(POSE_DEG[key]))
    out = []
    for part in spec.parts:
        X = plant.EvalBodyPoseInWorld(ctx, bodies[part.name])
        half = np.array(part.bbox_mm) * 0.5e-3
        v = (X.rotation().matrix() @ (CORNERS * half).T).T + X.translation()
        out.append((part.name, v))
    return out


def camera(az_deg, el_deg):
    """정사영 카메라. 화면 x/y 축과 시선 방향을 돌려준다."""
    a, e = np.radians(az_deg), np.radians(el_deg)
    fwd = np.array([np.sin(a) * np.cos(e), -np.cos(a) * np.cos(e), np.sin(e)])
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right)
    upv = np.cross(right, fwd)
    return right, upv, fwd


def projected_extent(geo, az, el, pad=0.06):
    """화면에 투영했을 때의 (가로, 세로) 크기와 중심.

    이미지 축을 이 비율로 잡아야 한다. 고정 비율로 잡으면 납작한 물체가
    세로로 긴 상자 안에 떠서 표와의 사이가 크게 빈다 (실제로 그랬다).
    """
    right, upv, _ = camera(az, el)
    allp = np.vstack([np.column_stack([v @ right, v @ upv]) for _, v in geo])
    lo, hi = allp.min(0), allp.max(0)
    size = (hi - lo) * (1.0 + 2 * pad)
    return size, (lo + hi) / 2


def draw_object(ax, geo, colors, az, el, center, size):
    """면을 뒤에서 앞으로 칠한다 (화가 알고리즘). 상자는 볼록이라 이걸로 정확하다."""
    right, upv, fwd = camera(az, el)
    polys = []
    for name, v in geo:
        p2 = np.column_stack([v @ right, v @ upv])
        depth = v @ fwd
        for f in FACES:
            quad = p2[list(f)]
            n = np.cross(v[f[1]] - v[f[0]], v[f[2]] - v[f[0]])
            n /= max(np.linalg.norm(n), 1e-12)
            if n @ fwd > 0:                       # 뒤통수 면은 안 그린다
                continue
            lam = max(float(n @ LIGHT), 0.0)
            # 색-값 대응이 깨지지 않게 음영은 72~100% 로만 (램프 그림과 같다)
            shade = 0.72 + 0.28 * (0.25 + 0.75 * lam)
            polys.append((depth[list(f)].mean(), quad,
                          np.clip(np.array(colors[name]) * shade, 0, 1)))
    for _, quad, col in sorted(polys, key=lambda t: -t[0]):
        ax.add_patch(Polygon(quad, closed=True, facecolor=col,
                             edgecolor=col * 0.82, lw=0.5, zorder=2))
    ax.set_xlim(center[0] - size[0] / 2, center[0] + size[0] / 2)
    ax.set_ylim(center[1] - size[1] / 2, center[1] + size[1] / 2)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.patch.set_alpha(0)


# ---------------------------------------------------------------------------
def build(key):
    d = DATA[key]
    spec = obj.OBJECTS[key]
    geo = box_faces(key)
    parts, hinge_g = d["parts"], d["hinge_g"]
    n_hinge = d["n_joint"]
    hinge_names = [f"{j.name}_hinge" for j in spec.joints]
    hinge_gt = spec.joints[0].hinge_density
    hinge_vol = [j.hinge_mass_kg / j.hinge_density for j in spec.joints]

    # 컬러 정의역: 이 그림에 나오는 모든 밀도를 담는다.
    #
    # **로그로 잡는다.** 3-link 는 700 ~ 5200 으로 7.4 배 범위라, 선형으로
    # 깔면 5200 하나가 눈금을 다 먹고 700/1150/1400 이 아래 20 % 에 뭉쳐
    # 서로 구별이 안 된다 (실제로 그렇게 나와서 고쳤다). 밀도는 비율척도라
    # 로그가 맞고, 그러면 네 값이 고르게 퍼진다.
    vals = list(d["gt_rho"]) + list(d["ignored"]["rho"]) + list(d["modelled"]["rho"])
    vals += [hinge_gt]
    lo, hi = np.log10(min(vals)), np.log10(max(vals))
    pad = 0.06 * (hi - lo)
    LMIN, LMAX = lo - pad, hi + pad
    DMIN, DMAX = 10 ** LMIN, 10 ** LMAX
    unit = lambda v: float(np.clip((np.log10(max(v, 1e-9)) - LMIN) / (LMAX - LMIN), 0, 1))
    ramp = lambda v: np.array(TURBO(unit(v))[:3])
    hexof = lambda v: matplotlib.colors.to_hex(ramp(v))

    def rows_for(case):
        """표에 들어갈 줄. (이름, ρ, M, ΔM, 색, 스와치)"""
        rows = []
        if case == "gt":
            rho, mass, dm = d["gt_rho"], d["gt_mass"], [None] * len(parts)
        else:
            c = d[case]
            rho, mass, dm = c["rho"], c["mass"], c["rel_mass_err"]
        for i, p in enumerate(parts):
            txt = "—" if dm[i] is None else f"{100*dm[i]:+.2f}%"
            col = INK2 if dm[i] is None else (
                GOOD if abs(dm[i]) < 0.01 else WARN if abs(dm[i]) < 0.05 else BAD)
            rows.append((p, f"{rho[i]:.1f}", f"{mass[i]:.4f}", txt, col, ramp(rho[i])))
        if case == "ignored":
            for hn in hinge_names:
                rows.append((hn, "—", "—", "not modelled", BAD, None))
        else:
            hr = d["modelled"]["hinge_rho"] if case == "modelled" else [hinge_gt] * n_hinge
            for hn, r, v in zip(hinge_names, hr, hinge_vol):
                txt = "—" if case == "gt" else f"{100*(r-hinge_gt)/hinge_gt:+.2f}%"
                rows.append((hn, f"{r:.1f}", f"{r*v:.4f}", txt,
                             INK2 if case == "gt" else GOOD, ramp(r)))
        return rows

    COLS = [
        dict(key="ignored", name="Hinge ignored",
             src=f"estimator has no {hinge_g:.0f} g hinge body",
             foot=f"claims ±{d['ignored']['claimed_half_pct']:.2f}%  "
                  f"({d['ignored']['rounds']:.1f} rounds)"),
        dict(key="modelled", name="Hinge modelled",
             src=f"hinge solved as an extra body",
             foot=f"claims ±{d['modelled']['claimed_half_pct']:.2f}%  "
                  f"({d['modelled']['rounds']:.1f} rounds)"),
        dict(key="gt", name="Ground truth",
             src="CAD volume × material density", foot="—"),
    ]
    for c in COLS:
        c["rows"] = rows_for(c["key"])
    n_row = max(len(c["rows"]) for c in COLS)

    # ---- 판형 --------------------------------------------------------------
    COL_W, X0, PAD = 3.30, 0.70, 0.20
    W = 12.6
    ext_size, ext_center = projected_extent(geo, *VIEW[key])
    _img_h = float(np.clip((COL_W - 0.20) * ext_size[1] / ext_size[0], 1.5, 3.4))
    H = 3.30 + _img_h + 0.42 * n_row
    fig = plt.figure(figsize=(W, H), dpi=100)
    fig.patch.set_facecolor("white")
    X = lambda v: v / W
    Y = lambda v: v / H

    TOP = H - 0.44
    fig.text(X(0.35), Y(TOP), f"Per-link density",
             ha="left", va="center", fontfamily=SANS, fontsize=17,
             fontweight="bold", color="#0f131a")
    fig.text(X(12.25), Y(TOP), d["label"], ha="right", va="center",
             fontfamily=SANS, fontsize=11, color="#4c5566")
    fig.add_artist(plt.Line2D([X(0.35), X(12.25)], [Y(TOP - 0.28)] * 2,
                              color="#dee3ec", lw=1))

    PL_TOP, PL_BOT = TOP - 0.58, 0.30
    fig.add_artist(FancyBboxPatch((X(0.35), Y(PL_BOT)), X(12.25) - X(0.35),
                                  Y(PL_TOP) - Y(PL_BOT),
                                  boxstyle="round,pad=0,rounding_size=0.004",
                                  facecolor=PLATE, edgecolor=RULE, lw=1, zorder=0))

    TITLE_Y = PL_TOP - 0.36
    SRC_Y = TITLE_Y - 0.24
    img_w = COL_W - 0.20
    img_h = float(np.clip(img_w * ext_size[1] / ext_size[0], 1.5, 3.4))
    IMG_TOP = SRC_Y - 0.22
    IMG_BOT = IMG_TOP - img_h
    HDR_Y = IMG_BOT - 0.34
    ROW_Y = [HDR_Y - 0.42 - 0.42 * i for i in range(n_row)]
    FOOT_Y = ROW_Y[-1] - 0.40

    for i, c in enumerate(COLS):
        x0 = X0 + i * COL_W
        if i:
            fig.add_artist(plt.Line2D([X(x0 - 0.05)] * 2, [Y(PL_BOT + 0.30), Y(PL_TOP - 0.18)],
                                      color=RULE, lw=1, zorder=1))
        fig.text(X(x0 + PAD), Y(TITLE_Y), c["name"], ha="left", va="center",
                 fontfamily=SANS, fontsize=13.5, fontweight="bold", color=INK, zorder=3)
        fig.text(X(x0 + PAD), Y(SRC_Y), c["src"], ha="left", va="center",
                 fontfamily=MONO, fontsize=8.6, color=INK2, zorder=3)

        ax = fig.add_axes([X(x0 + 0.10), Y(IMG_BOT), X(img_w), Y(img_h)],
                          zorder=2)
        colors = {}
        for name, rho in zip(parts, (d["gt_rho"] if c["key"] == "gt"
                                     else d[c["key"]]["rho"])):
            colors[name] = ramp(rho)
        draw_object(ax, geo, colors, *VIEW[key], ext_center, ext_size)

        cx = {"rho": x0 + 1.86, "m": x0 + 2.50, "d": x0 + COL_W - PAD}
        for lbl, k in [("PART", None), ("ρ KG/M³", "rho"), ("M KG", "m"), ("ΔM", "d")]:
            fig.text(X(x0 + PAD if k is None else cx[k]), Y(HDR_Y), lbl,
                     ha="left" if k is None else "right", va="center",
                     fontfamily=MONO, fontsize=7.4, color=INK2, zorder=3)
        fig.add_artist(plt.Line2D([X(x0 + PAD), X(x0 + COL_W - PAD)],
                                  [Y(HDR_Y - 0.13)] * 2, color=RULE, lw=1, zorder=1))

        for (part, rho, m, dm, dcol, swc), yy in zip(c["rows"], ROW_Y):
            if swc is None:
                fig.add_artist(Rectangle((X(x0 + PAD), Y(yy - 0.05)), X(0.105), Y(0.105),
                                         facecolor="none", edgecolor="#00000038",
                                         lw=.6, hatch="///", zorder=3))
            else:
                fig.add_artist(Rectangle((X(x0 + PAD), Y(yy - 0.05)), X(0.105), Y(0.105),
                                         facecolor=swc, edgecolor="#00000038",
                                         lw=.6, zorder=3))
            faint = swc is None
            fig.text(X(x0 + PAD + 0.18), Y(yy), part, ha="left", va="center",
                     fontfamily=MONO, fontsize=9.2,
                     color="#9aa3b0" if faint else INK2, zorder=3)
            for val, k, col in [(rho, "rho", INK), (m, "m", INK), (dm, "d", dcol)]:
                fig.text(X(cx[k]), Y(yy), val, ha="right", va="center",
                         fontfamily=MONO,
                         fontsize=7.6 if (k == "d" and len(val) > 8) else 9.2,
                         color="#9aa3b0" if (faint and k != "d") else col, zorder=3)
            fig.add_artist(plt.Line2D([X(x0 + PAD), X(x0 + COL_W - PAD)],
                                      [Y(yy - 0.21)] * 2, color=RULE, lw=.8, zorder=1))

        fig.text(X(x0 + PAD), Y(FOOT_Y), c["foot"], ha="left", va="center",
                 fontfamily=MONO, fontsize=8.4, color=INK2, zorder=3)

    # ---- 컬러바 ------------------------------------------------------------
    CB_X, CB_W = 10.78, 0.30
    CB_B, CB_H = IMG_BOT + 0.05, (IMG_TOP - IMG_BOT) * 0.94
    cax = fig.add_axes([X(CB_X), Y(CB_B), X(CB_W), Y(CB_H)], zorder=2)
    # 로그 매핑이므로 축은 0~1 정규화 좌표로 두고, 눈금만 값에서 환산한다.
    cax.imshow(np.linspace(1, 0, 512).reshape(-1, 1), cmap=TURBO, aspect="auto",
               extent=[0, 1, 0, 1])
    cax.set_xticks([]); cax.set_yticks([])
    for s in cax.spines.values():
        s.set_color(RULE); s.set_linewidth(1)
    fig.text(X(CB_X), Y(TITLE_Y), "EFFECTIVE\nDENSITY\nKG/M³", ha="left", va="top",
             fontfamily=MONO, fontsize=7.6, color=INK2, linespacing=1.5, zorder=3)

    ticks = [(DMIN, f"{DMIN:.0f}", False), (DMAX, f"{DMAX:.0f}", False)]
    for name, v in zip(parts, d["gt_rho"]):
        ticks.append((v, f"{v:.0f} {name.split('_')[0]}", True))
    ticks.append((hinge_gt, f"{hinge_gt:.0f} hinge", True))
    used = []
    # **부위 눈금이 먼저 자리를 잡는다.** 값 순으로 훑으면 범위 끝값(5891)이
    # 정작 중요한 5200 link0 을 밀어낸다 (실제로 그랬다). 이름 붙은 눈금이
    # 이 그림의 목적이고, 끝값은 남는 자리에만 넣는다.
    for v, lbl, strong in sorted(ticks, key=lambda t: (not t[2], -t[0])):
        yy = CB_B + unit(v) * CB_H
        if any(abs(yy - u) < 0.15 for u in used):
            continue
        used.append(yy)
        fig.add_artist(plt.Line2D([X(CB_X + CB_W), X(CB_X + CB_W + 0.07)], [Y(yy)] * 2,
                                  color=INK2 if strong else RULE, lw=1, zorder=3))
        fig.text(X(CB_X + CB_W + 0.12), Y(yy), lbl, ha="left", va="center",
                 fontfamily=MONO, fontsize=8.4 if strong else 7.8,
                 color=INK if strong else INK2,
                 fontweight="bold" if strong else "normal", zorder=3)

    for ext in ("png", "pdf"):
        p = HERE / f"linkage_density_panel_{key}.{ext}"
        fig.savefig(p, dpi=220, facecolor="white")
        print(f"wrote {p.name}  ({p.stat().st_size/1024:.0f} KB)")
    plt.close(fig)
    return {n: hexof(v) for n, v in zip(parts, d["gt_rho"])}


if __name__ == "__main__":
    for k in ("2link", "3link"):
        sw = build(k)
        print(f"  {k} 스와치: " + ", ".join(f"{n}={c}" for n, c in sw.items()))
