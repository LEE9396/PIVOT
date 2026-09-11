"""논문 Fig. 4 용 N-link 그림.

render_nlink.py 와 같은 물체·같은 seed 를 쓰되, 논문에 실을 형태로 그린다.
  - 왼쪽 3-link / 오른쪽 6-link, 각각 서로 다른 joint configuration 3개
  - 배경(격자·축·눈금·패널) 전부 제거, 투명 배경
  - 관절마다 그 관절의 각도를 숫자로 표기하고, 겹치면 밀어내며 지시선을 긋는다
  - one column 폭(약 3.4 in)에 맞춘 크기. 저장 후 빈 여백을 잘라낸다.
사용: ../robot_learning/scripts/run_drake_env.sh python -u render_nlink_paper.py
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import proj3d

import nlink, density_id_objects as obj, density_id_drake as alg

MM = 1e-3
FACES = [(0,1,2,3),(4,5,6,7),(0,1,5,4),(2,3,7,6),(0,3,7,4),(1,2,6,5)]
GREYS = ["#3f4a56", "#5d6b7a", "#7d8b99", "#9caab6", "#bcc7d0", "#dae1e7"]
RED = "#b3261e"

def box_corners(dims):
    d = np.array(dims) / 2.0
    return np.array([[sx*d[0], sy*d[1], sz*d[2]]
                     for sx,sy,sz in [(-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),
                                      (-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)]])

def draw(ax, spec, theta):
    alg.KIN_PLANT.SetPositions(alg.KIN_CTX, np.atleast_1d(theta))
    pts, origins = [], []
    for i, part in enumerate(spec.parts):
        X = alg.KIN_PLANT.EvalBodyPoseInWorld(alg.KIN_CTX, alg.KIN_BODIES[part.name])
        w = (X.rotation().matrix() @ box_corners([d*MM for d in part.bbox_mm]).T).T \
            + X.translation()
        pts.append(w); origins.append(np.asarray(X.translation(), dtype=float))
        ax.add_collection3d(Poly3DCollection(
            [w[list(f)] for f in FACES], facecolors=GREYS[i % len(GREYS)],
            edgecolors="#20262c", linewidths=0.35))
    P = np.vstack(pts)
    c, r = P.mean(0), (P.max(0)-P.min(0)).max()/2*1.02
    ax.set_xlim(c[0]-r, c[0]+r); ax.set_ylim(c[1]-r, c[1]+r); ax.set_zlim(c[2]-r, c[2]+r)
    ax.set_box_aspect((1,1,1)); ax.set_axis_off()
    ax.set_proj_type("ortho"); ax.view_init(elev=20, azim=-60)
    return origins

def label_joints(ax, origins, theta):
    """관절 각도를 2D 로 투영해 놓되, 겹치면 위아래로 밀고 지시선을 긋는다."""
    ax.get_figure().canvas.draw()               # 투영 행렬 확정
    placed = []
    for j, th in enumerate(np.atleast_1d(theta)):
        o = origins[j+1]
        x, y, _ = proj3d.proj_transform(o[0], o[1], o[2], ax.get_proj())
        ax_x, ax_y = ax.transLimits.transform((x, y)) if False else (x, y)
        # 데이터 좌표 -> 축 좌표
        px, py = ax.transData.transform((x, y))
        fx, fy = ax.transAxes.inverted().transform((px, py))
        # 후보 위치를 바깥쪽(오른쪽) 우선, 그다음 위/아래 번갈아 시도한다.
        cands = [(0.10, 0.0)] + [(0.10, s*d) for d in (0.09, 0.18, 0.27)
                                 for s in (1, -1)]
        tx, ty = fx + 0.10, fy
        for dx, dy in cands:
            tx, ty = fx + dx, min(max(fy + dy, 0.03), 0.93)
            if not any(abs(tx-ux) < 0.34 and abs(ty-uy) < 0.070
                       for ux, uy in placed):
                break
        placed.append((tx, ty))
        t = ax.annotate(f"$\\theta_{{{j+1}}}${np.degrees(th):.0f}$^\\circ$",
                        xy=(fx, fy), xycoords="axes fraction",
                        xytext=(tx, ty), textcoords="axes fraction",
                        fontsize=4.4, color=RED, weight="bold",
                        ha="left", va="center", zorder=30,
                        arrowprops=dict(arrowstyle="-", color=RED,
                                        lw=0.35, shrinkA=0.5, shrinkB=1.0))
        t.set_path_effects([pe.withStroke(linewidth=1.1, foreground="white")])

def crop(path):
    from PIL import Image
    im = Image.open(path).convert("RGBA")
    bbox = im.split()[-1].getbbox()             # 투명하지 않은 영역
    if bbox:
        im.crop(bbox).save(path)
        print("cropped ->", im.crop(bbox).size)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--out", default="figures/nlink_paper.png")
    a = ap.parse_args()

    COLS = [3, 6]
    CFG = {   # 도 단위. 정보이득 탐색이 실제로 고르는 종류의 서로 다른 자세들.
        3: [[20, 150], [110, 40], [165, 95]],
        6: [[20, 150, 60, 130, 35], [110, 40, 145, 25, 100], [165, 95, 30, 120, 70]],
    }
    nrow = 3
    fig = plt.figure(figsize=(3.45, 2.95))
    for col, p in enumerate(COLS):
        spec = nlink.make_spec(p, seed=a.seed)
        obj.set_measurement_averaging(); obj.bind_object(spec)
        nj = len(spec.joints)
        for row in range(nrow):
            th = np.radians(np.array(CFG[p][row][:nj], dtype=float))
            ax = fig.add_subplot(nrow, 2, row*2+col+1, projection="3d")
            origins = draw(ax, spec, th)
            H = 0.905 / nrow
            ax.set_position([col*0.5 - 0.105, (nrow-1-row)*H - 0.012, 0.5 + 0.21, H*1.30])
            label_joints(ax, origins, th)
    for col, p in enumerate(COLS):
        fig.text(0.25 + col*0.5, 0.995, f"{p}-link chain", ha="center", va="top",
                 fontsize=6.2, weight="bold")
    fig.savefig(a.out, dpi=500, transparent=True)
    crop(a.out)
    print("saved ->", a.out)

if __name__ == "__main__":
    main()
