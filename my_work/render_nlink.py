"""합성 사슬이 실제로 어떻게 생겼는지 그린다.

study_scaling 이 쓰는 것과 **같은 seed** 로 물체를 만들어, 펼침/중간/접힘
세 자세를 3차원으로 보여준다. 부위마다 색이 다르고, 손목(센서 원점)은 원점이다.
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import nlink, density_id_objects as obj, density_id_drake as alg

MM = 1e-3
FACES = [(0,1,2,3),(4,5,6,7),(0,1,5,4),(2,3,7,6),(0,3,7,4),(1,2,6,5)]

def box_corners(dims, center):
    d = np.array(dims)/2.0
    c = np.array(center)
    return np.array([[c[0]+sx*d[0], c[1]+sy*d[1], c[2]+sz*d[2]]
                     for sx,sy,sz in [(-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),
                                      (-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)]])

def draw(ax, spec, theta, title):
    alg.KIN_PLANT.SetPositions(alg.KIN_CTX, np.atleast_1d(theta))
    pts = []
    for i, part in enumerate(spec.parts):
        X = alg.KIN_PLANT.EvalBodyPoseInWorld(alg.KIN_CTX, alg.KIN_BODIES[part.name])
        loc = box_corners([d*MM for d in part.bbox_mm], (0.,0.,0.))
        w = (X.rotation().matrix() @ loc.T).T + X.translation()
        pts.append(w)
        col = plt.cm.turbo(i/max(len(spec.parts)-1,1))
        ax.add_collection3d(Poly3DCollection([w[list(f)] for f in FACES],
            facecolors=col, edgecolors="k", linewidths=0.4, alpha=0.85))
        ax.text(*w.mean(0), f"{i}", fontsize=7, weight="bold")
    P = np.vstack(pts)
    ax.scatter([0],[0],[0], c="r", s=40, marker="x")     # 손목
    c, r = P.mean(0), (P.max(0)-P.min(0)).max()/2*1.1
    for setl, m in ((ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)):
        setl(c[m]-r, c[m]+r)
    ax.set_title(title, fontsize=9); ax.set_box_aspect((1,1,1))
    ax.tick_params(labelsize=6)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", type=int, nargs="+", default=[4,5,6])
    ap.add_argument("--seed", type=int, default=1000, help="study_scaling 의 seed 0 번 물체")
    ap.add_argument("--out", default="figures/nlink_shapes.png")
    a = ap.parse_args()

    fig = plt.figure(figsize=(4.2*3, 3.8*len(a.parts)))
    for row, p in enumerate(a.parts):
        spec = nlink.make_spec(p, seed=a.seed)
        obj.set_measurement_averaging(); obj.bind_object(spec)
        b = [j.limits_rad for j in spec.joints]
        L = [f"{x:.0f}" for x in [pt.bbox_mm[0] for pt in spec.parts]]
        for col, (name, th) in enumerate([
                ("extended (all 0 deg)", np.zeros(len(b))),
                ("mid (all 90 deg)", np.array([0.5*(lo+hi) for lo,hi in b])),
                ("folded (all 180 deg)", np.array([hi for lo,hi in b]))]):
            ax = fig.add_subplot(len(a.parts), 3, row*3+col+1, projection="3d")
            draw(ax, spec, th, f"p={p}  {name}")
            if col == 0:
                ax.text2D(0.02, 0.94, f"lengths {L} mm", transform=ax.transAxes,
                          fontsize=7, family="monospace")
        print(f"p={p}  링크 길이 [mm]: {L}")
    fig.tight_layout(); fig.savefig(a.out, dpi=130)
    print(f"\n그림 -> {a.out}")

if __name__ == "__main__":
    main()
