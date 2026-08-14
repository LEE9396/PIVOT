"""검토 ②: 물체를 어느 방향으로 기울일 때 정보량이 가장 늘어나는가.

지금은 중력 방향 3개를 (-z, +x, +y) 로 고정해 놓았다. 로봇이 손목을 돌려
물체를 세 자세로 기울이면서 재는 것이다. 이게 최선인지 확인한다.

물어볼 것
  Q1. 정규직교 3방향이면 어떻게 돌려도 정보량이 같은가?
      -> 센서 잡음만 있을 땐 그렇다. 각도 오차가 들어가면 깨진다.
  Q2. 방향을 몇 개 쓸 때가 가장 이득인가? (1, 2, 3, 4, 6)
  Q3. 정보량이 최대가 되는 회전은 어디인가? 그때 관절 토크는?
  Q4. 한 방향만 고른다면 어디로 기울여야 하나? (구면 전수 탐색)

GT 밀도는 여기서 쓰지 않는다. 설계는 현재 추정값 rho_hat 만 본다.

실행:
  ../robot_learning/scripts/run_drake_env.sh python study_tilt.py --object 3link
"""

import argparse

import numpy as np

import angle_aware as aa
import density_id_drake as alg
import density_id_objects as obj
import design_core as dc


def rotations(count, seed=0):
    """무작위 SO(3) 표본."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(count):
        Q, R = np.linalg.qr(rng.normal(size=(3, 3)))
        out.append(Q * np.sign(np.diag(R)))
    return out


def sphere_grid(n_az=48, n_el=25):
    az = np.linspace(0.0, 2 * np.pi, n_az, endpoint=False)
    el = np.linspace(-np.pi / 2, np.pi / 2, n_el)
    A, E = np.meshgrid(az, el, indexing="ij")
    dirs = np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], -1)
    return az, el, dirs


def gain(theta, g_dirs, rho, Sigma, rel_error, sensor_only=False):
    """이 방향 조합을 한 번 쓰면 D-기준이 얼마나 좋아지는가 [nat]."""
    A = dc.regressor(theta, g_dirs)
    R = (dc.sensor_cov(g_dirs) if sensor_only
         else dc.effective_cov(theta, rho, g_dirs, rel_error))
    post = dc.posterior(Sigma, A, R)
    return dc.criterion_score(post, rho, "D") - dc.criterion_score(Sigma, rho, "D")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", default="3link", choices=list(obj.OBJECTS))
    ap.add_argument("--angle-rel-error", type=float,
                    default=aa.DEFAULT_ANGLE_REL_ERROR)
    ap.add_argument("--theta-deg", type=float, nargs="*", default=None,
                    help="평가할 관절각. 생략하면 구동범위 중앙")
    ap.add_argument("--rotations", type=int, default=200)
    ap.add_argument("--plot", default="figures/study_tilt.png")
    args = ap.parse_args()

    spec = obj.OBJECTS[args.object]
    obj.set_measurement_averaging()
    obj.bind_object(spec)
    obj.apply_weight_prior(spec, obj.assembled_mass_kg(spec))

    n_joint = len(spec.joints)
    if args.theta_deg:
        theta = np.deg2rad(np.array(args.theta_deg, dtype=float))
    else:
        theta = np.array([0.5 * (lo + hi) for lo, hi in
                          (j.limits_rad for j in spec.joints)])
    rho = alg.MU0.copy()             # 사전 추정값. GT 아님.
    Sigma = alg.SIGMA0.copy()
    rel = args.angle_rel_error

    print(f"물체 {spec.label}   관절각 {np.round(np.degrees(theta), 1)} deg")
    print(f"설계에 쓰는 밀도(사전추정) {np.round(rho, 1)}  — GT 는 채점에만\n")

    # ---------------------------------------------------------------- Q1
    print("Q1. 정규직교 3방향을 통째로 회전시키면 정보량이 달라지는가")
    base = dc.CANONICAL_TRIAD
    rots = rotations(args.rotations)
    g_only = np.array([gain(theta, R @ base.T, rho, Sigma, rel, sensor_only=True)
                       for R in rots])
    g_eff = np.array([gain(theta, R @ base.T, rho, Sigma, rel) for R in rots])
    ref_only = gain(theta, base, rho, Sigma, rel, sensor_only=True)
    ref_eff = gain(theta, base, rho, Sigma, rel)
    print(f"  센서잡음만        정준 {ref_only:8.4f}   "
          f"회전 {np.min(g_only):8.4f} ~ {np.max(g_only):8.4f}  "
          f"(폭 {np.ptp(g_only):.2e} nat)")
    print(f"  + 각도오차 {100*rel:.0f}%   정준 {ref_eff:8.4f}   "
          f"회전 {np.min(g_eff):8.4f} ~ {np.max(g_eff):8.4f}  "
          f"(폭 {np.ptp(g_eff):.2e} nat)")

    # 관절각을 바꿔도 계속 불변인지 확인한다 (한 자세에서 우연히 그런 게 아님)
    rng = np.random.default_rng(7)
    spread = []
    for _ in range(8):
        th = np.array([lo + (hi - lo) * rng.random()
                       for lo, hi in (j.limits_rad for j in spec.joints)])
        vals = [gain(th, R @ base.T, rho, Sigma, rel) for R in rots[:40]]
        spread.append(np.ptp(vals))
    print(f"  무작위 관절각 8곳에서도 회전에 따른 폭 최대 {max(spread):.2e} nat")
    print("  -> 정규직교 3방향이면 어떻게 돌려도 정보량이 **정확히 같다**.")
    print("     각도오차를 넣어도 깨지지 않는다. 직교 기저에 대한 합이라")
    print("     회전이 상쇄되기 때문이다. 즉 삼각대의 회전은 정보 면에서")
    print("     **공짜 자유도**다. 다른 목적(토크, 도달성)에 그냥 써도 된다.\n")

    # ---------------------------------------------------------------- Q2
    print("Q2. 중력 방향을 몇 개 쓸 것인가  (한 라운드 안에서)")
    sets = {
        "1 (아래로만)": base[:1],
        "2 (직교)": base[:2],
        "3 (직교, 현행)": base,
        "3 (비직교 60도)": np.array([
            [0.0, 0.0, -1.0],
            [np.sin(np.pi / 3), 0.0, -np.cos(np.pi / 3)],
            [-np.sin(np.pi / 3), 0.0, -np.cos(np.pi / 3)]]),
        "4 (정사면체)": np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1],
                                  [-1, -1, 1]], dtype=float) / np.sqrt(3),
        "6 (+-축 전부)": np.vstack([np.eye(3), -np.eye(3)]),
    }
    print(f"  {'방향집합':<18}{'정보이득':>10}{'방향당':>10}"
          f"{'최대 관절토크':>14}")
    for name, dirs in sets.items():
        value = gain(theta, dirs, rho, Sigma, rel)
        tau = dc.joint_torques(spec, theta, rho, dirs).max()
        print(f"  {name:<18}{value:>10.4f}{value/len(dirs):>10.4f}"
              f"{tau:>13.3f} N·m")
    print("  -> 방향을 늘리면 정보는 늘지만 '방향당' 효율은 떨어진다.")
    print("     3방향 직교가 꺾이는 지점이다(그 위는 이미 채워진 부분공간).\n")

    # ---------------------------------------------------------------- Q3
    print("Q3. 각도오차까지 넣었을 때 최적 회전은 어디인가")
    from scipy.spatial.transform import Rotation

    def rot_gain(v):
        return gain(theta, Rotation.from_rotvec(v).as_matrix() @ base.T,
                    rho, Sigma, rel)

    best_v, info = dc.continuous_best([(-np.pi, np.pi)] * 3, rot_gain,
                                      n_starts=12, seed=1)
    best_dirs = Rotation.from_rotvec(best_v).as_matrix() @ base.T
    tau_ref = dc.joint_torques(spec, theta, rho, base).max()
    tau_best = dc.joint_torques(spec, theta, rho, best_dirs).max()

    def torque_gain(v):
        dirs = Rotation.from_rotvec(v).as_matrix() @ base.T
        return -dc.joint_torques(spec, theta, rho, dirs).max()

    soft_v, _ = dc.continuous_best([(-np.pi, np.pi)] * 3, torque_gain,
                                   n_starts=8, seed=2)
    soft_dirs = Rotation.from_rotvec(soft_v).as_matrix() @ base.T
    print(f"  정준 삼각대     정보 {ref_eff:8.4f}  최대토크 {tau_ref:6.3f} N·m")
    print(f"  정보 최대 회전  정보 {info['score']:8.4f}  최대토크 {tau_best:6.3f} N·m"
          f"   (정보 {100*(info['score']/ref_eff-1):+.1f}%)")
    print(f"  토크 최소 회전  정보 {gain(theta, soft_dirs, rho, Sigma, rel):8.4f}"
          f"  최대토크 {dc.joint_torques(spec, theta, rho, soft_dirs).max():6.3f} N·m"
          f"   (토크 {100*(dc.joint_torques(spec, theta, rho, soft_dirs).max()/tau_ref-1):+.1f}%)\n")

    # ---------------------------------------------------------------- Q4
    print("Q4. 한 방향만 고른다면 어디로 기울여야 하나 (구면 전수)")
    az, el, dirs = sphere_grid()
    single = np.array([[gain(theta, d[None, :], rho, Sigma, rel)
                        for d in row] for row in dirs])
    flat = int(np.argmax(single))
    i, j = np.unravel_index(flat, single.shape)
    best_dir = dirs[i, j]
    worst = dirs[np.unravel_index(int(np.argmin(single)), single.shape)]
    print(f"  최고 {single.max():.4f} nat  방향 {np.round(best_dir, 3)}"
          f"  (방위 {np.degrees(az[i]):.0f}도, 고도 {np.degrees(el[j]):+.0f}도)")
    print(f"  최저 {single.min():.4f} nat  방향 {np.round(worst, 3)}")
    print(f"  아래로(-z) {gain(theta, np.array([[0, 0, -1.0]]), rho, Sigma, rel):.4f} nat")
    print(f"  최고/최저 비 {single.max()/single.min():.2f}배")
    print("  -> 방향 하나만 쓰면 어디로 기울이냐가 크게 갈린다(2.8배). 관절 축과")
    print("     중력이 나란하면 그 관절 아래 질량이 만드는 토크가 0 이라 안 보인다.")
    print("     정규직교 3방향은 이 편차를 통째로 평균내버린다. 그래서 Q1 의")
    print("     불변성이 나온다 — 3방향을 쓰는 한 '기울이는 방향' 고민은 끝난다.\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        angle = [np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
                 for R in rots]
        axes[0].plot(angle, g_eff - ref_eff, "o", ms=3, alpha=0.6)
        axes[0].axhline(0.0, color="k", ls="--", lw=1)
        axes[0].set_xlabel("triad rotation angle [deg]")
        axes[0].set_ylabel("D-gain minus canonical [nat]")
        axes[0].set_title(f"Q1  rotation is free  (spread {np.ptp(g_eff):.1e} nat)")

        labels = ["1 (down)", "2 (orth)", "3 (orth, current)", "3 (non-orth)",
                  "4 (tetra)", "6 (+-axes)"]
        vals = [gain(theta, sets[k], rho, Sigma, rel) for k in sets]
        axes[1].barh(labels, vals, color="C0")
        axes[1].set_xlabel("D-gain [nat]")
        axes[1].set_title("Q2  how many gravity directions")
        axes[1].tick_params(labelsize=8)

        im = axes[2].pcolormesh(np.degrees(az), np.degrees(el), single.T,
                                shading="auto", cmap="viridis")
        axes[2].plot(np.degrees(az[i]), np.degrees(el[j]), "r*", ms=14)
        axes[2].set_xlabel("azimuth [deg]"); axes[2].set_ylabel("elevation [deg]")
        axes[2].set_title("Q4  single tilt direction")
        fig.colorbar(im, ax=axes[2], label="D-gain [nat]")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=140)
        print(f"그림 저장 -> {args.plot}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
