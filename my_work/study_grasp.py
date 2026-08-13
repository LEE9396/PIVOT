"""파지점이 어긋났을 때 무슨 일이 나는지, 그리고 함께 풀면 되는지 **재본다**.

왜 문제인가
-----------
시뮬레이션에서는 "센서 원점이 물체의 이 점에 있다" 고 정확히 줄 수 있다.
실물에서는 못 준다. 지그를 대도 몇 mm 는 어긋나고, 사람이 눈대중으로 물리면
1 cm 도 어긋난다. 그 어긋남은 **모든 부위의 모멘트팔을 똑같이 밀어 놓는다.**

이건 잡음이 아니라 **치우침**이다. 라운드를 늘려도, 정지 문턱을 좁혀도
안 없어진다. 각도 오차가 만드는 치우침(study_tls.py)과 같은 구조다.

무엇을 재나
-----------
세 가지를 같은 조건에서 비교한다.

    (가) 어긋남 없음 + 밀도만 품          — 이상적인 경우 (시뮬레이션)
    (나) 어긋남 있음 + 밀도만 품          — 지금 실물에서 벌어질 일
    (다) 어긋남 있음 + 밀도와 함께 품     — 고친 것

(다)가 (가)에 가까우면 고쳐진 것이다. 덤으로 (다)를 어긋남이 **없을 때도**
써보고, 미지수를 3개 늘린 대가로 불확실성이 얼마나 커지는지도 잰다.
공짜로 좋아지는 것은 없으므로 그 값을 알아야 기본값을 정할 수 있다.

실행:
  ../robot_learning/scripts/run_drake_env.sh python study_grasp.py
  ../robot_learning/scripts/run_drake_env.sh python study_grasp.py --object 2link
"""

import argparse

import numpy as np

import density_id_drake as alg
import density_id_objects as obj
import design_core as dc

OFFSETS_MM = (0.0, 2.0, 5.0, 10.0)


def measure_with_offset(theta, delta_m, g_dirs, rng):
    """진짜 파지점이 delta 만큼 어긋난 채로 잰 렌치.

    센서 원점이 어긋나면 토크가 그만큼 달라진다. 힘은 그대로다.
    """
    alg.TRUTH_PLANT.SetPositions(alg.TRUTH_CTX, np.atleast_1d(theta))
    mass = alg.TRUTH_PLANT.CalcTotalMass(alg.TRUTH_CTX)
    com = alg.TRUTH_PLANT.CalcCenterOfMassPositionInWorld(alg.TRUTH_CTX)
    out = []
    for g_hat in g_dirs:
        force = mass * alg.G_ACC * np.asarray(g_hat, float)
        torque = np.cross(com - np.asarray(delta_m, float), force)
        noise = rng.normal(0.0, np.sqrt(alg.R_EPS_DIAG))
        out.append(np.concatenate([force, torque]) + noise)
    return np.concatenate(out)


def run_once(spec, rho_gt, offset_mm, estimate_grasp, seed, n_rounds=6,
             grasp_sigma_m=dc.GRASP_SIGMA_M):
    """라운드를 n_rounds 돌리고 최종 밀도 오차를 돌려준다."""
    rng = np.random.default_rng(seed)
    g_dirs = dc.CANONICAL_TRIAD
    rho_gt = np.asarray(rho_gt, dtype=float)
    total_mass = float(alg.TRUTH_PLANT.CalcTotalMass(alg.TRUTH_CTX))

    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    delta = direction * offset_mm * 1e-3

    bounds = alg.RHO_BOUNDS
    Sigma = alg.SIGMA0.copy()
    blocks = []
    thetas = obj.full_grid(spec)
    for index in range(n_rounds):
        # 자세는 파이프라인과 같은 방식으로 고른다 — 정보이득이 가장 큰 것.
        # 무작위로 고르면 조건수가 나빠져 오차가 실제보다 크게 나온다.
        gains = [alg.info_gain(alg.regressor(np.atleast_1d(th), g_dirs), Sigma)
                 for th in thetas]
        theta = thetas[int(np.argmax(gains))]
        A = alg.regressor(np.atleast_1d(theta), g_dirs)
        Sigma = alg.posterior_covariance(Sigma, A)
        y = measure_with_offset(theta, delta, g_dirs, rng)
        R = np.diag(np.tile(alg.R_EPS_DIAG, len(g_dirs)))
        blocks.append((A, y, R))

    if estimate_grasp:
        rho_hat, delta_hat, Sigma_full = dc.grasp_map(
            blocks, alg.MU0, alg.SIGMA0, bounds, g_dirs, total_mass,
            grasp_sigma_m=grasp_sigma_m)
        sigma_rho = np.sqrt(np.diag(Sigma_full)[:len(rho_gt)])
    else:
        rho_hat = dc.wls_map(blocks, alg.MU0, alg.SIGMA0, bounds)
        delta_hat = np.zeros(3)
        info = np.linalg.inv(alg.SIGMA0)
        for A, _, R in blocks:
            info = info + A.T @ np.linalg.solve(R, A)
        sigma_rho = np.sqrt(np.diag(np.linalg.inv(info)))

    error = np.abs(rho_hat - rho_gt) / rho_gt
    return dict(error=error, delta_true=delta, delta_hat=delta_hat,
                sigma_rel=sigma_rho / rho_hat)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--object", choices=("2link", "3link"), default="3link")
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--plot", default="study_grasp.png")
    args = parser.parse_args()

    spec = obj.OBJECTS[args.object]
    obj.set_measurement_averaging()
    rho_gt = obj.bind_object(spec)
    obj.apply_weight_prior(spec, obj.assembled_mass_kg(spec))

    print(f"{spec.label}   라운드 {args.rounds}, seed {args.seeds}개")
    print(f"파지점 사전분포 {1000 * dc.GRASP_SIGMA_M:.0f} mm\n")
    print(f"  {'어긋남':>8}  {'밀도만 품':>22}  {'함께 품':>22}")
    print(f"  {'[mm]':>8}  {'평균오차':>10}{'최악':>11}"
          f"  {'평균오차':>10}{'최악':>11}   {'delta 오차':>10}")

    table = []
    for offset in OFFSETS_MM:
        plain, joint, residual = [], [], []
        for seed in range(args.seeds):
            a = run_once(spec, rho_gt, offset, False, seed, args.rounds)
            b = run_once(spec, rho_gt, offset, True, seed, args.rounds)
            plain.append(a["error"].max())
            joint.append(b["error"].max())
            residual.append(np.linalg.norm(b["delta_hat"] - b["delta_true"]))
        table.append((offset, np.mean(plain), np.max(plain),
                      np.mean(joint), np.max(joint), np.mean(residual)))
        print(f"  {offset:>8.1f}  {100*table[-1][1]:>9.2f}%{100*table[-1][2]:>10.2f}%"
              f"  {100*table[-1][3]:>9.2f}%{100*table[-1][4]:>10.2f}%"
              f"   {1000*table[-1][5]:>8.3f} mm")

    print("\n읽는 법")
    zero = table[0]
    worst = table[-1]
    print(f"  어긋남이 없으면 둘이 비슷해야 한다 —"
          f" {100*zero[1]:.2f}% vs {100*zero[3]:.2f}%."
          f" 차이가 미지수 3개를 더 푼 대가다.")
    print(f"  어긋남 {worst[0]:.0f} mm 에서는 밀도만 풀면 {100*worst[1]:.2f}%,"
          f" 함께 풀면 {100*worst[3]:.2f}% 다.")
    if worst[3] < worst[1] * 0.7:
        print(f"  -> 함께 푸는 쪽이 확실히 낫다. 실물에서는 이것을 켜야 한다.")
    else:
        print(f"  -> 이 조건에서는 이득이 뚜렷하지 않다. 지그로 막는 편이 낫다.")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        offsets = [row[0] for row in table]
        fig, ax = plt.subplots(figsize=(7, 4.4))
        ax.plot(offsets, [100 * row[1] for row in table], "o-",
                label="density only (bias stays)")
        ax.plot(offsets, [100 * row[3] for row in table], "s-",
                label="density + grasp offset")
        ax.set_xlabel("true grasp-point offset [mm]")
        ax.set_ylabel("worst part density error [%]")
        ax.set_title(f"grasp-point offset — {args.object}")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.plot, dpi=130)
        print(f"\n그림 저장 -> {args.plot}")
    except Exception as exc:                          # noqa: BLE001
        print(f"(그림은 건너뜀: {exc})")


if __name__ == "__main__":
    main()
