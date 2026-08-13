"""검토 ③: 가중최소제곱(현행) vs 총최소제곱(오차변수 모형).

문제의 뿌리
-----------
측정 모형은  y = A(theta) rho + eps  다. 여기서 theta 는 FoundationPose 가
재준 값이라 +-5 % 오차가 있다. 즉 **계수행렬 A 자체가 틀렸다**.

  현행 WLS : A 는 맞다고 치고, A 가 틀린 몫을 R_eff = R + J Sigma_theta J^T 로
             '잡음'에 몰아넣는다. 잡음은 평균 0 이라고 가정하는데, 실제로는
             A 가 틀린 방향으로 rho 가 끌려간다. 이게 **치우침(bias)** 이고
             측정을 늘려도 안 없어진다.

  TLS      : 각 라운드의 진짜 각도 보정량 delta_i 를 rho 와 함께 미지수로 놓고
             같이 푼다. A 가 틀렸다는 사실이 모형 안에 있으므로 치우침이
             원리상 사라진다.

무엇을 재는가
-------------
같은 측정 데이터를 두 추정기에 똑같이 먹인 뒤,
  - 치우침   : seed 를 많이 돌려 평균낸 (추정 - GT). 0 에서 얼마나 벗어나나
  - 산포     : 표준편차
  - RMSE     : 둘의 합
  - 라운드 의존성 : 라운드를 늘리면 줄어드는가 (치우침이면 안 줄어든다)

GT 는 이 채점에만 쓴다. 두 추정기 모두 GT 를 못 본다.

실행:
  ../robot_learning/scripts/run_drake_env.sh python study_tls.py --seeds 60
"""

import argparse

import numpy as np

import angle_aware as aa
import density_id_drake as alg
import density_id_objects as obj
import design_core as dc
import nlink


def one_trial(spec, thetas, rho_gt, seed, rel, floor_deg, g_dirs):
    """한 번의 실험. 두 추정기가 완전히 같은 데이터를 본다."""
    rng = np.random.default_rng(seed)
    blocks, rounds = [], []
    rho_hat_wls = alg.MU0.copy()

    for theta_cmd in thetas:
        theta_cmd = np.atleast_1d(theta_cmd)
        sigma = np.sqrt(np.diag(aa.angle_covariance(theta_cmd, rel, floor_deg)))
        actual = theta_cmd + rng.normal(0.0, sigma)      # 작업자가 만든 실제 각도
        measured = actual + rng.normal(0.0, sigma)       # FoundationPose 가 읽은 값
        y = alg.measure(actual, g_dirs=list(g_dirs), rng=rng)

        A = dc.regressor(measured, g_dirs)
        R = dc.effective_cov(measured, rho_hat_wls, g_dirs, rel, floor_deg)
        blocks.append((A, y, R))
        rounds.append((measured, y))
        rho_hat_wls = dc.wls_map(blocks, alg.MU0, alg.SIGMA0, alg.RHO_BOUNDS)

    rho_hat_tls, info = dc.tls_map(rounds, alg.MU0, alg.SIGMA0, alg.RHO_BOUNDS,
                                   g_dirs, rho_init=rho_hat_wls,
                                   rel_error=rel, floor_deg=floor_deg)
    return rho_hat_wls, rho_hat_tls, info


def summarize(errors):
    """errors: (seeds, parts) 상대오차. 치우침/산포/RMSE 를 % 로."""
    bias = 100.0 * errors.mean(axis=0)
    spread = 100.0 * errors.std(axis=0)
    rmse = 100.0 * np.sqrt((errors ** 2).mean(axis=0))
    return bias, spread, rmse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", default="3link")
    ap.add_argument("--seeds", type=int, default=60)
    ap.add_argument("--rounds", type=int, nargs="*", default=[1, 2, 3, 5, 8])
    ap.add_argument("--angle-rel-error", type=float,
                    default=aa.DEFAULT_ANGLE_REL_ERROR)
    ap.add_argument("--plot", default="study_tls.png")
    args = ap.parse_args()

    spec = (obj.OBJECTS[args.object] if args.object in obj.OBJECTS
            else nlink.make_spec(int(args.object.rstrip("link"))))
    obj.set_measurement_averaging()
    rho_gt = obj.bind_object(spec)
    obj.apply_weight_prior(spec, obj.assembled_mass_kg(spec))
    g_dirs = dc.CANONICAL_TRIAD
    rel, floor_deg = args.angle_rel_error, aa.DEFAULT_ANGLE_FLOOR_DEG

    # 자세 계획: 연속 최적화로 라운드마다 새로 고른다 (검토 ① 의 결론을 적용).
    # GT 를 안 보고 순수하게 사전 추정값만으로 정한다.
    rho_design, Sigma = alg.MU0.copy(), alg.SIGMA0.copy()
    bounds = [j.limits_rad for j in spec.joints]
    plan = []
    for _ in range(max(args.rounds)):
        theta, _ = dc.continuous_best(
            bounds, lambda t: dc.utility(t, rho_design, Sigma, g_dirs, "D", rel),
            n_starts=8, seed=0)
        plan.append(theta)
        Sigma = dc.posterior(Sigma, dc.regressor(theta, g_dirs),
                             dc.effective_cov(theta, rho_design, g_dirs, rel))

    print(f"물체 {spec.label}   파트 {len(spec.parts)}개   "
          f"각도오차 +-{100*rel:.0f}%   seed {args.seeds}개")
    print("계획된 자세 [deg]: " +
          ", ".join(str(np.round(np.degrees(t), 1)) for t in plan[:8]))
    print("(GT 밀도는 아래 채점에만 쓰였고 두 추정기 모두 보지 못했다)\n")

    table = {}
    for n_round in args.rounds:
        err_w, err_t = [], []
        for seed in range(args.seeds):
            w, t, _ = one_trial(spec, plan[:n_round], rho_gt, 1000 + seed,
                                rel, floor_deg, g_dirs)
            err_w.append((w - rho_gt) / rho_gt)
            err_t.append((t - rho_gt) / rho_gt)
        table[n_round] = (np.array(err_w), np.array(err_t))

    print(f"{'라운드':>5} │ {'WLS(현행)':^28} │ {'TLS(제안)':^28}")
    print(f"{'':>5} │ {'치우침%':>9}{'산포%':>9}{'RMSE%':>9} │ "
          f"{'치우침%':>9}{'산포%':>9}{'RMSE%':>9}")
    print("─" * 70)
    for n_round in args.rounds:
        ew, et = table[n_round]
        bw, sw, rw = summarize(ew)
        bt, st, rt = summarize(et)
        # part 별 최댓값(절댓값 기준)으로 요약
        print(f"{n_round:>5} │ {np.abs(bw).max():>9.2f}{sw.max():>9.2f}"
              f"{rw.max():>9.2f} │ {np.abs(bt).max():>9.2f}{st.max():>9.2f}"
              f"{rt.max():>9.2f}")

    print("\n최종 라운드 파트별 상세")
    ew, et = table[max(args.rounds)]
    bw, sw, rw = summarize(ew)
    bt, st, rt = summarize(et)
    print(f"  {'파트':<14}{'GT':>8} │ {'WLS 치우침':>11}{'WLS RMSE':>10} │ "
          f"{'TLS 치우침':>11}{'TLS RMSE':>10}")
    for k, part in enumerate(spec.parts):
        print(f"  {part.name:<14}{rho_gt[k]:>8.0f} │ {bw[k]:>10.2f}%"
              f"{rw[k]:>9.2f}% │ {bt[k]:>10.2f}%{rt[k]:>9.2f}%")

    # ------------------------------------------------------------------
    print("\n치우침이 라운드를 늘려도 안 줄어드는가 (핵심 진단)")
    print(f"  {'라운드':>5}{'WLS 치우침%':>13}{'TLS 치우침%':>13}"
          f"{'WLS 산포%':>11}{'TLS 산포%':>11}")
    for n_round in args.rounds:
        ew, et = table[n_round]
        print(f"  {n_round:>5}{np.abs(summarize(ew)[0]).max():>13.2f}"
              f"{np.abs(summarize(et)[0]).max():>13.2f}"
              f"{summarize(ew)[1].max():>11.2f}{summarize(et)[1].max():>11.2f}")
    print("  산포는 1/sqrt(N) 로 줄어야 정상이다. 치우침이 안 줄면 그게 EIV 치우침이다.")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rounds = args.rounds
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        for label, idx, style in (("WLS (current)", 0, "o-"),
                                  ("TLS (proposed)", 1, "s-")):
            axes[0].plot(rounds, [np.abs(summarize(table[r][idx])[0]).max()
                                  for r in rounds], style, label=label)
            axes[1].plot(rounds, [summarize(table[r][idx])[1].max()
                                  for r in rounds], style, label=label)
            axes[2].plot(rounds, [summarize(table[r][idx])[2].max()
                                  for r in rounds], style, label=label)
        for ax, title in zip(axes, ("bias  |mean error|", "spread  std",
                                    "RMSE")):
            ax.set_xlabel("rounds"); ax.set_ylabel("max over parts [%]")
            ax.set_yscale("log"); ax.set_title(title)
            ax.grid(alpha=0.3, which="both"); ax.legend()
        fig.tight_layout()
        fig.savefig(args.plot, dpi=140)
        print(f"\n그림 저장 -> {args.plot}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
