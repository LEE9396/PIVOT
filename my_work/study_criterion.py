"""검토 ④: 자세 선택 기준을 D / A / E 중 무엇으로 할 것인가.

세 기준이 뭘 줄이려 하는지
--------------------------
사후 공분산 Sigma 를 상대 단위로 정규화한 S = Sigma / (rho rho^T) 위에서,

  D-최적 : det(S) 를 줄인다   -> 불확실성 '덩어리의 부피'. 모든 방향의 곱.
  A-최적 : trace(S) 를 줄인다 -> 분산의 합. 평균적으로 잘 맞추기.
  E-최적 : lambda_max(S) 를 줄인다 -> 가장 안 보이는 방향 하나. 최악 대비.

파트가 2~3개면 셋이 거의 같은 자세를 고른다. 미지수가 적어 방향 간 격차가
작기 때문이다. 차이는 파트가 늘어야 벌어진다. 그래서 실물 3-link 의 치수
규칙을 그대로 이어붙여 P = 2..8 짜리 물체를 만들어 비교한다.

무엇으로 채점하나
-----------------
정지 조건이 '최대 상대 반폭'이므로, 그것과 결이 같은 지표로 본다.
  - 목표 반폭에 닿기까지 걸린 라운드 수  (적을수록 좋다)
  - 그때의 실제 최대 상대오차            (GT 로 채점, 설계에는 안 씀)
  - 최악 파트의 반폭                     (E 가 직접 노리는 것)

실행:
  ../robot_learning/scripts/run_drake_env.sh python study_criterion.py
"""

import argparse

import numpy as np

import angle_aware as aa
import density_id_drake as alg
import density_id_objects as obj
import design_core as dc
import nlink


def closed_loop(spec, rho_gt, kind, target, max_rounds, seed, rel,
                g_dirs, n_starts, estimator="tls"):
    """한 기준으로 폐루프를 돌린다. GT 는 채점용으로만 반환한다."""
    rng = np.random.default_rng(seed)
    floor_deg = aa.DEFAULT_ANGLE_FLOOR_DEG
    bounds = [j.limits_rad for j in spec.joints]
    Sigma = alg.SIGMA0.copy()
    rho_hat = alg.MU0.copy()
    blocks, rounds = [], []
    history = []

    for round_index in range(1, max_rounds + 1):
        theta, _ = dc.continuous_best(
            bounds,
            lambda t: dc.utility(t, rho_hat, Sigma, g_dirs, kind, rel),
            n_starts=n_starts, seed=seed + round_index)

        sigma = np.sqrt(np.diag(aa.angle_covariance(theta, rel, floor_deg)))
        actual = np.atleast_1d(theta) + rng.normal(0.0, sigma)
        measured = actual + rng.normal(0.0, sigma)
        y = alg.measure(actual, g_dirs=list(g_dirs), rng=rng)

        A = dc.regressor(measured, g_dirs)
        R = dc.effective_cov(measured, rho_hat, g_dirs, rel, floor_deg)
        blocks.append((A, y, R))
        rounds.append((measured, y))
        Sigma = dc.posterior(Sigma, A, R)

        rho_wls = dc.wls_map(blocks, alg.MU0, alg.SIGMA0, alg.RHO_BOUNDS)
        if estimator == "tls":
            rho_hat, _ = dc.tls_map(rounds, alg.MU0, alg.SIGMA0,
                                    alg.RHO_BOUNDS, g_dirs,
                                    rho_init=rho_wls, rel_error=rel,
                                    floor_deg=floor_deg)
        else:
            rho_hat = rho_wls

        half = dc.half_width(Sigma, rho_hat)
        history.append(dict(
            round=round_index, theta=np.degrees(np.atleast_1d(theta)),
            half=float(half.max()),
            error=float(np.max(np.abs(rho_hat - rho_gt) / rho_gt)),   # 채점용
            worst_part=int(np.argmax(half))))
        if half.max() <= target:
            break
    return history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", type=int, nargs="*", default=[2, 3, 4, 5, 6, 8])
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--target", type=float, default=0.01)
    ap.add_argument("--max-rounds", type=int, default=12)
    ap.add_argument("--angle-rel-error", type=float,
                    default=aa.DEFAULT_ANGLE_REL_ERROR)
    ap.add_argument("--starts", type=int, default=8)
    ap.add_argument("--estimator", default="tls", choices=("tls", "wls"))
    ap.add_argument("--plot", default="figures/study_criterion.png")
    args = ap.parse_args()

    g_dirs = dc.CANONICAL_TRIAD
    print(f"목표 반폭 {100*args.target:.1f}%   최대 {args.max_rounds}라운드   "
          f"seed {args.seeds}개   추정기 {args.estimator.upper()}")
    print("자세 선택은 연속 최적화. GT 는 '실제오차' 열에만 쓰인다.\n")

    result = {}
    for n_part in args.parts:
        spec = nlink.make_spec(n_part)
        obj.set_measurement_averaging()
        rho_gt = obj.bind_object(spec)
        obj.apply_weight_prior(spec, obj.assembled_mass_kg(spec))

        print(f"── 파트 {n_part}개 (관절 {len(spec.joints)}개) "
              f"GT {np.round(rho_gt, 0)}")
        print(f"   {'기준':>4}{'라운드':>7}{'미달':>6}{'최종반폭%':>11}"
              f"{'실제오차%':>11}{'최악파트':>10}")
        for kind in dc.CRITERIA:
            runs = [closed_loop(spec, rho_gt, kind, args.target,
                                args.max_rounds, 100 * s, args.angle_rel_error,
                                g_dirs, args.starts, args.estimator)
                    for s in range(args.seeds)]
            n_round = np.array([len(h) for h in runs])
            failed = int(np.sum([h[-1]["half"] > args.target for h in runs]))
            half = np.array([h[-1]["half"] for h in runs])
            err = np.array([h[-1]["error"] for h in runs])
            worst = np.bincount([h[-1]["worst_part"] for h in runs],
                                minlength=n_part).argmax()
            result[(n_part, kind)] = dict(rounds=n_round, half=half, err=err)
            print(f"   {kind:>4}{n_round.mean():>7.1f}{failed:>6}"
                  f"{100*half.mean():>11.3f}{100*err.mean():>11.3f}"
                  f"{worst:>10}")
        print()

    # ------------------------------------------------------------------
    print("요약: 파트 수별 승자 (라운드 수 -> 동률이면 실제오차)")
    print(f"  {'파트':>4}" + "".join(f"{k:>10}" for k in dc.CRITERIA)
          + f"{'승자':>8}")
    for n_part in args.parts:
        rounds = {k: result[(n_part, k)]["rounds"].mean() for k in dc.CRITERIA}
        errs = {k: result[(n_part, k)]["err"].mean() for k in dc.CRITERIA}
        best = min(dc.CRITERIA, key=lambda k: (round(rounds[k], 1), errs[k]))
        print(f"  {n_part:>4}" + "".join(f"{rounds[k]:>10.1f}"
                                          for k in dc.CRITERIA)
              + f"{best:>8}")

    print("\n읽는 법")
    print("  파트 2~3개에서는 셋이 같은 라운드에 끝난다. 미지수가 적어 어느")
    print("  기준으로 골라도 사실상 같은 자세가 나오기 때문이다. 이 구간의")
    print("  '승자'는 실제오차 소수점 차이라 의미를 두면 안 된다.")
    print()
    print("  갈리는 건 파트 4개부터다. 여기서 D 가 이긴다.")
    print()
    print("  E 가 질 것 같지 않은데 지는 이유:")
    print("    E 는 가장 안 보이는 방향 '하나'만 본다. 그 하나를 고치면 다음")
    print("    라운드엔 다른 방향이 최악이 되고, 그걸 고치면 또 원래 방향이")
    print("    최악이 된다. 한 번에 하나씩만 보니까 돌려막기가 된다.")
    print("    게다가 lambda_max 는 고윳값이 뒤바뀌는 지점에서 미분이 끊긴다.")
    print("    연속 최적화기가 싫어하는 모양이라 최적점을 잘 못 찾는다.")
    print()
    print("  D 가 이기는 이유:")
    print("    logdet 은 모든 방향을 한꺼번에 본다. 매끄러워서 최적화도 잘 된다.")
    print("    '최악 하나'를 직접 노리지 않는데도 결과적으로 최악이 가장 빨리")
    print("    줄어든다 — 나머지를 같이 줄여두면 돌려막기가 안 생기기 때문이다.")
    print()
    print("  주의: 파트 5개부터는 셋 다 12라운드 안에 목표(1%)에 못 닿는다.")
    print("  기준을 바꿔서 해결될 문제가 아니라, 각도오차 +-5% 로는 그만큼의")
    print("  미지수를 가를 정보가 애초에 부족한 것이다. 파트가 많은 물체를")
    print("  다루려면 각도 정밀도를 올리거나 라운드 예산을 늘려야 한다.")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        for kind, style in zip(dc.CRITERIA, ("o-", "s-", "^-")):
            axes[0].plot(args.parts,
                         [result[(p, kind)]["rounds"].mean() for p in args.parts],
                         style, label=f"{kind}-optimal")
            axes[1].plot(args.parts,
                         [100 * result[(p, kind)]["err"].mean() for p in args.parts],
                         style, label=f"{kind}-optimal")
            axes[2].plot(args.parts,
                         [100 * result[(p, kind)]["half"].mean() for p in args.parts],
                         style, label=f"{kind}-optimal")
        axes[0].set_ylabel("rounds to target"); axes[0].set_title("cost")
        axes[1].set_ylabel("actual max rel. error [%]")
        axes[1].set_title("accuracy (scored with GT)"); axes[1].set_yscale("log")
        axes[2].set_ylabel("final half-width [%]")
        axes[2].set_title("claimed uncertainty"); axes[2].set_yscale("log")
        for ax in axes:
            ax.set_xlabel("number of parts"); ax.grid(alpha=0.3); ax.legend()
        fig.tight_layout()
        fig.savefig(args.plot, dpi=140)
        print(f"\n그림 저장 -> {args.plot}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
