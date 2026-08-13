"""검토 ⑤: 언제 탐색을 멈출 것인가.

전제
----
실물에서는 GT 를 모른다. 그러니 정지 조건은 **불확실성만 보고** 판단할 수밖에
없다. 이 스크립트는 그 전제를 그대로 지킨다. 정지 판단에 GT 는 한 글자도
들어가지 않는다. GT 는 마지막에 "그래서 그 판단이 맞았나"를 채점할 때만 쓴다.

무엇이 문제인가
---------------
현행 정지 조건은  1.96 * sqrt(diag Sigma) / |rho| <= 목표  다. Sigma 는 사후
공분산이고, 각도 오차는 R_eff = R + J Sigma_theta J^T 로 이미 반영돼 있다.
그런데도 실제 오차가 주장한 반폭보다 크게 나오는 일이 있다. 세 가지 이유다.

  (a) 계통 오차 : 각도 오차 중 일부는 라운드마다 새로 뽑히지 않는다.
                  FoundationPose 의 보정 치우침 같은 것은 매번 같은 방향으로
                  실린다. R_eff 는 이걸 '독립'으로 보므로 과소평가한다.
  (b) 선형화    : J 는 1차 근사다. 각도 오차가 크면 근사 오차가 남는다.
  (c) 모형 오차 : 균일밀도 가정, 부피 오차 등. 잔차에는 나타나지만 Sigma 에는 없다.

세 가지 정지 규칙을 비교한다.
  R1 현행        : 사후 공분산만
  R2 + 잔차 팽창 : 백색화 잔차가 모형보다 크면 그만큼 부풀린다 (c 를 잡는다)
  R3 + 치우침 몫 : 계통 각도오차를 따로 전파해 더한다 (a 를 잡는다) + R2

채점 기준은 **적용률(coverage)** 이다. "95 % 반폭"이라고 말했으면 실제 오차가
그 안에 들어오는 비율이 95 % 여야 한다. 그보다 낮으면 과신, 훨씬 높으면 과보수.

실행:
  ../robot_learning/scripts/run_drake_env.sh python study_stopping.py --seeds 200
"""

import argparse

import numpy as np

import angle_aware as aa
import density_id_drake as alg
import density_id_objects as obj
import design_core as dc
import nlink

RULES = ("R1 사후분산만", "R2 +잔차팽창", "R3 +치우침몫")


def trial(spec, plan, rho_gt, seed, rel, floor_deg, g_dirs, systematic,
          assumed_systematic, estimator):
    """한 번의 실험. 라운드마다 세 규칙의 반폭과 실제오차를 함께 기록한다."""
    rng = np.random.default_rng(seed)
    Sigma = alg.SIGMA0.copy()
    rho_hat = alg.MU0.copy()
    blocks, rounds, jacs, thetas = [], [], [], []

    # 계통 오차: 이 실험 내내 고정된 각도 치우침 (라운드가 늘어도 안 사라짐)
    n_joint = len(spec.joints)
    scale = np.sqrt(np.diag(aa.angle_covariance(
        np.atleast_1d(plan[0]), rel, floor_deg)))
    offset = np.sqrt(systematic) * rng.normal(0.0, scale)

    out = []
    for theta in plan:
        theta = np.atleast_1d(theta)
        sig = np.sqrt(np.diag(aa.angle_covariance(theta, rel, floor_deg)))
        jitter = np.sqrt(max(0.0, 1.0 - systematic)) * rng.normal(0.0, sig)
        actual = theta + offset + jitter
        measured = actual + offset + np.sqrt(
            max(0.0, 1.0 - systematic)) * rng.normal(0.0, sig)
        y = alg.measure(actual, g_dirs=list(g_dirs), rng=rng)

        A = dc.regressor(measured, g_dirs)
        R = dc.effective_cov(measured, rho_hat, g_dirs, rel, floor_deg)
        jac = dc.angle_jacobian(measured, rho_hat, g_dirs)
        blocks.append((A, y, R)); rounds.append((measured, y))
        jacs.append(jac); thetas.append(measured)
        Sigma = dc.posterior(Sigma, A, R)

        rho_wls = dc.wls_map(blocks, alg.MU0, alg.SIGMA0, alg.RHO_BOUNDS)
        if estimator == "tls":
            rho_hat, _ = dc.tls_map(rounds, alg.MU0, alg.SIGMA0,
                                    alg.RHO_BOUNDS, g_dirs, rho_init=rho_wls,
                                    rel_error=rel, floor_deg=floor_deg)
        else:
            rho_hat = rho_wls

        inflate = dc.residual_inflation(blocks, rho_hat)
        bias_cov = dc.bias_by_refit(rounds, alg.MU0, alg.SIGMA0,
                                    alg.RHO_BOUNDS, g_dirs, rho_hat,
                                    assumed_systematic, estimator,
                                    rel, floor_deg)
        widths = {
            RULES[0]: dc.half_width(Sigma, rho_hat),
            RULES[1]: dc.half_width(Sigma, rho_hat, inflate=inflate),
            RULES[2]: dc.half_width(Sigma, rho_hat, Cov_bias=bias_cov,
                                    inflate=inflate),
        }
        out.append(dict(widths=widths, inflate=inflate,
                        error=np.abs(rho_hat - rho_gt) / rho_gt))   # 채점용
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", default="3link")
    ap.add_argument("--seeds", type=int, default=120)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--angle-rel-error", type=float,
                    default=aa.DEFAULT_ANGLE_REL_ERROR)
    ap.add_argument("--systematic", type=float, nargs="*",
                    default=[0.0, 0.25, 0.5],
                    help="각도오차 분산 중 계통(안 없어지는) 몫")
    ap.add_argument("--assumed-systematic", type=float, default=0.3,
                    help="R3 규칙이 가정하는 계통 몫. 실물에서 우리가 정하는 값")
    ap.add_argument("--estimator", default="both",
                    choices=("tls", "wls", "both"))
    ap.add_argument("--plot", default="study_stopping.png")
    args = ap.parse_args()

    spec = (obj.OBJECTS[args.object] if args.object in obj.OBJECTS
            else nlink.make_spec(int(args.object.rstrip("link"))))
    obj.set_measurement_averaging()
    rho_gt = obj.bind_object(spec)
    obj.apply_weight_prior(spec, obj.assembled_mass_kg(spec))
    g_dirs = dc.CANONICAL_TRIAD
    rel, floor_deg = args.angle_rel_error, aa.DEFAULT_ANGLE_FLOOR_DEG

    rho_design, Sigma = alg.MU0.copy(), alg.SIGMA0.copy()
    bounds = [j.limits_rad for j in spec.joints]
    plan = []
    for _ in range(args.rounds):
        theta, _ = dc.continuous_best(
            bounds, lambda t: dc.utility(t, rho_design, Sigma, g_dirs, "D", rel),
            n_starts=8, seed=0)
        plan.append(theta)
        Sigma = dc.posterior(Sigma, dc.regressor(theta, g_dirs),
                             dc.effective_cov(theta, rho_design, g_dirs, rel))

    print(f"물체 {spec.label}   seed {args.seeds}개   각도오차 +-{100*rel:.0f}%")
    print(f"R3 가 가정하는 계통 몫 = {args.assumed_systematic:.2f} "
          f"(실물에서 우리가 정해 넣는 값)")
    print("정지 판단에 GT 는 안 쓴다. 아래 '적용률'은 사후 채점이다.\n")

    estimators = ("wls", "tls") if args.estimator == "both" else (args.estimator,)
    store = {}
    for estimator in estimators:
        for systematic in args.systematic:
            runs = [trial(spec, plan, rho_gt, 5000 + s, rel, floor_deg, g_dirs,
                          systematic, args.assumed_systematic, estimator)
                    for s in range(args.seeds)]
            store[(estimator, systematic)] = runs

            print(f"── 추정기 {estimator.upper()}   실제 계통 몫 {systematic:.2f}"
                  f"{'  (전부 라운드마다 새로 뽑힘)' if systematic == 0 else ''}")
            print(f"   {'라운드':>5}{'실제오차%':>10}" +
                  "".join(f"{r:>16}" for r in RULES))
            print(f"   {'':>5}{'':>10}" + "".join(f"{'반폭%':>8}{'적용률':>8}"
                                                  for _ in RULES))
            for k in range(args.rounds):
                err = np.array([run[k]["error"] for run in runs])
                line = f"   {k+1:>5}{100*err.max(axis=1).mean():>10.3f}"
                for rule in RULES:
                    w = np.array([run[k]["widths"][rule] for run in runs])
                    covered = np.mean(np.all(err <= w, axis=1))
                    line += f"{100*w.max(axis=1).mean():>8.3f}{100*covered:>7.0f}%"
                print(line)
            print()

    print("읽는 법")
    print("  적용률이 95 % 근처여야 '95 % 반폭'이라는 말이 참이 된다.")
    print("  95 % 보다 한참 낮으면 과신 — 목표에 닿았다고 멈췄는데 실제로는 아니다.")
    print("  100 % 로 붙으면 과보수 — 필요 이상으로 오래 재게 된다.")
    print("  R3 의 '가정하는 계통 몫'은 실물에서 우리가 정하는 안전계수다.")
    print("  FoundationPose 를 고정 장면에서 반복 측정해 분산을 나눠보면 잴 수 있다.")
    print()
    print("  주의: 정지 조건만 손봐서는 WLS 를 구제할 수 없다. WLS 의 오차는")
    print("  분산이 아니라 치우침이라, 반폭을 부풀리면 '언제 멈춰도 틀린' 상태가")
    print("  '언제 멈춰도 틀렸다고 인정하는' 상태로 바뀔 뿐이다. 추정기를 TLS 로")
    print("  바꾸면 치우침 자체가 사라져서 R1 만으로도 적용률이 제자리를 찾는다.")
    print("  TLS 에서 R3 = R2 인 것은 우연이 아니다. TLS 는 각도 보정량을 이미")
    print("  풀고 있어서, 각도를 공통으로 밀어도 답이 안 움직인다(치우침 몫 = 0).\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_col = len(args.systematic)
        fig, axes = plt.subplots(len(estimators), n_col,
                                 figsize=(5 * n_col, 4.2 * len(estimators)),
                                 squeeze=False)
        rounds = np.arange(1, args.rounds + 1)
        for row, estimator in enumerate(estimators):
            for ax, systematic in zip(axes[row], args.systematic):
                runs = store[(estimator, systematic)]
                for rule, style in zip(RULES, ("o-", "s-", "^-")):
                    cov = []
                    for k in range(args.rounds):
                        err = np.array([run[k]["error"] for run in runs])
                        w = np.array([run[k]["widths"][rule] for run in runs])
                        cov.append(100 * np.mean(np.all(err <= w, axis=1)))
                    ax.plot(rounds, cov, style, label=rule.split()[0])
                ax.axhline(95, color="k", ls="--", lw=1)
                ax.set_ylim(0, 105)
                ax.set_xlabel("rounds"); ax.set_ylabel("coverage [%]")
                ax.set_title(f"{estimator.upper()}  systematic={systematic:.2f}")
                ax.grid(alpha=0.3); ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=140)
        print(f"그림 저장 -> {args.plot}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
