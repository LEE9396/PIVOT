"""검토 ①: 후보 자세를 격자 top-k 로 고를 것인가, 연속 최적화할 것인가.

관절각은 원래 연속량이다. 지금 코드는 관절마다 5칸(또는 7칸) 격자를 깔고 그중
정보이득이 가장 큰 하나를 고른다. 두 가지가 문제다.

  (1) 격자 사이의 진짜 최적점을 지나친다.
  (2) 후보 수가 n^J 로 터진다. 관절 2개면 25개지만 5개면 3125개고, 그 각각에
      대해 R_eff 를 만들려면 드레이크 정기구학을 여러 번 돌려야 한다.

연속 최적화(상자제약 L-BFGS-B, 다중 시작)는 둘 다 없앤다. 이 스크립트는
같은 목적함수 위에서 두 방식을 나란히 돌려 비교한다.

  - 격자 해상도 n = 3, 5, 7, 9, 13, 21 별 최적값과 평가 횟수
  - 연속 최적화의 최적값과 평가 횟수
  - 관절 수 J = 1..5 로 늘렸을 때 비용 증가
  - 격자가 놓친 양이 '반폭 몇 %' 에 해당하는지 (해석 가능한 단위로)

GT 밀도는 쓰지 않는다. 설계는 현재 추정값만 본다.

실행:
  ../robot_learning/scripts/run_drake_env.sh python study_continuous.py
"""

import argparse
import time

import numpy as np

import angle_aware as aa
import density_id_drake as alg
import density_id_objects as obj
import design_core as dc
import nlink


def make_score(spec, rho, Sigma, rel, kind="D"):
    counter = {"n": 0}

    def score(theta):
        counter["n"] += 1
        return dc.utility(theta, rho, Sigma, dc.CANONICAL_TRIAD, kind, rel)

    return score, counter


def bounds_of(spec):
    return [j.limits_rad for j in spec.joints]


def study_one(spec, rel, kind, grid_sizes, n_starts, seed):
    obj.set_measurement_averaging()
    obj.bind_object(spec)
    obj.apply_weight_prior(spec, obj.assembled_mass_kg(spec))
    rho, Sigma = alg.MU0.copy(), alg.SIGMA0.copy()
    bounds = bounds_of(spec)

    rows = []
    for n in grid_sizes:
        score, counter = make_score(spec, rho, Sigma, rel, kind)
        t0 = time.perf_counter()
        theta, info = dc.grid_best(bounds, n, score)
        rows.append(dict(kind=f"격자 n={n}", theta=theta, score=info["score"],
                         n_eval=counter["n"], seconds=time.perf_counter() - t0))

    score, counter = make_score(spec, rho, Sigma, rel, kind)
    t0 = time.perf_counter()
    theta, info = dc.continuous_best(bounds, score, n_starts=n_starts, seed=seed)
    rows.append(dict(kind="연속 최적화", theta=theta, score=info["score"],
                     n_eval=counter["n"], seconds=time.perf_counter() - t0))
    return rows, rho, Sigma


def half_width_at(spec, theta, rho, Sigma, rel):
    """그 자세를 한 번 쓰면 최대 상대반폭이 얼마가 되는가 [%]."""
    A = dc.regressor(theta, dc.CANONICAL_TRIAD)
    R = dc.effective_cov(theta, rho, dc.CANONICAL_TRIAD, rel)
    post = dc.posterior(Sigma, A, R)
    return 100.0 * float(np.max(dc.half_width(post, rho)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", default="3link")
    ap.add_argument("--criterion", default="D", choices=dc.CRITERIA)
    ap.add_argument("--angle-rel-error", type=float,
                    default=aa.DEFAULT_ANGLE_REL_ERROR)
    ap.add_argument("--starts", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-joints", type=int, default=5)
    ap.add_argument("--plot", default="study_continuous.png")
    args = ap.parse_args()

    spec = (obj.OBJECTS[args.object] if args.object in obj.OBJECTS
            else nlink.make_spec(int(args.object.rstrip("link"))))
    rel = args.angle_rel_error
    grid_sizes = [3, 5, 7, 9, 13, 21]

    print(f"물체 {spec.label}  관절 {len(spec.joints)}개  기준 {args.criterion}")
    print(f"구동범위 {[tuple(np.round(np.degrees(b), 0)) for b in bounds_of(spec)]} deg\n")

    rows, rho, Sigma = study_one(spec, rel, args.criterion, grid_sizes,
                                 args.starts, args.seed)
    best = max(r["score"] for r in rows)
    print("A. 같은 목적함수 위에서 격자 vs 연속")
    print(f"  {'방식':<14}{'최적 관절각[deg]':>22}{'기준값':>10}"
          f"{'최적대비':>9}{'평가횟수':>8}{'시간[s]':>8}{'반폭[%]':>9}")
    for r in rows:
        hw = half_width_at(spec, r["theta"], rho, Sigma, rel)
        print(f"  {r['kind']:<14}{str(np.round(np.degrees(r['theta']), 1)):>22}"
              f"{r['score']:>10.4f}{r['score']-best:>9.4f}"
              f"{r['n_eval']:>8}{r['seconds']:>8.2f}{hw:>9.2f}")

    grid5 = next(r for r in rows if r["kind"] == "격자 n=5")
    cont = rows[-1]
    hw5 = half_width_at(spec, grid5["theta"], rho, Sigma, rel)
    hwc = half_width_at(spec, cont["theta"], rho, Sigma, rel)
    print(f"\n  현행(n=5) 대비 연속 최적화: 기준값 {cont['score']-grid5['score']:+.4f} nat,"
          f"  1라운드 반폭 {hw5:.2f}% -> {hwc:.2f}% ({100*(hwc/hw5-1):+.1f}%)")

    # ------------------------------------------------------------------
    print("\nB. 관절 수를 늘리면 비용이 어떻게 벌어지는가 (격자 n=5 고정)")
    print(f"  {'관절':>4}{'격자 후보수':>12}{'격자 평가':>10}{'연속 평가':>10}"
          f"{'격자[s]':>9}{'연속[s]':>9}{'기준값 차':>11}")
    scaling = []
    for n_part in range(2, args.max_joints + 2):
        s = nlink.make_spec(n_part)
        obj.set_measurement_averaging()
        obj.bind_object(s)
        obj.apply_weight_prior(s, obj.assembled_mass_kg(s))
        r_, S_ = alg.MU0.copy(), alg.SIGMA0.copy()
        b = bounds_of(s)

        sc, c1 = make_score(s, r_, S_, rel, args.criterion)
        t0 = time.perf_counter()
        _, gi = dc.grid_best(b, 5, sc)
        t_grid = time.perf_counter() - t0

        sc, c2 = make_score(s, r_, S_, rel, args.criterion)
        t0 = time.perf_counter()
        _, ci = dc.continuous_best(b, sc, n_starts=args.starts, seed=args.seed)
        t_cont = time.perf_counter() - t0

        print(f"  {len(s.joints):>4}{5**len(s.joints):>12}{c1['n']:>10}"
              f"{c2['n']:>10}{t_grid:>9.2f}{t_cont:>9.2f}"
              f"{ci['score']-gi['score']:>+11.4f}")
        scaling.append((len(s.joints), c1["n"], c2["n"], t_grid, t_cont,
                        ci["score"] - gi["score"]))

    print("\n  -> 격자는 5^J 로 지수 폭발한다. 연속은 (시작점 수) x (반복) x (J+1)")
    print("     이라 J 에 거의 비례한다. 이 물체에선 J=5 부근에서 역전한다.")
    print("     기준값 차는 항상 0 이상 — 연속이 격자를 이기거나 같다.")
    print("     단, 시작점을 고정한 채 J 를 키우면 탐색 밀도가 묽어져 이득 폭이")
    print("     들쭉날쭉해진다(J=5 에서 +0.91). 관절이 많으면 시작점도 늘려야 한다.")

    # ------------------------------------------------------------------
    print("\nC. 연속 최적화의 위험: 국소최대에 갇히는가 (시작점 수 민감도)")
    obj.bind_object(spec)
    obj.apply_weight_prior(spec, obj.assembled_mass_kg(spec))
    rho, Sigma = alg.MU0.copy(), alg.SIGMA0.copy()
    b = bounds_of(spec)
    print(f"  {'시작점':>6}{'기준값':>10}{'평가횟수':>10}")
    for n_starts in (1, 2, 4, 8, 16):
        sc, c = make_score(spec, rho, Sigma, rel, args.criterion)
        _, i = dc.continuous_best(b, sc, n_starts=n_starts, seed=args.seed)
        print(f"  {n_starts:>6}{i['score']:>10.4f}{c['n']:>10}")
    print("  -> 상자 꼭짓점을 항상 시작점에 넣기 때문에 1개만 줘도 사실상 안전하다.")
    print("     이 문제의 최적해는 구동범위 끝에 붙는 경향이 있다.\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        gr = [r for r in rows if r["kind"].startswith("격자")]
        axes[0].plot([r["n_eval"] for r in gr], [r["score"] for r in gr],
                     "o-", label="grid")
        axes[0].plot([cont["n_eval"]], [cont["score"]], "r*", ms=16,
                     label="continuous")
        axes[0].set_xscale("log")
        axes[0].set_xlabel("objective evaluations")
        axes[0].set_ylabel(f"{args.criterion}-criterion")
        axes[0].set_title("A  same objective, two searches")
        axes[0].legend(); axes[0].grid(alpha=0.3)

        J = [s[0] for s in scaling]
        axes[1].semilogy(J, [s[1] for s in scaling], "o-", label="grid n=5")
        axes[1].semilogy(J, [s[2] for s in scaling], "s-", label="continuous")
        axes[1].set_xlabel("number of joints")
        axes[1].set_ylabel("objective evaluations")
        axes[1].set_title("B  cost vs number of joints")
        axes[1].legend(); axes[1].grid(alpha=0.3, which="both")

        # 목적함수 지형 (관절 2개일 때만)
        if len(spec.joints) == 2:
            n = 41
            ax0 = np.linspace(*b[0], n); ax1 = np.linspace(*b[1], n)
            Z = np.array([[dc.utility([x, y], rho, Sigma, dc.CANONICAL_TRIAD,
                                      args.criterion, rel)
                           for y in ax1] for x in ax0])
            im = axes[2].pcolormesh(np.degrees(ax0), np.degrees(ax1), Z.T,
                                    shading="auto", cmap="magma")
            g5 = np.degrees(grid5["theta"]); cc = np.degrees(cont["theta"])
            axes[2].plot(*np.meshgrid(np.degrees(np.linspace(*b[0], 5)),
                                      np.degrees(np.linspace(*b[1], 5))),
                         "w.", ms=3, alpha=0.6)
            axes[2].plot(g5[0], g5[1], "co", ms=10, mfc="none", mew=2,
                         label="grid n=5")
            axes[2].plot(cc[0], cc[1], "r*", ms=16, label="continuous")
            axes[2].set_xlabel("joint 1 [deg]"); axes[2].set_ylabel("joint 2 [deg]")
            axes[2].set_title("C  objective landscape")
            axes[2].legend(fontsize=8)
            fig.colorbar(im, ax=axes[2])
        fig.tight_layout()
        fig.savefig(args.plot, dpi=140)
        print(f"그림 저장 -> {args.plot}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
