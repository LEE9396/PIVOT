"""Fig.5(a)(b): 탐색 절차 비교 — ours / offline-D / random.

무엇을 비교하는가
-----------------
셋 다 **같은 D-최적 기준**을 쓴다 (design_core.criterion_score, kind="D").
갈리는 것은 그 기준을 **언제, 무엇에 대해** 적용하느냐다.

  ours       매 라운드 갱신된 posterior 아래에서 다음 자세를 D-최적으로
             재선택한다 (design_core.closed_loop 의 select="continuous",
             기본 그대로 — 폐루프).
  offline-D  고전 OED 의 이산 유비. **시작 시점의 prior(Sigma0, MU0) 하나에
             대해서만** 자세 budget 개를 한 번에 D-최적으로 고른다 (아래
             offline_batch_design). 실측 데이터는 이 배치 설계 단계에
             전혀 들어가지 않는다 — rho_hat 은 MU0 로 고정, Sigma 는
             '이 자세를 재면 정보가 이만큼 늘 것이다'라는 가상 누적치만
             본다. 이후 그 자세열을 design_core.closed_loop(select="fixed")
             로 그대로 실행한다 — 라운드마다 실측치로 추정(TLS)하고 정지
             판단은 하지만, **다음 자세를 절대 다시 고르지 않는다.**
  random     매 라운드 실현 가능 집합에서 균일 무작위로 고른다
             (select="random", design_core.sample_feasible).

세 arm 모두 추정기(TLS)·정지 규칙(residual)·기준(D)은 동일하게 둔다.
그래야 벌어지는 차이가 '추정기'가 아니라 '절차'에서만 온다.

지표 두 가지
------------
  1. 목표(tau(p) = 0.005*p) 도달 라운드 수 — design_core.closed_loop 의
     rounds/converged 를 그대로 쓴다.
  2. 목표 도달까지 누적 연산 시간 — 자세 선택 시간과 추정 시간을 분리해서
     기록한다. 관절을 맞추는 시간(가정 상수)은 넣지 않는다.

     offline-D 는 자세열 전체를 라운드 1 이전에 한 번에 만든다. 그 배치
     설계는 자세를 budget(30)개 다 고르고 나서야 끝나므로, 실행에 필요한
     자세 i 를 만드는 데 걸린 시간을 라운드 i 의 '선택 시간'으로 그대로
     붙인다 (offline_batch_design 이 자세마다 개별로 시간을 잰다). 즉
     "목표가 라운드 k 에서 달성됐다"면 그 누적 선택 시간은 배치 설계에서
     자세 1..k 를 만드는 데 쓴 시간의 합이다 — 실측 데이터를 전혀 안 쓰고도
     사전에 다 계산해 둘 수 있는 몫이라는 점은 그대로 유지하면서, ours와
     같은 '라운드별로 얼마나 썼나' 잣대로 잰다.

물체 생성은 study_scaling.py 와 같은 규칙 · 같은 시드를 쓴다 (그 파일은
import 하지 않는다 — 세 줄만 그대로 옮긴다).

실행:
  ../robot_learning/scripts/run_drake_env.sh python -u study_strategy.py --smoke
  ../robot_learning/scripts/run_drake_env.sh python -u study_strategy.py
"""

import argparse
import json
import time

import numpy as np

import angle_aware as aa
import density_id_drake as alg
import density_id_objects as obj
import design_core as dc
import nlink

ARMS = ("ours", "offline-D", "random")


# study_scaling.py 의 target_for() 와 같은 규칙 (그 파일은 import 하지 않는다).
def target_for(n_part, per_link=0.005):
    return per_link * n_part


def setup(n_part, seed):
    """물체 하나를 세우고 GT 를 돌려준다. study_scaling.run_cell 과 같은 방식,
    같은 시드(1000+seed) — 두 표가 같은 물체를 가리켜야 한다."""
    spec = nlink.make_spec(n_part, seed=1000 + seed)
    obj.set_measurement_averaging()
    rho_gt = obj.bind_object(spec)
    obj.apply_weight_prior(spec, obj.assembled_mass_kg(spec))
    return spec, rho_gt


def offline_batch_design(spec, budget, seed, g_dirs, criterion,
                         rel_error, floor_deg, n_starts, feasible):
    """고전 OED 의 이산 유비: prior 하나로 budget 개 자세를 한 번에 고른다.

    실측 데이터는 전혀 안 쓴다. rho_hat 은 처음부터 끝까지 alg.MU0 로
    고정하고, Sigma 는 '이 자세를 재면 정보가 이만큼 늘 것이다'라는 가상의
    누적치만 본다 (design_core.posterior 로 하이포테티컬 갱신, 실측 y 는
    한 번도 안 들어간다). 그래서 두 번째 자세를 고를 때 첫 번째 자세와
    겹치지 않게는 되지만("이미 이 방향은 정보가 쌓였다"), 실측값이 실제로
    무엇을 보여줬는지는 전혀 반영하지 않는다 — 그게 '재선택 없음'의 의미다.

    자세마다 걸린 시간을 따로 재서 돌려준다. closed_loop(select="fixed")
    실행 단계에서는 이 자세열을 그냥 인덱싱만 하므로 t_select 가 거의 0이
    된다 — 진짜 선택 비용은 여기, 배치 설계 단계에서 다 치른다.
    """
    bounds = [j.limits_rad for j in spec.joints]
    Sigma = alg.SIGMA0.copy()
    rho_nom = alg.MU0.copy()
    sequence, select_times = [], []
    for i in range(budget):
        def score(theta, Sigma=Sigma):
            return dc.utility(theta, rho_nom, Sigma, g_dirs, criterion,
                              rel_error, floor_deg)
        t0 = time.perf_counter()
        theta, _ = dc.continuous_best(bounds, score, n_starts=n_starts,
                                      seed=seed + i, feasible=feasible)
        select_times.append(time.perf_counter() - t0)
        theta = np.atleast_1d(theta)
        sequence.append(theta)
        A = dc.regressor(theta, g_dirs)
        R_eff = dc.effective_cov(theta, rho_nom, g_dirs, rel_error, floor_deg)
        Sigma = dc.posterior(Sigma, A, R_eff)
    return sequence, select_times


def run_arm(arm, spec, target, budget, seed, rel, n_starts):
    """한 (arm, spec, seed) 를 돌려 라운드별 select/est 시간을 돌려준다."""
    g_dirs = dc.CANONICAL_TRIAD
    common = dict(target=target, max_rounds=budget, seed=100 * seed,
                 rel_error=rel, n_starts=n_starts,
                 criterion="D", estimator="tls", stop_rule="residual")

    batch_select_times = None
    if arm == "ours":
        out = dc.closed_loop(spec, select="continuous", **common)
    elif arm == "random":
        out = dc.closed_loop(spec, select="random", **common)
    elif arm == "offline-D":
        sequence, batch_select_times = offline_batch_design(
            spec, budget, seed=100 * seed, g_dirs=g_dirs, criterion="D",
            rel_error=rel, floor_deg=aa.DEFAULT_ANGLE_FLOOR_DEG,
            n_starts=n_starts, feasible=None)
        out = dc.closed_loop(spec, select="fixed", theta_sequence=sequence,
                             **common)
    else:
        raise ValueError(arm)

    hist = out["history"]
    t_select = [h["t_select"] for h in hist]
    t_est = [h["t_est"] for h in hist]
    if batch_select_times is not None:
        # 배치 설계 비용을 '그 자세를 쓴 라운드'에 붙인다 (모듈 docstring 참고).
        t_select = [b + s for b, s in zip(batch_select_times[:len(hist)], t_select)]

    return dict(rounds=out["rounds"], converged=bool(out["converged"]),
               t_select=t_select, t_est=t_est,
               cum_select=float(np.sum(t_select)),
               cum_est=float(np.sum(t_est)),
               rho_hat=out["rho_hat"])


def score_error(rho_hat, rho_gt, n_part):
    part = slice(0, n_part)
    return float(np.max(np.abs(rho_hat[part] - rho_gt[part]) / rho_gt[part]))


def run_cell(n_part, seeds, rel, budget, n_starts):
    target = target_for(n_part)
    cell = {arm: dict(rounds=[], converged=[], cum_select=[], cum_est=[],
                      error=[]) for arm in ARMS}
    for s in range(seeds):
        spec, rho_gt = setup(n_part, s)
        for arm in ARMS:
            r = run_arm(arm, spec, target, budget, s, rel, n_starts)
            cell[arm]["rounds"].append(r["rounds"])
            cell[arm]["converged"].append(r["converged"])
            cell[arm]["cum_select"].append(r["cum_select"])
            cell[arm]["cum_est"].append(r["cum_est"])
            cell[arm]["error"].append(score_error(r["rho_hat"], rho_gt, n_part))
    return target, cell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--rel", type=float, default=0.05)
    ap.add_argument("--budget", type=int, default=30)
    ap.add_argument("--starts", type=int, default=6)
    ap.add_argument("--smoke", action="store_true",
                    help="p=2,3 seed=2 로 비용만 잰다 (실행 안 함)")
    ap.add_argument("--json", default="figures/strategy_v1.json")
    args = ap.parse_args()

    if args.smoke:
        parts, seeds = [2, 3], 2
    else:
        parts, seeds = args.parts, args.seeds

    print(f"arms={ARMS}  parts={parts}  seeds={seeds}  rel={100*args.rel:g}%  "
          f"budget={args.budget}라운드  starts={args.starts}")
    t_start = time.time()
    table = {}
    for p in parts:
        t0 = time.time()
        target, cell = run_cell(p, seeds, args.rel, args.budget, args.starts)
        dt = time.time() - t0
        table[str(p)] = dict(target=target, cells=cell, wall_seconds=dt)
        line = f"  p={p:<2} 목표 {100*target:>4.2f}%  [{dt:>6.1f}s]  "
        for arm in ARMS:
            c = cell[arm]
            ok = [r for r, cv in zip(c["rounds"], c["converged"]) if cv]
            med_r = f"{np.median(ok):.0f}" if ok else "-"
            n_ok = sum(c["converged"])
            line += (f"{arm} R={med_r}({n_ok}/{seeds}) "
                    f"sel={np.median(c['cum_select']):.3f}s "
                    f"est={np.median(c['cum_est']):.3f}s   ")
        print(line, flush=True)

    total = time.time() - t_start
    print(f"\n총 {total:.1f}초")

    if args.smoke:
        n_cells = len(parts) * seeds * len(ARMS)
        per_cell = total / n_cells
        full_cells = len(args.parts) * args.seeds * len(ARMS)
        est_full = per_cell * full_cells
        print(f"\n[스모크] 셀당 평균 {per_cell:.2f}초 "
              f"({n_cells}셀 = {len(parts)}p x {seeds}seed x {len(ARMS)}arm,"
              f" budget={args.budget})")
        print(f"[스모크] 본실행(parts={args.parts}, seed={args.seeds}) 예상: "
              f"{full_cells}셀 x {per_cell:.2f}s ~= {est_full:.0f}초 "
              f"({est_full/60:.1f}분)")
        print("  (주의: p 가 클수록/수렴 안 될수록 셀당 비용이 더 크다 —"
              " 이 추정은 하한에 가깝다)")
        return

    payload = dict(arms=list(ARMS), parts=parts, seeds=seeds, rel=args.rel,
                   budget=args.budget, n_starts=args.starts,
                   per_link_target=0.005, table=table,
                   total_seconds=total)
    with open(args.json, "w") as fh:
        json.dump(payload, fh, indent=1)
    print(f"수치 -> {args.json}")


if __name__ == "__main__":
    main()
