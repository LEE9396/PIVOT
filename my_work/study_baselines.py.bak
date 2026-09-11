"""IV-D 의 두 표를 한 번에 뽑는다 — 시뮬 baseline 비교와 Ablation.

왜 시뮬 baseline 이 실물과 다른가
--------------------------------
PUGS·SiPhy 는 이미지가 필요해 시뮬에서 못 돌린다. 대신 여기서는 **정리 1이
예측한 실패를 직접 보이는** 대조군을 쓴다. 오히려 이쪽이 정보량이 크다 —
"형상을 안 바꾸면 안 된다" 를 vision baseline 이 아니라 우리 이론으로
설명되는 대조군으로 보이는 자리다.

  uniform      물체가 균질하다고 가정. 저울값 / 총부피
               -> Sum of Its Parts 가 쓴 가정과 같다
  single       우리 추정기인데 형상을 한 번만 쓴다 (R=1)
               -> 정리 1의 rank 4 천장을 직접 때린다
  wls          우리 탐색인데 각도를 미지수로 안 푼다
               -> EIV 치우침이 라운드로 안 사라지는 것을 보인다
  ours         TLS + D-최적 폐루프

Ablation 은 같은 물체·같은 seed 로 돌려야 비교가 선다. 4-9 에 흩어져 있던
수치들이 서로 다른 조건에서 나온 것이라 표로 묶을 수 없었다.

실행:
  ../robot_learning/scripts/run_drake_env.sh python study_baselines.py
"""
import argparse
import json
import time

import numpy as np

import density_id_objects as obj
import design_core as dc
import nlink

TARGET_BY_P = {2: 0.01, 3: 0.01, 4: 0.015, 5: 0.02, 6: 0.02}


def setup(n_part, seed):
    """물체 하나를 세우고 GT 를 돌려준다. GT 는 채점에만 쓴다."""
    spec = nlink.make_spec(n_part, seed=1000 + seed)
    obj.set_measurement_averaging()
    rho_gt = obj.bind_object(spec)
    obj.apply_weight_prior(spec, obj.assembled_mass_kg(spec))
    return spec, rho_gt


def score(rho_hat, rho_gt, n_part):
    """부위만 채점한다. 힌지는 저울로 아는 값이라 성적에 넣지 않는다."""
    part = slice(0, n_part)
    return float(np.max(np.abs(rho_hat[part] - rho_gt[part]) / rho_gt[part]))


def uniform_estimate(spec, rho_gt):
    """균질 가정 — 저울에 읽힌 총질량을 총부피로 나눈다.

    저울값과 부피는 우리도 아는 것이므로 이 대조군은 GT 를 훔치지 않는다.
    """
    table = obj.body_table(spec)
    vols = np.array([row["volume_m3"] for row in table])
    total_kg = obj.assembled_mass_kg(spec)
    return np.full(len(rho_gt), total_kg / float(vols.sum()))


def run_baselines(parts, seeds, rel, max_rounds, n_starts):
    rows = {}
    for p in parts:
        target = TARGET_BY_P.get(p, 0.02)
        acc = {k: [] for k in ("uniform", "single", "wls", "ours")}
        rounds = {k: [] for k in ("single", "wls", "ours")}
        for s in range(seeds):
            spec, gt = setup(p, s)

            acc["uniform"].append(score(uniform_estimate(spec, gt), gt, p))

            one = dc.closed_loop(spec, target=0.0, max_rounds=1, seed=100 * s,
                                 rel_error=rel, n_starts=n_starts)
            acc["single"].append(score(one["rho_hat"], gt, p))
            rounds["single"].append(1)

            for key, est in (("wls", "wls"), ("ours", "tls")):
                out = dc.closed_loop(spec, target=target, max_rounds=max_rounds,
                                     seed=100 * s, rel_error=rel,
                                     n_starts=n_starts, estimator=est)
                acc[key].append(score(out["rho_hat"], gt, p))
                rounds[key].append(out["rounds"])

        rows[p] = dict(target=target,
                       err={k: [float(v) for v in acc[k]] for k in acc},
                       rounds={k: rounds[k] for k in rounds})
        line = f"  p={p:<2} 목표 {100*target:>4.1f}%   "
        for k in ("uniform", "single", "wls", "ours"):
            line += f"{k} {100*np.median(acc[k]):>8.2f}%   "
        print(line, flush=True)
    return rows


ABLATIONS = (
    ("ours",        dict()),
    ("WLS",         dict(estimator="wls")),
    ("E-optimal",   dict(criterion="E")),
    ("grid search", dict(select="grid")),
    ("no inflation", dict(stop_rule="variance")),
)


def run_ablation(parts, seeds, rel, max_rounds, n_starts):
    rows = {}
    for name, kw in ABLATIONS:
        cells = {}
        for p in parts:
            target = TARGET_BY_P.get(p, 0.02)
            errs, rds, ok = [], [], 0
            for s in range(seeds):
                spec, gt = setup(p, s)
                out = dc.closed_loop(spec, target=target, max_rounds=max_rounds,
                                     seed=100 * s, rel_error=rel,
                                     n_starts=n_starts, **kw)
                errs.append(score(out["rho_hat"], gt, p))
                rds.append(out["rounds"])
                ok += int(out["converged"])
            cells[p] = dict(err=[float(e) for e in errs], rounds=rds,
                            converged=ok, n=seeds)
        rows[name] = cells
        line = f"  {name:<14}"
        for p in parts:
            c = cells[p]
            line += f"p={p} {100*np.median(c['err']):>7.2f}% ({c['converged']}/{c['n']})  "
        print(line, flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--rel", type=float, default=0.05)
    ap.add_argument("--max-rounds", type=int, default=14)
    ap.add_argument("--starts", type=int, default=6)
    ap.add_argument("--json", default="figures/baselines.json")
    a = ap.parse_args()

    t0 = time.time()
    print(f"각도오차 {100*a.rel:g}%   seed {a.seeds}개   예산 {a.max_rounds}라운드")
    print(f"목표 반폭은 부위 수마다 다르다: {TARGET_BY_P}\n")

    print("[1/2] 시뮬 baseline — 부위 최악 질량오차 (중앙값)")
    base = run_baselines(a.parts, a.seeds, a.rel, a.max_rounds, a.starts)

    print("\n[2/2] Ablation — 같은 물체·같은 seed")
    abl = run_ablation(a.parts, a.seeds, a.rel, a.max_rounds, a.starts)

    json.dump(dict(parts=a.parts, seeds=a.seeds, rel=a.rel,
                   max_rounds=a.max_rounds, target=TARGET_BY_P,
                   baselines=base, ablation=abl),
              open(a.json, "w"), indent=1)
    print(f"\n총 {time.time()-t0:.0f}초   수치 -> {a.json}")
    print("\n읽는 법")
    print("  uniform 이 크게 틀리는 것이 정리 1의 직접 증거다 —")
    print("  형상을 안 바꾸면 부위를 가를 방법이 없다.")
    print("  single 은 우리 추정기인데 형상을 한 번만 쓴 것이라,")
    print("  차이가 '추정기' 가 아니라 '형상을 바꾼 것' 에서 온다는 뜻이다.")


if __name__ == "__main__":
    main()
