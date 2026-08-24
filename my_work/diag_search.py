"""p>=4 의 실패가 '정보 부족'인가 '탐색/최적화 실패'인가.

증상: p=4 오차 8.6% 인데 p=5 는 1.0%. 미지수가 늘었는데 나아지는 것은
정보량으로 설명이 안 된다. 원인 후보가 둘이다.

  (a) 자세 탐색 실패  — continuous_best 가 평평한 지형에서 국소해에 갇힘
  (b) TLS 최적화 실패 — 미지수가 P + R*J 로 커져 나쁜 국소해로 감

(a) 는 시작점을 늘려 보면 갈리고, (b) 는 WLS 와 비교하면 갈린다.
정보 부족이라면 셋 다 비슷하게 나빠야 한다.
"""
import argparse, numpy as np, nlink, density_id_objects as obj, design_core as dc, density_id_drake as alg

def run(p, starts, estimator, seeds, rounds, rel):
    spec = nlink.make_spec(p); npart = len(spec.parts)
    errs, rs, ok = [], [], 0
    for s in range(seeds):
        obj.set_measurement_averaging(); g = obj.bind_object(spec)
        obj.apply_weight_prior(spec, obj.assembled_mass_kg(spec))
        r = dc.closed_loop(spec, target=0.01, max_rounds=rounds, seed=100*s,
                           rel_error=rel, n_starts=starts, estimator=estimator)
        errs.append(100*np.max(np.abs(r["rho_hat"][:npart]-g[:npart])/g[:npart]))
        rs.append(r["rounds"]); ok += int(r["converged"])
    return np.mean(errs), np.max(errs), ok, seeds

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", type=int, nargs="+", default=[4, 5, 6])
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--rel", type=float, default=0.05)
    a = ap.parse_args()
    cases = [("기준 (starts=6, TLS)", 6, "tls"),
             ("시작점 24개 (TLS)",    24, "tls"),
             ("WLS (starts=6)",       6, "wls")]
    print(f"각도오차 {100*a.rel:g}%, 최대 {a.rounds}라운드, seed {a.seeds}개\n")
    print(f"  {'설정':<22}" + "".join(f"{f'p={p}':>22}" for p in a.parts))
    print(f"  {'':<22}" + "".join(f"{'오차평균/최악 (수렴)':>22}" for _ in a.parts))
    for name, st, est in cases:
        cells = []
        for p in a.parts:
            m, w, ok, n = run(p, st, est, a.seeds, a.rounds, a.rel)
            cells.append(f"{m:>7.2f}/{w:<7.2f}({ok}/{n})".rjust(22))
        print(f"  {name:<22}" + "".join(cells))
    print("\n  셋이 비슷하게 나쁘면 정보 부족. 시작점을 늘려 좋아지면 탐색 실패.")
    print("  WLS 가 더 안정적이면 TLS 국소해 문제.")
