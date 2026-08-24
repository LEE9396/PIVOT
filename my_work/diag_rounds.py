"""라운드를 늘리면 정말 줄어드는가 — 반폭이 1/sqrt(R) 로 떨어지는지 재고,
목표 1 % 에 닿는 데 몇 라운드가 필요한지 외삽한다.

각도 오차가 라운드마다 독립으로 뽑히면 평균으로 줄어야 한다(치우침 없음).
그렇다면 원리상 가능하고, 문제는 '몇 번이냐' 뿐이다. 정말 그런지 본다.
"""
import numpy as np, nlink, density_id_objects as obj, design_core as dc, density_id_drake as alg
import angle_aware as aa
REL, G = 0.05, dc.CANONICAL_TRIAD

def trace(p, R=40, seed=0):
    spec = nlink.make_spec(p)
    obj.set_measurement_averaging(); rho = obj.bind_object(spec)
    obj.apply_weight_prior(spec, obj.assembled_mass_kg(spec))
    bounds = [j.limits_rad for j in spec.joints]
    Sigma = alg.SIGMA0.copy(); rh = alg.MU0.copy()
    rng = np.random.default_rng(seed); npart = len(spec.parts)
    out = []
    for it in range(1, R+1):
        th,_ = dc.continuous_best(bounds, lambda t: dc.utility(t, rh, Sigma, G, "D", REL),
                                  n_starts=6, seed=seed*100+it)
        th = np.atleast_1d(th)
        sig = np.sqrt(np.diag(aa.angle_covariance(th, REL, aa.DEFAULT_ANGLE_FLOOR_DEG)))
        actual = th + rng.normal(0, sig)
        A = dc.regressor(actual + rng.normal(0, sig), G)
        Re = dc.effective_cov(th, rh, G, REL)
        Sigma = dc.posterior(Sigma, A, Re)
        out.append(dc.stopping_width(dc.half_width(Sigma, rho), npart))
    return np.array(out)

print("사후 반폭이 라운드에 따라 어떻게 줄어드는가 (각도오차 5%)\n")
print(f"{'p':>2}" + "".join(f"{f'R={r}':>10}" for r in (2,5,10,20,40))
      + f"{'기울기':>9}{'1% 도달 필요':>14}")
print("-"*72)
for p in (3, 4, 5, 6, 8):
    h = trace(p)
    picks = [h[r-1] for r in (2,5,10,20,40)]
    R = np.arange(1, len(h)+1)
    m = R >= 5
    slope = np.polyfit(np.log(R[m]), np.log(h[m]), 1)[0]     # h ~ R^slope
    need = float(np.exp((np.log(1.0)-np.log(h[-1]))/slope + np.log(40)))
    print(f"{p:>2}" + "".join(f"{100*v:>10.2f}" for v in picks)
          + f"{slope:>9.2f}{need:>14,.0f} 라운드")
print("\n기울기 -0.5 이면 1/sqrt(R) 로 정상 감소 (평균으로 줄어든다는 뜻).")
print("반폭 단위는 %.")
