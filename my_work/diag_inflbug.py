"""잔차 팽창이 TLS 의 각도 보정을 이중으로 세고 있는가.

closed_loop 은 TLS 로 rho 를 풀면서 각 라운드의 각도 보정량 delta 를 함께
얻는데, 그 delta 를 버리고 잔차를 A(측정각도) 로 잰다. 그러면 TLS 가 찾아낸
보정량이 통째로 잔차로 잡혀 팽창이 커진다.

  (a) 지금  : 잔차 = y − A(측정각도)·rho
  (b) 수정  : 잔차 = y − A(측정각도 + delta)·rho     <- TLS 가 실제로 맞춘 모형
"""
import numpy as np, nlink, density_id_objects as obj, design_core as dc, density_id_drake as alg
import angle_aware as aa
G = dc.CANONICAL_TRIAD

def run(p, R=15, seed=0, rel=0.05):
    spec = nlink.make_spec(p); obj.set_measurement_averaging()
    gt = obj.bind_object(spec); obj.apply_weight_prior(spec, obj.assembled_mass_kg(spec))
    npart = len(spec.parts); b = [j.limits_rad for j in spec.joints]
    Sigma = alg.SIGMA0.copy(); rh = alg.MU0.copy()
    rng = np.random.default_rng(seed); blocks, rounds = [], []
    out = []
    for it in range(1, R+1):
        th,_ = dc.continuous_best(b, lambda t: dc.utility(t, rh, Sigma, G, "D", rel),
                                  n_starts=6, seed=seed*100+it)
        th = np.atleast_1d(th)
        sg = np.sqrt(np.diag(aa.angle_covariance(th, rel, aa.DEFAULT_ANGLE_FLOOR_DEG)))
        actual = th + rng.normal(0, sg); meas = actual + rng.normal(0, sg)
        y = alg.measure(actual, g_dirs=list(G), rng=rng)
        A = dc.regressor(meas, G); Re = dc.effective_cov(meas, rh, G, rel)
        blocks.append((A, y, Re)); rounds.append((meas, y))
        Sigma = dc.posterior(Sigma, A, Re)
        rw = dc.wls_map(blocks, alg.MU0, alg.SIGMA0, alg.RHO_BOUNDS)
        rh, info = dc.tls_map(rounds, alg.MU0, alg.SIGMA0, alg.RHO_BOUNDS, G,
                              rho_init=rw, rel_error=rel)
        # (a) 지금 방식
        inf_now = dc.residual_inflation(blocks, rh)
        # (b) TLS 가 실제로 맞춘 모형으로 잔차를 잰다
        corr = [(dc.regressor(np.atleast_1d(t)+d, G), yy, RR)
                for (t, yy), d, (_,_,RR) in zip(rounds, info["delta"], blocks)]
        inf_fix = dc.residual_inflation(corr, rh, n_extra=info["delta"].size)
        err = 100*np.max(np.abs(rh[:npart]-gt[:npart])/gt[:npart])
        h = dc.half_width(Sigma, rh)
        out.append((it, inf_now, inf_fix,
                    100*dc.stopping_width(h*inf_now, npart),
                    100*dc.stopping_width(h*inf_fix, npart), err))
    return out

print("잔차 팽창 배율과 그 결과 반폭 (각도오차 5%)\n")
for p in (4, 5, 6):
    print(f"── p={p} (미지수 {2*p-1})")
    print(f"   {'R':>3}{'팽창(지금)':>12}{'팽창(수정)':>12}{'반폭(지금)%':>13}{'반폭(수정)%':>13}{'실제오차%':>11}")
    for it, a, b_, ha, hb, e in run(p):
        if it in (1, 3, 5, 10, 15):
            print(f"   {it:>3}{a:>12.1f}{b_:>12.2f}{ha:>13.2f}{hb:>13.3f}{e:>11.3f}")
    print()
