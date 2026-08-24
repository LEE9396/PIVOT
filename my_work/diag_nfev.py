"""p>=6 에서 추정이 흔들리는 이유 — TLS 최적화가 잘려 나가는가.

tls_map 의 미지수는 P + R*J 로 라운드에 비례해 늘어난다. 그런데 반복 상한이
max_nfev=400 으로 고정돼 있다. 수치 미분이라 야코비안 한 번에 미지수 개수만큼
평가가 드니, 미지수가 100개면 400회로 반복을 서너 번밖에 못 한다.

  실제 쓴 nfev 가 상한에 붙어 있으면 -> 잘려 나간 것 (고칠 수 있다)
  상한 훨씬 아래면                  -> 수렴은 했고 다른 원인
"""
import numpy as np, nlink, density_id_objects as obj, design_core as dc, density_id_drake as alg
import angle_aware as aa
G = dc.CANONICAL_TRIAD

def run(p, R, cap, seed=0, rel=0.05):
    spec = nlink.make_spec(p); obj.set_measurement_averaging()
    gt = obj.bind_object(spec); obj.apply_weight_prior(spec, obj.assembled_mass_kg(spec))
    npart = len(spec.parts); b = [j.limits_rad for j in spec.joints]
    Sigma = alg.SIGMA0.copy(); rh = alg.MU0.copy()
    rng = np.random.default_rng(seed); blocks, rounds = [], []
    nf = 0
    for it in range(1, R+1):
        th,_ = dc.continuous_best(b, lambda t: dc.utility(t, rh, Sigma, G, "D", rel),
                                  n_starts=6, seed=seed*100+it)
        th = np.atleast_1d(th)
        sg = np.sqrt(np.diag(aa.angle_covariance(th, rel, aa.DEFAULT_ANGLE_FLOOR_DEG)))
        actual = th + rng.normal(0, sg); meas = actual + rng.normal(0, sg)
        y = alg.measure(actual, g_dirs=list(G), rng=rng)
        A = dc.regressor(meas, G); Re = dc.effective_cov(meas, rh, G, rel)
        blocks.append((A,y,Re)); rounds.append((meas,y))
        Sigma = dc.posterior(Sigma, A, Re)
        rw = dc.wls_map(blocks, alg.MU0, alg.SIGMA0, alg.RHO_BOUNDS)
        rh, info = dc.tls_map(rounds, alg.MU0, alg.SIGMA0, alg.RHO_BOUNDS, G,
                              rho_init=rw, rel_error=rel, max_nfev=cap)
        nf = info["nfev"]
    n_unk = len(rh) + R*len(b)
    err = 100*np.max(np.abs(rh[:npart]-gt[:npart])/gt[:npart])
    return n_unk, nf, err

print("TLS 최적화가 상한에 잘리는가 (15라운드)\n")
print(f"{'p':>2}{'미지수':>7}{'상한400: nfev':>15}{'오차%':>9}   |{'상한20000: nfev':>17}{'오차%':>9}")
print("-"*68)
for p in (4, 5, 6, 8):
    u1, n1, e1 = run(p, 15, 400)
    u2, n2, e2 = run(p, 15, 20000)
    mark = " <- 잘림" if n1 >= 400 else ""
    print(f"{p:>2}{u1:>7}{n1:>15}{e1:>9.2f}{mark}   |{n2:>17}{e2:>9.2f}")
