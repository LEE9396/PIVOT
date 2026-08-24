"""D 는 옳다. 문제는 '같은 자세를 반복하는 것' 이다 — 다양성을 더하면 되는가.

증거:  D+시작점6 (자세 다양)  0.83%   <  D+시작점24 (자세 비슷)  25.64%
                             <  SPREAD (완전 고정) 194%
자세가 반복될수록 나빠진다. 그래서 D 를 유지하되 다양성을 강제해 본다.

  D          지금 그대로
  D+PENALTY  이미 쓴 자세 근방이면 효용을 깎는다
  D+RANDOM   확률 eps 로 무작위 자세를 섞는다 (가장 단순한 다양화)
"""
import numpy as np, nlink, density_id_objects as obj, design_core as dc, density_id_drake as alg
G = dc.CANONICAL_TRIAD

def loop(p, mode, R=20, seed=0, tau=np.deg2rad(30.0), eps=0.35):
    spec = nlink.make_spec(p); obj.set_measurement_averaging()
    gt = obj.bind_object(spec); obj.apply_weight_prior(spec, obj.assembled_mass_kg(spec))
    npart = len(spec.parts); b = np.array([j.limits_rad for j in spec.joints])
    Sigma = alg.SIGMA0.copy(); rh = alg.MU0.copy(); used = []
    rng = np.random.default_rng(seed)
    for it in range(1, R+1):
        if mode == "RANDOM" and used and rng.random() < eps:
            th = b[:,0] + (b[:,1]-b[:,0])*rng.random(len(b))
        else:
            def f(t, _u=used):
                u = dc.utility(t, rh, Sigma, G, "D", 0.05)
                if mode == "PENALTY" and _u:
                    d = min(np.linalg.norm(np.atleast_1d(t)-q) for q in _u)
                    u -= 3.0*np.exp(-(d/tau)**2)      # 가까울수록 깎는다
                return u
            th,_ = dc.continuous_best(list(map(tuple,b)), f, n_starts=6, seed=seed*100+it)
        th = np.atleast_1d(th); used.append(th.copy())
        Sigma = dc.posterior(Sigma, dc.regressor(th, G),
                             dc.effective_cov(th, rh, G, 0.05))
    T = np.array(used); iu = np.triu_indices(len(T), 1)
    div = float(np.degrees(np.linalg.norm(T[:,None,:]-T[None,:,:],axis=-1)[iu]).mean())
    return 100*dc.stopping_width(dc.half_width(Sigma, gt), npart), div

print("20라운드 후 최악 반폭 [%] 와 자세 다양성 [deg]. 목표 1%.\n")
print(f"{'p':>2}" + "".join(f"{m:>22}" for m in ("D (지금)","D+PENALTY","D+RANDOM")))
print(f"{'':>2}" + "".join(f"{'반폭 / 다양성':>22}" for _ in range(3)))
print("-"*68)
for p in (4, 5, 6, 8):
    cells=[]
    for m in ("D","PENALTY","RANDOM"):
        h, d = loop(p, m)
        cells.append(f"{h:>9.2f} /{d:>8.1f}".rjust(22))
    print(f"{p:>2}" + "".join(cells))
