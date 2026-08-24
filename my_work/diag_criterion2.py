"""기준을 바꾸면 정말 좋아지는가 — D-최적 vs '잘 벌어진 자세' 를 직접 비교.

D 효용은 200개 자세에 걸쳐 13% 범위인데 신호/잡음은 18배 차이가 난다.
지형이 평평해서 순위가 무의미하다. 그렇다면 '실제로 중요한 것' 을 직접
최대화하면 나아져야 한다.

  D        : 지금 기준 (log det 정보이득)
  SPREAD   : 부위 도심이 손목 기준 거리에서 가장 넓게 흩어지는 자세
  SIGMIN   : 백색화·정규화한 회귀행렬의 최소 특이값 (가장 안 보이는 방향을 키움)
"""
import numpy as np, nlink, density_id_objects as obj, design_core as dc, density_id_drake as alg
G = dc.CANONICAL_TRIAD

def score_spread(th, rho, npart):
    c = alg.part_centroids_in_S(np.atleast_1d(th))[:npart]
    return float(np.std(np.linalg.norm(c, axis=1)))

def score_sigmin(th, rho, npart):
    A = dc.regressor(th, G); Re = dc.effective_cov(th, rho, G, 0.05)
    W = np.linalg.cholesky(np.linalg.inv(Re)).T
    An = (W @ A) * rho                                   # 상대 단위
    return float(np.linalg.svd(An, compute_uv=False).min())

def loop(p, kind, R=20, seed=0):
    spec = nlink.make_spec(p); obj.set_measurement_averaging()
    rho_gt = obj.bind_object(spec); obj.apply_weight_prior(spec, obj.assembled_mass_kg(spec))
    npart = len(spec.parts); b = [j.limits_rad for j in spec.joints]
    Sigma = alg.SIGMA0.copy(); rh = alg.MU0.copy()
    for it in range(1, R+1):
        if kind == "D":
            f = lambda t: dc.utility(t, rh, Sigma, G, "D", 0.05)
        elif kind == "SPREAD":
            f = lambda t: score_spread(t, rh, npart)
        else:
            f = lambda t: score_sigmin(t, rh, npart)
        th,_ = dc.continuous_best(b, f, n_starts=6, seed=seed*100+it)
        Sigma = dc.posterior(Sigma, dc.regressor(th, G),
                             dc.effective_cov(th, rh, G, 0.05))
    return 100*dc.stopping_width(dc.half_width(Sigma, rho_gt), npart)

print("20라운드 후 최악 방향 반폭 [%] — 낮을수록 좋다. 목표 1%.\n")
print(f"{'p':>2}{'D (지금)':>12}{'SPREAD':>12}{'SIGMIN':>12}")
print("-"*38)
for p in (4, 5, 6, 8):
    r = [loop(p, k) for k in ("D", "SPREAD", "SIGMIN")]
    print(f"{p:>2}{r[0]:>12.2f}{r[1]:>12.2f}{r[2]:>12.2f}")
