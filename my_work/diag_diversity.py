"""시작점을 늘리면 왜 나빠지나 — 자세가 서로 비슷해지는가?

가설: 정리 3은 쌓인 C(theta_r) 들의 **행공간이 상보적**이어야 한다고 말한다.
그런데 한 걸음 D-최적은 매 라운드 '지금 가장 좋은 한 점'을 고른다. 목적함수가
라운드 사이에 조금밖에 안 변하므로, 최적화를 잘할수록 **매번 거의 같은 자세**를
고르게 된다. 시작점이 적으면 국소해가 매번 달라져 우연히 다양해진다.

그렇다면 나쁜 최적화가 정칙화 노릇을 하고 있는 것이고, 진짜 고칠 것은
'다양성을 명시적으로 강제하는 것' 이다.
"""
import numpy as np, nlink, density_id_objects as obj, design_core as dc, density_id_drake as alg
G = dc.CANONICAL_TRIAD

def chosen(p, starts, R=10, seed=1):
    spec = nlink.make_spec(p)
    obj.set_measurement_averaging(); rho = obj.bind_object(spec)
    obj.apply_weight_prior(spec, obj.assembled_mass_kg(spec))
    bounds = [j.limits_rad for j in spec.joints]
    Sigma = alg.SIGMA0.copy(); rh = alg.MU0.copy(); ths = []
    for it in range(1, R+1):
        th,_ = dc.continuous_best(bounds, lambda t: dc.utility(t, rh, Sigma, G, "D", 0.05),
                                  n_starts=starts, seed=seed*100+it)
        th = np.atleast_1d(th); ths.append(th)
        Sigma = dc.posterior(Sigma, dc.regressor(th, G),
                             dc.effective_cov(th, rh, G, 0.05))
    return np.degrees(np.array(ths))

print("고른 자세들이 서로 얼마나 다른가 (10라운드). 클수록 다양하다.\n")
print(f"{'p':>2}{'시작점':>7}{'자세 간 평균거리 deg':>22}{'최소거리':>10}{'누적 rank':>11}")
print("-"*54)
for p in (4, 5, 6):
    for st in (6, 24):
        T = chosen(p, st)
        D = np.linalg.norm(T[:, None, :]-T[None, :, :], axis=-1)
        iu = np.triu_indices(len(T), 1)
        # 누적 정보의 rank (상보성의 직접 지표)
        spec = nlink.make_spec(p); obj.set_measurement_averaging()
        rho = obj.bind_object(spec); obj.apply_weight_prior(spec, obj.assembled_mass_kg(spec))
        M = sum(dc.regressor(np.radians(t), G).T @ dc.regressor(np.radians(t), G) for t in T)
        rk = np.linalg.matrix_rank(M, tol=1e-9*np.abs(M).max())
        print(f"{p:>2}{st:>7}{D[iu].mean():>22.1f}{D[iu].min():>10.2f}{rk:>7}/{len(rho)}")
