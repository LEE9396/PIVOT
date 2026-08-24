"""누적 정보에서 '어느 방향'이 안 잡히는가, 그리고 잡음의 출처는 어디인가."""
import numpy as np, nlink, density_id_objects as obj, design_core as dc, density_id_drake as alg
REL, G = 0.05, dc.CANONICAL_TRIAD
np.set_printoptions(precision=2, suppress=True, linewidth=200)

def study(p, R=8, seed=0):
    spec = nlink.make_spec(p)
    obj.set_measurement_averaging(); rho = obj.bind_object(spec)
    obj.apply_weight_prior(spec, obj.assembled_mass_kg(spec))
    bounds = [j.limits_rad for j in spec.joints]
    Sigma = alg.SIGMA0.copy(); rh = alg.MU0.copy()
    amp_f, amp_t = [], []
    for it in range(1, R+1):                       # 알고리즘이 실제로 고르는 자세로
        th,_ = dc.continuous_best(bounds, lambda t: dc.utility(t, rh, Sigma, G, "D", REL),
                                  n_starts=6, seed=it)
        A = dc.regressor(th, G); Re = dc.effective_cov(th, rh, G, REL)
        Rs = dc.sensor_cov(G)
        d = np.diag(Re)/np.diag(Rs)
        idx = np.arange(len(d)) % 6
        amp_f.append(d[idx < 3].max()); amp_t.append(d[idx >= 3].max())
        Sigma = dc.posterior(Sigma, A, Re)
    S = Sigma/np.outer(rho, rho)                   # 상대 단위 사후 공분산
    w, V = np.linalg.eigh(S)
    npart = len(spec.parts)
    worst = V[:, -1]                               # 가장 안 잡히는 방향
    return (np.sqrt(w[-1])*1.96*100, np.sqrt(w[0])*1.96*100,
            worst/np.abs(worst).max(), npart, max(amp_f), max(amp_t), rho)

print(f"라운드 8회 누적 후. 각도오차 {100*REL:g}%.\n")
print(f"{'p':>2} {'최악방향 반폭%':>13} {'최선방향 반폭%':>13} {'각도잡음 배율(힘/토크)':>22}")
print("-"*58)
rows=[]
for p in (3, 4, 5, 6, 8):
    hi, lo, worst, npart, af, at, rho = study(p)
    rows.append((p, worst, npart))
    print(f"{p:>2} {hi:>13.3f} {lo:>13.4f} {af:>13.1f} /{at:>7.1f}")
print("\n가장 안 잡히는 방향의 성분 (1 에 가까울수록 그 부위가 그 방향에 크게 기여):")
for p, worst, npart in rows:
    print(f"  p={p}  부위 {np.round(worst[:npart],2)}   힌지 {np.round(worst[npart:],2)}")
