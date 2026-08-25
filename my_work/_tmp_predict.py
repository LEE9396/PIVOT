"""측정 전에 반폭을 예측할 수 있는가 — 예측 vs 실제.

정리 1의 따름결과: M(θ) 에 밀도가 없으므로 정보 축적은 형상만으로 계산된다.
R_eff 에만 밀도가 들어오는데 그건 저울에서 얻은 사전분포로 대신할 수 있다.
그러면 **렌치를 한 번도 안 재고** R 라운드 뒤 반폭을 예측할 수 있어야 한다.

예측 : 사전분포 밀도(MU0)만 쓰고 y 는 안 씀
실제 : 전체 폐루프 (측정 + TLS 추정)
"""
import numpy as np, nlink, density_id_objects as obj, design_core as dc, density_id_drake as alg
G=dc.CANONICAL_TRIAD

def predict(p, R, seed=0):
    """렌치를 안 재고 반폭 궤적을 예측한다. 저울(총질량)만 쓴다."""
    spec=nlink.make_spec(p, seed=1000+seed); obj.set_measurement_averaging()
    obj.bind_object(spec); obj.apply_weight_prior(spec, obj.assembled_mass_kg(spec))
    b=[j.limits_rad for j in spec.joints]; n=len(spec.parts)
    Sigma=alg.SIGMA0.copy(); rh=alg.MU0.copy(); out=[]
    for it in range(1,R+1):
        th,_=dc.continuous_best(b, lambda t: dc.utility(t,rh,Sigma,G,"D",0.05), n_starts=6, seed=it)
        Sigma=dc.posterior(Sigma, dc.regressor(th,G), dc.effective_cov(th,rh,G,0.05))
        out.append(100*dc.stopping_width(dc.half_width(Sigma,rh), n))
    return out

def actual(p, R, seed=0):
    spec=nlink.make_spec(p, seed=1000+seed); obj.set_measurement_averaging()
    gt=obj.bind_object(spec); obj.apply_weight_prior(spec, obj.assembled_mass_kg(spec))
    n=len(spec.parts)
    r=dc.closed_loop(spec, target=1e-9, max_rounds=R, seed=0, rel_error=0.05, n_starts=6)
    return [100*h["worst"] for h in r["history"]], \
           100*np.max(np.abs(r["rho_hat"][:n]-gt[:n])/gt[:n])

R=8
print(f"측정 전 예측 vs 실제 (반폭 %, {R}라운드)\n")
print(f"{'p':>2}{'':>4}" + "".join(f"{f'R{k}':>9}" for k in (1,2,3,5,8)) + f"{'실제오차%':>11}")
print("-"*62)
for p in (3,4,5):
    pr=predict(p,R); ac,err=actual(p,R)
    print(f"{p:>2}{'예측':>4}" + "".join(f"{pr[k-1]:>9.2f}" for k in (1,2,3,5,8)))
    print(f"{'':>2}{'실제':>4}" + "".join(f"{ac[k-1]:>9.2f}" for k in (1,2,3,5,8)) + f"{err:>11.3f}")
    print()
