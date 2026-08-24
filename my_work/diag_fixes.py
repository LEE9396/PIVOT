"""해결방안 후보들이 실제로 최악 방향 반폭을 줄이는가 — 8라운드 고정 비교.

원인: 부위별 정보는 토크에만 있는데(정리 1), 토크가 각도 오차에 10^6 배
증폭된다. 그래서 후보는 '각도 오차의 영향을 줄이는' 쪽으로 모인다.

  (0) 기준            지금 설정
  (1) 각도 정밀도     rel 5% -> 1%   : 오차 자체를 줄인다
  (2) 자세 다양화     E-최적 기준     : 최악 방향을 직접 노린다
  (3) 미지수 축소     힌지를 하나로 묶는다 (같은 부품이므로 같은 밀도)
  (4) 사전분포 강화   부위 밀도에 물리적 상한/하한을 좁게 준다
"""
import numpy as np, nlink, density_id_objects as obj, design_core as dc, density_id_drake as alg
G = dc.CANONICAL_TRIAD

def worst_halfwidth(spec, rho, rel=0.05, crit="D", R=8, seed=1, sigma0_scale=1.0):
    bounds = [j.limits_rad for j in spec.joints]
    Sigma = alg.SIGMA0.copy() * sigma0_scale**2
    rh = alg.MU0.copy(); npart = len(spec.parts)
    for it in range(1, R+1):
        th,_ = dc.continuous_best(bounds, lambda t: dc.utility(t, rh, Sigma, G, crit, rel),
                                  n_starts=6, seed=seed*100+it)
        A = dc.regressor(th, G); Re = dc.effective_cov(th, rh, G, rel)
        Sigma = dc.posterior(Sigma, A, Re)
    return 100*dc.stopping_width(dc.half_width(Sigma, rho), npart)

def setup(p, hinge_kg=None):
    spec = nlink.make_spec(p) if hinge_kg is None else nlink.make_spec(p, hinge_kg=hinge_kg)
    obj.set_measurement_averaging(); rho = obj.bind_object(spec)
    obj.apply_weight_prior(spec, obj.assembled_mass_kg(spec))
    return spec, rho

print("8라운드 후 최악 방향 반폭 [%] — 낮을수록 좋다. 목표 1%.\n")
print(f"{'p':>2}{'(0) 기준':>10}{'(1) 각도1%':>12}{'(2) E-최적':>12}"
      f"{'(3) 힌지제거':>13}{'(4) 사전분포x0.2':>17}")
print("-"*68)
for p in (4, 5, 6, 8):
    spec, rho = setup(p)
    base = worst_halfwidth(spec, rho)
    fine = worst_halfwidth(spec, rho, rel=0.01)
    eopt = worst_halfwidth(spec, rho, crit="E")
    tight = worst_halfwidth(spec, rho, sigma0_scale=0.2)
    spec2, rho2 = setup(p, hinge_kg=0.0)          # 힌지를 아예 뺀 사양
    nohinge = worst_halfwidth(spec2, rho2)
    print(f"{p:>2}{base:>10.2f}{fine:>12.2f}{eopt:>12.2f}{nohinge:>13.2f}{tight:>17.2f}")
print("\n(3) 은 미지수를 2p-1 에서 p 로 줄인 경우 — 효과의 상한을 보는 것이다.")
