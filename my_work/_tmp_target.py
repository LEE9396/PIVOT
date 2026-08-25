"""목표 반폭을 얼마로 잡으면 라운드가 몇 개인가.

한 번 돌리면서 라운드마다 반폭을 기록하고, 여러 목표선을 언제 넘는지 읽는다.
"""
import numpy as np, nlink, density_id_objects as obj, design_core as dc, density_id_drake as alg
G=dc.CANONICAL_TRIAD; TARGETS=(0.005,0.01,0.015,0.02,0.03,0.05)

def trace(p, R=30, seed=0):
    spec=nlink.make_spec(p, seed=1000+seed); obj.set_measurement_averaging()
    gt=obj.bind_object(spec); obj.apply_weight_prior(spec, obj.assembled_mass_kg(spec))
    b=[j.limits_rad for j in spec.joints]; n=len(spec.parts)
    Sigma=alg.SIGMA0.copy(); rh=alg.MU0.copy(); hs=[]
    for it in range(1,R+1):
        th,_=dc.continuous_best(b, lambda t: dc.utility(t,rh,Sigma,G,"D",0.05), n_starts=6, seed=it)
        Sigma=dc.posterior(Sigma, dc.regressor(th,G), dc.effective_cov(th,rh,G,0.05))
        hs.append(dc.stopping_width(dc.half_width(Sigma,gt), n))
    return np.array(hs)

print("목표 반폭별 필요 라운드 (seed 2개 중앙값, 30라운드 예산, — 는 미달)\n")
print(f"{'p':>2}" + "".join(f"{f'{100*t:g}%':>8}" for t in TARGETS))
print("-"*52)
for p in (3,4,5,6):
    per=[]
    for s in range(2):
        h=trace(p, seed=s)
        per.append([next((i+1 for i,v in enumerate(h) if v<=t), None) for t in TARGETS])
    row=""
    for k in range(len(TARGETS)):
        vals=[x[k] for x in per if x[k]]
        row += f"{(str(int(np.median(vals))) if len(vals)==2 else '—'):>8}"
    print(f"{p:>2}{row}")
