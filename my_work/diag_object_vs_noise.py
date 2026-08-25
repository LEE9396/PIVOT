"""편차가 물체 탓인가 잡음 탓인가.

지금 seed 는 물체(링크 길이)와 잡음(센서·각도·최적화 시작점)을 동시에 바꾼다.
갈라서 본다.

  A) 물체 고정 + 잡음만 8가지   -> 편차가 크면 '잡음 탓'
  B) 물체 8가지 + 잡음 고정     -> 편차가 크면 '물체 탓'
"""
import numpy as np, nlink, density_id_objects as obj, design_core as dc

def run(p, obj_seed, noise_seed, R=30):
    spec=nlink.make_spec(p, seed=obj_seed); obj.set_measurement_averaging()
    gt=obj.bind_object(spec); obj.apply_weight_prior(spec, obj.assembled_mass_kg(spec))
    r=dc.closed_loop(spec, target=0.02, max_rounds=R, seed=noise_seed,
                     rel_error=0.05, n_starts=6)
    return r["rounds"], r["converged"]

for p in (5,):
    print(f"── p={p},  목표 2 %,  30라운드 예산\n")
    A=[run(p, 1000, 100*s) for s in range(8)]      # 물체 고정, 잡음만
    B=[run(p, 1000+s, 0)   for s in range(8)]      # 물체 바꿈, 잡음 고정
    for lab, X in (("A 물체고정+잡음변화", A), ("B 물체변화+잡음고정", B)):
        R=np.array([x[0] for x in X]); ok=np.array([x[1] for x in X])
        r=R[ok]
        print(f"  {lab}: {np.array2string(R)}  수렴 {ok.sum()}/8")
        if len(r): print(f"       {'':<18} 범위 {r.min()}~{r.max()}  중앙 {int(np.median(r))}  표준편차 {r.std():.1f}")
