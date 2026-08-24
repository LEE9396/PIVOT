"""정보는 1/sqrt(R) 로 쌓인다. 추정기가 실제로 따라가는가?

diag_rounds 는 posterior 만 갱신했다 — 정보 누적은 보였지만 TLS 가 그걸
따라간다는 증거는 아니다. TLS 미지수는 P + R*J 로 라운드에 비례해 늘어나므로
라운드를 늘리는 것 자체가 추정을 어렵게 만든다.

그래서 전체 폐루프를 예산을 크게 주고 돌린다. 예측 필요 라운드는
p=4 약 44, p=5 약 148 이다.
"""
import argparse, time, numpy as np, nlink, density_id_objects as obj, design_core as dc

ap = argparse.ArgumentParser()
ap.add_argument("--parts", type=int, nargs="+", default=[4])
ap.add_argument("--rounds", type=int, default=80)
ap.add_argument("--seeds", type=int, default=2)
a = ap.parse_args()

print(f"각도오차 5%, 최대 {a.rounds}라운드, seed {a.seeds}개\n")
print(f"{'p':>2}{'seed':>6}{'라운드':>8}{'수렴':>7}{'최종반폭%':>11}{'부위오차%':>11}{'초':>8}")
print("-"*55)
for p in a.parts:
    spec = nlink.make_spec(p); npart = len(spec.parts)
    for s in range(a.seeds):
        obj.set_measurement_averaging(); g = obj.bind_object(spec)
        obj.apply_weight_prior(spec, obj.assembled_mass_kg(spec))
        t0 = time.time()
        r = dc.closed_loop(spec, target=0.01, max_rounds=a.rounds, seed=100*s,
                           rel_error=0.05, n_starts=6)
        err = 100*np.max(np.abs(r["rho_hat"][:npart]-g[:npart])/g[:npart])
        print(f"{p:>2}{s:>6}{r['rounds']:>8}{str(r['converged']):>7}"
              f"{100*r['worst']:>11.3f}{err:>11.3f}{time.time()-t0:>8.0f}")
