"""정지 조건이 얼마나 보수적인가 — 보고하는 반폭 vs 실제 오차.

정지 조건은 '반폭 <= 목표' 다. 그런데 반폭이 실제 오차보다 계속 크면,
추정이 이미 목표를 넘어섰는데도 루프가 안 끝난다. 얼마나 그런지 잰다.

  배율 = 보고 반폭 / 실제 오차
     1 에 가까우면 잘 맞춘 것
     크면 보수적 (안전하지만 라운드를 낭비)
     1 보다 작으면 과신 (위험)
"""
import numpy as np, nlink, density_id_objects as obj, design_core as dc
np.seterr(all="ignore")

def run(p, R, seeds=4, rel=0.05):
    spec = nlink.make_spec(p); npart = len(spec.parts)
    rows = []
    for s in range(seeds):
        obj.set_measurement_averaging(); gt = obj.bind_object(spec)
        obj.apply_weight_prior(spec, obj.assembled_mass_kg(spec))
        r = dc.closed_loop(spec, target=1e-9, max_rounds=R, seed=100*s,
                           rel_error=rel, n_starts=6)   # 목표를 0 으로 두어 끝까지 돈다
        for h in r["history"]:
            err = float(np.max(np.abs(h["rho"][:npart]-gt[:npart])/gt[:npart]))
            rows.append((h["round"], h["worst"], err))
    return np.array(rows)

print("보고 반폭 / 실제 오차 = 몇 배 보수적인가 (각도오차 5%)\n")
print(f"{'p':>2}{'라운드':>7}{'보고 반폭%':>12}{'실제 오차%':>12}{'배율':>8}"
      f"{'오차가 1% 아래인가':>20}{'반폭이 1% 아래인가':>20}")
print("-"*82)
for p in (3, 4, 5, 6):
    A = run(p, R=25)
    for R in (5, 10, 15, 25):
        m = A[:,0] == R
        if not m.any(): continue
        hw, er = A[m,1].mean(), A[m,2].mean()
        print(f"{p:>2}{R:>7}{100*hw:>12.2f}{100*er:>12.3f}{hw/max(er,1e-12):>8.1f}"
              f"{('예' if er<0.01 else '아니오'):>20}{('예' if hw<0.01 else '아니오'):>20}")
    print()
