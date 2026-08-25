"""링크 길이가 원인인가 — 등길이로 바꾸면 빨리 수렴하는가."""
import numpy as np, nlink, density_id_objects as obj, design_core as dc
orig = nlink.link_lengths
def run(p, lengths, label, R=40, seeds=2):
    nlink.link_lengths = lambda n: lengths(n)
    rs=[]
    for s in range(seeds):
        spec=nlink.make_spec(p); obj.set_measurement_averaging()
        gt=obj.bind_object(spec); obj.apply_weight_prior(spec,obj.assembled_mass_kg(spec))
        n=len(spec.parts)
        r=dc.closed_loop(spec,target=0.01,max_rounds=R,seed=100*s,rel_error=0.05,n_starts=6)
        e=100*np.max(np.abs(r["rho_hat"][:n]-gt[:n])/gt[:n])
        rs.append((r["rounds"], r["converged"], 100*r["worst"], e))
    ok=[x[0] for x in rs if x[1]]
    return (f"{int(np.median(ok))}R ({len(ok)}/{seeds})" if ok else f"— ({0}/{seeds})",
            np.mean([x[2] for x in rs]), np.mean([x[3] for x in rs]))
print("40라운드 예산, seed 2.  링크 길이만 바꾼 비교\n")
print(f"{'p':>2}{'현재(150->60 감소)':>26}{'등길이 100mm':>26}")
print(f"{'':>2}{'라운드 / 반폭% / 오차%':>26}{'라운드 / 반폭% / 오차%':>26}")
print("-"*56)
for p in (4,5):
    a=run(p, orig, "감소")
    b=run(p, lambda n: [100.0]*n, "등길이")
    print(f"{p:>2}{f'{a[0]} / {a[1]:.2f} / {a[2]:.2f}':>26}{f'{b[0]} / {b[1]:.2f} / {b[2]:.2f}':>26}")
nlink.link_lengths = orig
