"""비용이 라운드 번호에 따라 어떻게 자라는가."""
import time, numpy as np, nlink, density_id_objects as obj, design_core as dc

class Timer:
    def __init__(self): self.marks = []
for p in (5, 6):
    spec = nlink.make_spec(p, seed=1000)
    obj.set_measurement_averaging(); obj.bind_object(spec)
    obj.apply_weight_prior(spec, obj.assembled_mass_kg(spec))
    prev, marks = time.time(), []
    for R in (4, 8, 12, 16, 20, 24):
        t0 = time.time()
        r = dc.closed_loop(spec, target=0.0, max_rounds=R, seed=0,
                           rel_error=0.05, n_starts=6)
        marks.append((R, time.time()-t0, 100*r["history"][-1]["worst"]))
    print(f"p={p}")
    for R, dt, w in marks:
        print(f"   {R:2d}라운드 누적 {dt:7.1f}s   마지막 반폭 {w:8.3f} %")
