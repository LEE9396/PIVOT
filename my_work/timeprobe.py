"""라운드당 실제 비용을 잰다. 예산을 정하기 전에 이것부터."""
import time, numpy as np, nlink, density_id_objects as obj, design_core as dc
for p in (4, 5, 6):
    spec = nlink.make_spec(p, seed=1000)
    obj.set_measurement_averaging(); obj.bind_object(spec)
    obj.apply_weight_prior(spec, obj.assembled_mass_kg(spec))
    t0 = time.time()
    r = dc.closed_loop(spec, target=0.0, max_rounds=6, seed=0,
                       rel_error=0.05, n_starts=6)
    dt = time.time() - t0
    w = [h["worst"] for h in r["history"]]
    print(f"p={p}  6라운드 {dt:6.1f}s   라운드당 {dt/6:5.1f}s   "
          f"반폭 {100*w[0]:8.2f} -> {100*w[-1]:6.3f} %")
