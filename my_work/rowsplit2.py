import numpy as np, nlink, density_id_objects as obj, design_core as dc
REL, G = 0.05, dc.CANONICAL_TRIAD
for p in (3, 5):
    spec = nlink.make_spec(p)
    obj.set_measurement_averaging(); rho = obj.bind_object(spec)
    obj.apply_weight_prior(spec, obj.assembled_mass_kg(spec))
    b = np.array([j.limits_rad for j in spec.joints])
    rng = np.random.default_rng(0); best = None
    for _ in range(24):
        th = b[:,0] + (b[:,1]-b[:,0])*rng.random(len(b))
        A = dc.regressor(th, G); R = dc.effective_cov(th, rho, G, REL)
        sc = np.linalg.slogdet(A.T @ np.linalg.solve(R, A)/np.outer(rho, rho))[1]
        if best is None or sc > best[0]: best = (sc, th)
    R = dc.effective_cov(best[1], rho, G, REL); Rs = dc.sensor_cov(G)
    r = np.diag(R)/np.diag(Rs)
    print(f"p={p}  행 {r.size}개, median={np.median(r):.4g}")
    print("  " + "  ".join(f"{v:9.4g}" for v in r))
