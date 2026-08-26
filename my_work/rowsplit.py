"""각도 오차가 힘 행과 토크 행을 각각 몇 배로 부풀리는가 — 행 종류로 갈라서 잰다."""
import numpy as np, nlink, density_id_objects as obj, design_core as dc
REL, G = 0.05, dc.CANONICAL_TRIAD

print(f"각도오차 {100*REL:g}%,  중력 {len(G)}방향,  무작위 자세 24개 중 D-최적")
print("비율 = R_eff 대각 / R_sensor 대각  (분산 기준). 괄호는 sqrt = 오차 크기 배율\n")
print(f"{'p':>2}{'행수':>6}{'힘 행 (분산배 / 크기배)':>28}{'토크 행 (분산배 / 크기배)':>30}")
print("-"*70)
for p in (2, 3, 4, 5, 6, 7, 8):
    spec = nlink.make_spec(p)
    obj.set_measurement_averaging(); rho = obj.bind_object(spec)
    obj.apply_weight_prior(spec, obj.assembled_mass_kg(spec))
    b = np.array([j.limits_rad for j in spec.joints])
    rng = np.random.default_rng(0)
    best = None
    for _ in range(24):
        th = b[:, 0] + (b[:, 1]-b[:, 0])*rng.random(len(b))
        A = dc.regressor(th, G); R = dc.effective_cov(th, rho, G, REL)
        sc = np.linalg.slogdet(A.T @ np.linalg.solve(R, A) / np.outer(rho, rho))[1]
        if best is None or sc > best[0]:
            best = (sc, th)
    th = best[1]
    R  = dc.effective_cov(th, rho, G, REL)
    Rs = dc.sensor_cov(G)
    ratio = np.diag(R) / np.diag(Rs)
    n = ratio.size
    # 한 중력 방향마다 힘 3줄 + 토크 3줄 순서로 쌓인다
    idx = np.arange(n) % 6
    f, t = ratio[idx < 3], ratio[idx >= 3]
    print(f"{p:>2}{n:>6}"
          f"{f.mean():>16.2f} / {np.sqrt(f.mean()):>7.2f}"
          f"{t.mean():>18.1f} / {np.sqrt(t.mean()):>7.1f}")
