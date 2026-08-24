"""D-최적이 왜 '잘 벌어진 자세' 를 안 고르는가.

가설: 부위를 잘 벌리는 자세는 관절을 돌릴 때 부위가 크게 움직이는 자세이기도
하다. 그러면 각도 오차가 만드는 잡음 J Sigma J^T 도 같이 커진다. 신호와 잡음이
같이 커지면 D-최적 지형이 평평해지고, 최적점이 '벌어짐' 과 무관해진다.

무작위 자세를 많이 뽑아서 세 값을 같이 본다.
   벌어짐  : 부위 도심들이 손목 기준 거리에서 얼마나 흩어져 있나 (표준편차)
   신호    : 회귀행렬의 토크 블록 크기
   잡음    : 각도 오차가 만드는 유효 잡음 (R_eff / R_sensor)
   효용    : D-최적 점수
"""
import numpy as np, nlink, density_id_objects as obj, design_core as dc, density_id_drake as alg
G = dc.CANONICAL_TRIAD

for p in (5, 8):
    spec = nlink.make_spec(p); obj.set_measurement_averaging()
    rho = obj.bind_object(spec); obj.apply_weight_prior(spec, obj.assembled_mass_kg(spec))
    npart = len(spec.parts)
    b = np.array([j.limits_rad for j in spec.joints])
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(200):
        th = b[:,0] + (b[:,1]-b[:,0])*rng.random(len(b))
        c = alg.part_centroids_in_S(th)[:npart]
        d = np.linalg.norm(c, axis=1)
        spread = float(np.std(d))*1000                      # 벌어짐 [mm]
        A = dc.regressor(th, G)
        sig = float(np.linalg.norm(A[3::6])+np.linalg.norm(A[4::6])+np.linalg.norm(A[5::6]))
        Re = dc.effective_cov(th, rho, G, 0.05); Rs = dc.sensor_cov(G)
        idx = np.arange(Re.shape[0]) % 6
        noise = float(np.sqrt(np.diag(Re)[idx>=3].max()/np.diag(Rs)[idx>=3].max()))
        u = dc.utility(th, rho, alg.SIGMA0, G, "D", 0.05)
        rows.append((spread, sig, noise, sig/noise, u))
    R = np.array(rows)
    o = np.argsort(-R[:,4])                                  # 효용 높은 순
    print(f"\n=== p={p} — 무작위 자세 200개 ===")
    print(f"  벌어짐 범위 {R[:,0].min():.0f}~{R[:,0].max():.0f} mm")
    print(f"  {'':>12}{'벌어짐mm':>10}{'토크신호':>10}{'각도잡음배율':>12}{'신호/잡음':>10}{'D효용':>10}")
    for lab, i in (("효용 1위", o[0]), ("효용 2위", o[1]), ("효용 최하", o[-1]),
                   ("벌어짐 1위", int(np.argmax(R[:,0]))),
                   ("신호/잡음 1위", int(np.argmax(R[:,3])))):
        v = R[i]
        print(f"  {lab:>12}{v[0]:>10.0f}{v[1]:>10.2e}{v[2]:>12.0f}{v[3]:>10.2e}{v[4]:>10.2f}")
    cc = lambda a,b_: float(np.corrcoef(a,b_)[0,1])
    print(f"  상관: D효용 vs 벌어짐 {cc(R[:,4],R[:,0]):+.2f}   "
          f"D효용 vs 신호/잡음 {cc(R[:,4],R[:,3]):+.2f}   "
          f"신호 vs 잡음 {cc(R[:,1],R[:,2]):+.2f}")
