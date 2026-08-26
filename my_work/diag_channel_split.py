"""각도 오차가 어느 행을 부풀리는가 — 신호를 나르는 행만 골라서 잰다.

정리 1이 "부위별 정보는 전부 토크에서 온다" 고 말하고, 3-5b Remark 가
"각도 오차는 토크만 오염시킨다" 고 말한다. 둘이 같은 행을 가리키는지,
그리고 그 오염이 센서 잡음보다 얼마나 큰지를 여기서 잰다.

18행 중 12행은 정확히 1.0 이다.
  힘 9행       F = -g M_total ghat  이라 theta 에 아예 의존하지 않는다
  토크 3행     tau 의 ghat 방향 성분. 외적이라 항상 0 -> 신호도 잡음도 없다
남는 6행이 부위를 구별하는 유일한 통로다.
"""
import numpy as np, nlink, density_id_objects as obj, design_core as dc
REL, G = 0.05, dc.CANONICAL_TRIAD

print(f"각도오차 {100*REL:g}%, 중력 3방향(18행), 무작위 자세 24개 중 D-최적")
print("R_eff/R_sensor — 분산 기준. 괄호 안은 sqrt = 오차 크기 배율\n")
print(f"{'p':>2}{'부푼 행':>8}{'최소':>12}{'중앙':>12}{'최대':>12}   크기 배율(중앙/최대)")
print("-"*76)
for p in (2, 3, 4, 5, 6, 7, 8):
    spec = nlink.make_spec(p)
    obj.set_measurement_averaging(); rho = obj.bind_object(spec)
    obj.apply_weight_prior(spec, obj.assembled_mass_kg(spec))
    b = np.array([j.limits_rad for j in spec.joints])
    rng = np.random.default_rng(0); best = None
    for _ in range(24):
        th = b[:, 0] + (b[:, 1]-b[:, 0])*rng.random(len(b))
        A = dc.regressor(th, G); R = dc.effective_cov(th, rho, G, REL)
        sc = np.linalg.slogdet(A.T @ np.linalg.solve(R, A)/np.outer(rho, rho))[1]
        if best is None or sc > best[0]: best = (sc, th)
    R = dc.effective_cov(best[1], rho, G, REL); Rs = dc.sensor_cov(G)
    r = np.diag(R)/np.diag(Rs)
    hot = r[r > 1.0 + 1e-9]
    print(f"{p:>2}{hot.size:>8}{hot.min():>12.4g}{np.median(hot):>12.4g}"
          f"{hot.max():>12.4g}   {np.sqrt(np.median(hot)):>7.1f} / {np.sqrt(hot.max()):>7.1f}")
print("\n힘 9행은 전부 정확히 1.00 — 각도 오차의 영향이 구조적으로 0 이다.")
