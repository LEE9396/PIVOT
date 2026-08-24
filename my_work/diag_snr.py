"""p 가 커지면 왜 안 되는가 — 폐루프를 돌리지 않고 정보 구조로 직접 진단한다.

세 가지를 잰다.

  1. 부위별 신호대잡음 (SNR)
     "그 부위 밀도가 1 % 바뀌면 렌치가 얼마나 움직이는가" 를 유효 잡음과
     비교한다. 1 보다 작으면 **아무리 잘 탐색해도 그 부위는 안 보인다.**
     이건 알고리즘 문제가 아니라 물리 문제다.

  2. 정보행렬의 조건수와 고윳값 분포
     상대 단위로 정규화한 사후 공분산에서, 가장 안 보이는 방향이 가장 잘
     보이는 방향보다 몇 배 나쁜가.

  3. 잡음의 출처 분해
     유효 잡음 R_eff = R_sensor + J Sigma_theta J^T 에서 각도 몫이 몇 배인가.
     각도 몫이 지배하면 센서를 좋게 해도 소용없다.
"""
import numpy as np, nlink, density_id_objects as obj, design_core as dc, density_id_drake as alg
import angle_aware as aa

REL = 0.05
G = dc.CANONICAL_TRIAD

def analyse(p, n_theta=24, seed=0):
    spec = nlink.make_spec(p)
    obj.set_measurement_averaging(); rho = obj.bind_object(spec)
    obj.apply_weight_prior(spec, obj.assembled_mass_kg(spec))
    bounds = np.array([j.limits_rad for j in spec.joints])
    rng = np.random.default_rng(seed)

    best = None
    for _ in range(n_theta):                      # 무작위 자세 중 가장 정보량 큰 것
        th = bounds[:, 0] + (bounds[:, 1]-bounds[:, 0])*rng.random(len(bounds))
        A = dc.regressor(th, G)
        R = dc.effective_cov(th, rho, G, REL)
        M = A.T @ np.linalg.solve(R, A)
        sc = np.linalg.slogdet(M / np.outer(rho, rho))[1]
        if best is None or sc > best[0]:
            best = (sc, th, A, R)
    _, th, A, R = best

    # 1. 부위별 SNR — 밀도 1 % 변화가 만드는 렌치 vs 유효 잡음
    sd = np.sqrt(np.diag(R))
    snr = np.array([np.linalg.norm(A[:, i]*rho[i]*0.01 / sd) for i in range(len(rho))])

    # 2. 조건수 (상대 단위)
    M = A.T @ np.linalg.solve(R, A)
    w = np.linalg.eigvalsh(M / np.outer(rho, rho))
    w = np.clip(w, 1e-300, None)
    cond = w.max()/w.min()

    # 3. 잡음 출처 — 각도 몫이 센서 몫의 몇 배인가
    Rs = dc.sensor_cov(G)
    ratio = float(np.median(np.diag(R)/np.diag(Rs)))
    return spec, rho, snr, cond, ratio, th

print(f"각도오차 {100*REL:g}%. 부위별 SNR = '밀도 1 % 변화'가 유효 잡음의 몇 배인가.\n")
print(f"{'p':>2}{'미지수':>6}{'조건수':>12}{'각도잡음/센서잡음':>18}   부위별 SNR (base -> tip)")
print("-"*100)
for p in (2, 3, 4, 5, 6, 7, 8):
    spec, rho, snr, cond, ratio, th = analyse(p)
    npart = len(spec.parts)
    s = " ".join(f"{v:6.1f}" for v in snr[:npart])
    weak = int(np.sum(snr[:npart] < 1.0))
    print(f"{p:>2}{len(rho):>6}{cond:>12.2e}{ratio:>18.1f}   {s}"
          + (f"   <- SNR<1 인 부위 {weak}개" if weak else ""))
print("\nSNR < 1 이면 그 부위는 잡음에 묻힌다 — 탐색을 아무리 잘해도 안 보인다.")
