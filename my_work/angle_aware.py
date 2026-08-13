"""관절각 측정 오차를 반영한 추정과 자세 추천.

무엇이 달라지는가
-----------------
지금까지는 관절각을 정확히 안다고 보고 y = A(theta) rho + 센서노이즈 로 풀었다.
실제로는 FoundationPose 가 각도를 재주고 거기에 상대오차가 있다 (기본 +-5 %).
그러면 알고리즘이 믿는 각도와 실제 각도가 달라서, 예측한 렌치 자체가 틀린다.

이 오차는 센서 노이즈와 성질이 다르다.
  - 센서 노이즈: 샘플마다 무작위 -> 여러 번 재서 평균내면 사라진다
  - 각도 오차  : 그 라운드 내내 고정 -> 아무리 오래 재도 안 사라진다
                 대신 **다른 자세**에서 재면 다른 방향으로 들어가 상쇄된다

그래서 각도 불확실성을 측정 공분산에 더한다.

    y = A(theta_hat) rho + eps_sensor + (dA/dtheta) delta_theta rho
    R_eff(theta, rho) = R_sensor + J_theta Sigma_theta J_theta^T

여기서 J_theta = d(A(theta) rho)/dtheta 다.

부수 효과 하나가 중요하다. R_eff 가 rho 에 의존하므로 **정보이득이 현재
추정값에 의존하게 된다**. 선형·가우시안에서 최적 설계가 데이터와 무관했던
문제가 여기서 풀린다. 즉 자세 추천이 매 라운드 실제로 달라진다.
"""

import numpy as np

import density_id_drake as alg
import density_id_objects as obj

DEFAULT_ANGLE_REL_ERROR = 0.05      # FoundationPose 상대오차 +-5 %
DEFAULT_ANGLE_FLOOR_DEG = 0.5       # 각도가 0 에 가까워도 남는 절대 오차


# ---------------------------------------------------------------------------
def angle_jacobian(spec, theta, rho, step_rad=1e-4):
    """d(A(theta) rho) / dtheta. 크기 (6*방향수, 관절수)."""
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    columns = []
    for k in range(theta.size):
        plus, minus = theta.copy(), theta.copy()
        plus[k] += step_rad
        minus[k] -= step_rad
        columns.append((alg.regressor(plus) @ rho
                        - alg.regressor(minus) @ rho) / (2.0 * step_rad))
    return np.column_stack(columns)


def angle_covariance(theta, rel_error=DEFAULT_ANGLE_REL_ERROR,
                     floor_deg=DEFAULT_ANGLE_FLOOR_DEG):
    """각도 측정 오차의 공분산. 상대오차 + 절대 하한."""
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    sigma = np.maximum(rel_error * np.abs(theta), np.deg2rad(floor_deg))
    return np.diag(sigma ** 2)


def effective_covariance(spec, theta, rho, rel_error=DEFAULT_ANGLE_REL_ERROR,
                         floor_deg=DEFAULT_ANGLE_FLOOR_DEG):
    """센서 노이즈에 각도 오차 기여를 더한 측정 공분산."""
    jac = angle_jacobian(spec, theta, rho)
    return (np.diag(alg.R_STACK_DIAG)
            + jac @ angle_covariance(theta, rel_error, floor_deg) @ jac.T)


# ---------------------------------------------------------------------------
# R 이 더 이상 대각이 아니므로 추정기 쪽 식도 일반화한다.
# ---------------------------------------------------------------------------
def info_gain(A, Sigma, R_eff):
    """기대 정보이득. 파라미터 공간 형태(P x P)라 싸다."""
    M = np.eye(Sigma.shape[0]) + Sigma @ (A.T @ np.linalg.solve(R_eff, A))
    return 0.5 * float(np.linalg.slogdet(M)[1])


def posterior_covariance(Sigma_prev, A, R_eff):
    info = np.linalg.inv(Sigma_prev) + A.T @ np.linalg.solve(R_eff, A)
    return np.linalg.inv(info)


def constrained_map(blocks, mu0, Sigma0, bounds):
    """누적 측정 + 사전분포의 박스 제약 MAP.

    blocks 는 (A_i, y_i, R_i) 목록. R 이 블록마다 다르므로 각각 촐레스키로
    백색화한 뒤 쌓는다.
    """
    from scipy.optimize import lsq_linear

    rows, rhs = [], []
    for A, y, R in blocks:
        L = np.linalg.cholesky(R)
        rows.append(np.linalg.solve(L, A))
        rhs.append(np.linalg.solve(L, y))
    L0 = np.linalg.cholesky(np.linalg.inv(Sigma0))
    rows.append(L0.T)
    rhs.append(L0.T @ mu0)
    result = lsq_linear(np.vstack(rows), np.concatenate(rhs),
                        bounds=bounds, method="bvls")
    return result.x


# ---------------------------------------------------------------------------
def recommend(spec, candidates, rho_hat, Sigma, rel_error=DEFAULT_ANGLE_REL_ERROR,
              floor_deg=DEFAULT_ANGLE_FLOOR_DEG, torque_filter=None):
    """정보이득이 가장 큰 관절각을 추천한다.

    R_eff 가 rho_hat 에 의존하므로 추천 결과가 매 라운드 달라진다.
    torque_filter 를 주면 안전하다고 판정된 자세만 후보로 삼는다.
    """
    allowed = list(range(len(candidates)))
    if torque_filter is not None:
        safe = [i for i in allowed if torque_filter(i)]
        if safe:
            allowed = safe

    gains, details = [], []
    for index in allowed:
        theta = candidates[index]
        A = alg.regressor(theta)
        R_eff = effective_covariance(spec, theta, rho_hat, rel_error, floor_deg)
        gains.append(info_gain(A, Sigma, R_eff))
        # 각도 오차가 센서 노이즈 대비 얼마나 지배적인지 기록
        sensor = float(np.mean(alg.R_STACK_DIAG))
        total = float(np.mean(np.diag(R_eff)))
        details.append(dict(index=index,
                            angle_share=1.0 - sensor / max(total, 1e-30)))
    order = int(np.argmax(gains))
    best = allowed[order]
    return best, dict(gain=gains[order],
                      gains=np.array(gains),
                      allowed=allowed,
                      angle_share=details[order]["angle_share"])


# ---------------------------------------------------------------------------
def run(spec, candidates, rho_gt, target=0.02, max_rounds=30, seed=0,
        rel_error=DEFAULT_ANGLE_REL_ERROR, floor_deg=DEFAULT_ANGLE_FLOOR_DEG,
        torque_filter=None, verbose=True):
    """각도 오차를 반영한 폐루프. 매 라운드 밀도·불확실성을 갱신하고 추천한다.

    측정은 '실제 각도'에서 이루어지고, 알고리즘은 FoundationPose 가 알려준
    '측정된 각도'만 안다. 둘의 차이가 곧 모형에 넣은 오차다.
    """
    rng = np.random.default_rng(seed)
    Sigma = alg.SIGMA0.copy()
    rho_hat = alg.MU0.copy()
    blocks = []
    history = []

    for round_index in range(1, max_rounds + 1):
        index, info = recommend(spec, candidates, rho_hat, Sigma,
                                rel_error, floor_deg, torque_filter)
        commanded = np.atleast_1d(candidates[index])

        # 실제 관절각은 명령과 조금 다르고, FoundationPose 가 그것을
        # 다시 오차를 섞어 알려준다. 알고리즘은 measured 만 쓴다.
        sigma_theta = np.sqrt(np.diag(angle_covariance(
            commanded, rel_error, floor_deg)))
        actual = commanded + rng.normal(0.0, sigma_theta)
        measured = actual + rng.normal(0.0, sigma_theta)

        A = alg.regressor(measured)
        y = alg.measure(actual, rng=rng)
        R_eff = effective_covariance(spec, measured, rho_hat,
                                     rel_error, floor_deg)
        blocks.append((A, y, R_eff))
        Sigma = posterior_covariance(Sigma, A, R_eff)
        rho_hat = constrained_map(blocks, alg.MU0, alg.SIGMA0, alg.RHO_BOUNDS)

        half = float(np.max(1.96 * np.sqrt(np.diag(Sigma))
                            / np.maximum(np.abs(rho_hat), 1e-9)))
        actual_err = float(np.max(np.abs(rho_hat - rho_gt) / np.abs(rho_gt)))
        history.append(dict(round=round_index,
                            commanded_deg=np.degrees(commanded).tolist(),
                            half_width=half, actual_error=actual_err,
                            angle_share=info["angle_share"],
                            rho=rho_hat.copy()))
        if verbose:
            print(f"  round {round_index}: 추천 q="
                  f"{np.round(np.degrees(commanded), 1)}"
                  f"  각도오차 기여 {100*info['angle_share']:.0f}%"
                  f"  주장 {100*half:.2f}%  실제 {100*actual_err:.2f}%")
        if half <= target:
            break
    return history
