"""검토 지점 비교 실험을 위한 공용 계산 모듈.

기존 파이프라인(density_id_drake + angle_aware)은 세 군데가 굳어 있다.

  1. 후보 자세      : 격자를 깔고 그중 최고를 고른다 (top-k)
  2. 중력 3방향     : (-z, +x, +y) 고정
  3. 추정기         : 가중최소제곱 + 각도오차를 잡음으로 흡수 (R_eff)
  4. 선택 기준      : D-최적 하나
  5. 정지 조건      : 사후 공분산만 보고 판단

이 모듈은 다섯 가지를 모두 **매개변수**로 바꿔서, 어느 쪽이 나은지 실험으로
비교할 수 있게 한다. GT(rho_gt)는 여기서 아무 데도 쓰지 않는다. 실험 스크립트가
마지막에 "그래서 얼마나 맞았나"를 채점할 때만 쓴다.
"""

import numpy as np

import density_id_drake as alg
import angle_aware as aa

# ---------------------------------------------------------------------------
# 1. g_dirs 를 인자로 받을 수 있게 일반화한 기본량
#    (alg / aa 는 G_DIRS 전역을 쓰므로 여기서 다시 쓴다)
# ---------------------------------------------------------------------------
CANONICAL_TRIAD = np.array([[0.0, 0.0, -1.0],
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0]])


def sensor_cov(g_dirs):
    """센서 잡음만의 측정 공분산. 방향 수만큼 블록이 쌓인다."""
    return np.diag(np.tile(alg.R_EPS_DIAG, len(g_dirs)))


def regressor(theta, g_dirs):
    return alg.regressor(np.atleast_1d(theta), [np.asarray(g) for g in g_dirs])


def angle_jacobian(theta, rho, g_dirs, step_rad=1e-4):
    """d(A(theta) rho) / dtheta. 중앙차분."""
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    cols = []
    for k in range(theta.size):
        plus, minus = theta.copy(), theta.copy()
        plus[k] += step_rad
        minus[k] -= step_rad
        cols.append((regressor(plus, g_dirs) @ rho
                     - regressor(minus, g_dirs) @ rho) / (2.0 * step_rad))
    return np.column_stack(cols)


def effective_cov(theta, rho, g_dirs,
                  rel_error=aa.DEFAULT_ANGLE_REL_ERROR,
                  floor_deg=aa.DEFAULT_ANGLE_FLOOR_DEG):
    """R_eff = R_sensor + J Sigma_theta J^T."""
    jac = angle_jacobian(theta, rho, g_dirs)
    return sensor_cov(g_dirs) + jac @ aa.angle_covariance(
        theta, rel_error, floor_deg) @ jac.T


def posterior(Sigma_prev, A, R_eff):
    info = np.linalg.inv(Sigma_prev) + A.T @ np.linalg.solve(R_eff, A)
    return np.linalg.inv(info)


def joint_torques(spec, theta, rho, g_dirs):
    """중력 방향들에 대해 각 관절이 받는 토크의 최댓값 [N·m].

    힌지 한계값은 실물에서 모르는 값이라 판정에는 안 쓰지만, "정보량이 같다면
    덜 무리하는 쪽"을 고를 때 비교 지표로 쓴다.
    """
    import density_id_objects as obj

    plant, _ = obj.build_plant(spec, np.asarray(rho, dtype=float))
    context = plant.CreateDefaultContext()
    worst = np.zeros(len(spec.joints))
    for g_hat in g_dirs:
        plant.mutable_gravity_field().set_gravity_vector(
            alg.G_ACC * np.asarray(g_hat, dtype=float))
        plant.SetPositions(context, np.atleast_1d(theta))
        worst = np.maximum(worst, np.abs(
            plant.CalcGravityGeneralizedForces(context)))
    return worst


# ---------------------------------------------------------------------------
# 2. 선택 기준 D / A / E
#    셋 다 '상대 단위'로 정규화한 공분산 위에서 계산한다. 밀도가 700~5200 처럼
#    자릿수가 다르면 정규화 없이는 A·E 가 제일 무거운 part 에 끌려간다.
#    정지 조건도 상대 반폭이므로 이렇게 맞추는 편이 일관된다.
#    (D 는 정규화가 상수 오프셋이라 순위가 바뀌지 않는다.)
# ---------------------------------------------------------------------------
CRITERIA = ("D", "A", "E")


def normalized(Sigma, rho_hat):
    scale = np.maximum(np.abs(rho_hat), 1e-9)
    return Sigma / np.outer(scale, scale)


def criterion_score(Sigma_post, rho_hat, kind):
    """클수록 좋다. 사후 불확실성이 작아진 정도를 재는 값."""
    S = normalized(Sigma_post, rho_hat)
    if kind == "D":                       # 부피(모든 방향 곱)를 줄인다
        return -float(np.linalg.slogdet(S)[1])
    if kind == "A":                       # 분산 합(평균 오차)을 줄인다
        return -float(np.trace(S))
    if kind == "E":                       # 최악 방향 하나를 줄인다
        return -float(np.linalg.eigvalsh(S).max())
    raise ValueError(kind)


def utility(theta, rho_hat, Sigma, g_dirs, kind="D",
            rel_error=aa.DEFAULT_ANGLE_REL_ERROR,
            floor_deg=aa.DEFAULT_ANGLE_FLOOR_DEG):
    """이 자세를 한 번 더 재면 기준값이 얼마나 좋아지는가."""
    A = regressor(theta, g_dirs)
    R = effective_cov(theta, rho_hat, g_dirs, rel_error, floor_deg)
    return criterion_score(posterior(Sigma, A, R), rho_hat, kind)


# ---------------------------------------------------------------------------
# 3. 후보 선택: 격자 top-k  vs  연속 최적화
# ---------------------------------------------------------------------------
def grid_best(bounds, n_per_joint, score_fn, feasible=None):
    """격자를 깔고 최고 하나를 고른다 (현행 방식)."""
    axes = [np.linspace(lo, hi, n_per_joint) for lo, hi in bounds]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, len(bounds))
    if feasible is not None:
        grid = np.array([t for t in grid if feasible(t)])
    scores = np.array([score_fn(t) for t in grid])
    best = int(np.argmax(scores))
    return grid[best], dict(score=scores[best], n_eval=len(grid), grid=grid,
                            scores=scores)


def continuous_best(bounds, score_fn, n_starts=8, seed=0, feasible=None,
                    maxiter=60):
    """상자 제약 위에서 연속 최적화한다.

    관절각은 원래 연속량이다. 격자는 그걸 임의로 잘라놓은 것이라 진짜 최적점을
    지나칠 수밖에 없고, 관절이 늘어나면 후보 수가 n^J 로 터진다. 연속
    최적화는 그 두 문제를 동시에 없앤다.

    다중 시작(multi-start) L-BFGS-B 를 쓴다. 정보이득 지형은 국소최대가
    여러 개라 한 점에서 출발하면 갇힌다.
    """
    from scipy.optimize import minimize

    bounds = np.asarray(bounds, dtype=float)
    lo, hi = bounds[:, 0], bounds[:, 1]
    rng = np.random.default_rng(seed)

    # 시작점: 중앙 + 상자 꼭짓점 + 난수. 꼭짓점을 넣는 이유는 이 문제에서
    # 최적해가 구동범위 끝에 붙는 일이 잦기 때문이다. 다만 꼭짓점은 2^J 개라
    # 관절이 늘면 그것만으로 지수 폭발하므로 n_starts 개까지만 표집한다.
    n_dim = len(bounds)
    seeds = [0.5 * (lo + hi)]
    n_corner = 2 ** n_dim
    picks = (range(n_corner) if n_corner <= n_starts
             else rng.choice(n_corner, size=n_starts, replace=False))
    for code in picks:
        bit = [(int(code) >> k) & 1 for k in range(n_dim)]
        seeds.append(np.where(np.array(bit, dtype=bool), hi, lo))
    while len(seeds) < n_starts:
        seeds.append(lo + (hi - lo) * rng.random(n_dim))
    seeds = seeds[:max(n_starts, 2)]

    calls = {"n": 0}

    def negative(x):
        calls["n"] += 1
        return -score_fn(np.clip(x, lo, hi))

    best_x, best_s = None, -np.inf
    for start in seeds:
        res = minimize(negative, np.asarray(start, dtype=float),
                       method="L-BFGS-B", bounds=list(zip(lo, hi)),
                       options=dict(maxiter=maxiter, eps=1e-3))
        x = np.clip(res.x, lo, hi)
        if feasible is not None and not feasible(x):
            continue
        s = -res.fun
        if s > best_s:
            best_x, best_s = x, s
    if best_x is None:                       # 전부 실현 불가면 격자로 후퇴
        return grid_best(bounds, 5, score_fn, feasible)
    return best_x, dict(score=best_s, n_eval=calls["n"], n_starts=len(seeds))


# ---------------------------------------------------------------------------
# 4. 추정기: 가중최소제곱(현행)  vs  총최소제곱(오차변수 모형)
# ---------------------------------------------------------------------------
def wls_map(blocks, mu0, Sigma0, bounds):
    """현행. 각도 오차를 R_eff 에 잡음처럼 흡수시키고 rho 만 푼다.

    blocks = [(A_i, y_i, R_i), ...]
    A 자체가 틀렸다는 사실은 모형에 없다. 그래서 남는 치우침이 있다.
    """
    return aa.constrained_map(blocks, mu0, Sigma0, bounds)


def tls_map(rounds, mu0, Sigma0, bounds, g_dirs, rho_init=None,
            rel_error=aa.DEFAULT_ANGLE_REL_ERROR,
            floor_deg=aa.DEFAULT_ANGLE_FLOOR_DEG, max_nfev=None,
            grasp_sigma_m=0.0, total_mass_kg=None, grasp_init=None):
    """총최소제곱(구조화 TLS = 오차변수 최대우도).

    rounds = [(theta_measured_i, y_i), ...]

    각도가 틀렸다는 사실을 **모형 안에** 넣는다. 즉 각 라운드의 진짜 각도
    보정량 delta_i 를 rho 와 **함께** 미지수로 놓고 푼다.

        min over rho, {delta_i}
            sum_i || y_i - A(theta_i + delta_i) rho ||^2_{R_sensor^-1}
          + sum_i || delta_i ||^2_{Sigma_theta^-1}
          + || rho - mu0 ||^2_{Sigma0^-1}

    WLS 는 delta 를 '평균 0인 잡음'으로 뭉개서 A 를 고정으로 본다. 그래서
    측정을 아무리 늘려도 안 없어지는 치우침이 남는다. TLS 는 delta 를
    데이터에서 되찾으므로 그 치우침이 원리상 사라진다.

    대가는 두 가지다. (1) 비선형 최적화라 국소해가 있을 수 있고,
    (2) 미지수가 P + R*J 개로 늘어 라운드가 적으면 과적합 위험이 있다.
    두 번째는 delta 에 붙은 사전분포 항이 눌러준다.
    """
    from scipy.optimize import least_squares

    n_joint = len(np.atleast_1d(rounds[0][0]))
    n_part = len(mu0)
    R = len(rounds)

    w_sensor = 1.0 / np.sqrt(np.tile(alg.R_EPS_DIAG, len(g_dirs)))
    L0 = np.linalg.cholesky(np.linalg.inv(Sigma0)).T
    sig_theta = [np.sqrt(np.diag(aa.angle_covariance(
        np.atleast_1d(th), rel_error, floor_deg))) for th, _ in rounds]

    # 파지점 어긋남도 같이 풀 것인가. 각도 보정 delta_i 뒤에 3개를 더 붙인다.
    # 둘은 서로 다른 것을 고친다 — 각도는 라운드마다 다르고, 파지점은 모든
    # 라운드에 똑같이 실린다. 한쪽만 고치면 다른 쪽 치우침이 그대로 남는다.
    n_grasp = 3 if (grasp_sigma_m and total_mass_kg) else 0
    grasp_cols = (grasp_columns(g_dirs, total_mass_kg) if n_grasp else None)

    lo, hi = bounds
    n_angle = R * n_joint
    rho0 = np.asarray(rho_init if rho_init is not None else mu0, dtype=float)
    grasp0 = (np.zeros(3) if grasp_init is None
              else np.asarray(grasp_init, dtype=float))
    x0 = np.concatenate([np.clip(rho0, lo, hi), np.zeros(n_angle),
                         grasp0[:n_grasp]])
    x_lo = np.concatenate([np.full(n_part, lo),
                           np.full(n_angle, -np.inf),
                           np.full(n_grasp, -0.05)])
    x_hi = np.concatenate([np.full(n_part, hi),
                           np.full(n_angle, np.inf),
                           np.full(n_grasp, 0.05)])

    def residual(x):
        rho = x[:n_part]
        deltas = x[n_part:n_part + n_angle].reshape(R, n_joint)
        grasp = x[n_part + n_angle:]
        offset = grasp_cols @ grasp if n_grasp else 0.0
        parts = []
        for (theta, y), delta, sig in zip(rounds, deltas, sig_theta):
            A = regressor(np.atleast_1d(theta) + delta, g_dirs)
            parts.append((y - A @ rho - offset) * w_sensor)
            parts.append(delta / sig)
        parts.append(L0 @ (rho - mu0))
        if n_grasp:
            parts.append(grasp / grasp_sigma_m)
        return np.concatenate(parts)

    # max_nfev 를 고정하면 안 된다. 미지수가 P + R*J 로 라운드에 비례해 늘어나고,
    # 수치 미분이라 야코비안 한 번에 미지수 개수만큼 평가가 든다. 400 으로 묶어
    # 두었더니 p=6 (미지수 86) 이 547 회면 수렴하는데 잘려서 오차가 1.11 % 대신
    # 13.36 % 로 나왔다. None 이면 scipy 가 100*n 을 쓴다 — 문제 크기에 맞춰 준다.
    res = least_squares(residual, x0, bounds=(x_lo, x_hi), max_nfev=max_nfev)
    rho_hat = res.x[:n_part]
    return rho_hat, dict(delta=res.x[n_part:n_part + n_angle].reshape(R, n_joint),
                         grasp=res.x[n_part + n_angle:],
                         cost=float(res.cost), nfev=int(res.nfev),
                         jac=res.jac)


# ---------------------------------------------------------------------------
# 4-b. 파지점이 어긋난 몫도 함께 푼다
# ---------------------------------------------------------------------------
# 시뮬레이션에서는 "센서 원점이 물체의 이 점에 있다" 고 정확히 줄 수 있다.
# 실물에서는 못 준다. 지그를 대도 몇 mm 는 어긋나고, 그 어긋남 delta 는
# **모든 부위의 모멘트팔을 똑같이 밀어 놓는다.** 라운드를 늘려도 안 없어지는
# 치우침이다 (도심 오차 5 mm 가 Arm 질량 42% 를 틀리게 만든 것과 같은 구조).
#
# 그런데 이 몫은 **모형 안에 넣을 수 있다.** 센서 원점에서 잰 토크는
#
#     tau_true = sum_i m_i (c_i - delta) x g = tau_model - M (delta x g)
#              = tau_model + M G [g]_x delta
#
# 이고 총질량 M 은 저울로 이미 안다. 즉 delta 는 **선형 미지수 3개**로
# 들어온다. 힘 성분에는 안 나타난다 (F = M g 는 어디서 재든 같다).
#
# 중력 방향 하나에서 [g]_x 의 계수는 g 에 수직한 평면 2차원만 잡는다.
# 직교 3방향을 다 쓰면 3차원이 다 잡히므로 delta 는 원리상 식별된다.
GRASP_SIGMA_M = 0.005          # 지그를 대고 잡았을 때 남는 어긋남 [m]


def grasp_columns(g_dirs, total_mass_kg):
    """파지점 어긋남 delta 가 렌치에 만드는 몫. (6J x 3)"""
    blocks = []
    for g_hat in np.asarray(g_dirs, dtype=float):
        skew = np.array([[0.0, -g_hat[2], g_hat[1]],
                         [g_hat[2], 0.0, -g_hat[0]],
                         [-g_hat[1], g_hat[0], 0.0]])
        blocks.append(np.vstack([np.zeros((3, 3)),
                                 total_mass_kg * alg.G_ACC * skew]))
    return np.vstack(blocks)


def augmented(A, g_dirs, total_mass_kg):
    """[밀도 열들 | 파지점 열들]. 미지수는 [rho; delta] 가 된다."""
    return np.hstack([A, grasp_columns(g_dirs, total_mass_kg)])


def grasp_map(blocks, mu0, Sigma0, bounds, g_dirs, total_mass_kg,
              grasp_sigma_m=GRASP_SIGMA_M, grasp_bound_m=0.05):
    """밀도와 파지점 어긋남을 **함께** 푼다.

    blocks = [(A_i, y_i, R_i), ...]  — 기존 WLS 와 같은 입력.

    delta 에는 사전분포를 준다. 지그로 대는 정확도가 얼마인지는 사람이 아는
    값이고 (기본 5 mm), 그 안에서만 움직이게 눌러 두어야 밀도와 헷갈리지
    않는다. 사전분포가 없으면 delta 가 밀도 오차를 흡수해 버린다.

    돌려주는 것: (rho_hat, delta_hat, Sigma_full)
    """
    n_part = len(mu0)
    lo, hi = bounds
    rows, ys, weights = [], [], []
    for A, y, R in blocks:
        L = np.linalg.cholesky(np.linalg.inv(np.atleast_2d(R)))
        rows.append(L.T @ augmented(A, g_dirs, total_mass_kg))
        ys.append(L.T @ np.asarray(y, dtype=float))
    # 사전분포 항 (rho 와 delta 각각)
    L0 = np.linalg.cholesky(np.linalg.inv(Sigma0)).T
    prior_rows = np.hstack([L0, np.zeros((n_part, 3))])
    prior_y = L0 @ mu0
    grasp_rows = np.hstack([np.zeros((3, n_part)), np.eye(3) / grasp_sigma_m])
    rows += [prior_rows, grasp_rows]
    ys += [prior_y, np.zeros(3)]

    M = np.vstack(rows)
    b = np.concatenate(ys)
    x_lo = np.concatenate([np.full(n_part, lo), np.full(3, -grasp_bound_m)])
    x_hi = np.concatenate([np.full(n_part, hi), np.full(3, grasp_bound_m)])

    from scipy.optimize import lsq_linear
    result = lsq_linear(M, b, bounds=(x_lo, x_hi))
    Sigma_full = np.linalg.inv(M.T @ M)
    return result.x[:n_part], result.x[n_part:], Sigma_full


def tls_covariance(info, n_part):
    """TLS 해 주변의 사후 공분산 (가우스-뉴턴 근사).

    delta 를 소거한 뒤의 rho 블록만 뽑는다. delta 와 rho 의 상관이 여기서
    자동으로 반영되므로, WLS 처럼 R_eff 로 억지로 부풀릴 필요가 없다.
    """
    J = info["jac"]
    H = J.T @ J
    A = H[:n_part, :n_part]
    B = H[:n_part, n_part:]
    D = H[n_part:, n_part:]
    if B.size:
        schur = A - B @ np.linalg.solve(D, B.T)
    else:
        schur = A
    return np.linalg.inv(schur)


# ---------------------------------------------------------------------------
# 5. 정지 조건: 분산만  vs  분산 + 치우침 몫
# ---------------------------------------------------------------------------
def bias_covariance(blocks, jacobians, Sigma0, systematic_fraction,
                    thetas, rel_error=aa.DEFAULT_ANGLE_REL_ERROR,
                    floor_deg=aa.DEFAULT_ANGLE_FLOOR_DEG):
    """라운드마다 안 사라지는(계통) 각도 오차가 rho 추정에 남기는 치우침.

    각도 오차를 두 몫으로 나눈다.
      - 라운드마다 새로 뽑히는 몫  : 라운드를 늘리면 평균으로 줄어든다.
                                     이건 이미 R_eff 가 반영한다.
      - 매 라운드 똑같이 실리는 몫 : FoundationPose 자체의 보정 치우침 같은 것.
                                     아무리 재도 안 줄어든다. R_eff 는 이걸
                                     '독립'으로 보기 때문에 과소평가한다.

    여기서는 두 번째 몫만 따로 전파한다.
        S = M^-1 sum_r A_r^T R_r^-1 J_r        (공통 오프셋에 대한 민감도)
        Cov_bias = f * S Sigma_theta S^T
    """
    n_part = Sigma0.shape[0]
    M = np.linalg.inv(Sigma0)
    S = np.zeros((n_part, jacobians[0].shape[1]))
    for (A, _, R), jac in zip(blocks, jacobians):
        M = M + A.T @ np.linalg.solve(R, A)
    Minv = np.linalg.inv(M)
    for (A, _, R), jac in zip(blocks, jacobians):
        S = S + Minv @ (A.T @ np.linalg.solve(R, jac))
    Sigma_theta = aa.angle_covariance(np.atleast_1d(thetas[-1]),
                                      rel_error, floor_deg)
    return systematic_fraction * (S @ Sigma_theta @ S.T)


def bias_by_refit(rounds, mu0, Sigma0, bounds, g_dirs, rho_hat,
                  systematic_fraction, estimator="wls",
                  rel_error=aa.DEFAULT_ANGLE_REL_ERROR,
                  floor_deg=aa.DEFAULT_ANGLE_FLOOR_DEG):
    """계통 각도오차가 '실제로' 추정값을 얼마나 움직이는지 직접 재본다.

    위의 bias_covariance 는 1차 민감도인데, 가중치 R_eff 안에 이미 J Sigma J^T
    가 들어 있어서 그 방향이 억눌린다. 즉 "각도가 틀리면 답이 얼마나 흔들리나"를
    과소평가한다. 그래서 아예 **다시 풀어본다**.

      모든 라운드의 각도를 공통으로 +1 sigma / -1 sigma 만큼 밀어놓고 재추정한다.
      두 답의 벌어진 폭의 절반이 곧 그 관절 축이 만드는 치우침 몫이다.

    GT 는 안 쓴다. FoundationPose 가 자기 오차라고 밝힌 sigma 만 쓴다.
    관절 축마다 따로 밀어보고 제곱합으로 합친다.
    """
    if systematic_fraction <= 0.0:
        return np.zeros((len(rho_hat), len(rho_hat)))

    n_joint = len(np.atleast_1d(rounds[0][0]))
    sigma = np.sqrt(np.diag(aa.angle_covariance(
        np.atleast_1d(rounds[-1][0]), rel_error, floor_deg)))
    sigma = sigma * np.sqrt(systematic_fraction)

    def refit(shift):
        blocks, shifted = [], []
        for theta, y in rounds:
            th = np.atleast_1d(theta) + shift
            A = regressor(th, g_dirs)
            blocks.append((A, y, effective_cov(th, rho_hat, g_dirs,
                                               rel_error, floor_deg)))
            shifted.append((th, y))
        if estimator == "tls":
            out, _ = tls_map(shifted, mu0, Sigma0, bounds, g_dirs,
                             rho_init=rho_hat, rel_error=rel_error,
                             floor_deg=floor_deg)
            return out
        return wls_map(blocks, mu0, Sigma0, bounds)

    columns = []
    for k in range(n_joint):
        shift = np.zeros(n_joint)
        shift[k] = sigma[k]
        plus = refit(shift)
        minus = refit(-shift)
        columns.append(0.5 * (plus - minus))
    S = np.column_stack(columns)
    return S @ S.T


def residual_inflation(blocks, rho_hat, dof_floor=1, offset=None,
                       n_extra=0):
    """잔차가 모형보다 크면 그만큼 불확실성을 부풀린다.

    모형이 맞으면 백색화 잔차 제곱합 / 자유도 = 1 이어야 한다. 이게 3 이면
    "내 잡음 모형이 실제보다 sqrt(3) 배 낙관적이었다"는 뜻이고, GT 없이
    데이터만으로 알 수 있다. 1 미만이면 부풀리지 않는다(줄이지도 않는다).

    offset 을 주면 그 몫을 예측값에 더해서 잔차를 잰다. 파지점 어긋남을
    함께 풀 때 반드시 필요하다 — 안 넣으면 어긋남이 통째로 잔차로 잡혀
    팽창이 100배까지 뛰고, 정지 조건이 영영 만족되지 않는다 (실제로 그랬다).
    n_extra 는 늘어난 미지수 개수로, 자유도에서 빼 준다.
    """
    total, rows = 0.0, 0
    for A, y, R in blocks:
        r = y - A @ rho_hat
        if offset is not None:
            r = r - offset
        total += float(r @ np.linalg.solve(R, r))
        rows += len(y)
    dof = max(rows - len(rho_hat) - n_extra, dof_floor)
    return max(1.0, np.sqrt(total / dof))


def residual_scale(blocks, rounds, rho_hat, g_dirs, tls_info,
                   stop_rule="residual", grasp_offset=None, n_grasp=0):
    """정지 판단에 쓸 잔차 팽창 배율. **추정기가 실제로 맞춘 모형으로 잰다.**

    TLS 는 각 라운드의 각도 보정량 delta 를 rho 와 함께 푼다. 즉 맞춘 모형은
    y ~ A(theta + delta) rho 이지 y ~ A(theta) rho 가 아니다. 그런데 잔차를
    보정 전 회귀행렬로 재면 **TLS 가 찾아낸 delta 가 통째로 잔차로 잡힌다.**
    그러면 팽창이 폭주하고, 추정이 이미 목표를 넘었는데도 정지 조건이 영영
    만족되지 않는다.

    실제로 그랬다 (각도오차 5 %, 10 라운드):

        p=4   팽창 79.3 배 -> 1.05 배    반폭 89.17 % -> 1.18 %  (실제오차 0.28 %)
        p=5   팽창 94.3 배 -> 1.13 배    반폭 156.1 % -> 1.87 %  (실제오차 0.89 %)

    같은 실수를 파지점 어긋남에서 이미 한 번 했다 (residual_inflation 주석).
    미지수로 함께 푼 것은 무엇이든 잔차를 잴 때도 반영해야 한다.
    """
    if stop_rule == "variance":
        return 1.0
    if tls_info is None or "delta" not in tls_info:
        return residual_inflation(blocks, rho_hat, offset=grasp_offset,
                                  n_extra=n_grasp)
    delta = np.atleast_2d(tls_info["delta"])
    fitted = [(regressor(np.atleast_1d(theta) + d, g_dirs), y, R_eff)
              for (theta, y), d, (_, _, R_eff) in zip(rounds, delta, blocks)]
    return residual_inflation(fitted, rho_hat, offset=grasp_offset,
                              n_extra=int(delta.size) + n_grasp)


def half_width(Sigma, rho_hat, Cov_bias=None, inflate=1.0, z=1.96):
    """95 % 상대 반폭. 치우침 몫과 잔차 팽창을 선택적으로 더한다."""
    var = np.diag(Sigma) * inflate ** 2
    if Cov_bias is not None:
        var = var + np.diag(Cov_bias)
    return z * np.sqrt(var) / np.maximum(np.abs(rho_hat), 1e-9)


def stopping_width(half, n_wanted=None):
    """정지 판단에 쓰는 대표값. **알고 싶은 것** 만 본다.

    힌지처럼 이미 저울로 재서 아는 양은 rho 벡터에 같이 들어가지만, 탐색의
    목적이 아니다. 그런 성가신 매개변수(nuisance parameter)까지 최대값에
    넣으면, 부위 밀도가 다 수렴했는데도 힌지의 사전분포 폭(2 %) 때문에
    루프가 안 끝난다. 실제로 그 일이 일어났다.

      부위 반폭 [0.074, 0.639, 0.361] %  <- 이미 목표 1 % 달성
      힌지 반폭 [3.712, 3.712] %         <- 사전분포 폭 그대로, 안 줄어듦
      전체 최대 3.712 %                  -> 목표 미달로 판정, 계속 돌음

    n_wanted 를 주면 앞쪽 그만큼만 본다 (부위가 항상 앞에 온다).
    """
    half = np.atleast_1d(half)
    return float(np.max(half[:n_wanted] if n_wanted else half))


# ---------------------------------------------------------------------------
# 5-b. 재기 전에 "이 목표가 가능한가" 를 계산한다
#
# 사후 공분산은 **측정값에 의존하지 않는다.** posterior(Sigma, A, R) 를 보면
# y 가 없다 — A(theta) 와 R 만 들어간다. 그래서 로봇을 움직이기 전에 "이
# 물체·이 센서로 N 라운드를 돌면 반폭이 얼마까지 내려가나" 를 정확히 계산할
# 수 있다.
#
# 이게 왜 필요한가. session_20260904_1736 은 목표를 5 % 로 잡고 돌았는데,
# 계산해 보면 램프(571 g) + AFT200 조합에서는 모든 것이 완벽해도 40 % 가
# 바닥이다. 부위 셋 중 둘이 100 g 미만이고, 100 g 은 약 1 N 인데 센서 잡음이
# 0.4 N 이다. **원리적으로 도달 불가능한 목표를 놓고 8 라운드를 돌았다.**
#
# 여기서 나오는 값은 **최선의 경우**다. 잔차 팽창을 1.0 으로 놓고, 각도도
# 매번 최적점을 정확히 맞춘다고 본다. 실제는 이보다 나쁘다. 그러므로 목표가
# 이 값보다 작으면 **확실히** 도달 못 한다.
# ---------------------------------------------------------------------------
def forecast_covariance(designs, mu0, Sigma0, g_dirs, total_mass_kg=None,
                        grasp_sigma_m=0.0):
    """설계(A, R) 목록만으로 밀도 사후 공분산을 만든다. 측정값이 필요 없다.

    grasp_sigma_m 을 주면 파지점 어긋남을 미지수로 함께 푼 뒤의 밀도 블록을
    돌려준다 — grasp_map 과 같은 구성이고 y 만 빠진 것이다. 이 몫을 빼먹으면
    실제보다 좋게 나온다 (자유변수 3 개가 밀도 정보를 나눠 갖기 때문).
    """
    n_part = len(mu0)
    use_grasp = bool(grasp_sigma_m and total_mass_kg)
    rows = []
    for A, R in designs:
        L = np.linalg.cholesky(np.linalg.inv(np.atleast_2d(R)))
        rows.append(L.T @ (augmented(A, g_dirs, total_mass_kg) if use_grasp
                           else A))
    L0 = np.linalg.cholesky(np.linalg.inv(Sigma0)).T
    if use_grasp:
        rows.append(np.hstack([L0, np.zeros((n_part, 3))]))
        rows.append(np.hstack([np.zeros((3, n_part)),
                               np.eye(3) / grasp_sigma_m]))
    else:
        rows.append(L0)
    M = np.vstack(rows)
    return np.linalg.inv(M.T @ M)[:n_part, :n_part]


def achievable_half_width(bounds, mu0, Sigma0, g_dirs, max_rounds=6,
                          criterion="D", feasible=None, n_wanted=None,
                          rel_error=aa.DEFAULT_ANGLE_REL_ERROR,
                          floor_deg=aa.DEFAULT_ANGLE_FLOOR_DEG,
                          total_mass_kg=None, grasp_sigma_m=0.0,
                          n_starts=8, seed=0):
    """라운드를 늘려 가며 도달 가능한 최선의 반폭. [(라운드, 반폭), ...]

    매 라운드 가장 좋은 자세를 고른다 — 실제 탐색과 같은 기준이다. 다만
    측정을 안 하므로 rho 는 사전분포 평균에 머문다. R_eff 가 rho 에 의존하니
    정확한 예보는 아니지만, **목표가 가능한지 판정하기에는 충분하다.**
    """
    Sigma = np.asarray(Sigma0, dtype=float).copy()
    designs, out = [], []
    for index in range(1, max_rounds + 1):
        def score(theta, _S=Sigma):
            return utility(theta, mu0, _S, g_dirs, criterion,
                           rel_error, floor_deg)

        theta, _ = continuous_best(bounds, score, n_starts=n_starts,
                                   seed=seed + index, feasible=feasible)
        theta = np.atleast_1d(theta)
        A = regressor(theta, g_dirs)
        R = effective_cov(theta, mu0, g_dirs, rel_error, floor_deg)
        Sigma = posterior(Sigma, A, R)
        designs.append((A, R))
        cov = forecast_covariance(designs, mu0, Sigma0, g_dirs,
                                  total_mass_kg, grasp_sigma_m)
        half = half_width(cov, mu0, inflate=1.0)
        out.append((index, stopping_width(half, n_wanted)))
    return out


# ---------------------------------------------------------------------------
# 6. 한 곳으로 모은 폐루프
#
# 위 다섯 가지 선택을 인자로 받는 유일한 탐색 루프다. 로봇 없이 숫자만 볼 때
# (export_urdf, study_criterion) 여기를 쓰고, 화면 두 개로 볼 때는 dual_view 가
# 같은 조각들을 화면 갱신과 섞어 쓴다. 알고리즘은 한 벌뿐이다.
#
# 기본값은 study_*.py 로 확인한 결론이다.
#   select="continuous"  격자 top-1 대신 연속 최적화        (study_continuous)
#   criterion="D"        A/E 보다 라운드가 짧다              (study_criterion)
#   estimator="tls"      오차변수 치우침이 사라진다          (study_tls)
#   stop_rule="residual" 적용률이 88 % -> 98 % 로 제자리     (study_stopping)
#
# GT 는 이 함수에 들어오지 않는다. 호출한 쪽이 반환값을 채점할 뿐이다.
# ---------------------------------------------------------------------------
def closed_loop(spec, target=0.01, max_rounds=12, seed=0,
                rel_error=aa.DEFAULT_ANGLE_REL_ERROR,
                floor_deg=aa.DEFAULT_ANGLE_FLOOR_DEG,
                g_dirs=None, select="continuous", criterion="D",
                estimator="tls", stop_rule="residual", systematic=0.3,
                n_starts=8, grid_steps=5, feasible=None, n_wanted=None,
                verbose=False):
    """자세 고르기 -> 재기 -> 갱신 을 목표 불확실성까지 반복한다."""
    g_dirs = CANONICAL_TRIAD if g_dirs is None else np.asarray(g_dirs)
    if n_wanted is None:
        n_wanted = len(spec.parts)      # 힌지 등 부속은 정지 판단에서 뺀다
    rng = np.random.default_rng(seed)
    bounds = [j.limits_rad for j in spec.joints]
    Sigma = alg.SIGMA0.copy()
    rho_hat = alg.MU0.copy()
    blocks, rounds, history = [], [], []
    inflate, bias_cov = 1.0, None

    for index in range(1, max_rounds + 1):
        def score(theta):
            return utility(theta, rho_hat, Sigma, g_dirs, criterion,
                           rel_error, floor_deg)

        if select == "continuous":
            theta, _ = continuous_best(bounds, score, n_starts=n_starts,
                                       seed=seed + index, feasible=feasible)
        else:
            theta, _ = grid_best(bounds, grid_steps, score, feasible=feasible)
        theta = np.atleast_1d(theta)

        # 작업자가 맞춘 실제 각도와, FoundationPose 가 읽어준 각도는 다르다.
        # 알고리즘은 measured 만 본다.
        sigma = np.sqrt(np.diag(aa.angle_covariance(theta, rel_error, floor_deg)))
        actual = theta + rng.normal(0.0, sigma)
        measured = actual + rng.normal(0.0, sigma)
        y = alg.measure(actual, g_dirs=list(g_dirs), rng=rng)

        A = regressor(measured, g_dirs)
        R = effective_cov(measured, rho_hat, g_dirs, rel_error, floor_deg)
        blocks.append((A, y, R))
        rounds.append((measured, y))
        Sigma = posterior(Sigma, A, R)

        rho_wls = wls_map(blocks, alg.MU0, alg.SIGMA0, alg.RHO_BOUNDS)
        tls_info = None
        if estimator == "tls":
            rho_hat, tls_info = tls_map(rounds, alg.MU0, alg.SIGMA0,
                                        alg.RHO_BOUNDS, g_dirs,
                                        rho_init=rho_wls,
                                        rel_error=rel_error,
                                        floor_deg=floor_deg)
        else:
            rho_hat = rho_wls

        inflate = residual_scale(blocks, rounds, rho_hat, g_dirs, tls_info,
                                 stop_rule)
        if stop_rule == "bias":
            bias_cov = bias_by_refit(rounds, alg.MU0, alg.SIGMA0,
                                     alg.RHO_BOUNDS, g_dirs, rho_hat,
                                     systematic, estimator, rel_error,
                                     floor_deg)
        half = half_width(Sigma, rho_hat, Cov_bias=bias_cov, inflate=inflate)
        worst = stopping_width(half, n_wanted)
        history.append(dict(round=index, theta=theta.copy(),
                            rho=rho_hat.copy(), half=half.copy(),
                            worst=worst, inflate=inflate))
        if verbose:
            print(f"  round {index}: q={np.round(np.degrees(theta), 1)} deg"
                  f"  반폭 {100*worst:.3f}%"
                  + (f"  (잔차팽창 x{inflate:.2f})" if inflate > 1.005 else ""))
        if worst <= target:
            break

    return dict(rho_hat=rho_hat, Sigma=Sigma, half=half, worst=worst,
                inflate=inflate, rounds=len(history),
                converged=bool(worst <= target), history=history)
