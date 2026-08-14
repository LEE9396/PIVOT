"""상황에 따라 달라지는 자세 선택.

왜 지금까지 같은 각도만 나왔는가
--------------------------------
선형·가우시안 모델에서 정보행렬 A^T R^-1 A 는 측정값 y 에 전혀 의존하지
않는다. 그래서 D-optimal 최적 설계는 데이터를 보기 전에 이미 정해져 있고,
매 라운드 같은 답이 나오는 것이 수학적으로 당연하다. 지금의 "능동 선택"은
사실상 오프라인 설계였다.

무엇이 알고리즘을 실제로 적응하게 만드는가
------------------------------------------
추정값에 의존하는 제약. 힌지가 버티는지는 **부위 질량**에 달렸는데, 그건
바로 우리가 추정하려는 값이다. 실제 실험에서는 정답을 모르므로,
현재 믿음(rho_hat, Sigma)으로 안전 여부를 판단해야 한다.

  - 초반: 불확실성이 크다 -> 무거울 수도 있다고 보고 보수적인 자세만 고른다
  - 후반: 불확실성이 줄었다 -> 정보량이 큰 공격적인 자세도 허용된다

따라서 고르는 자세가 라운드마다 실제로 바뀐다.

기존 코드는 make_is_feasible(..., rho_gt) 처럼 정답 밀도로 안전을 판단했다.
로봇이 가질 수 없는 정보이므로, 이 모듈이 그것을 대체한다.
"""

import sys as _sys, pathlib as _pathlib
# 이 폴더는 my_work 밖이라 형제 모듈이 안 보인다. run_drake_env.sh 가
# PYTHONPATH 를 지우므로 (ROS 오염 제거) 환경변수로는 못 넣는다.
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))


import numpy as np

import density_id_drake as alg
import density_id_objects as obj

DEFAULT_CONFIDENCE_K = 2.0     # 평균 + k*표준편차 를 최악으로 본다
DEFAULT_TIE_BAND = 0.02        # 정보이득이 최댓값의 2 % 안이면 동률로 본다

# 안전에서 정보로 넘어가는 기준. 부위 밀도의 상대 불확실성
# r = max_i (sigma_i / rho_hat_i) 가 이 값 아래로 내려오면 정보이득만 본다.
DEFAULT_TRUST_LEVEL = 0.20     # 20 %


def blend_weight(rho_hat, Sigma, trust_level=DEFAULT_TRUST_LEVEL):
    """정보이득에 줄 가중치 w. 0 이면 안전만, 1 이면 정보만 본다.

    첫 라운드에는 휴리스틱 사전분포뿐이라 상대 불확실성이 크다 -> w=0.
    측정이 쌓여 불확실성이 trust_level 아래로 내려오면 w=1 이 된다.
    """
    rho = np.maximum(np.abs(np.asarray(rho_hat, dtype=float)), 1e-9)
    relative = float(np.max(np.sqrt(np.diag(Sigma)) / rho))
    return float(np.clip(1.0 - relative / trust_level, 0.0, 1.0)), relative


# ---------------------------------------------------------------------------
def torque_jacobian(spec, theta, gravity_dirs=None):
    """관절 중력토크를 밀도의 선형함수로 표현한다.

        tau(theta, g) = J[g] @ rho          J[g] 는 (관절 수 x 부위 수)

    밀도 e_i (부위 i 만 1) 로 만든 plant 의 중력토크가 곧 J 의 i 번째 열이다.
    질량이 밀도에 선형이고 중력토크가 질량에 선형이므로 정확하다.
    """
    if gravity_dirs is None:
        gravity_dirs = alg.G_DIRS
    n_parts = len(spec.parts)
    columns = {tuple(g): [] for g in gravity_dirs}
    for index in range(n_parts):
        unit = np.full(n_parts, 1e-9)
        unit[index] = 1.0
        plant, _ = obj.build_plant(spec, unit)
        context = plant.CreateDefaultContext()
        plant.SetPositions(context, np.atleast_1d(np.asarray(theta)))
        for g_hat in gravity_dirs:
            plant.mutable_gravity_field().set_gravity_vector(alg.G_ACC * g_hat)
            columns[tuple(g_hat)].append(
                plant.CalcGravityGeneralizedForces(context).copy())
    return {g: np.column_stack(cols) for g, cols in columns.items()}


class TorqueModel:
    """후보 자세마다 토크 야코비안을 한 번만 계산해 재사용한다."""

    def __init__(self, spec, candidates, gravity_dirs=None):
        self.spec = spec
        self.candidates = [np.atleast_1d(np.asarray(c)) for c in candidates]
        self.jacobians = [torque_jacobian(spec, c, gravity_dirs)
                          for c in self.candidates]

    def worst_torque(self, index, rho):
        """이 자세에서 관절이 받는 토크의 최댓값 (중력 3방향, 관절 전체)."""
        return max(float(np.max(np.abs(J @ rho)))
                   for J in self.jacobians[index].values())

    def torque_upper_bound(self, index, rho_hat, Sigma, k=DEFAULT_CONFIDENCE_K):
        """현재 믿음에서 토크의 상한 (평균 + k*표준편차).

        tau 가 rho 에 선형이므로 각 행 j 에 대해
            평균 = J[j] @ rho_hat,  분산 = J[j] Sigma J[j]^T
        """
        worst = 0.0
        for J in self.jacobians[index].values():
            mean = J @ rho_hat
            var = np.einsum("ij,jk,ik->i", J, Sigma, J)
            bound = np.abs(mean) + k * np.sqrt(np.maximum(var, 0.0))
            worst = max(worst, float(np.max(bound)))
        return worst


# ---------------------------------------------------------------------------
class AdaptiveSelector:
    """현재 믿음으로 안전을 판정하고, 그 안에서 정보이득이 큰 자세를 고른다.

    동률 구간에서는 이미 쓴 자세와 가장 먼 것을 고른다. 정보이득을 거의
    잃지 않으면서 자세가 다양해지고, 모델 오차가 한 자세에 몰리는 것도 막는다.
    """

    def __init__(self, spec, candidates, hinge, safety=obj.DEFAULT_SAFETY,
                 confidence_k=DEFAULT_CONFIDENCE_K, tie_band=DEFAULT_TIE_BAND,
                 diversify=True, trust_level=DEFAULT_TRUST_LEVEL):
        self.spec = spec
        self.candidates = [np.atleast_1d(np.asarray(c)) for c in candidates]
        self.model = TorqueModel(spec, self.candidates)
        self.limit = hinge.holding_torque_nm / safety
        self.confidence_k = confidence_k
        self.tie_band = tie_band
        self.diversify = diversify
        self.trust_level = trust_level
        self.used = []
        self.log = []

    # ------------------------------------------------------------------
    def feasible_indices(self, rho_hat, Sigma):
        """현재 믿음에서 힌지가 버틸 것으로 보이는 자세들."""
        return [i for i in range(len(self.candidates))
                if self.model.torque_upper_bound(
                    i, rho_hat, Sigma, self.confidence_k) <= self.limit]

    def select(self, rho_hat, Sigma):
        """안전과 정보이득을 섞어 고른다. 섞는 비율이 확신에 따라 변한다.

        1라운드는 휴리스틱 사전분포뿐이라 w=0 -> 토크가 가장 낮은 자세.
        측정이 쌓여 불확실성이 줄면 w -> 1 -> 정보이득이 가장 큰 자세.
        """
        n = len(self.candidates)
        bounds = np.array([self.model.torque_upper_bound(
            i, rho_hat, Sigma, self.confidence_k) for i in range(n)])
        gains = np.array([alg.info_gain(alg.regressor(c), Sigma)
                          for c in self.candidates])
        w, relative = blend_weight(rho_hat, Sigma, self.trust_level)

        # 안전 제약을 통과한 자세가 있으면 그 안에서만 고른다.
        allowed = np.flatnonzero(bounds <= self.limit)
        guaranteed = allowed.size > 0
        if not guaranteed:
            # 아직 어떤 자세도 안전을 보장할 수 없다 (사전분포가 넓어서).
            # 이때는 후보 전체를 두고 안전 쪽에 무게를 실어 고른다.
            allowed = np.arange(n)

        def normalize(values):
            span = values.max() - values.min()
            return (np.zeros_like(values) if span < 1e-12
                    else (values - values.min()) / span)

        safety_score = 1.0 - normalize(bounds[allowed])   # 토크가 낮을수록 높다
        info_score = normalize(gains[allowed])
        score = w * info_score + (1.0 - w) * safety_score

        best = score.max()
        tied = [int(allowed[j]) for j in range(allowed.size)
                if score[j] >= best - self.tie_band]
        if self.diversify and self.used and len(tied) > 1:
            def distance(index):
                return min(np.linalg.norm(self.candidates[index] - u)
                           for u in self.used)
            chosen = max(tied, key=distance)
        else:
            chosen = int(allowed[int(np.argmax(score))])

        self.used.append(self.candidates[chosen])
        self.log.append(dict(
            n_feasible=int(np.sum(bounds <= self.limit)),
            n_tied=len(tied),
            chosen_deg=np.degrees(self.candidates[chosen]).tolist(),
            gain=float(gains[chosen]),
            gain_rank=int(np.sum(gains > gains[chosen]) + 1),
            weight=w,
            relative_uncertainty=relative,
            guaranteed_safe=bool(guaranteed),
            torque_bound=float(bounds[chosen]),
        ))
        return self.candidates[chosen]


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 토크 한계를 모르고 실험하기
#
# 힌지 유지토크는 제조사가 공개하지 않고 나사 조절식이라 매번 달라진다.
# 그것을 미리 알지 않고도 실험을 성립시키는 세 가지 장치.
#
#   1) 필요 자체를 줄인다 — 측정 삼각대를 회전시켜 토크를 낮춘다.
#      직교 삼각대와 등방 노이즈에서는 정보행렬이 삼각대 회전에 불변이므로,
#      정보를 하나도 잃지 않고 관절토크만 줄일 수 있다.
#   2) 예측 대신 탐지한다 — 관절이 미끄러지면 가정한 각도와 측정이 어긋난다.
#      노이즈를 아는 상태이므로 잔차 카이제곱 검정으로 잡아낼 수 있다.
#   3) 한계를 온라인으로 좁힌다 — 버틴 자세는 하한, 미끄러진 자세는 상한.
# ---------------------------------------------------------------------------
def best_measurement_triad(spec, theta, rho_reference, n_trials=400, seed=0):
    """관절토크를 최소로 만드는 직교 측정 삼각대를 찾는다.

    정보량은 삼각대 회전에 불변이므로 이 최적화는 정보를 전혀 희생하지 않는다.
    rho_reference 는 현재 추정치를 쓰면 된다 (정답이 아니어도 무방).
    """
    from scipy.spatial.transform import Rotation

    canonical = [np.array(v, dtype=float)
                 for v in ([0, 0, -1], [1, 0, 0], [0, 1, 0])]
    rng = np.random.default_rng(seed)
    best = (np.inf, canonical)
    for _ in range(n_trials):
        rotation = Rotation.random(
            random_state=int(rng.integers(1 << 30))).as_matrix()
        dirs = [rotation @ g for g in canonical]
        jac = torque_jacobian(spec, theta, gravity_dirs=dirs)
        worst = max(float(np.max(np.abs(J @ rho_reference)))
                    for J in jac.values())
        if worst < best[0]:
            best = (worst, dirs)
    return best[1], best[0]


def slip_chi2(A, y, rho_hat, Sigma, r_stack_diag=None):
    """측정이 '가정한 관절각'과 맞는지 검정한다.

    관절이 미끄러지면 실제 각도가 달라져 잔차가 노이즈로 설명되지 않는다.
    잔차 공분산은 센서 노이즈와 밀도 불확실성을 함께 반영한다.

        r = y - A rho_hat,   S = R + A Sigma A^T,   chi2 = r^T S^-1 r

    반환한 chi2 를 자유도 len(y) 와 비교한다. 크게 넘으면 미끄러진 것이다.
    """
    if r_stack_diag is None:
        r_stack_diag = alg.R_STACK_DIAG
    residual = np.asarray(y) - A @ rho_hat
    covariance = np.diag(r_stack_diag) + A @ Sigma @ A.T
    return float(residual @ np.linalg.solve(covariance, residual)), len(residual)


def slip_detected(A, y, rho_hat, Sigma, threshold_ratio=3.0):
    """chi2 가 자유도의 threshold_ratio 배를 넘으면 미끄러진 것으로 본다."""
    chi2, dof = slip_chi2(A, y, rho_hat, Sigma)
    return chi2 > threshold_ratio * dof, chi2 / dof


def relative_half_width(rho_hat, Sigma, z=1.96):
    """부위별 95 % 신뢰구간의 상대 반폭 중 최댓값. 정지 조건에 쓴다."""
    rho = np.maximum(np.abs(np.asarray(rho_hat, dtype=float)), 1e-9)
    return float(np.max(z * np.sqrt(np.diag(Sigma)) / rho))


def run_until_converged(spec, candidates, hinge, rho_gt, target=0.01,
                        max_rounds=20, seed=0, safety=obj.DEFAULT_SAFETY,
                        confidence_k=DEFAULT_CONFIDENCE_K,
                        tie_band=DEFAULT_TIE_BAND, diversify=True,
                        trust_level=DEFAULT_TRUST_LEVEL, verbose=False):
    """불확실성이 목표 아래로 내려갈 때까지 탐색한 뒤 GT 와 비교한다.

    정답은 측정 생성에만 쓰이고 정지 판단에는 절대 쓰이지 않는다. 그래야
    "불확실성이 충분히 줄었다"는 알고리즘의 판단이 실제 오차와 맞는지를
    검사할 수 있다. 둘이 어긋나면 탐색에 쓴 후보 자세가 무의미했다는 뜻이다.
    """
    selector = AdaptiveSelector(spec, candidates, hinge, safety,
                                confidence_k, tie_band, diversify, trust_level)
    rng = np.random.default_rng(seed)
    Sigma = alg.SIGMA0.copy()
    rho_hat = alg.MU0.copy()
    A_all = np.empty((0, alg.P))
    y_all = np.empty(0)
    used, breached = [], False

    for index in range(1, max_rounds + 1):
        theta = selector.select(rho_hat, Sigma)
        entry = selector.log[-1]
        used.append(tuple(np.round(entry["chosen_deg"], 1)))
        nearest = int(np.argmin([np.linalg.norm(c - theta)
                                 for c in selector.candidates]))
        if selector.model.worst_torque(nearest, rho_gt) > selector.limit:
            breached = True

        A_all = np.vstack([A_all, alg.regressor(theta)])
        y_all = np.concatenate([y_all, alg.measure(theta, rng=rng)])
        Sigma = alg.posterior_covariance(Sigma, alg.regressor(theta))
        rho_hat = alg.constrained_map(A_all, y_all)

        half = relative_half_width(rho_hat, Sigma)
        if verbose:
            print(f"    round {index}: q={used[-1]}"
                  f"  95% 상대반폭 {100*half:.3f}%")
        if half <= target:
            break

    predicted = 1.96 * np.sqrt(np.diag(Sigma))
    actual = np.abs(rho_hat - rho_gt)
    return dict(
        rounds=index,
        converged=half <= target,
        poses=used,
        n_unique=len(set(used)),
        rho_hat=rho_hat,
        rho_gt=np.asarray(rho_gt, dtype=float),
        predicted_95=predicted,          # 알고리즘이 주장한 오차 한계
        actual_error=actual,             # 실제 오차 (GT 로만 계산 가능)
        max_rel_predicted=half,
        max_rel_actual=float(np.max(actual / np.abs(rho_gt))),
        calibration=float(np.max(actual / np.maximum(predicted, 1e-12))),
        torque_breached=breached,
    )


def run_adaptive(spec, candidates, hinge, rho_gt, n_rounds=6, seed=0,
                 safety=obj.DEFAULT_SAFETY, confidence_k=DEFAULT_CONFIDENCE_K,
                 tie_band=DEFAULT_TIE_BAND, diversify=True, verbose=True,
                 trust_level=DEFAULT_TRUST_LEVEL):
    """추정과 안전 판정을 함께 돌리는 폐루프. 정답 밀도는 측정 생성에만 쓴다."""
    selector = AdaptiveSelector(spec, candidates, hinge, safety,
                                confidence_k, tie_band, diversify, trust_level)
    rng = np.random.default_rng(seed)
    Sigma = alg.SIGMA0.copy()
    rho_hat = alg.MU0.copy()
    A_all = np.empty((0, alg.P))
    y_all = np.empty(0)
    history = []

    for index in range(1, n_rounds + 1):
        theta = selector.select(rho_hat, Sigma)
        entry = selector.log[-1]
        A = alg.regressor(theta)
        y = alg.measure(theta, rng=rng)
        A_all = np.vstack([A_all, A])
        y_all = np.concatenate([y_all, y])
        Sigma = alg.posterior_covariance(Sigma, A)
        rho_hat = alg.constrained_map(A_all, y_all)

        rmse = float(np.sqrt(np.mean((rho_hat - rho_gt) ** 2)))
        # 실제 토크(정답 밀도)로 안전이 지켜졌는지 사후 감사한다.
        actual = selector.model.worst_torque(
            int(np.argmin([np.linalg.norm(c - theta)
                           for c in selector.candidates])), rho_gt)
        history.append(dict(round=index, theta_deg=entry["chosen_deg"],
                            n_feasible=entry["n_feasible"],
                            n_tied=entry["n_tied"],
                            weight=entry["weight"],
                            relative_uncertainty=entry["relative_uncertainty"],
                            guaranteed_safe=entry["guaranteed_safe"],
                            gain_rank=entry["gain_rank"],
                            torque_bound=entry["torque_bound"],
                            torque_actual=actual,
                            rho=rho_hat.copy(), rmse=rmse))
        if verbose:
            mode = ("정보" if entry["weight"] > 0.99 else
                    "안전" if entry["weight"] < 0.01 else "혼합")
            mark = "" if actual <= selector.limit else "  <- 실제 토크 초과!"
            print(f"  round {index}: q={np.round(entry['chosen_deg'], 1)}"
                  f"  w={entry['weight']:.2f}({mode})"
                  f"  상대불확실성 {100*entry['relative_uncertainty']:6.1f}%"
                  f"  정보순위 {entry['gain_rank']}/{len(candidates)}"
                  f"  토크 실제 {actual:.3f} (상한 {entry['torque_bound']:.3f})"
                  f" / 한계 {selector.limit:.3f}"
                  f"  RMSE {rmse:.2f}{mark}")
    return history
