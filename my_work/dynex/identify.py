"""가우시안 사후분포 추정기와 물리 일관성 투영.

미지수 Φ = [phi_1; ...; phi_P] (부위당 10개). 측정은 y = Y Φ + b + ε 이고
b 는 궤적마다 하나씩 붙는 센서 영점(6개)이다. 사전분포가 가우시안이고
모형이 선형이므로 사후분포는 닫힌 형태다. 물리적으로 불가능한 해
(유사관성이 양정치가 아님)는 준정부호 계획으로 가장 가까운 가능한 점에
투영한다 (Wensing 2017, Scalable Real2Sim 의 방식).
"""
from dataclasses import dataclass

import numpy as np

from .inertial import (N_PARAM, central_inertia, com, is_physical,
                       principal_moments, pseudo_basis, solid_box_phi,
                       point_mass_phi, scale_vector, split)


@dataclass
class GaussianEstimate:
    """상태 = [phi_1; ...; phi_P; b]. b 는 세션 전체에 하나뿐인 센서 영점 6개.

    영점을 궤적마다 따로 두면 느린 궤적의 일정한 중력 토크(1차 모멘트 정보의
    대부분)가 영점으로 흡수된다. 실물에서도 영점은 타어로 좁게 알고 세션
    동안 천천히 흐르는 값이므로, 상태에 한 번 넣고 모든 궤적이 공유한다.
    """
    mean: np.ndarray
    cov: np.ndarray
    n_bias: int = 0

    @property
    def n_parts(self):
        return (self.mean.size - self.n_bias) // N_PARAM

    @property
    def n_phi(self):
        return self.mean.size - self.n_bias

    def part(self, k):
        s = slice(N_PARAM * k, N_PARAM * (k + 1))
        return self.mean[s], self.cov[s, s]

    def std(self):
        return np.sqrt(np.clip(np.diag(self.cov), 0.0, None))

    def copy(self):
        return GaussianEstimate(self.mean.copy(), self.cov.copy(), self.n_bias)

    def bias(self):
        return self.mean[self.n_phi:], self.cov[self.n_phi:, self.n_phi:]


# ---------------------------------------------------------------------------
def prior_from_spec(spec, total_mass_kg=None, mass_rel_std=0.5,
                    com_frac_std=0.5, inertia_rel_std=1.0,
                    bias_sigma=(0.2, 0.2, 0.2, 0.01, 0.01, 0.01)):
    """부위마다 균일 밀도 직육면체를 평균으로 둔 사전분포.

    total_mass_kg 를 주면 부피 비례로 나눠 평균 질량을 정한다 (논문의
    '저울 총무게만 아는' 출발점). 힌지 질량은 자식 부위에 더한다.
    """
    parts = spec.parts
    volumes = np.array([p.volume_m3 for p in parts])
    hinge_mass = {p.name: 0.0 for p in parts}
    hinge_pos = {}
    for joint in spec.joints:
        if joint.hinge_mass_kg > 0.0:
            hinge_mass[joint.child] += joint.hinge_mass_kg
            child = next(p for p in parts if p.name == joint.child)
            origin = np.array(joint.origin_in_child_link_mm
                              if joint.origin_in_child_link_mm is not None
                              else (0.0, 0.0, 0.0))
            hinge_pos[joint.child] = (origin - np.array(child.bbox_center_in_link_mm)) * 1e-3
    if total_mass_kg is None:
        masses = np.array([p.rho_gt * p.volume_m3 for p in parts])
    else:
        free = total_mass_kg - sum(hinge_mass.values())
        masses = free * volumes / volumes.sum()
    means, stds = [], []
    for part, mass in zip(parts, masses):
        dims = np.array(part.bbox_mm) * 1e-3
        phi = solid_box_phi(mass, dims)
        if hinge_mass[part.name] > 0.0:
            phi = phi + point_mass_phi(hinge_mass[part.name], hinge_pos[part.name])
        m, h, I = split(phi)
        s = np.empty(N_PARAM)
        s[0] = mass_rel_std * m
        s[1:4] = com_frac_std * m * dims / 2.0
        s[4:10] = inertia_rel_std * np.abs(np.array([I[0, 0], I[0, 1], I[0, 2],
                                                   I[1, 1], I[1, 2], I[2, 2]]))
        s[4:10] = np.maximum(s[4:10], inertia_rel_std * 0.1 * np.trace(I) / 3.0)
        means.append(phi)
        stds.append(s)
    n_bias = 0 if bias_sigma is None else len(bias_sigma)
    mean = np.concatenate(means + ([np.zeros(n_bias)] if n_bias else []))
    stds = stds + ([np.asarray(bias_sigma, float)] if n_bias else [])
    cov = np.diag(np.concatenate(stds) ** 2)
    return GaussianEstimate(mean, cov, n_bias)


def apply_total_mass(est, total_mass_kg, rel_error=0.002):
    """저울 총무게를 선형 측정 하나로 넣는다."""
    row = np.zeros((1, est.n_phi))
    row[0, ::N_PARAM] = 1.0
    return update(est, row, np.array([total_mass_kg]),
                  np.array([(rel_error * total_mass_kg) ** 2]), with_bias=False)


# ---------------------------------------------------------------------------
def _augment(est, Y, with_bias):
    """[Y | B]: B 는 행 i 에 영점 i%6 의 열. with_bias=False 면 영점 열을 0 으로."""
    Y = np.asarray(Y, float)
    if est.n_bias == 0:
        return Y
    rows = Y.shape[0]
    B = np.zeros((rows, est.n_bias))
    if with_bias:
        B[np.arange(rows), np.arange(rows) % 6] = 1.0
    return np.hstack([Y, B])


def update(est, Y, y, r_diag, with_bias=True):
    """y = [Y | B] [Φ; b] + ε  (정보 형태로 닫힌 갱신)."""
    A = _augment(est, Y, with_bias)
    y = np.asarray(y, float)
    w = 1.0 / np.asarray(r_diag, float)
    prior_info = np.linalg.inv(est.cov)
    info = prior_info + (A.T * w) @ A
    rhs = prior_info @ est.mean + (A.T * w) @ y
    cov = np.linalg.inv(info)
    cov = 0.5 * (cov + cov.T)
    return GaussianEstimate(cov @ rhs, cov, est.n_bias)


def information_gain(est, Y, r_diag, with_bias=True):
    """½ log det(I + Σ^{1/2} Aᵀ R⁻¹ A Σ^{1/2}),  A = [Y | B]."""
    A = _augment(est, Y, with_bias)
    w = 1.0 / np.asarray(r_diag, float)
    F = (A.T * w) @ A
    L = np.linalg.cholesky(est.cov + 1e-18 * np.eye(est.cov.shape[0]))
    M = np.eye(est.cov.shape[0]) + L.T @ F @ L
    return 0.5 * np.linalg.slogdet(M)[1]


# ---------------------------------------------------------------------------
def project_physical(est, eps_rel=1e-6):
    """사후 평균을 물리적으로 가능한 가장 가까운 점(마할라노비스 거리)으로.

    min (x−μ)ᵀ Σ⁻¹ (x−μ)  s.t.  J(x_k) ⪰ ε·I  (부위마다).
    Drake MathematicalProgram + Clarabel (없으면 SCS).
    """
    from pydrake.solvers import (ClarabelSolver, MathematicalProgram, ScsSolver)
    n = est.mean.size
    P = est.n_parts
    if all(is_physical(est.mean[N_PARAM * k:N_PARAM * (k + 1)]) for k in range(P)):
        return est, False
    prog = MathematicalProgram()
    x = prog.NewContinuousVariables(n, "x")
    L = np.linalg.cholesky(np.linalg.inv(est.cov))
    r = L.T @ (x - est.mean)
    prog.AddQuadraticCost(r.dot(r))
    basis = pseudo_basis()
    for k in range(P):
        xk = x[N_PARAM * k:N_PARAM * (k + 1)]
        m0 = max(est.mean[N_PARAM * k], 1e-3)
        eps = eps_rel * m0
        F0 = -eps * np.eye(4)
        prog.AddLinearMatrixInequalityConstraint([F0] + basis, xk)
    prog.SetInitialGuess(x, est.mean)
    result = ClarabelSolver().Solve(prog)
    if not result.is_success():
        result = ScsSolver().Solve(prog)
    if not result.is_success():
        return est, False
    out = est.copy()
    out.mean = result.GetSolution(x)
    return out, True


# ---------------------------------------------------------------------------
def part_errors(phi_est, phi_true):
    m_e, h_e, _ = split(phi_est)
    m_t, h_t, _ = split(phi_true)
    c_e, c_t = h_e / m_e, h_t / m_t
    Ie, It = central_inertia(phi_est), central_inertia(phi_true)
    return dict(
        mass_pct=100.0 * abs(m_e - m_t) / m_t,
        com_mm=1000.0 * np.linalg.norm(c_e - c_t),
        inertia_pct=100.0 * np.linalg.norm(Ie - It) / np.linalg.norm(It),
        principal_pct=100.0 * np.abs(principal_moments(phi_est) - principal_moments(phi_true))
        / np.abs(principal_moments(phi_true)),
    )


def part_uncertainty(est, k):
    """부위 k 의 질량 상대 표준편차, 무게중심 표준편차 [mm], 관성 상대 표준편차."""
    mu, cov = est.part(k)
    m = mu[0]
    sd = np.sqrt(np.clip(np.diag(cov), 0, None))
    com_sd_mm = 1000.0 * np.linalg.norm(sd[1:4]) / m
    I = central_inertia(mu)
    inertia_sd = np.linalg.norm(sd[4:10]) / max(np.linalg.norm(I), 1e-12)
    return dict(mass_rel=sd[0] / m, com_mm=com_sd_mm, inertia_rel=inertia_sd)
