"""힌지 제약 아래 정보이득 최대 여기 궤적 설계.

결정 변수  x : 푸리에 자유 계수 (관절 6 × 2(K−1))
목적      IG(x) = ½ log det(I + Σ^{1/2} F(x) Σ^{1/2}),  F = Σ_t Y_tᵀ R⁻¹ Y_t
제약      관절 위치·속도·가속도, 작업공간 상자, 최소 간격(충돌),
          힌지 축 토크 |g_i(t)·Φ| ≤ 예산_i  (Φ 의 불확실성까지 k·σ 로 보수적으로),
          파지 마찰 용량.
풀이      벌칙 연속법 + L-BFGS-B (유한차분) → 조밀 격자 검증 → 진폭 축소.
          Scalable Real2Sim 이 조건수 + E-최적성을 쓴 자리에 논문 식 (6) 의
          정보이득을 그대로 쓴다 (사전분포와 잡음 모형이 있으니 D-최적).

"힌지에 평행하면 빨리, 수직이면 천천히" 는 여기서 규칙이 아니라 결과다:
힌지 축 토크 계수 g_i(t) 가 0 인 운동 방향은 제약에 걸리지 않으므로
최적화가 스스로 그쪽 진폭을 키운다.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from .identify import information_gain
from .regressor import hinge_axis_row, rigid_body_regressor, shift_wrench_rows
from .trajectory import FourierTrajectory


@dataclass
class Limits:
    qd_max: np.ndarray = field(default_factory=lambda: np.full(6, np.deg2rad(180.0)))
    qdd_max: np.ndarray = field(default_factory=lambda: np.full(6, np.deg2rad(300.0)))
    q_margin_rad: float = 0.05
    q_abs_max_rad: float = 2.0 * np.pi     # RB5-850E 관절 범위 ±360° (URDF 의 ±180° 는 IK 용)
    workspace_lower: np.ndarray = None
    workspace_upper: np.ndarray = None
    min_distance_m: float = 0.010
    hinge_budget_nm: np.ndarray = None        # 관절별 허용 힌지 축 토크
    robust_k: float = 2.0                     # 힌지 토크의 k·σ 보수성
    grasp_force_cap_n: float = 60.0           # 2F-85 마찰 파지 용량 (보수)
    grasp_torque_cap_nm: float = 0.6
    cable_wrist_range_rad: float = np.deg2rad(120.0)   # F/T 케이블 보호: 손목3 이 시작값에서 벗어날 수 있는 폭 (가정)


@dataclass
class Noise:
    sigma_f: float = 0.4          # [N]   샘플당 표준편차
    sigma_t: float = 0.025        # [N·m]
    rate_hz: float = 50.0         # 실제로 기록되는 샘플 속도

    @property
    def r_diag6(self):
        return np.array([self.sigma_f ** 2] * 3 + [self.sigma_t ** 2] * 3)


class ExcitationDesigner:
    def __init__(self, model, theta, q0, est, noise, limits, period_s=8.0,
                 n_harmonics=4, n_samples=40, seed=0):
        self.model = model
        self.theta = np.atleast_1d(np.asarray(theta, float))
        self.traj = FourierTrajectory(q0, period_s, n_harmonics)
        self.est = est
        self.noise = noise
        self.limits = limits
        self.n_samples = int(n_samples)
        self.rng = np.random.default_rng(seed)
        self.M = model.part_transports(self.theta)
        self.Mstack = np.hstack(self.M)                 # 10 x 10P
        self.hinges = model.hinge_geometry(self.theta)
        self.r_g = model.grasp_point_in_S(self.theta)
        self.down = model.downstream
        self.n_parts = model.n_parts
        self.P = 10 * self.n_parts
        # 힌지 i 의 부위 선택 행렬 (10P x 10P): 하류 부위만 남긴다
        self.down_mask = []
        for parts in self.down:
            mask = np.zeros(self.P)
            for j in parts:
                mask[10 * j:10 * (j + 1)] = 1.0
            self.down_mask.append(mask)
        self.times = self.traj.times(self.n_samples)
        self.rate_factor = noise.rate_hz * period_s / self.n_samples
        self.r_diag = np.tile(noise.r_diag6, self.n_samples) / self.rate_factor
        if limits.hinge_budget_nm is None:
            limits.hinge_budget_nm = np.full(len(self.hinges), 0.1)
        self.n_free = self.traj.n_free
        self.last = None

    # ------------------------------------------------------------------
    def rows_at(self, q, qd, qdd):
        kin = self.model.evaluate(q, qd, qdd, self.theta)
        Y = rigid_body_regressor(kin["a_o"], kin["omega"], kin["alpha"], kin["g"])
        Yp = Y @ self.Mstack                                # 6 x 10P
        hinge_rows = []
        for (r_h, a_hat), mask in zip(self.hinges, self.down_mask):
            hinge_rows.append((hinge_axis_row(Y, r_h, a_hat) @ self.Mstack) * mask)
        grasp_rows = shift_wrench_rows(Y, self.r_g) @ self.Mstack
        p_WG = kin["p_WS"] + kin["X_WS"].rotation().matrix() @ self.r_g
        return Yp, np.array(hinge_rows), grasp_rows, p_WG

    def evaluate(self, x, times=None):
        times = self.times if times is None else np.asarray(times, float)
        q, qd, qdd = self.traj.evaluate(x, times)
        n = times.size
        Yp = np.zeros((6 * n, self.P))
        hinge_mean = np.zeros((n, len(self.hinges)))
        hinge_bound = np.zeros((n, len(self.hinges)))
        grasp = np.zeros((n, 6))
        dist = np.zeros(n)
        pos = np.zeros((n, 3))
        mu, cov = self.est.mean[:self.P], self.est.cov[:self.P, :self.P]
        for k in range(n):
            Yk, hk, gk, p = self.rows_at(q[k], qd[k], qdd[k])
            Yp[6 * k:6 * k + 6] = Yk
            hinge_mean[k] = hk @ mu
            hinge_bound[k] = np.abs(hinge_mean[k]) + self.limits.robust_k * np.sqrt(
                np.einsum("ij,jk,ik->i", hk, cov, hk))
            grasp[k] = gk @ mu
            dist[k] = self.model.min_distance(q[k], self.theta)
            pos[k] = p
        return dict(times=times, q=q, qd=qd, qdd=qdd, Yp=Yp, hinge_mean=hinge_mean,
                    hinge_bound=hinge_bound, grasp=grasp, dist=dist, pos=pos)

    def info_gain(self, Yp, n_times=None):
        n_times = Yp.shape[0] // 6 if n_times is None else n_times
        factor = self.noise.rate_hz * self.traj.T / n_times
        r_diag = np.tile(self.noise.r_diag6, n_times) / factor
        return information_gain(self.est, Yp, r_diag)

    # ------------------------------------------------------------------
    def violations(self, ev):
        lim, m = self.limits, self.model
        q, qd, qdd = ev["q"], ev["qd"], ev["qdd"]
        v = []
        q_lo = np.maximum(m.q_lower, -lim.q_abs_max_rad) if lim.q_abs_max_rad < np.pi else -lim.q_abs_max_rad
        q_hi = np.minimum(m.q_upper, lim.q_abs_max_rad) if lim.q_abs_max_rad < np.pi else lim.q_abs_max_rad
        v.append(np.maximum(0.0, (q_lo + lim.q_margin_rad) - q) / np.pi)
        v.append(np.maximum(0.0, q - (q_hi - lim.q_margin_rad)) / np.pi)
        v.append(np.maximum(0.0, np.abs(qd) - lim.qd_max) / lim.qd_max)
        v.append(np.maximum(0.0, np.abs(qdd) - lim.qdd_max) / lim.qdd_max)
        if lim.workspace_lower is not None:
            v.append(np.maximum(0.0, lim.workspace_lower - ev["pos"]) / 0.1)
            v.append(np.maximum(0.0, ev["pos"] - lim.workspace_upper) / 0.1)
        v.append(np.maximum(0.0, lim.min_distance_m - ev["dist"]) / lim.min_distance_m)
        v.append(np.maximum(0.0, np.abs(q[:, 5] - self.traj.q0[5]) - lim.cable_wrist_range_rad) / np.pi)
        v.append(np.maximum(0.0, ev["hinge_bound"] - lim.hinge_budget_nm) / lim.hinge_budget_nm)
        f = np.linalg.norm(ev["grasp"][:, :3], axis=1)
        t = np.linalg.norm(ev["grasp"][:, 3:], axis=1)
        v.append(np.maximum(0.0, f - lim.grasp_force_cap_n) / lim.grasp_force_cap_n)
        v.append(np.maximum(0.0, t - lim.grasp_torque_cap_nm) / lim.grasp_torque_cap_nm)
        return np.concatenate([a.ravel() for a in v])

    def objective(self, x, mu_pen):
        ev = self.evaluate(x)
        ig = self.info_gain(ev["Yp"])
        viol = self.violations(ev)
        return -ig + mu_pen * float(np.sum(viol ** 2))

    # ------------------------------------------------------------------
    def static_feasible(self):
        ev = self.evaluate(np.zeros(self.n_free), times=np.array([0.0]))
        return (self.violations(ev).max() <= 0.0), ev

    def optimize(self, n_starts=3, amp=0.10, mu_schedule=(1e2, 1e3, 1e4),
                 maxiter=(30, 30, 45), bound=0.6, verbose=False):
        ok, ev0 = self.static_feasible()
        if not ok:
            raise RuntimeError("시작 자세가 이미 제약을 어긴다 (정지 힌지 토크·충돌·작업공간)")
        best = None
        bounds = [(-bound, bound)] * self.n_free
        for s in range(n_starts):
            # 첫 시작점은 작게(안전 쪽), 나머지는 amp 로 넓게 뿌린다
            x = self.rng.normal(0.0, amp * (0.5 if s == 0 else 1.0), self.n_free)
            for mu_pen, iters in zip(mu_schedule, maxiter):
                res = minimize(self.objective, x, args=(mu_pen,), method="L-BFGS-B",
                               jac="2-point", bounds=bounds,
                               options=dict(maxiter=iters, eps=1e-4))
                x = res.x
            x = self.backoff(x)
            ver = self.verify(x)
            if verbose:
                print(f"    start {s}: IG {ver['ig']:.2f} nat, feasible {ver['feasible']},"
                      f" hinge peak {ver['hinge_peak_ratio']:.2f}")
            if best is None or (ver["feasible"] and ver["ig"] > best[1]["ig"]):
                best = (x, ver)
        self.last = best
        return best

    def backoff(self, x, n_dense=200):
        """조밀 격자에서 제약을 어기면 진폭을 이분법으로 줄인다."""
        def feasible(scale):
            ev = self.evaluate(scale * x, times=self.traj.times(n_dense))
            return self.violations(ev).max() <= 0.0
        if feasible(1.0):
            return x
        lo, hi = 0.0, 1.0
        for _ in range(12):
            mid = 0.5 * (lo + hi)
            if feasible(mid):
                lo = mid
            else:
                hi = mid
        return lo * x

    def verify(self, x, n_dense=200):
        ev = self.evaluate(x, times=self.traj.times(n_dense))
        viol = self.violations(ev)
        ratio = (ev["hinge_bound"] / self.limits.hinge_budget_nm).max() if len(self.hinges) else 0.0
        return dict(
            ig=self.info_gain(ev["Yp"]), feasible=bool(viol.max() <= 1e-9),
            max_violation=float(viol.max()),
            hinge_peak_ratio=float(ratio),
            hinge_peak_nm=float(np.abs(ev["hinge_mean"]).max()) if len(self.hinges) else 0.0,
            qd_peak_deg=float(np.degrees(np.abs(ev["qd"]).max())),
            qdd_peak_deg=float(np.degrees(np.abs(ev["qdd"]).max())),
            min_distance_m=float(ev["dist"].min()),
            grasp_force_peak_n=float(np.linalg.norm(ev["grasp"][:, :3], axis=1).max()),
            grasp_torque_peak_nm=float(np.linalg.norm(ev["grasp"][:, 3:], axis=1).max()),
            ev=ev,
        )
