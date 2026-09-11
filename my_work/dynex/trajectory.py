"""경계 조건이 자동으로 맞는 주기 푸리에 궤적.

관절 j:  q_j(t) = q0_j + Σ_{k=1..K} [ a_jk sin(kω₀t) + b_jk (cos(kω₀t) − 1) ],
ω₀ = 2π/T.  q(0)=q0 는 자동. 시작·끝 속도와 가속도를 0 으로 만들기 위해
    Σ_k k a_jk = 0,   Σ_k k² b_jk = 0
을 첫 번째 계수로 흡수한다 (a_j1, b_j1 이 나머지에서 정해진다). 주기
함수라 t=T 에서도 같은 조건이 성립한다. 자유 계수는 관절당 2(K−1) 개.
"""
import numpy as np


class FourierTrajectory:
    def __init__(self, q0, period_s, n_harmonics=4):
        assert n_harmonics >= 2
        self.q0 = np.asarray(q0, float)
        self.n_joints = self.q0.size
        self.T = float(period_s)
        self.K = int(n_harmonics)
        self.w0 = 2.0 * np.pi / self.T
        self.n_free = self.n_joints * 2 * (self.K - 1)

    def coefficients(self, x):
        x = np.asarray(x, float).reshape(self.n_joints, 2, self.K - 1)
        a = np.zeros((self.n_joints, self.K))
        b = np.zeros((self.n_joints, self.K))
        ks = np.arange(2, self.K + 1)
        a[:, 1:] = x[:, 0, :]
        b[:, 1:] = x[:, 1, :]
        a[:, 0] = -(ks * a[:, 1:]).sum(axis=1)
        b[:, 0] = -(ks ** 2 * b[:, 1:]).sum(axis=1)
        return a, b

    def evaluate(self, x, t):
        """(q, qd, qdd) 각각 (len(t), n_joints)."""
        t = np.atleast_1d(np.asarray(t, float))
        a, b = self.coefficients(x)
        ks = np.arange(1, self.K + 1)
        wk = ks * self.w0                                   # (K,)
        arg = np.outer(t, wk)                               # (N, K)
        s, c = np.sin(arg), np.cos(arg)
        q = self.q0[None, :] + s @ a.T + (c - 1.0) @ b.T
        qd = (s * 0 + c * wk) @ a.T - (s * wk) @ b.T
        qdd = -(s * wk ** 2) @ a.T - (c * wk ** 2) @ b.T
        return q, qd, qdd

    def times(self, n):
        return np.linspace(0.0, self.T, n, endpoint=False) + 0.5 * self.T / n

    def self_test(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0, 0.1, self.n_free)
        for t in (0.0, self.T):
            q, qd, qdd = self.evaluate(x, [t])
            assert np.abs(q - self.q0).max() < 1e-12
            assert np.abs(qd).max() < 1e-12, qd
            assert np.abs(qdd).max() < 1e-12, qdd
        h = 1e-6
        t0 = 0.37 * self.T
        q0, qd0, qdd0 = self.evaluate(x, [t0])
        qp, _, _ = self.evaluate(x, [t0 + h])
        qm, _, _ = self.evaluate(x, [t0 - h])
        assert np.abs((qp - qm) / (2 * h) - qd0).max() < 1e-5
        assert np.abs((qp - 2 * q0 + qm) / h ** 2 - qdd0).max() < 1e-3
        return True


if __name__ == "__main__":
    print(FourierTrajectory(np.zeros(6), 8.0, 4).self_test())
