"""강체 뉴턴-오일러 회귀행렬 (센서 프레임 S, 물체가 S 에 고정).

센서가 부하에 가하는 렌치 (힘 f, 토크 τ, S 원점 기준, S 축 표현):
    f = m (a_o − g) + α × h + ω × (ω × h)
    τ = h × (a_o − g) + I α + ω × (I ω)
  a_o : S 원점의 선가속도, ω/α : S 의 각속도/각가속도, g : S 에서 본 중력.
phi = [m, h, I] 에 선형이므로 [f; τ] = Y(a_o, ω, α, g) phi, Y ∈ R^{6x10}.

정지 상태에서는 f = −m g, τ = −h × g 로, 논문의 준정적 모형과 같다.
병진 가속 a_o 는 언제나 (a_o − g) 로만 들어간다. 즉 병진 가속은 "중력을
바꾸는 것" 이상의 정보를 주지 않고, 새 정보(관성 텐서)는 ω, α 에서만 온다.
"""
import numpy as np

from .inertial import skew


def L_matrix(v):
    """L(v) ivec = I v  (ivec = [Ixx, Ixy, Ixz, Iyy, Iyz, Izz])."""
    x, y, z = v
    return np.array([[x, y, z, 0, 0, 0],
                     [0, x, 0, y, z, 0],
                     [0, 0, x, 0, y, z]], dtype=float)


def rigid_body_regressor(a_o, omega, alpha, g):
    lin = np.asarray(a_o, float) - np.asarray(g, float)
    W = skew(omega)
    Y = np.zeros((6, 10))
    Y[:3, 0] = lin
    Y[:3, 1:4] = skew(alpha) + W @ W
    Y[3:, 1:4] = -skew(lin)
    Y[3:, 4:] = L_matrix(alpha) + W @ L_matrix(omega)
    return Y


def hinge_axis_row(Y, r_h, a_hat):
    """힌지 축 토크의 계수 행 (1x10).

    S 원점 렌치 [f; τ_o] = Y phi 를 힌지 점 h 로 옮기면 τ_h = τ_o − r_h × f.
    마찰이 버텨야 하는 것은 축 성분 â·τ_h 뿐이다 (나머지는 베어링이 전달).
    """
    a_hat = np.asarray(a_hat, float)
    return a_hat @ (Y[3:] - skew(r_h) @ Y[:3])


def shift_wrench_rows(Y, r):
    """S 원점 기준 Y 를 점 r (S 좌표) 기준 토크로 바꾼 6x10."""
    out = Y.copy()
    out[3:] = Y[3:] - skew(r) @ Y[:3]
    return out


def self_test():
    """회귀행렬 = Drake 가 계산한 강체 운동량 변화율인지 확인.

    강체 하나를 6-DOF 자유 관절로 두고, 임의의 (V, A) 에서
    Drake 의 CalcInverseDynamics 로 필요한 일반화 힘을 구해 비교한다.
    """
    from pydrake.math import RigidTransform, RollPitchYaw, RotationMatrix
    from pydrake.multibody.plant import MultibodyPlant
    from pydrake.multibody.tree import SpatialInertia, UnitInertia
    from pydrake.multibody.math import SpatialVelocity, SpatialAcceleration
    from .inertial import from_spatial_inertia
    rng = np.random.default_rng(1)
    plant = MultibodyPlant(0.0)
    c = np.array([0.03, -0.02, 0.05])
    M = SpatialInertia(0.8, c, UnitInertia.SolidBox(0.12, 0.05, 0.2)
                       .ShiftFromCenterOfMass(-c))
    body = plant.AddRigidBody("b", M)      # 기본: 자유 부동체 (quaternion)
    plant.Finalize()
    ctx = plant.CreateDefaultContext()
    R_WB = RotationMatrix(RollPitchYaw(*rng.uniform(-1, 1, 3)))
    p_WB = rng.uniform(-0.5, 0.5, 3)
    plant.SetFreeBodyPose(ctx, body, RigidTransform(R_WB, p_WB))
    w_W = rng.uniform(-2, 2, 3)
    v_W = rng.uniform(-1, 1, 3)
    plant.SetFreeBodySpatialVelocity(ctx, body, SpatialVelocity(w_W, v_W))
    aw_W = rng.uniform(-5, 5, 3)
    av_W = rng.uniform(-5, 5, 3)
    # vdot: 자유체의 일반화 속도는 [ω_W; v_W] (world 표현), vdot 도 같은 순서
    vdot = np.concatenate([aw_W, av_W])
    from pydrake.multibody.tree import MultibodyForces
    forces = MultibodyForces(plant)
    plant.CalcForceElementsContribution(ctx, forces)   # 중력 포함
    tau_gen = plant.CalcInverseDynamics(ctx, vdot, forces)
    # tau_gen = 원점 Bo 기준, world 표현 스패셜 힘 [τ; f] 이 필요한 외력
    tau_W, f_W = tau_gen[:3], tau_gen[3:]
    R = R_WB.matrix()
    a_o_S = R.T @ av_W
    w_S = R.T @ w_W
    al_S = R.T @ aw_W
    g_S = R.T @ np.array([0, 0, -9.81])
    Y = rigid_body_regressor(a_o_S, w_S, al_S, g_S)
    phi = from_spatial_inertia(M)
    wrench_S = Y @ phi
    ours = np.concatenate([R @ wrench_S[:3], R @ wrench_S[3:]])
    ref = np.concatenate([f_W, tau_W])
    err = np.abs(ours - ref).max() / max(1.0, np.abs(ref).max())
    assert err < 1e-9, (ours, ref)
    return err


if __name__ == "__main__":
    print("regressor self-test rel err", self_test())
