"""10-파라미터 관성 벡터와 프레임 변환.

phi = [m, hx, hy, hz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz]
  m  : 질량
  h  : 1차 모멘트 m·c (c = 프레임 원점 기준 무게중심)
  I  : 프레임 원점 기준, 프레임 축으로 표현한 관성 텐서 (6개 고유 성분)

유사관성 J(phi) = [[½tr(I)·1 − I, h], [hᵀ, m]] 은 phi 에 선형이고,
프레임 변환 T = [[R, p],[0, 1]] 아래에서 J' = T J Tᵀ 로 옮겨진다
(Wensing, Kim, Slotine 2017). 그래서 프레임 변환도 phi 에 대한 10x10 선형
사상이며, 물리적으로 가능한 phi 는 J ≻ 0 과 동치다.
"""
import numpy as np

N_PARAM = 10


def skew(v):
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def inertia_matrix(ivec):
    xx, xy, xz, yy, yz, zz = ivec
    return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])


def inertia_vector(I):
    return np.array([I[0, 0], I[0, 1], I[0, 2], I[1, 1], I[1, 2], I[2, 2]])


def split(phi):
    phi = np.asarray(phi, dtype=float)
    return float(phi[0]), phi[1:4].copy(), inertia_matrix(phi[4:10])


def join(m, h, I):
    return np.concatenate([[m], np.asarray(h, float), inertia_vector(I)])


def pseudo_inertia(phi):
    m, h, I = split(phi)
    sigma = 0.5 * np.trace(I) * np.eye(3) - I
    J = np.zeros((4, 4))
    J[:3, :3] = sigma
    J[:3, 3] = h
    J[3, :3] = h
    J[3, 3] = m
    return J


def from_pseudo_inertia(J):
    sigma = J[:3, :3]
    I = np.trace(sigma) * np.eye(3) - sigma
    return join(J[3, 3], J[:3, 3], I)


def pseudo_basis():
    """J(phi) = Σ_k phi_k J_k 의 상수 행렬 J_k 열 개. LMI 제약에 쓴다."""
    basis = []
    for k in range(N_PARAM):
        e = np.zeros(N_PARAM)
        e[k] = 1.0
        basis.append(pseudo_inertia(e))
    return basis


def transport_matrix(R, p):
    """프레임 A 의 phi 를 프레임 B 의 phi 로 옮기는 10x10 행렬.

    x_B = R x_A + p  (R = R_BA, p = A 원점의 B 좌표).
    """
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, float)
    T[:3, 3] = np.asarray(p, float)
    M = np.zeros((N_PARAM, N_PARAM))
    for k in range(N_PARAM):
        e = np.zeros(N_PARAM)
        e[k] = 1.0
        M[:, k] = from_pseudo_inertia(T @ pseudo_inertia(e) @ T.T)
    return M


def transport_from_rigid_transform(X_BA):
    """pydrake RigidTransform X_BA 로 transport_matrix 를 만든다."""
    return transport_matrix(X_BA.rotation().matrix(), X_BA.translation())


def from_spatial_inertia(M):
    """pydrake SpatialInertia (점 P 기준, 프레임 E 표현) -> phi."""
    m = float(M.get_mass())
    c = np.asarray(M.get_com(), float)
    I = M.CalcRotationalInertia().CopyToFullMatrix3()
    return join(m, m * c, I)


def to_spatial_inertia(phi):
    from pydrake.multibody.tree import SpatialInertia, UnitInertia
    m, h, I = split(phi)
    G = I / m
    unit = UnitInertia(G[0, 0], G[1, 1], G[2, 2], G[0, 1], G[0, 2], G[1, 2])
    return SpatialInertia(m, h / m, unit)


def com(phi):
    m, h, _ = split(phi)
    return h / m


def central_inertia(phi):
    """무게중심 기준 관성 텐서 (평행축 정리)."""
    m, h, I = split(phi)
    c = h / m
    return I - m * (np.dot(c, c) * np.eye(3) - np.outer(c, c))


def principal_moments(phi):
    return np.sort(np.linalg.eigvalsh(central_inertia(phi)))


def is_physical(phi, eps=0.0):
    return bool(np.linalg.eigvalsh(pseudo_inertia(phi)).min() > eps)


def scale_vector(mass_scale, length_scale):
    """사전분포와 정보량 계산에 쓰는 단위 스케일 [kg, kg·m×3, kg·m²×6]."""
    s = np.empty(N_PARAM)
    s[0] = mass_scale
    s[1:4] = mass_scale * length_scale
    s[4:10] = mass_scale * length_scale ** 2
    return s


def solid_box_phi(mass, dims, com_offset=None):
    """균일 밀도 직육면체. 원점은 상자 중심에서 com_offset 만큼 떨어진 곳."""
    lx, ly, lz = dims
    Ic = mass / 12.0 * np.diag([ly**2 + lz**2, lx**2 + lz**2, lx**2 + ly**2])
    c = np.zeros(3) if com_offset is None else np.asarray(com_offset, float)
    I = Ic + mass * (np.dot(c, c) * np.eye(3) - np.outer(c, c))
    return join(mass, mass * c, I)


def point_mass_phi(mass, position):
    p = np.asarray(position, float)
    return join(mass, mass * p, mass * (np.dot(p, p) * np.eye(3) - np.outer(p, p)))


def self_test():
    """Drake SpatialInertia 의 Shift/ReExpress 와 transport_matrix 가 같은지."""
    from pydrake.math import RigidTransform, RollPitchYaw, RotationMatrix
    from pydrake.multibody.tree import SpatialInertia, UnitInertia
    rng = np.random.default_rng(0)
    M = SpatialInertia(0.7, np.array([0.02, -0.01, 0.05]),
                       UnitInertia.SolidBox(0.1, 0.05, 0.2).ShiftFromCenterOfMass(
                           -np.array([0.02, -0.01, 0.05])))
    phi_A = from_spatial_inertia(M)
    R = RotationMatrix(RollPitchYaw(*rng.uniform(-1, 1, 3)))
    p = rng.uniform(-0.3, 0.3, 3)
    # Drake: 먼저 원점을 옮기고(Shift 는 같은 프레임에서 새 점까지의 벡터),
    # 그다음 축을 바꾼다. x_B = R x_A + p 이므로 B 원점은 A 좌표로 -Rᵀp.
    p_AB_A = -R.matrix().T @ p
    M_B = M.Shift(p_AB_A).ReExpress(R)
    phi_B_drake = from_spatial_inertia(M_B)
    phi_B_ours = transport_matrix(R.matrix(), p) @ phi_A
    err = np.abs(phi_B_drake - phi_B_ours).max()
    assert err < 1e-12, err
    assert is_physical(phi_A) and is_physical(phi_B_ours)
    # 왕복
    back = transport_from_rigid_transform(RigidTransform(R, p).inverse()) @ phi_B_ours
    assert np.abs(back - phi_A).max() < 1e-12
    return err


if __name__ == "__main__":
    print("transport self-test max err", self_test())
