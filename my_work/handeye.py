"""손-눈 캘리브레이션 솔버 (AX = XB). numpy 만 쓴다.

왜 직접 푸나
------------
opencv 의 calibrateHandEye 를 쓰면 되지만, 그러면 **고정된 Drake 환경에
opencv 가 들어와야 한다.** 이 파이프라인에서 opencv 가 필요한 곳은 태그를
'보는' 일 하나뿐이고, 그건 어차피 카메라에 붙은 별도 프로세스에서 한다
(calib_detect.py, FoundationPose 와 같은 구조). 그러니 푸는 쪽은 numpy 로
닫아 두는 편이 경계가 깨끗하다.

푸는 문제
---------
카메라가 실험실에 고정돼 로봇을 바라본다 (eye-to-hand). 보정판은 그리퍼에
볼트로 붙어 로봇과 함께 움직인다.

    W  월드 = 로봇 베이스
    E  팔의 tcp 프레임      X_WE 는 관절각 + FK 로 **안다**
    T  보정판               X_ET 는 모르는 상수 (판을 어디에 붙였나)
    C  카메라               X_WC 는 모르는 상수 (**우리가 구하는 것**)

매 자세에서 성립하는 관계는 하나다.

    X_WC · X_CT = X_WE · X_ET

자세 두 개(i, j)를 빼면 미지수 하나가 사라진다.

    A X = X B      A = X_WE_j · X_WE_i^-1     (팔이 움직인 양)
                   B = X_CT_j · X_CT_i^-1     (카메라가 본 움직인 양)
                   X = X_WC                   (구하는 것)

즉 **같은 운동을 두 좌표계에서 본 것**이고, 그 둘을 잇는 변환이 답이다.

Park & Martin (1994) 의 닫힌 해를 쓴다.
  회전    log(R_A) = a, log(R_B) = b 로 두면 R_X b = a 이므로
          M = sum b a^T, R_X = (M^T M)^(-1/2) M^T
  평행이동 (R_A - I) t_X = R_X t_B - t_A 를 쌓아 최소제곱

왜 자세를 많이, 그리고 **다양하게** 찍어야 하나
-----------------------------------------------
회전축이 전부 나란하면 (M^T M) 가 특이해져 답이 안 나온다. 축이 서로
크게 벌어진 자세쌍이 있어야 한다. generate_poses 가 그걸 노리고 만든다.

자기 검사:
  ../robot_learning/scripts/run_drake_env.sh python handeye.py
"""

import numpy as np


# ---------------------------------------------------------------------------
# SO(3) 로그/지수 — 회전을 '축 x 각도' 벡터로 오간다
# ---------------------------------------------------------------------------
def log_so3(R):
    """회전행렬 -> 회전벡터 (축 * 각도). 각도가 0 이나 pi 근처를 견딘다."""
    R = np.asarray(R, dtype=float)
    cos = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cos)
    if angle < 1e-9:
        return np.zeros(3)
    if np.pi - angle < 1e-6:
        # 180도 근처에서는 (R - R^T) 가 0 이 되어 축을 못 뽑는다.
        # 대신 (R + I) 의 열 중 가장 긴 것이 축 방향이다.
        A = R + np.eye(3)
        axis = A[:, int(np.argmax(np.linalg.norm(A, axis=0)))]
        axis = axis / np.linalg.norm(axis)
        return axis * angle
    vee = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return vee * (angle / (2.0 * np.sin(angle)))


def exp_so3(w):
    """회전벡터 -> 회전행렬 (로드리게스)."""
    w = np.asarray(w, dtype=float)
    angle = float(np.linalg.norm(w))
    if angle < 1e-12:
        return np.eye(3)
    k = w / angle
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return (np.eye(3) + np.sin(angle) * K
            + (1.0 - np.cos(angle)) * (K @ K))


def nearest_rotation(M):
    """가장 가까운 회전행렬 (SVD 사영). 잡음 때문에 직교성이 깨진 것을 고친다."""
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0.0:          # 반사가 섞이면 뒤집는다
        U[:, -1] *= -1.0
        R = U @ Vt
    return R


def transform(R, t):
    X = np.eye(4)
    X[:3, :3] = R
    X[:3, 3] = np.asarray(t, dtype=float)
    return X


def invert(X):
    R, t = X[:3, :3], X[:3, 3]
    return transform(R.T, -R.T @ t)


# ---------------------------------------------------------------------------
# 솔버
# ---------------------------------------------------------------------------
def solve_ax_xb(pairs):
    """AX = XB 를 푼다. pairs = [(A, B), ...] 4x4 동차변환들."""
    if len(pairs) < 2:
        raise ValueError("자세쌍이 최소 2개 필요하다 (실제로는 훨씬 많아야 한다)")

    M = np.zeros((3, 3))
    for A, B in pairs:
        a = log_so3(A[:3, :3])
        b = log_so3(B[:3, :3])
        M += np.outer(b, a)

    # R_X = (M^T M)^(-1/2) M^T
    eigval, eigvec = np.linalg.eigh(M.T @ M)
    if eigval.min() < 1e-12:
        raise ValueError(
            "회전축이 충분히 다양하지 않다 — 자세들이 거의 같은 축으로만 "
            "돌았다. 손목을 여러 방향으로 크게 꺾은 자세를 섞어야 한다.")
    inv_sqrt = eigvec @ np.diag(1.0 / np.sqrt(eigval)) @ eigvec.T
    R_X = nearest_rotation(inv_sqrt @ M.T)

    rows, rhs = [], []
    for A, B in pairs:
        rows.append(A[:3, :3] - np.eye(3))
        rhs.append(R_X @ B[:3, 3] - A[:3, 3])
    t_X, *_ = np.linalg.lstsq(np.vstack(rows), np.concatenate(rhs), rcond=None)
    return transform(R_X, t_X)


def pose_pairs(X_WE_list, X_CT_list, min_angle_deg=10.0):
    """자세 목록에서 (A, B) 쌍을 만든다.

    모든 조합을 쓰되 **회전이 거의 없는 쌍은 버린다.** 그런 쌍은 A X = X B
    에서 양변이 거의 항등식이라 정보가 없고, 잡음만 보태 답을 흐린다.
    """
    pairs = []
    n = len(X_WE_list)
    for i in range(n):
        for j in range(i + 1, n):
            A = X_WE_list[j] @ invert(X_WE_list[i])
            B = X_CT_list[j] @ invert(X_CT_list[i])
            if np.degrees(np.linalg.norm(log_so3(A[:3, :3]))) < min_angle_deg:
                continue
            pairs.append((A, B))
    return pairs


def solve_eye_to_hand(X_WE_list, X_CT_list, min_angle_deg=10.0):
    """고정 카메라 + 로봇에 붙은 보정판.

    X_WE_list : 자세마다 팔 tcp 의 월드 기준 자세 (FK)
    X_CT_list : 자세마다 카메라가 본 보정판 자세 (검출기)

    돌려주는 것: dict(X_WC, X_ET, n_pairs, residual_mm, residual_deg)

    residual 은 **자기 일관성**이다. 구한 X_WC, X_ET 로 X_WC·X_CT 와
    X_WE·X_ET 를 각각 만들어 얼마나 어긋나는지 잰다. 정답을 모르는 실물에서
    유일하게 볼 수 있는 품질 지표다.
    """
    X_WE_list = [np.asarray(X, dtype=float) for X in X_WE_list]
    X_CT_list = [np.asarray(X, dtype=float) for X in X_CT_list]
    if len(X_WE_list) != len(X_CT_list):
        raise ValueError("자세 개수와 검출 개수가 다르다")

    pairs = pose_pairs(X_WE_list, X_CT_list, min_angle_deg)
    if len(pairs) < 2:
        raise ValueError(
            f"쓸 만한 자세쌍이 {len(pairs)}개뿐이다 (회전 {min_angle_deg} deg 이상). "
            "자세를 더 찍거나 서로 더 다르게 잡아야 한다.")
    X_WC = solve_ax_xb(pairs)

    # X_ET = X_WE^-1 · X_WC · X_CT — 자세마다 나오는 값을 평균낸다.
    estimates = [invert(E) @ X_WC @ T for E, T in zip(X_WE_list, X_CT_list)]
    t_ET = np.mean([X[:3, 3] for X in estimates], axis=0)
    R_ET = nearest_rotation(np.mean([X[:3, :3] for X in estimates], axis=0))
    X_ET = transform(R_ET, t_ET)

    errors_mm, errors_deg = [], []
    for E, T in zip(X_WE_list, X_CT_list):
        left, right = X_WC @ T, E @ X_ET
        errors_mm.append(1000.0 * np.linalg.norm(left[:3, 3] - right[:3, 3]))
        errors_deg.append(np.degrees(np.linalg.norm(
            log_so3(left[:3, :3].T @ right[:3, :3]))))
    return dict(X_WC=X_WC, X_ET=X_ET, n_pairs=len(pairs),
                residual_mm=float(np.sqrt(np.mean(np.square(errors_mm)))),
                residual_deg=float(np.sqrt(np.mean(np.square(errors_deg)))),
                worst_mm=float(np.max(errors_mm)),
                worst_deg=float(np.max(errors_deg)))


# ---------------------------------------------------------------------------
# 자기 검사 — 하드웨어 없이 솔버가 맞는지 확인한다
#
# 이게 왜 중요한가: 손-눈 캘리브레이션은 **변환 방향을 뒤집어도 답이 그럴싸하게
# 나온다.** X_WC 대신 X_CW 를 구해 놓고 모르면, 실물에서 각도 측정 자세가
# 통째로 엉뚱한 곳으로 가는데 원인을 찾기 어렵다. 정답을 아는 합성 자료로
# 규약을 못 박아 둔다.
# ---------------------------------------------------------------------------
def _random_pose(rng, angle_deg=180.0, span_m=1.0):
    axis = rng.normal(size=3)
    axis = axis / np.linalg.norm(axis)
    angle = np.deg2rad(rng.uniform(-angle_deg, angle_deg))
    return transform(exp_so3(axis * angle), rng.uniform(-span_m, span_m, 3))


def self_test(n_pose=20, seed=0, noise_mm=0.0, noise_deg=0.0):
    """정답을 만들어 놓고 되찾아지는지 본다."""
    rng = np.random.default_rng(seed)
    X_WC_true = _random_pose(rng, 180.0, 1.5)
    X_ET_true = _random_pose(rng, 180.0, 0.1)

    X_WE_list, X_CT_list = [], []
    for _ in range(n_pose):
        X_WE = _random_pose(rng, 180.0, 0.6)
        X_CT = invert(X_WC_true) @ X_WE @ X_ET_true
        if noise_mm or noise_deg:
            # 검출기 잡음은 **카메라가 본 값** 에만 실린다. FK 는 정확하다고 본다.
            X_CT = X_CT @ transform(
                exp_so3(rng.normal(0.0, np.deg2rad(noise_deg), 3)),
                rng.normal(0.0, noise_mm * 1e-3, 3))
        X_WE_list.append(X_WE)
        X_CT_list.append(X_CT)

    out = solve_eye_to_hand(X_WE_list, X_CT_list)
    err_mm = 1000.0 * np.linalg.norm(out["X_WC"][:3, 3] - X_WC_true[:3, 3])
    err_deg = np.degrees(np.linalg.norm(
        log_so3(out["X_WC"][:3, :3].T @ X_WC_true[:3, :3])))
    return dict(err_mm=err_mm, err_deg=err_deg, **out)


def main():
    print("손-눈 솔버 자기 검사 — 정답을 만들어 놓고 되찾는다\n")
    print(f"  {'검출기 잡음':<22}{'자세수':>6}{'X_WC 위치오차':>15}"
          f"{'X_WC 각도오차':>15}{'자기일관성':>12}")
    for noise_mm, noise_deg, n_pose in ((0.0, 0.0, 20), (0.0, 0.0, 6),
                                        (0.5, 0.05, 20), (1.0, 0.1, 20),
                                        (2.0, 0.2, 20), (2.0, 0.2, 60)):
        errs = [self_test(n_pose=n_pose, seed=s, noise_mm=noise_mm,
                          noise_deg=noise_deg) for s in range(12)]
        label = ("없음" if not (noise_mm or noise_deg)
                 else f"{noise_mm} mm / {noise_deg} deg")
        print(f"  {label:<22}{n_pose:>6}"
              f"{np.mean([e['err_mm'] for e in errs]):>12.4f} mm"
              f"{np.mean([e['err_deg'] for e in errs]):>12.4f} deg"
              f"{np.mean([e['residual_mm'] for e in errs]):>9.3f} mm")

    print("\n  잡음이 0 이면 오차도 0 이어야 한다 (규약이 맞다는 뜻).")
    print("  자세를 늘리면 잡음이 있어도 좋아진다 — 이 오차는 치우침이 아니다.")

    print("\n회전축이 한 방향뿐이면 어떻게 되나 (일부러 실패시켜 본다)")
    rng = np.random.default_rng(0)
    X_WC_true, X_ET_true = _random_pose(rng, 180.0, 1.5), _random_pose(rng)
    axis = np.array([0.0, 0.0, 1.0])
    Es = [transform(exp_so3(axis * np.deg2rad(a)), rng.uniform(-0.5, 0.5, 3))
          for a in np.linspace(-90, 90, 12)]
    Ts = [invert(X_WC_true) @ E @ X_ET_true for E in Es]
    try:
        solve_eye_to_hand(Es, Ts)
        print("  [문제] 실패해야 하는데 답을 냈다")
    except ValueError as exc:
        print(f"  제대로 거부한다: {exc}")


if __name__ == "__main__":
    main()
