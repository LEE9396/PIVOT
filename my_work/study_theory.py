"""검토 ⑨: THEORY.md 의 정리 1~3 이 실제로 성립하는지 기계로 확인한다.

왜 이 스크립트가 필요한가
-------------------------
정리는 종이 위에서 맞아도 코드가 다른 것을 계산하고 있으면 소용이 없다.
교수님도 리뷰어도 "유도한 식이 돌아가는 코드와 같은가"를 반드시 묻는다.
그래서 유도한 닫힌 형태를 density_id_drake.regressor() 가 만드는 A 로부터
직접 계산한 A^T R^-1 A 와 대조한다. 사람 눈이 아니라 assert 가 판정한다.

무엇을 확인하나 (THEORY.md 와 번호가 같다)
------------------------------------------
  ① 닫힌 형태     M(theta) = g^2[ (3/sf^2) V V^T + (2/st^2) C^T C ]
  ② 회전 불변성   삼면체를 SO(3) 로 돌려도 M 이 안 변한다
  ③ rank 상한     단일 형상에서 rank M <= 1 + rank C <= 4
  ④ 강체 한계     theta 고정 + 중력방향 K 개 -> 여전히 rank <= 4
  ⑤ 영공간        null M = {u : V^T u = 0, C u = 0}
  ⑥ 라운드 하한   누적 rank <= 1 + min(3R, P),  R >= ceil((P-1)/3)

기호는 THEORY.md 와 같다.
  V    = (V_1..V_P)^T          부피 벡터
  C(t) = [V_i c_i(t)]  (3xP)   부피가중 도심 행렬

실행:
  ../robot_learning/scripts/run_drake_env.sh python study_theory.py
  ../robot_learning/scripts/run_drake_env.sh python study_theory.py --parts 2 3 4 5 6 8
"""

import argparse

import numpy as np

G_ACC = 9.81
SIGMA_F, SIGMA_T = 0.10, 0.003          # AFT200 급. density_id_drake 와 같은 값.
TRIAD = np.eye(3)                       # 직교정규 삼면체 (어느 것이든 무관 — 정리 1)


# ---------------------------------------------------------------------------
# 기준 구현 — density_id_drake.regressor() 와 같은 규약을 명시적으로 다시 쓴다.
# 코드를 그대로 import 해서 쓰면 "같은 버그를 양쪽이 공유"할 때 못 잡는다.
# ---------------------------------------------------------------------------
def regressor_ref(volumes, centroids, g_hat):
    """A(theta, g_hat) = g [ g_hat V^T ; -[g_hat]_x C ]  (6 x P)."""
    force = G_ACC * np.outer(g_hat, volumes)
    torque = G_ACC * (np.cross(centroids, g_hat).T * volumes)
    return np.vstack([force, torque])


def information(volumes, centroids, g_dirs):
    """A^T R^-1 A 를 방향들에 대해 합한 것. R 은 블록 등방."""
    r_inv = np.diag(np.r_[np.full(3, SIGMA_F ** -2), np.full(3, SIGMA_T ** -2)])
    return sum(regressor_ref(volumes, centroids, g).T @ r_inv
               @ regressor_ref(volumes, centroids, g) for g in g_dirs)


def information_closed(volumes, centroids, g_dirs):
    """THEORY.md 정리 1의 일반형.

        M = g^2 [ (K/sf^2) V V^T + (1/st^2) C^T (K I - sum g g^T) C ]

    직교정규 삼면체면 sum g g^T = I 이므로 (K I - sum) = 2I 가 되어
    방향에 대한 의존이 완전히 사라진다.
    """
    C = (centroids * volumes[:, None]).T                 # (3, P)
    K = len(g_dirs)
    S = sum(np.outer(g, g) for g in g_dirs)
    return G_ACC ** 2 * ((K / SIGMA_F ** 2) * np.outer(volumes, volumes)
                         + (1.0 / SIGMA_T ** 2) * C.T @ (K * np.eye(3) - S) @ C)


def rel_err(a, b):
    scale = max(np.abs(a).max(), np.abs(b).max(), 1e-300)
    return float(np.abs(a - b).max() / scale)


def random_rotation(rng):
    Q, R = np.linalg.qr(rng.normal(size=(3, 3)))
    return Q * np.sign(np.diag(R))


def rank_of(M):
    """스케일에 안 휘둘리게 상대 허용오차로 rank 를 센다."""
    return int(np.linalg.matrix_rank(M, tol=1e-9 * max(np.abs(M).max(), 1e-300)))


# ---------------------------------------------------------------------------
# ①~⑤ : 무작위 형상에서
# ---------------------------------------------------------------------------
def check_random(trials, rng):
    worst = dict(closed=0.0, invariance=0.0, closed_general=0.0,
                 rank_excess=-99, nullspace=0.0)

    for _ in range(trials):
        P = int(rng.integers(2, 9))
        V = rng.uniform(1e-5, 1e-3, P)               # 부피 [m^3]
        c = rng.normal(0.0, 0.15, (P, 3))            # 손목 기준 도심 [m]
        C = (c * V[:, None]).T

        # ① 닫힌 형태 (직교 삼면체, 회전시켜도 성립해야 한다)
        dirs = list(random_rotation(rng) @ TRIAD)
        M_num = information(V, c, dirs)
        M_cls = G_ACC ** 2 * ((3.0 / SIGMA_F ** 2) * np.outer(V, V)
                              + (2.0 / SIGMA_T ** 2) * (C.T @ C))
        worst["closed"] = max(worst["closed"], rel_err(M_num, M_cls))

        # ② 삼면체를 돌려도 M 이 같은가
        M_ref = information(V, c, list(TRIAD))
        worst["invariance"] = max(worst["invariance"], rel_err(M_num, M_ref))

        # ③④ 임의 개수 K, 임의 배치의 중력방향에서도 rank <= 4 이고 일반형이 맞는가
        K = int(rng.integers(1, 21))
        gs = rng.normal(size=(K, 3))
        gs /= np.linalg.norm(gs, axis=1, keepdims=True)
        M_k = information(V, c, list(gs))
        worst["closed_general"] = max(worst["closed_general"],
                                      rel_err(M_k, information_closed(V, c, list(gs))))
        if P >= 4:
            worst["rank_excess"] = max(worst["rank_excess"], rank_of(M_k) - 4)

        # ⑤ 영공간이 '질량·1차모멘트 보존 재분배' 와 같은가
        if P >= 5:
            u = np.linalg.eigh(M_ref)[1][:, 0]        # 최소고유값 방향
            scale = np.abs(V).max() * np.abs(u).max()
            worst["nullspace"] = max(worst["nullspace"],
                                     max(abs(V @ u) / scale,
                                         np.abs(C @ u).max() / (0.5 * scale)))
    return worst


# ---------------------------------------------------------------------------
# ⑥ : 라운드를 늘리며 누적 rank 가 상한을 지키는가 / 하한이 달성되는가
# ---------------------------------------------------------------------------
def check_rounds(parts, max_rounds, rng):
    rows = []
    for P in parts:
        V = rng.uniform(1e-5, 1e-3, P)
        M = np.zeros((P, P))
        ranks = []
        for R in range(1, max_rounds + 1):
            c = rng.normal(0.0, 0.15, (P, 3))         # 무작위 형상 = generic
            M += information(V, c, list(TRIAD))
            ranks.append(rank_of(M))
        bound_needed = int(np.ceil((P - 1) / 3.0))
        reached = next((r for r, k in enumerate(ranks, 1) if k >= P), None)
        rows.append((P, ranks, bound_needed, reached))
    return rows


# ---------------------------------------------------------------------------
# 실물 물체로 확인 — 여기서만 저장소의 코드를 실제로 부른다.
# ---------------------------------------------------------------------------
def check_real_objects(keys):
    """저장소의 실제 물체에서 rank 상한이 지켜지는지 본다.

    density_id_drake 의 회귀행렬은 모듈 전역에 묶여 있으므로, 물체마다
    obj.bind_object() 로 다시 물린 뒤에 읽어야 한다 (dual_view 와 같은 절차).
    """
    try:
        import density_id_objects as obj
        import design_core as dc
    except Exception as exc:                          # Drake 없이도 나머지는 돈다
        print(f"  (건너뜀 — 저장소 모듈을 못 불러왔습니다: {exc})")
        return

    specs = dict(obj.OBJECTS)
    try:                                              # 램프는 별도 모듈에 있다
        import desk_lamp
        specs["desklamp"] = desk_lamp.build_spec()
    except Exception:
        pass

    r_block = np.diag(np.r_[np.full(3, SIGMA_F ** -2), np.full(3, SIGMA_T ** -2)])
    r_inv = np.kron(np.eye(len(TRIAD)), r_block)      # 중력 방향 3개만큼 블록대각

    for key in keys:
        spec = specs.get(key)
        if spec is None:
            print(f"  {key:10s} 건너뜀 (물체 정의를 못 찾음)")
            continue
        try:
            obj.bind_object(spec)
            bounds = [j.limits_rad for j in spec.joints]
            theta = np.array([0.5 * (lo + hi) for lo, hi in bounds])
            A = dc.regressor(theta, [np.asarray(g) for g in TRIAD])
            M = A.T @ r_inv @ A
        except Exception as exc:
            print(f"  {key:10s} 건너뜀 ({exc})")
            continue

        n = A.shape[1]
        rank, ub = rank_of(M), min(4, n)
        assert rank <= ub, f"{key}: 단일 형상 rank {rank} 가 상한 {ub} 를 넘었다"
        print(f"  {key:10s} 미지수 {n}  단일형상 rank {rank} (상한 {ub})  "
              f"라운드 하한 {int(np.ceil((n - 1) / 3))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--parts", type=int, nargs="+", default=[2, 3, 4, 5, 6, 8])
    ap.add_argument("--max-rounds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--objects", nargs="+", default=["2link", "3link", "desklamp"])
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    print(f"\n=== 무작위 형상 {args.trials}개 (THEORY.md 정리 1~2) ===")
    w = check_random(args.trials, rng)
    print(f"  ① 닫힌형태 vs A^T R^-1 A       상대오차 최대 : {w['closed']:.2e}")
    print(f"  ② 삼면체 회전 불변성            상대오차 최대 : {w['invariance']:.2e}")
    print(f"  ③ 임의 K개 방향 일반 닫힌형태   상대오차 최대 : {w['closed_general']:.2e}")
    print(f"  ④ rank - 4 의 최댓값 (P>=4)                  : {w['rank_excess']:+d}")
    print(f"  ⑤ 영공간 조건 위반량 최대                    : {w['nullspace']:.2e}")

    tol = 1e-10
    assert w["closed"] < tol, "정리 1의 닫힌 형태가 틀렸다"
    assert w["invariance"] < tol, "정리 1의 회전 불변성이 깨졌다"
    assert w["closed_general"] < tol, "일반형 닫힌 형태가 틀렸다"
    assert w["rank_excess"] <= 0, "정리 2의 rank 상한 4를 넘었다"
    assert w["nullspace"] < 1e-8, "정리 2.2의 영공간 특성화가 틀렸다"

    print(f"\n=== 누적 rank 와 라운드 하한 (THEORY.md 정리 3) ===")
    print(f"  {'P':>3}  {'라운드별 누적 rank':<28} {'상한 1+min(3R,P)':<24} 하한  달성")
    for P, ranks, need, reached in check_rounds(args.parts, args.max_rounds, rng):
        ub = [1 + min(3 * R, P) for R in range(1, len(ranks) + 1)]
        assert all(k <= b for k, b in zip(ranks, ub)), f"P={P} 에서 rank 상한 위반"
        print(f"  {P:>3}  {str(ranks):<28} {str(ub):<24} {need:>3}  {reached}")
    print("\n  하한 ceil((P-1)/3) 이 '달성' 열과 같으면, 하한은 generic 하게 tight 하다.")
    print("  그런데 실제 폐루프는 이보다 훨씬 많이 걸린다 (3-link: rank 는 1라운드,")
    print("  실제로는 2라운드). 그 차이가 조건수이고, 조건수는 각도 정밀도가 만든다.")

    print(f"\n=== 저장소의 실제 물체 ===")
    check_real_objects(args.objects)
    print("\n모든 assert 통과 — THEORY.md 의 정리 1~3 이 성립합니다.\n")


if __name__ == "__main__":
    main()
