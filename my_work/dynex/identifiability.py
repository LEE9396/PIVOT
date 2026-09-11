#!/usr/bin/env python
"""구조적 식별 가능성 — 관절 사슬에서 부위 파라미터 10P 개 중 몇 개(조합)가 보이는가.

한 형상 θ 에서 센서가 보는 것은 조립체의 10개 φ_S(θ) = Σ_j M_j(θ) φ_j 뿐이다.
형상을 아무리 많이 바꿔도 stack_θ [M_1(θ) … M_P(θ)] 의 계수(rank)를 넘지 못한다.
회전 관절 하나마다 축 둘레 회전에 불변인 조합 4개가 부모 부위와 뭉쳐서
(Gautier–Khalil 의 base parameter 재편성) 보이지 않는다. 파지 부위를 바꿔도
같다 — 조립체에 고정된 어느 프레임이든 같은 10개를 다른 좌표로 볼 뿐이다.

    cd my_work && ../robot_learning/scripts/run_drake_env.sh python -m dynex.identifiability --object 3link
"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import density_id_objects as obj                     # noqa: E402

from .inertial import transport_from_rigid_transform  # noqa: E402
from .kinematics import SceneModel                   # noqa: E402

COMPONENTS = ("m", "hx", "hy", "hz", "Ixx", "Ixy", "Ixz", "Iyy", "Iyz", "Izz")


def assembly_maps(model, thetas, anchor=None):
    rows = []
    for theta in thetas:
        poses = model.part_poses_in_S(theta)
        if anchor is None:
            rows.append(np.hstack([transport_from_rigid_transform(X) for X in poses]))
        else:
            X_SA = poses[anchor]
            rows.append(np.hstack([transport_from_rigid_transform(X_SA.inverse() @ X)
                                   for X in poses]))
    return np.vstack(rows)


def analyse(spec, n_configs=40, seed=0, tol=1e-9):
    model = SceneModel(spec)
    rng = np.random.default_rng(seed)
    lims = [j.limits_rad for j in spec.joints]
    thetas = [np.array([rng.uniform(lo + 0.15, hi - 0.15) for lo, hi in lims])
              for _ in range(n_configs)]
    A = assembly_maps(model, thetas)
    n = A.shape[1]
    U, S, Vt = np.linalg.svd(A)
    rank = int((S > tol * S[0]).sum())
    null = Vt[rank:]
    names = [f"{p.name}.{c}" for p in spec.parts for c in COMPONENTS]
    per_config = []
    for k in (1, 2, 3, 4, 6):
        Ak = assembly_maps(model, thetas[:k])
        s = np.linalg.svd(Ak, compute_uv=False)
        per_config.append((k, int((s > tol * s[0]).sum())))
    return dict(n_params=n, rank=rank, null_dim=n - rank, null=null, names=names,
                per_config=per_config, n_parts=len(spec.parts), n_joints=len(spec.joints))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--object", default="3link", choices=tuple(obj.OBJECTS))
    args = ap.parse_args()
    spec = obj.OBJECTS[args.object]
    r = analyse(spec)
    print(f"{spec.label}: 부위 {r['n_parts']}, 관절 {r['n_joints']} → 파라미터 {r['n_params']}개 중"
          f" 식별 가능 {r['rank']}개, 보이지 않는 조합 {r['null_dim']}개 (관절당 {r['null_dim'] // max(r['n_joints'], 1)})")
    print("형상 수에 따른 rank:", ", ".join(f"{k}개→{rk}" for k, rk in r["per_config"]))
    print("보이지 않는 조합 (큰 성분 순):")
    for k, v in enumerate(r["null"]):
        big = np.argsort(-np.abs(v))[:5]
        print(f"  {k + 1}: " + ", ".join(f"{r['names'][i]} {v[i]:+.2f}" for i in big))


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
def base_basis(model, spec, n_configs=40, seed=0, tol=1e-9):
    """식별 가능 부분공간의 정규직교 기저 (열벡터), 정적으로 보이는 부분과 동적으로만 보이는 부분.

    돌려주는 것: V_all (n x r), V_static (n x r_s), V_dyn (n x (r − r_s))
      V_static ⊂ V_all,  V_dyn = V_all 에서 V_static 을 뺀 직교 여공간.
    """
    rng = np.random.default_rng(seed)
    lims = [j.limits_rad for j in spec.joints]
    thetas = [np.array([rng.uniform(lo + 0.15, hi - 0.15) for lo, hi in lims])
              for _ in range(n_configs)]
    A = assembly_maps(model, thetas)
    A_static = np.vstack([A[10 * k:10 * k + 4] for k in range(n_configs)])

    def row_space(M):
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        r = int((S > tol * S[0]).sum())
        return Vt[:r].T                                     # n x r, 정규직교 열

    V_all = row_space(A)
    V_static = row_space(A_static)
    # 동적 전용 = V_all 안에서 V_static 에 직교하는 부분
    P = V_all @ V_all.T - V_static @ V_static.T
    U, S, _ = np.linalg.svd(P)
    r_dyn = int((S > 1e-9).sum())
    V_dyn = U[:, :r_dyn]
    return V_all, V_static, V_dyn


def base_report(est, phis_true, model, spec):
    """사후분포를 식별 가능 좌표로 투영해 오차·불확실성을 잰다.

    각 기저 방향 v 에 대해 참값 좌표 t = vᵀΦ_true, 추정 e = vᵀμ, 표준편차 s = √(vᵀΣv).
    상대값은 그 방향의 참값 크기 대신 사전 스케일(부위 질량·길이) 로 나눈다.
    """
    from .inertial import scale_vector
    Phi = np.concatenate(phis_true)
    n = Phi.size
    mu, cov = est.mean[:n], est.cov[:n, :n]
    V_all, V_static, V_dyn = base_basis(model, spec)
    out = {}
    for name, V in (("static", V_static), ("dynamic_only", V_dyn), ("all", V_all)):
        t = V.T @ Phi
        e = V.T @ mu
        s = np.sqrt(np.einsum("ij,jk,ik->i", V.T, cov, V.T))
        scale = np.maximum(np.abs(t), 1e-12)
        out[name] = dict(n=V.shape[1], err_rel=np.abs(e - t) / scale, sd_rel=s / scale,
                         err_abs=np.abs(e - t), sd_abs=s, truth=t)
    return out
