"""정지 손목 F/T로 빈 공구와 하중 상태를 식별해 물체의 질량·1차 모멘트를 분리한다.

Scalable Real2Sim의 파라미터 차감을 정지 렌치에 적용한다. 회전 관성은 정지
측정으로 식별하지 않는다. 센서 바이어스는 빈 상태에서만 식별하여 하중 힘을
새 영점으로 흡수하지 않는다. 힘·토크 축 간 간섭을 교정하는 모형은 아니다.
"""

import numpy as np

import density_id_drake as alg
from tare_check import _skew, fit_force, fit_torque


def static_matrix(directions):
    """미지수 [질량, 질량×무게중심 xyz]의 정지 중력 행렬."""
    g = np.asarray(directions, dtype=float)
    if (g.ndim != 2 or g.shape[1] != 3 or len(g) == 0
            or not np.all(np.isfinite(g))
            or not np.allclose(np.linalg.norm(g, axis=1), 1., atol=1e-5)):
        raise ValueError("센서 좌표계의 단위 중력 방향들이 필요합니다")
    return alg.FORCE_SIGN * alg.G_ACC * np.vstack([
        np.block([[v[:, None], np.zeros((3, 3))],
                  [np.zeros((3, 1)), -_skew(v)]]) for v in g])


def fit_static(directions, wrenches, bias=None, force_limit_n=0.5,
               torque_limit_nm=0.02, condition_limit=100.):
    """빈 상태는 바이어스까지, 하중 상태는 고정 바이어스로 관성 파라미터만 맞춘다."""
    A = static_matrix(directions)
    values = np.asarray(wrenches, dtype=float)
    n = len(directions)
    if values.shape != (n, 6) or not np.all(np.isfinite(values)):
        raise ValueError("각 자세마다 유한한 6축 원시 렌치가 필요합니다")
    limits = np.array([force_limit_n, torque_limit_nm, condition_limit])
    if not np.all(np.isfinite(limits)) or np.any(limits <= 0):
        raise ValueError("잔차·조건수 한계는 양의 유한수여야 합니다")
    design = np.column_stack([np.tile(np.eye(6), (n, 1)), A]) if bias is None else A
    # N과 N·m 및 파라미터 단위의 차이를 자세 부족으로 오인하지 않는다.
    norms = np.linalg.norm(design, axis=0)
    normalized = design / np.maximum(norms, 1e-15)
    rank = int(np.linalg.matrix_rank(normalized))
    condition = float(np.linalg.cond(normalized))
    result = dict(passed=False, identifiable=False, rank=rank,
                  required_rank=design.shape[1],
                  condition=condition if np.isfinite(condition) else None)
    if rank < design.shape[1] or condition > condition_limit:
        return dict(result, reason="중력 방향 분포가 부족합니다. 동일/인접 자세만으로 식별하지 않습니다")
    if bias is None:
        bf, weight, _ = fit_force(directions, values[:, :3])
        bt, moment, _ = fit_torque(directions, values[:, 3:])
        bias = np.r_[bf, bt]
        params = np.r_[weight / (alg.FORCE_SIGN * alg.G_ACC),
                       moment / alg.FORCE_SIGN]
    else:
        bias = np.asarray(bias, dtype=float)
        if bias.shape != (6,) or not np.all(np.isfinite(bias)):
            raise ValueError("빈 상태에서 식별한 유한한 6축 바이어스가 필요합니다")
        params = np.linalg.lstsq(A, (values - bias).ravel(), rcond=None)[0]
    residual = values - bias - (A @ params).reshape(n, 6)
    force = float(np.max(np.linalg.norm(residual[:, :3], axis=1)))
    torque = float(np.max(np.linalg.norm(residual[:, 3:], axis=1)))
    passed = params[0] > 0 and force <= force_limit_n and torque <= torque_limit_nm
    return dict(result, identifiable=True, passed=bool(passed),
                reason="ok" if passed else "중력 모형 잔차 또는 양의 질량 조건 불합격",
                mass_kg=float(params[0]), first_moment_kg_m=params[1:].tolist(),
                com_m=(params[1:] / params[0]).tolist() if params[0] > 0 else None,
                bias=bias.tolist(), residual_wrench=residual.tolist(),
                max_force_residual_n=force, max_torque_residual_nm=torque)


def separate_payload(empty_g, empty_wrench, loaded_g, loaded_wrench, **limits):
    """파라미터 차감과 측정 렌치 차감을 모두 보존한다. 불합격은 추정기에 보내지 않는다."""
    empty = fit_static(empty_g, empty_wrench, **limits)
    result = dict(passed=False, method="static_inertial_parameter_difference",
                  empty=empty, rotational_inertia_identified=False,
                  uncertainty_calibrated=False, estimator_reply=None)
    if not empty["passed"]:
        return dict(result, reason="빈 공구 식별 실패")
    loaded = fit_static(loaded_g, loaded_wrench, bias=empty["bias"], **limits)
    result["loaded"] = loaded
    if not loaded["identifiable"]:
        return dict(result, reason="하중 상태 식별 실패")
    empty_params = np.r_[empty["mass_kg"], empty["first_moment_kg_m"]]
    loaded_params = np.r_[loaded["mass_kg"], loaded["first_moment_kg_m"]]
    delta = loaded_params - empty_params
    A = static_matrix(loaded_g)
    offset = np.asarray(empty["bias"]) + (A @ empty_params).reshape(-1, 6)
    raw = np.asarray(loaded_wrench, dtype=float)
    corrected = raw - offset
    # 맞춤값 A@delta로 센서값을 대체하면 큰 Fz 잔차가 사라지므로 원 차분을 쓴다.
    result.update(object_mass_kg=float(delta[0]),
                  object_first_moment_kg_m=delta[1:].tolist(),
                  object_com_m=(delta[1:] / delta[0]).tolist() if delta[0] > 0 else None,
                  corrected_wrench=corrected.tolist(),
                  object_model_residual=(corrected - (A @ delta).reshape(-1, 6)).tolist())
    result["passed"] = bool(loaded["passed"] and delta[0] > 0)
    result["reason"] = "ok" if result["passed"] else "물체 질량 또는 하중 모형 잔차 불합격"
    if result["passed"]:
        result["estimator_reply"] = dict(
            wrench=corrected.ravel().tolist(), wrench_raw=raw.ravel().tolist(),
            tare_applied=offset.ravel().tolist(), tare_required=True,
            wrench_frame="ft_mount", g_dirs=np.asarray(loaded_g).tolist())
    return result
