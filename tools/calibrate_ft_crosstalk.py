"""AFT200 논문 식 (6), (8)의 토크→힘 간섭을 같은 자세의 하중 차분에 맞춘다.

오프라인 도구이며 센서 설정·활성 영점·로봇을 변경하지 않는다.
원문의 공구 CoM/베이스 기울기 비선형 동시 식별 전체를 구현한 것은 아니다.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from identify_ft_payload import load_record
import density_id_drake as alg


def correct_difference(delta_wrench, matrix):
    """빈 공구 차감 뒤 적용한다. 물체의 실제 토크는 보존한다."""
    delta = np.asarray(delta_wrench, dtype=float)
    k = np.asarray(matrix, dtype=float)
    if (delta.ndim != 2 or delta.shape[1] != 6 or k.shape != (3, 3)
            or not np.all(np.isfinite(delta)) or not np.all(np.isfinite(k))
            or not np.allclose(np.diag(k), 0., atol=1e-12, rtol=0)):
        raise ValueError("유한한 N×6 차분과 대각이 0인 3×3 간섭 행렬이 필요합니다")
    corrected = delta.copy()
    corrected[:, :3] -= delta[:, 3:] @ k.T
    return corrected


def fit_crosstalk(delta_wrench, expected_force, condition_limit=100.,
                  min_torque_excitation_nm=0.05):
    """논문의 대각 0 제약으로 축마다 2개 계수를 식별한다. 계수 단위는 1/m."""
    delta = correct_difference(delta_wrench, np.zeros((3, 3)))
    expected = np.asarray(expected_force, dtype=float)
    if (expected.shape != (len(delta), 3) or not np.all(np.isfinite(expected))
            or len(delta) < 3):
        raise ValueError("서로 다른 하중 조건의 6축/기대 힘 기록이 최소 3개 필요합니다")
    if not all(np.isfinite(v) and v > 0 for v in
               (condition_limit, min_torque_excitation_nm)):
        raise ValueError("조건수와 토크 가진 한계는 양의 유한수여야 합니다")
    k, diagnostics = np.zeros((3, 3)), []
    for axis in range(3):
        columns = [j for j in range(3) if j != axis]
        x = delta[:, 3:][:, columns]
        norms = np.linalg.norm(x, axis=0)
        normalized = x / np.maximum(norms, 1e-15)
        rank = int(np.linalg.matrix_rank(normalized))
        condition = float(np.linalg.cond(normalized))
        # 표본을 복제해도 약한 토크 방향이 충분한 가진으로 바뀌지 않게 한다.
        excitation = float(np.linalg.svd(x, compute_uv=False)[-1] / np.sqrt(len(x)))
        ok = rank == 2 and condition <= condition_limit and excitation >= min_torque_excitation_nm
        diagnostics.append(dict(force_axis="xyz"[axis], rank=rank,
                                condition=condition if np.isfinite(condition) else None,
                                min_torque_excitation_nm=excitation, passed=bool(ok)))
        if ok:
            k[axis, columns] = np.linalg.lstsq(x, delta[:, axis] - expected[:, axis], rcond=None)[0]
    identifiable = all(d["passed"] for d in diagnostics)
    return dict(identifiable=identifiable, axes=diagnostics,
                matrix=k.tolist() if identifiable else None)


def evaluate(delta_wrench, expected_force, matrix):
    expected = np.asarray(expected_force, dtype=float)
    corrected = correct_difference(delta_wrench, matrix)
    if (expected.shape != (len(corrected), 3) or not len(expected)
            or not np.all(np.isfinite(expected))
            or np.any(np.linalg.norm(expected, axis=1) <= 0)):
        raise ValueError("질량이 양수인 교정용 물체의 기대 힘이 필요합니다")
    force = corrected[:, :3]
    residual = force - expected
    g = expected / np.linalg.norm(expected, axis=1)[:, None]
    projected = np.sum(force * g, axis=1)
    parallel_torque = np.sum(corrected[:, 3:] * g, axis=1)
    return dict(corrected_wrench=corrected.tolist(),
                force_residual_n=residual.tolist(),
                max_force_residual_n=float(np.max(np.linalg.norm(residual, axis=1))),
                rms_force_residual_n=float(np.sqrt(np.mean(np.sum(residual**2, axis=1)))),
                norm_mass_kg=(np.linalg.norm(force, axis=1) / alg.G_ACC).tolist(),
                projected_mass_kg=(projected / alg.G_ACC).tolist(),
                torque_parallel_gravity_nm=parallel_torque.tolist(),
                max_parallel_torque_nm=float(np.max(np.abs(parallel_torque))),
                absolute_mass_error_kg=(np.abs(np.linalg.norm(force, axis=1)
                                             - np.linalg.norm(expected, axis=1)) / alg.G_ACC).tolist(),
                perpendicular_force_n=np.linalg.norm(force - projected[:, None]*g, axis=1).tolist())


def load_cycle(item, directory, force_limit_n=0.5, torque_limit_nm=0.02):
    """E→L→E의 자세·세션·실제 개구·시간·복귀값을 검사한다."""
    mass = float(item["reference_mass_kg"])
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError("교정용 기준 질량은 양의 유한수여야 합니다")
    paths = [(directory / item[k]).resolve() for k in ("empty_before", "loaded", "empty_after")]
    if len(set(paths)) != 3:
        raise ValueError("빈 기준을 복귀 기록으로 재사용할 수 없습니다")
    records = [load_record(p, s, force_limit_n, torque_limit_nm)
               for p, s in zip(paths, ("empty", "loaded", "empty"))]
    signatures = {(r["sensor_sha256"], r["kinematics_sha256"], r["ft_session_id"]) for r in records}
    if len(signatures) != 1:
        raise ValueError("주기 중 센서·운동학·영점/장착 세션이 다릅니다")
    if np.ptp([r["opening_count"] for r in records]) > 3:
        raise ValueError("빈 상태와 하중의 실제 개구가 3카운트 넘게 다릅니다")
    raw = [json.loads(p.read_text()) for p in paths]
    poses, times = [], []
    for data in raw:
        poses.extend(b["joint_before_deg"] for b in data["blocks"])
        stamps = np.asarray([t for b in data["blocks"] for t in b["sample_time_s"]], dtype=float)
        if not len(stamps) or not np.all(np.isfinite(stamps)) or np.any(np.diff(stamps) < 0):
            raise ValueError("유효하고 시간순인 기록 시각이 필요합니다")
        times.append([float(stamps[0]), float(stamps[-1])])
    poses = np.asarray(poses)
    if np.max(np.abs((poses - poses[0] + 180.) % 360. - 180.)) > 0.1:
        raise ValueError("빈 상태와 하중 사이 팔 자세 차이가 0.1°를 넘습니다")
    if not (times[0][1] < times[1][0] and times[1][1] < times[2][0]
            and times[2][1] - times[0][0] <= 1200):
        raise ValueError("20분 이내의 빈 상태→하중→빈 복귀 순서가 필요합니다")
    w = np.asarray([r["wrench"] for r in records])
    drift = w[2] - w[0]
    if np.linalg.norm(drift[:3]) > force_limit_n or np.linalg.norm(drift[3:]) > torque_limit_nm:
        raise ValueError("빈 상태 복귀 변화가 한계를 넘어 간섭과 드리프트를 분리할 수 없습니다")
    # 복귀값으로 변화량을 한정하고 두 빈 평균의 중간값을 사용한다. 시간 드리프트를 가정해 맞추지 않는다.
    delta = w[1] - (w[0] + w[2])/2
    expected = alg.FORCE_SIGN * alg.G_ACC * mass * np.asarray(records[1]["g"])
    return dict(records=records, signature=list(next(iter(signatures))),
                delta=delta.tolist(), expected_force=expected.tolist(),
                reference_mass_kg=mass, empty_return_change=drift.tolist())


def calibrate_manifest(manifest, directory, force_limit_n=0.5, torque_limit_nm=0.02):
    if not all(np.isfinite(v) and v > 0 for v in (force_limit_n, torque_limit_nm)):
        raise ValueError("잔차 한계는 양의 유한수여야 합니다")
    groups = {key: [load_cycle(item, directory, force_limit_n, torque_limit_nm)
                    for item in manifest[key]] for key in ("train", "validation")}
    if any(len(rows) < 3 for rows in groups.values()):
        raise ValueError("식별 및 독립 검증에 각각 최소 3개 측정 주기가 필요합니다")
    all_cycles = groups["train"] + groups["validation"]
    if len({tuple(c["signature"]) for c in all_cycles}) != 1:
        raise ValueError("식별과 검증의 센서·운동학·영점/장착 세션이 다릅니다")
    loaded_hashes = [c["records"][1]["sha256"] for c in all_cycles]
    if len(set(loaded_hashes)) != len(loaded_hashes):
        raise ValueError("하중 기록을 복제해서 독립 측정으로 셀 수 없습니다")
    hashes = {key: {r["sha256"] for c in rows for r in c["records"]} for key, rows in groups.items()}
    if hashes["train"] & hashes["validation"]:
        raise ValueError("식별 기록을 독립 검증에 재사용할 수 없습니다")
    arrays = {key: (np.array([c["delta"] for c in rows]),
                    np.array([c["expected_force"] for c in rows])) for key, rows in groups.items()}
    fit = fit_crosstalk(*arrays["train"])
    result = dict(model="AFT200_paper_eq6_eq8_paired_difference", source_doi="10.3390/app15031510",
                  fit=fit, cycles=groups, force_validation_passed=False,
                  calibration_valid=False, uncertainty_calibrated=False,
                  torque_accuracy_validated=False, robot_commands_sent=False,
                  limits=dict(force_limit_n=force_limit_n, torque_limit_nm=torque_limit_nm),
                  reason="토크 방향의 정보량 부족", estimator_reply=None)
    if not fit["identifiable"]:
        return result
    validation_design = fit_crosstalk(*arrays["validation"])
    result["validation_design"] = {k: v for k, v in validation_design.items() if k != "matrix"}
    result["before"] = {key: evaluate(*data, np.zeros((3, 3))) for key, data in arrays.items()}
    result["after"] = {key: evaluate(*data, fit["matrix"]) for key, data in arrays.items()}
    passed = (validation_design["identifiable"] and
              all(r["max_force_residual_n"] <= force_limit_n for r in result["after"].values()))
    result.update(force_validation_passed=bool(passed),
                  reason="기록된 범위의 힘 검증 통과; 전체 6축 교정은 미검증" if passed
                  else "간섭 보정 후 잔차 또는 독립 검증 정보량 불합격")
    # 순수 중력 모멘트는 중력에 수직이다. 힘만 맞춰 토크/좌표계 오류를 숨기지 않는다.
    result['torque_gravity_consistent'] = all(
        r['max_parallel_torque_nm'] <= torque_limit_nm for r in result['after'].values())
    if passed and not result['torque_gravity_consistent']:
        result['reason'] = "힘 검증만 통과; 중력 평행 토크가 커서 토크·축 정렬 검증 필요"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force-limit-n", type=float, default=0.5)
    parser.add_argument("--torque-limit-nm", type=float, default=0.02)
    args = parser.parse_args()
    try:
        result = calibrate_manifest(json.loads(args.manifest.read_text()), args.manifest.parent,
                                    args.force_limit_n, args.torque_limit_nm)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        result = dict(force_validation_passed=False, calibration_valid=False,
                      robot_commands_sent=False, reason=str(exc), estimator_reply=None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(result["reason"])
    print(args.output)
    return 0 if result["force_validation_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
