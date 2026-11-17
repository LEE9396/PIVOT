"""저장된 정지 F/T 기록으로 빈 공구/물체 파라미터를 차감한다. 로봇 연결 없음.

my_work에서 실행:
  ../robot_learning/scripts/run_drake_env.sh python ../tools/identify_ft_payload.py \
    --empty empty_1.json empty_2.json empty_3.json \
    --loaded load_1.json load_2.json load_3.json --output separated.json
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "my_work"))
from payload_identification import separate_payload


def load_record(path, state, force_span_n=0.5, torque_span_nm=0.02):
    data = json.loads(Path(path).read_text())
    if (data.get("status") != "recorded" or data.get("load_state") != state
            or data.get("wrench_frame") != "ft_mount"
            or not all(data.get(k) for k in ("sensor_sha256", "kinematics_sha256",
                                            "ft_session_id", "configuration_id"))):
        raise ValueError(f"{path}: 완료 상태·좌표계·하중/파지/센서 세션 정보가 필요합니다")
    blocks = data["blocks"]
    if len(blocks) < 3:
        raise ValueError(f"{path}: 반복성을 확인할 정지 구간이 3개 이상 필요합니다")
    means = np.asarray([b["wrench_mean"] for b in blocks], dtype=float)
    directions = np.asarray([b["achieved_g_hat"] for b in blocks], dtype=float)
    poses = np.asarray([q for b in blocks for q in
                        [b["joint_before_deg"], b["joint_after_deg"]]
                        + b.get("controller_joint_deg", [])], dtype=float)
    counts = np.asarray([b[k]["position"] for b in blocks
                         for k in ("gripper_before", "gripper_after")], dtype=float)
    if (means.shape != (len(blocks), 6) or directions.shape != (len(blocks), 3)
            or poses.ndim != 2 or poses.shape[1] != 6
            or not all(np.all(np.isfinite(v)) for v in (means, directions, poses, counts))
            or not np.allclose(np.linalg.norm(directions, axis=1), 1., atol=1e-5)
            or np.any(counts < 0) or np.any(counts > 255) or np.ptp(counts) > 1):
        raise ValueError(f"{path}: 유효한 고정 자세·개구 및 6축 기록이 필요합니다")
    if np.max(np.abs((poses - poses[0] + 180.) % 360. - 180.)) > 0.1:
        raise ValueError(f"{path}: 정지 기록 전체에서 팔 자세가 변했습니다")
    if np.max(np.linalg.norm(directions - directions[0], axis=1)) > 0.01:
        raise ValueError(f"{path}: 정지 기록의 중력 방향이 달라졌습니다")
    span = means[:, None] - means[None, :]
    if (np.max(np.linalg.norm(span[:, :, :3], axis=2)) > force_span_n
            or np.max(np.linalg.norm(span[:, :, 3:], axis=2)) > torque_span_nm):
        raise ValueError(f"{path}: 반복 구간의 힘/토크 변화가 안정화 한계를 넘었습니다")
    weights = np.asarray([len(b["wrench_samples"]) for b in blocks])
    if np.any(weights < 2):
        raise ValueError(f"{path}: 구간별 원시 표본이 부족합니다")
    for block, mean, count in zip(blocks, means, weights):
        raw = np.asarray(block["wrench_samples"], dtype=float)
        if (raw.shape != (count, 6) or not np.all(np.isfinite(raw))
                or not np.allclose(raw.mean(axis=0), mean, atol=1e-9, rtol=1e-9)):
            raise ValueError(f"{path}: 원시 표본과 구간 평균이 일치하지 않습니다")
    g = np.average(directions, axis=0, weights=weights)
    return dict(path=str(Path(path).resolve()),
                sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                sensor_sha256=data["sensor_sha256"], ft_session_id=data["ft_session_id"],
                kinematics_sha256=data["kinematics_sha256"],
                configuration_id=data["configuration_id"],
                opening_count=float(np.mean(counts)),
                g=(g / np.linalg.norm(g)).tolist(),
                wrench=np.average(means, axis=0, weights=weights).tolist())


def identify(empty, loaded, max_opening_delta_counts=3., **limits):
    """같은 센서 세션에서 실제 개구가 가장 가까운 빈 공구 모델을 고른다."""
    if (not empty or not loaded or not np.isfinite(max_opening_delta_counts)
            or max_opening_delta_counts < 0):
        raise ValueError("빈 공구/하중 기록 및 유효한 개구 허용차가 필요합니다")
    if len({(r["sensor_sha256"], r["kinematics_sha256"], r["ft_session_id"])
            for r in empty + loaded}) != 1:
        raise ValueError("센서 코드 또는 영점·장착 세션이 다릅니다")
    if len({r["configuration_id"] for r in loaded}) != 1:
        raise ValueError("파지 위치 또는 물체 관절각이 다른 하중 기록을 섞을 수 없습니다")
    if np.ptp([r["opening_count"] for r in loaded]) > 1:
        raise ValueError("하중 기록 사이에 실제 그리퍼 개구가 달라졌습니다")
    groups = {}
    for r in empty:
        groups.setdefault(r["configuration_id"], []).append(r)
    for group in groups.values():
        if np.ptp([r["opening_count"] for r in group]) > 1:
            raise ValueError("빈 공구 그룹 안에서 그리퍼 개구가 달라졌습니다")
    opening = float(np.mean([r["opening_count"] for r in loaded]))
    key = min(groups, key=lambda k: abs(np.mean([r["opening_count"] for r in groups[k]]) - opening))
    selected = groups[key]
    gap = max(abs(r["opening_count"] - opening) for r in selected)
    if gap > max_opening_delta_counts:
        raise ValueError("실제 파지 개구에 충분히 가까운 빈 공구 기록이 없습니다")
    result = separate_payload([r["g"] for r in selected], [r["wrench"] for r in selected],
                              [r["g"] for r in loaded], [r["wrench"] for r in loaded], **limits)
    result.update(selected_empty_group=key, opening_difference_counts=gap,
                  empty_records=selected, loaded_records=loaded,
                  robot_commands_sent=False, calibration_valid=False,
                  pass_meaning="식별 가능성과 중력 모델 잔차 검사이며 실물 정확도 인증이 아님",
                  limits=dict(max_opening_delta_counts=max_opening_delta_counts, **limits))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--empty", type=Path, nargs="+", required=True)
    parser.add_argument("--loaded", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-opening-delta-counts", type=float, default=3.)
    parser.add_argument("--force-limit-n", type=float, default=0.5)
    parser.add_argument("--torque-limit-nm", type=float, default=0.02)
    parser.add_argument("--condition-limit", type=float, default=100.)
    parser.add_argument("--known-mass-kg", type=float, help="평가 전용. 식별에는 사용하지 않음")
    args = parser.parse_args()
    limits = dict(force_limit_n=args.force_limit_n, torque_limit_nm=args.torque_limit_nm,
                  condition_limit=args.condition_limit)
    if (not all(np.isfinite(v) and v > 0 for v in limits.values())
            or (args.known_mass_kg is not None and
                (not np.isfinite(args.known_mass_kg) or args.known_mass_kg <= 0))):
        parser.error("한계값과 평가용 질량은 양의 유한수여야 합니다")
    # 기존 측정·결과 파일을 덮어쓰지 않고 실패 이유도 저장한다.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        try:
            paths = args.empty + args.loaded
            if len({p.resolve() for p in paths}) != len(paths):
                raise ValueError("같은 기록을 중복 사용하거나 빈 상태와 하중에 함께 쓸 수 없습니다")
            empty = [load_record(p, "empty", args.force_limit_n, args.torque_limit_nm) for p in args.empty]
            loaded = [load_record(p, "loaded", args.force_limit_n, args.torque_limit_nm) for p in args.loaded]
            report = identify(empty, loaded, args.max_opening_delta_counts, **limits)
            if args.known_mass_kg is not None and "object_mass_kg" in report:
                error = abs(report["object_mass_kg"] - args.known_mass_kg)
                report["evaluation"] = dict(known_mass_kg=args.known_mass_kg,
                                             absolute_mass_error_kg=error,
                                             relative_mass_error=error / args.known_mass_kg)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            report = dict(passed=False, reason=str(exc), calibration_valid=False,
                          robot_commands_sent=False, estimator_reply=None)
        json.dump(report, output, indent=2, ensure_ascii=False, allow_nan=False)
        output.write("\n")
    print(json.dumps({k: report[k] for k in ("passed", "reason")}, ensure_ascii=False))
    print(args.output)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
