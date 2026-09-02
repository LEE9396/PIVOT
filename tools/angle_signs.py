#!/usr/bin/env python
"""FoundationPose 가 읽어 주는 각도를 PIVOT 관절각에 맞춘다 (부호와 영점).

왜 필요한가
-----------
창 2 는 "부위 장축 사이의 기하 각" 을 준다. **부호가 없다** (foundationpose
README). PIVOT 관절각은 URDF 축 기준의 **부호 있는** 값이고 영점도 다르다.
둘을 안 맞추면 `adjust_by_pose` 가 목표와 반대로 판정한다 — 사용자가 각도를
반대로 돌리고, 탐색은 엉뚱한 자세에서 돈다. 예외도 경고도 안 난다
(my_work/NAMING.md).

어떻게 맞추나 — 별도 계측기가 필요 없다
----------------------------------------
배달물 메시와 URDF 만 있으면 "관절값 q 일 때 두 부위 장축 사이 각이
얼마인가" 를 **계산할 수 있다**. 그것이 창 2 가 재는 바로 그 양이다.

    model(q) = ∠(장축_부모(q), 장축_자식(q))     <- 이 스크립트가 만든다
    창 2      alpha                              <- 실물에서 읽는다

alpha 하나로는 부호를 못 정한다 (model 이 q 와 -q 에서 같은 값일 수 있다).
그래서 **쓸어담는다**: 작업자가 관절을 한쪽 끝에서 다른 끝까지 천천히 접는
동안 alpha 를 계속 기록하고, 그 궤적을 model 곡선에 맞춘다.

    q_PIVOT = s * (q_from_model) + b,   s in {+1, -1}
    잔차 RMS = **실제 각도 오차** -> 그대로 ANGLE_FLOOR_DEG 에 넣으면 된다

쓰는 법
-------
  1) 모델 곡선만 확인 (실물 불필요)
       $R python tools/angle_signs.py --model-only

  2) 실물에서 쓸어담기 — 로봇은 안 움직인다. 사람이 관절을 돌린다.
       $R python tools/angle_signs.py --pose-file /tmp/lamp_foundationpose_live/latest.json \
           --joint 1 --seconds 40 --output calibration/angle_signs.json

     화면 안내대로 그 관절 하나만 한쪽 끝에서 다른 끝까지 **천천히** 접는다.
     관절마다 따로 한다.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "my_work"))


def long_axis(points):
    """점구름의 최장 주축 (단위벡터)."""
    centred = points - points.mean(axis=0)
    _, _, axes = np.linalg.svd(centred, full_matrices=False)
    spans = [(float(np.ptp(centred @ axes[k])), k) for k in range(3)]
    return axes[max(spans)[1]]


def rotate(points, axis, angle, pivot):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
    return (points - pivot) @ R.T + pivot


def model_curve(spec_parts, joint, samples=361, directed=False):
    """관절값 -> 두 부위 장축 사이 각(도). 창 2 가 재는 양과 같은 정의.

    장축은 SVD 가 주는 **방향 없는** 축이라 부호가 임의다. 그래서 두 규약이
    가능하다.
      undirected  |cos| 을 쓴다 -> 0~90 도. 축을 선분으로 본다.
      directed    부모 장축 방향을 한 번 고정하고 cos 을 쓴다 -> 0~180 도.
    팀원 코드가 어느 쪽인지 모르므로 둘 다 만들어 보고 더 잘 맞는 쪽을 쓴다.
    """
    parent = spec_parts[joint["parent"]]
    child = spec_parts[joint["child"]]
    axis, pivot = joint["axis"], joint["origin"]
    lo, hi = joint["limits"]
    a_parent = long_axis(parent)
    reference = None
    grid, angles = np.linspace(lo, hi, samples), []
    for q in grid:
        a_child = long_axis(rotate(child, axis, q, pivot))
        if directed:
            # 축 부호가 표본마다 뒤집히지 않도록 직전 값에 맞춰 붙인다.
            if reference is not None and a_child @ reference < 0:
                a_child = -a_child
            reference = a_child
            cos = float(np.clip(a_parent @ a_child, -1.0, 1.0))
        else:
            cos = float(np.clip(abs(a_parent @ a_child), 0.0, 1.0))
        angles.append(np.degrees(np.arccos(cos)))
    return grid, np.asarray(angles)


def fit(grid, model_deg, observed_deg, n_coarse=61):
    """쓸어담은 궤적을 모델 곡선에 맞춘다.

    모델은 단조가 아니다 — 기하각이 0 도와 90 도에서 접힌다. 그래서 관측값
    하나하나에 가장 가까운 q 를 붙이면 가지 사이를 뛰어다닌다. 대신 **궤적
    전체**를 맞춘다.

    작업자가 q 를 한 방향으로 고르게 접었다고 보고
        q(t) = q0 + (q1 - q0) * t,   t = 0..1 (표본 순서)
    로 두고 (q0, q1) 을 격자 탐색한다. q1 < q0 이면 반대 방향으로 접은 것이고,
    그것이 곧 부호다. 남는 잔차가 **실제 각도 오차**다.
    """
    observed = np.asarray(observed_deg, dtype=float)
    t = np.linspace(0.0, 1.0, len(observed))
    lo, hi = float(grid[0]), float(grid[-1])
    coarse = np.linspace(lo, hi, n_coarse)
    best = None
    for q0 in coarse:
        for q1 in coarse:
            if abs(q1 - q0) < np.deg2rad(20.0):
                continue                      # 너무 조금 접은 것은 못 믿는다
            predicted = np.interp(q0 + (q1 - q0) * t, grid, model_deg)
            rms = float(np.sqrt(np.mean((predicted - observed) ** 2)))
            if best is None or rms < best["residual_rms_deg"]:
                best = dict(q_start_deg=float(np.degrees(q0)),
                            q_end_deg=float(np.degrees(q1)),
                            residual_rms_deg=rms,
                            sign=1.0 if q1 > q0 else -1.0)
    best["span_deg"] = abs(best["q_end_deg"] - best["q_start_deg"])
    best["n"] = int(len(observed))
    return best


def read_pose(path, key=None):
    data = json.loads(Path(path).read_text())
    if key and key in data:
        return float(data[key])
    for name in ("angles_deg", "joint_angles_deg", "angles"):
        if name in data:
            values = data[name]
            return ([float(v) for v in values] if isinstance(values, list)
                    else values)
    return {k: v for k, v in data.items() if k.endswith("_deg")}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-only", action="store_true",
                    help="모델 곡선만 계산해 보여준다 (실물 불필요)")
    ap.add_argument("--pose-file", type=Path,
                    help="FoundationPose latest.json")
    ap.add_argument("--angle-key", default=None,
                    help="latest.json 안에서 읽을 키 (예: support_head_deg)")
    ap.add_argument("--joint", type=int, default=1, help="관절 번호 (1부터)")
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--rate-hz", type=float, default=5.0)
    ap.add_argument("--output", type=Path,
                    default=Path("calibration/angle_signs.json"))
    args = ap.parse_args()

    import desk_lamp
    spec = desk_lamp.build_spec()
    parts = {}
    for part in spec.parts:
        cloud = []
        for path in part.collision_meshes:
            cloud.append(np.array([[float(t) for t in line.split()[1:4]]
                                   for line in open(path)
                                   if line.startswith("v ")]))
        parts[part.name] = np.vstack(cloud)
    joint = spec.joints[args.joint - 1]
    info = dict(parent=joint.parent, child=joint.child,
                axis=np.asarray(joint.axis, float),
                origin=np.asarray(joint.origin_in_parent_link_mm, float) / 1000.0,
                limits=joint.limits_rad)
    curves = {mode: model_curve(parts, info, directed=(mode == "directed"))
              for mode in ("undirected", "directed")}
    print(f"관절 {joint.name}  ({joint.parent} -> {joint.child})")
    print(f"  구동범위 {np.degrees(info['limits'][0]):.1f} ~ "
          f"{np.degrees(info['limits'][1]):.1f} deg")
    for mode, (grid, model) in curves.items():
        turning = int((np.diff(np.sign(np.diff(model))) != 0).sum())
        print(f"  [{mode:10s}] 기하각 {model.min():5.1f} ~ {model.max():5.1f} deg,"
              f" 꺾이는 점 {turning}개"
              + ("  (단조 — 한 점만 읽어도 q 가 정해진다)" if turning == 0
                 else "  (접힘 — 쓸어담아야 q 가 정해진다)"))
    if args.model_only:
        grid, model = curves["undirected"]
        for q, a in zip(grid[::40], model[::40]):
            print(f"    q {np.degrees(q):7.1f} deg  ->  기하각 {a:6.1f} deg")
        return

    if args.pose_file is None:
        sys.exit("--pose-file 이 필요합니다 (또는 --model-only)")
    print(f"\n지금부터 {args.seconds:.0f}초 동안 **{joint.parent}-{joint.child}"
          f" 관절만** 한쪽 끝에서 다른 끝까지 천천히 접으세요.")
    print("로봇은 움직이지 않습니다. 준비되면 Enter.")
    input()
    observed, stamps = [], []
    end = time.time() + args.seconds
    while time.time() < end:
        try:
            value = read_pose(args.pose_file, args.angle_key)
        except Exception as exc:                            # noqa: BLE001
            print(f"  [주의] 자세를 못 읽었습니다: {exc}")
            time.sleep(1.0 / args.rate_hz)
            continue
        if isinstance(value, dict):
            sys.exit(f"어느 값을 쓸지 --angle-key 로 정하세요: {list(value)}")
        if isinstance(value, list):
            value = value[args.joint - 1]
        observed.append(float(value))
        stamps.append(time.time())
        print(f"\r  {len(observed):4d} 표본  현재 {value:6.1f} deg", end="")
        time.sleep(1.0 / args.rate_hz)
    print()
    if len(observed) < 20:
        sys.exit("표본이 너무 적습니다 (20개 미만)")

    fits = {}
    for mode, (grid, model) in curves.items():
        fits[mode] = fit(grid, model, observed)
        fits[mode]["axis_mode"] = mode
    mode = min(fits, key=lambda m: fits[m]["residual_rms_deg"])
    result = fits[mode]
    result.update(joint=joint.name, parent=joint.parent, child=joint.child,
                  observed_min_deg=float(min(observed)),
                  observed_max_deg=float(max(observed)),
                  alternative=fits["directed" if mode == "undirected"
                                   else "undirected"])
    print(f"\n축 규약  {mode}"
          f"  (다른 규약의 잔차 {result['alternative']['residual_rms_deg']:.2f} deg)")
    print(f"쓸어담은 구간  q {result['q_start_deg']:+.1f} -> "
          f"{result['q_end_deg']:+.1f} deg  (폭 {result['span_deg']:.1f} deg)")
    print(f"부호 s = {result['sign']:+.0f}")
    print(f"잔차 RMS = {result['residual_rms_deg']:.2f} deg"
          f"   <- 이 값을 ANGLE_FLOOR_DEG 에 쓰세요")
    ratio = result["residual_rms_deg"] / max(
        result["alternative"]["residual_rms_deg"], 1e-9)
    if ratio > 0.7:
        print("  [주의] 두 축 규약의 잔차가 비슷합니다 — 어느 쪽인지 못"
              " 갈랐습니다. 더 넓게, 더 천천히 다시 쓸어담으세요.")
    if result["span_deg"] < 40.0:
        print("  [주의] 쓸어담은 폭이 좁습니다 — 구동범위 끝에서 끝까지"
              " 접어야 부호가 확실해집니다.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged = {}
    if args.output.is_file():
        merged = json.loads(args.output.read_text())
    merged[joint.name] = result
    args.output.write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n")
    print(f"\n{args.output} 에 저장했습니다.")


if __name__ == "__main__":
    main()
