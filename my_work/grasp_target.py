"""파지점 추천 — 어떤 워터타이트 부위 메시에서든 '실제로 물리는 자리'를 고른다.

왜 desk_lamp.pinch_grasp 를 그대로 못 쓰나
------------------------------------------
`desk_lamp.pinch_grasp` 는 배달물의 **볼록 분해 조각**을 훑는다. 조각 하나가
팔을 토막 낸 것이라 거의 원기둥이어서, 조각의 중심과 주축이 곧 파지 자세가
된다. 좋은 방법이지만 볼록 분해가 같이 와야 한다.

실물 화면에 파지점을 띄우려면 **FoundationPose 가 추적하는 그 메시** 위의
좌표가 필요하다. 그 메시(MeshPCA `*_metric_watertight.ply`)에는 볼록 분해가
없다. 그래서 여기서는 볼록 분해 없이 같은 것을 찾는다.

  1. 부위의 장축(PCA 1축)을 따라 얇은 판(slab)으로 훑는다
  2. 판마다 **가로 두 방향의 실제 폭**을 잰다 (장축에 수직인 평면에서)
  3. 두 폭 중 좁은 쪽이 죠가 물 방향이고, 그 값이 필요한 개구량이다
  4. 개구량 안에 들어오는 판 중 **가장 좁은** 자리를 고른다

판마다 재는 것이 핵심이다. 부위 전체의 AABB 나 도심을 쓰면 굽은 팔에서
파지점이 재료 밖 허공에 놓인다 (desk_lamp.pinch_grasp 주석의 그 문제).

무엇이 나오나
-------------
파지 목표를 **그 메시 자신의 좌표계**로 돌려준다. 그래야 FoundationPose 가
준 자세(카메라 <- 메시)에 그대로 곱해 화면에 띄울 수 있다.

    {"point": [x,y,z],        파지점 (m, 메시 좌표)
     "jaw_axis": [x,y,z],     죠가 오므라드는 방향 (단위)
     "long_axis": [x,y,z],    부위가 뻗은 방향 (단위)
     "width_m": 0.0412,       그 자리의 실제 단면 폭
     "opening_m": 0.0452}     죠를 벌려 둘 개구량 (폭 + 여유)

실행:
    cd ~/Desktop/PIVOT/my_work
    ../robot_learning/scripts/run_drake_env.sh python grasp_target.py \\
        --mesh ~/MeshPCA/foundationpose/assets/lamp/support_metric_watertight.ply \\
        --part support --out outputs/grasp_target_lamp.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

import mesh_props as mp

# 죠를 물체 폭보다 이만큼 더 벌려 둔다 [m]. 넣을 때 긁히지 않을 만큼만.
JAW_CLEARANCE_M = 0.004
# 판 두께 [m]. 너무 얇으면 정점이 모자라고, 너무 두꺼우면 굽은 데서 폭이 부푼다.
SLAB_M = 0.012
# 판 하나가 쓸모 있으려면 이만큼은 정점이 있어야 한다.
MIN_SLAB_POINTS = 24
# 양 끝은 건드리지 않는다. 부위 끝은 관절이거나 얇은 파편이라 잡을 자리가 아니다.
END_MARGIN = 0.15


def principal_axes(points):
    """(중심, 축 3개(열), 표준편차) — 큰 축이 먼저."""
    center = points.mean(axis=0)
    centered = points - center
    values, vectors = np.linalg.eigh(centered.T @ centered / len(centered))
    order = np.argsort(values)[::-1]
    return center, vectors[:, order], np.sqrt(values[order])


def slab_width(points, jaw_dir, other_dir):
    """이 판을 두 가로 방향으로 재서 (좁은 폭, 죠 방향, 넓은 폭)."""
    a = float(np.ptp(points @ jaw_dir))
    b = float(np.ptp(points @ other_dir))
    return (a, jaw_dir, b) if a <= b else (b, other_dir, a)


def recommend(vertices, max_opening_m, slab_m=SLAB_M,
              end_margin=END_MARGIN, clearance_m=JAW_CLEARANCE_M):
    """파지 목표 하나를 고른다. 못 고르면 None.

    max_opening_m 은 그리퍼가 벌릴 수 있는 최대 개구량이다. 그보다 두꺼운
    자리는 애초에 후보가 아니다 — 시뮬레이션은 용접이라 돌아가지만 실물은
    못 잡는다.
    """
    vertices = np.asarray(vertices, dtype=float)
    center, axes, _ = principal_axes(vertices)
    long_dir = axes[:, 0]
    cross = axes[:, 1], axes[:, 2]

    along = (vertices - center) @ long_dir
    lo, hi = along.min(), along.max()
    span = hi - lo
    if span <= slab_m:
        return None
    # 양 끝을 빼고 판 중심을 훑는다.
    starts = np.arange(lo + end_margin * span, hi - end_margin * span,
                       slab_m / 2.0)

    # 판마다 물 수 있는지 재 둔다. 여기서 **가장 좁은 자리를 고르면 안 된다.**
    # 램프 팔처럼 전 구간이 9~12 mm 로 고른 부위에서는 폭에 변별력이 없어서,
    # 최솟값이 늘 끝자락(관절 근처)에 붙는다. 관절 위를 잡으면 그 관절이
    # 파지점 기준으로 돌아 모멘트팔이 라운드마다 달라진다.
    slabs = []
    for position in starts:
        near = vertices[np.abs(along - position) <= slab_m / 2.0]
        if len(near) < MIN_SLAB_POINTS:
            slabs.append(None)
            continue
        local = near - near.mean(axis=0)
        width, jaw_dir, wide = slab_width(local, cross[0], cross[1])
        ok = width <= max_opening_m
        slabs.append((near.mean(axis=0), jaw_dir, width, wide) if ok else None)

    # 물 수 있는 판이 **연달아 가장 길게 이어지는 구간**을 고르고, 그
    # 한가운데를 잡는다. 그게 부위의 몸통이다 (desk_lamp.pinch_grasp 가
    # 볼록 조각 중 가장 긴 것을 고른 것과 같은 뜻이다). 끝자락의 얇은
    # 파편이나 관절 목은 구간이 짧아 저절로 밀린다.
    best_run, run_start = None, None
    for index in range(len(slabs) + 1):
        alive = index < len(slabs) and slabs[index] is not None
        if alive and run_start is None:
            run_start = index
        elif not alive and run_start is not None:
            length = index - run_start
            if best_run is None or length > best_run[0]:
                best_run = (length, run_start, index)
            run_start = None
    if best_run is None:
        return None
    _, first, last = best_run
    point, jaw_dir, width, wide = slabs[(first + last - 1) // 2]
    # 죠 축과 장축을 직교화한다 (판 주축은 정확히 직교하지 않는다).
    jaw = jaw_dir - (jaw_dir @ long_dir) * long_dir
    jaw /= np.linalg.norm(jaw)
    return dict(
        point=[float(v) for v in point],
        jaw_axis=[float(v) for v in jaw],
        long_axis=[float(v) for v in long_dir],
        width_m=float(width),
        across_m=float(wide),
        opening_m=float(min(width + clearance_m, max_opening_m)),
    )


def from_mesh(path, max_opening_m, **kwargs):
    vertices, _ = mp.read_mesh(Path(path))
    target = recommend(vertices, max_opening_m, **kwargs)
    if target is None:
        raise RuntimeError(
            f"{Path(path).name}: 개구 {1000*max_opening_m:.0f} mm 안에 드는"
            f" 파지 자리를 못 찾았습니다 — 이 부위는 이 그리퍼로 못 잡습니다")
    target["mesh"] = str(Path(path).resolve())
    return target


def from_pivot_part(path, part, max_opening_m):
    """PIVOT 밀도 모델이 쓰는 명목 파지점을 같은 메시 좌표로 내보낸다."""
    import desk_lamp

    expected = desk_lamp.visual_mesh_path(part).resolve()
    if Path(path).resolve() != expected:
        raise RuntimeError(f"FoundationPose mesh must be PIVOT's {expected}")
    pinch = desk_lamp.pinch_grasp(part)
    if pinch is None:
        raise RuntimeError(f"PIVOT has no pinch grasp for {part}")
    width = pinch["width_mm"] * 1e-3
    return dict(
        point=[float(v) for v in pinch["point"]],
        jaw_axis=[float(v) for v in pinch["jaw_axis"]],
        long_axis=[float(v) for v in pinch["long_axis"]],
        width_m=width,
        opening_m=float(min(width + JAW_CLEARANCE_M, max_opening_m)),
        mesh=str(expected),
        pivot_part=part,
    )


def describe(target):
    point = np.array(target["point"]) * 1000.0
    across = (f" (가로 {1000*target['across_m']:.1f} mm)"
              if "across_m" in target else "")
    return (f"파지점 {np.round(point, 1)} mm"
            f"  단면 {1000*target['width_m']:.1f} mm{across}"
            f"  개구 {1000*target['opening_m']:.1f} mm")


def main():
    import grippers as gr

    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=Path, required=True,
                        help="부위 하나의 워터타이트 메시 (.ply/.obj/.stl)")
    parser.add_argument("--part", default=None,
                        help="FoundationPose 가 쓰는 부위 이름 (예: support)")
    parser.add_argument("--pivot-part", default=None,
                        help="PIVOT 명목 파지점을 그대로 사용 (예: link_3)")
    parser.add_argument("--gripper", default="robotiq2f85",
                        choices=tuple(gr.GRIPPERS))
    parser.add_argument("--slab-mm", type=float, default=1000.0 * SLAB_M)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    spec = gr.GRIPPERS[args.gripper]
    target = (from_pivot_part(args.mesh, args.pivot_part, spec.max_opening_m)
              if args.pivot_part else
              from_mesh(args.mesh, spec.max_opening_m,
                        slab_m=args.slab_mm * 1e-3))
    target["part"] = args.part or Path(args.mesh).stem
    target["gripper"] = spec.key
    print(f"{target['part']}  ({spec.label}, 최대 개구"
          f" {1000*spec.max_opening_m:.0f} mm)")
    print(f"  {describe(target)}")
    print(f"  죠 축   {np.round(target['jaw_axis'], 4)}")
    print(f"  장축    {np.round(target['long_axis'], 4)}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(target, indent=2) + "\n")
        print(f"  저장 -> {args.out}")


if __name__ == "__main__":
    main()
