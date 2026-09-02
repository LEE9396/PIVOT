#!/usr/bin/env python
"""배달물의 통짜 충돌 메시를 **볼록 조각**으로 쪼개 굽는다.

왜 필요한가
-----------
Drake 의 `Convex` 는 파일 하나를 **볼록 껍질 하나로** 감싼다. 비볼록한 부위
메시를 그대로 넘기면 두 가지가 동시에 망가진다.

  1. 형상이 부푼다. 이 램프 배달물에서 1.5 ~ 2.9 배였다.
  2. `desk_lamp.pinch_grasp()` 가 **조각의 정점 평균**을 파지점으로 삼는데,
     조각이 하나뿐이면 그 평균이 부위 한가운데가 된다. 330 mm 짜리 팔의
     한가운데를 잡은 것으로 모형화되고, 반대쪽 끝이 손목을 관통한다.

2026-09-02 에 이것 때문에 head 가 AFT200 마운트를 41 mm 파고든 채
"힌지·도달·충돌을 모두 통과하는 자세가 없다" 만 나왔다.

쓰는 법
-------
    python -m pip install coacd
    python tools/make_convex.py <배달물>/collision_meshes/support.obj
    python tools/make_convex.py <배달물>/collision_meshes          # 폴더 통째로

결과는 `<collision_meshes>/convex/<부위이름>/part_NN.obj` 로 나간다 —
`desk_lamp.collision_meshes()` 가 찾는 바로 그 자리다.

조각 수
-------
기본값(threshold 0.05)은 굽은 팔을 3~8 조각으로 자른다. 조각이 적으면
하나가 부위 대부분을 차지해 위 2번 문제가 남는다. `--report` 가 조각별
길이를 찍어 주니, **가장 긴 조각이 부위 전체 길이의 절반을 넘으면**
--threshold 를 낮춰 다시 구워라.
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def read_obj(path):
    vertices, faces = [], []
    for line in open(path, errors="ignore"):
        if line.startswith("v "):
            vertices.append([float(t) for t in line.split()[1:4]])
        elif line.startswith("f "):
            index = [int(t.split("/")[0]) for t in line.split()[1:]]
            for k in range(1, len(index) - 1):
                faces.append([index[0] - 1, index[k] - 1, index[k + 1] - 1])
    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=int)


def write_obj(path, vertices, faces, name):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as out:
        out.write(f"o {name}\n")
        for v in vertices:
            out.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for f in faces:
            out.write(f"f {f[0]+1} {f[1]+1} {f[2]+1}\n")


def span_mm(vertices):
    """조각의 최장 주축 길이 [mm]."""
    centred = vertices - vertices.mean(axis=0)
    _, _, axes = np.linalg.svd(centred, full_matrices=False)
    return 1000.0 * max(float(np.ptp(centred @ axes[k])) for k in range(3))


def decompose(source, out_root, threshold, report):
    try:
        import coacd
    except ImportError:
        sys.exit("coacd 가 없다.  python -m pip install coacd")

    vertices, faces = read_obj(source)
    if len(faces) == 0:
        sys.exit(f"{source} 에 면이 없다")
    parts = coacd.run_coacd(coacd.Mesh(vertices, faces), threshold=threshold)

    folder = out_root / source.stem
    for stale in sorted(folder.glob("part_*.obj")):
        stale.unlink()
    spans = []
    for index, (piece_v, piece_f) in enumerate(parts):
        piece_v = np.asarray(piece_v, dtype=float)
        write_obj(folder / f"part_{index:02d}.obj", piece_v,
                  np.asarray(piece_f, dtype=int), f"{source.stem}_{index:02d}")
        spans.append(span_mm(piece_v))

    whole = span_mm(vertices)
    longest = max(spans)
    print(f"{source.name:>16s} -> {len(parts):2d} 조각   "
          f"부위 길이 {whole:6.1f} mm, 가장 긴 조각 {longest:6.1f} mm "
          f"({100*longest/whole:4.0f} %)   {folder}")
    if report:
        for index, s in enumerate(spans):
            print(f"      part_{index:02d}  {s:6.1f} mm")
    if longest > 0.5 * whole:
        print(f"      [주의] 가장 긴 조각이 부위의 절반을 넘는다. 파지점이"
              f" 부위 한가운데로 잡힐 수 있다 — --threshold 를"
              f" {threshold/2:.3f} 로 낮춰 다시 구워 보라.")
    return longest <= 0.5 * whole


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", type=Path,
                    help="충돌 .obj 파일 하나, 또는 collision_meshes 폴더")
    ap.add_argument("--threshold", type=float, default=0.05,
                    help="CoACD 오차 문턱. 작을수록 조각이 많다 (기본 0.05)")
    ap.add_argument("--report", action="store_true", help="조각별 길이도 찍는다")
    args = ap.parse_args()

    target = args.target.expanduser()
    if target.is_dir():
        sources = sorted(p for p in target.glob("*.obj") if p.parent.name != "convex")
        out_root = target / "convex"
    else:
        sources = [target]
        out_root = target.parent / "convex"
    if not sources:
        sys.exit(f"{target} 에서 .obj 를 못 찾았다")

    print(f"결과 위치: {out_root}\n")
    ok = all([decompose(s, out_root, args.threshold, args.report) for s in sources])
    print("\n조각 배치 끝. desk_lamp.collision_meshes() 가 이제 이걸 읽는다.")
    if not ok:
        print("조각이 굵은 부위가 있다 — 위 [주의] 를 보고 다시 구울지 정하라.")


if __name__ == "__main__":
    main()
