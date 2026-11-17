#!/usr/bin/env python3
"""부위 이름표 그림을 굽는다 — "어느 것이 base 인가" 를 눈으로 알려 주는 그림.

왜 필요한가
-----------
창 2 는 FoundationPose 를 켜기 전에 SAM3 로 base/support/head 마스크를 딴다.
사람이 박스를 그려 주는 방식(`--manual`)인데, 화면에는 "1/3 base: 왼쪽 드래그"
라는 글자만 나온다. **어느 덩어리가 base 인지 알려 주는 것이 아무것도 없다.**

이 이름표를 틀리면 조용히 망가진다. FoundationPose 가 엉뚱한 부위를 추적하고,
PIVOT 은 그 각도를 그대로 믿는다. 예외도 경고도 안 난다. 램프는 특히 위험한데,
예전 배달물의 파일 이름이 실제 부위와 뒤바뀌어 있었던 전력이 있다
(my_work/NAMING.md).

그래서 배달물의 **부위별로 분리된 visual mesh** 를 부위마다 다른 색으로,
여러 방향에서 그려 한 장으로 만든다. 색은 마스크 미리보기와 같은 값을 쓴다.

    base    파랑    (20, 100, 255)
    support 초록    (40, 210,  40)
    head    빨강    (240, 40,  30)

쓰는 법
-------
    python tools/make_part_legend.py <배달물>/visual_meshes -o /tmp/lamp_parts.png
    python tools/make_part_legend.py <배달물>/visual_meshes --parts base,support,head
"""

import argparse
from pathlib import Path

import numpy as np

COLORS = {"base": (0.078, 0.392, 1.0), "support": (0.157, 0.824, 0.157),
          "head": (0.941, 0.157, 0.118), "moving_link": (0.941, 0.157, 0.118)}
FALLBACK = [(0.078, 0.392, 1.0), (0.157, 0.824, 0.157), (0.941, 0.157, 0.118),
            (0.95, 0.7, 0.1), (0.6, 0.3, 0.9)]


def read_vertices(path, limit=40000):
    """정점만 읽는다. 형상을 알아보는 데는 점구름으로 충분하다."""
    vertices = [[float(t) for t in line.split()[1:4]]
                for line in open(path, errors="ignore") if line.startswith("v ")]
    vertices = np.asarray(vertices, dtype=float)
    if len(vertices) > limit:                      # 균일하게 솎는다
        step = len(vertices) // limit + 1
        vertices = vertices[::step]
    return vertices


def read_ply_cloud(path, limit=40000):
    """삼각 메시 PLY와 3DGS PLY에서 위치와 표시색을 읽는다."""
    types = {"char": "i1", "uchar": "u1", "int8": "i1", "uint8": "u1",
             "short": "<i2", "ushort": "<u2", "int16": "<i2", "uint16": "<u2",
             "int": "<i4", "uint": "<u4", "int32": "<i4", "uint32": "<u4",
             "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8"}
    properties, count, vertex = [], 0, False
    with open(path, "rb") as handle:
        header = []
        while True:
            line = handle.readline().decode("ascii").strip()
            header.append(line)
            if line.startswith("format "):
                binary = "binary_little_endian" in line
            elif line.startswith("element "):
                vertex = line.startswith("element vertex ")
                if vertex:
                    count = int(line.split()[2])
            elif vertex and line.startswith("property ") and " list " not in line:
                _, kind, name = line.split()
                properties.append((name, types[kind]))
            elif line == "end_header":
                break
        if binary:
            rows = np.fromfile(handle, np.dtype(properties), count=count)
            values = {name: rows[name] for name, _ in properties}
        else:
            data = np.loadtxt(handle, max_rows=count, ndmin=2)
            values = {name: data[:, index]
                      for index, (name, _) in enumerate(properties)}
    points = np.column_stack([values[key] for key in ("x", "y", "z")])
    if all(key in values for key in ("red", "green", "blue")):
        colors = np.column_stack([values[key] for key in ("red", "green", "blue")])
    elif all(key in values for key in ("f_dc_0", "f_dc_1", "f_dc_2")):
        colors = 255.0 * np.clip(0.5 + 0.28209479177387814 * np.column_stack(
            [values[key] for key in ("f_dc_0", "f_dc_1", "f_dc_2")]), 0.0, 1.0)
    else:
        colors = np.full((len(points), 3), 180.0)
    step = max(1, len(points) // limit + 1)
    return points[::step], np.asarray(colors[::step], dtype=np.uint8)


def render(clouds, out_path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    everything = np.vstack(list(clouds.values()))
    centre = everything.mean(axis=0)
    radius = float(np.abs(everything - centre).max())

    # 세 방향에서 본다. 하나만 그리면 가려진 부위를 못 알아본다.
    # matplotlib 기본 글꼴에 한글이 없다. 이름표 그림이라 글자가 깨지면
    # 안 되므로 뷰 이름과 제목은 ASCII 로 쓴다 (부위 이름은 원래 ASCII).
    views = [(20, -60, "oblique"), (0, -90, "front"), (89, -90, "top")]
    figure = plt.figure(figsize=(13.5, 5.0), dpi=110)
    figure.suptitle(title, fontsize=13)
    for index, (elev, azim, label) in enumerate(views):
        ax = figure.add_subplot(1, 3, index + 1, projection="3d")
        for name, points in clouds.items():
            colour = COLORS.get(name, FALLBACK[list(clouds).index(name)
                                               % len(FALLBACK)])
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.6,
                       c=[colour], depthshade=True, linewidths=0,
                       label=name if index == 0 else None)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(label, fontsize=11)
        for setter, value in ((ax.set_xlim, centre[0]), (ax.set_ylim, centre[1]),
                              (ax.set_zlim, centre[2])):
            setter(value - radius, value + radius)
        ax.set_axis_off()
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:                                    # noqa: BLE001
            pass
        if index == 0:
            legend = ax.legend(loc="upper left", fontsize=12, markerscale=14,
                               framealpha=0.9)
            for text in legend.get_texts():
                text.set_fontweight("bold")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path)
    plt.close(figure)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mesh_dir", type=Path, help="부위별 메시가 있는 폴더")
    ap.add_argument("-o", "--output", type=Path,
                    default=Path("/tmp/part_legend.png"))
    ap.add_argument("--parts", default=None,
                    help="쉼표로 구분한 부위 이름 (기본: 폴더의 .obj/.ply 전부)")
    ap.add_argument("--files", default=None,
                    help="표시명=파일명 목록. 예: base=gaussian_1.ply,head=gaussian_2.ply")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    folder = args.mesh_dir.expanduser()
    file_map = dict(item.split("=", 1) for item in args.files.split(",")) \
        if args.files else {}
    if file_map:
        names = list(file_map)
    elif args.parts:
        names = [n.strip() for n in args.parts.split(",") if n.strip()]
    else:
        names = sorted({p.stem for p in folder.iterdir()
                        if p.suffix.lower() in (".obj", ".ply")})
    clouds = {}
    for name in names:
        if name in file_map:
            path = folder / file_map[name]
            if path.is_file():
                clouds[name] = read_ply_cloud(path)[0]
            continue
        for suffix in (".obj", ".ply"):
            path = folder / f"{name}{suffix}"
            if path.is_file() and suffix == ".obj":
                clouds[name] = read_vertices(path)
                break
            if path.is_file():                    # .ply 는 헤더를 건너뛴다
                clouds[name] = read_ply_cloud(path)[0]
                break
    missing = [n for n in names if n not in clouds]
    if missing:
        raise SystemExit(f"{folder} 에서 못 찾은 부위: {missing}")

    for name, points in clouds.items():
        size = (points.max(axis=0) - points.min(axis=0)) * 1000.0
        print(f"  {name:12s} 점 {len(points):6d}  크기 "
              f"{size[0]:5.0f} x {size[1]:5.0f} x {size[2]:5.0f} mm")
    render(clouds, args.output,
           args.title or f"part labels for SAM3 masking - {folder.parent.name}")
    print(f"\n{args.output} 에 저장했다. 창 2 의 마스크 단계에서 이 그림을 옆에"
          " 띄워 어느 덩어리가 어느 이름인지 보고 박스를 그리면 된다.")


def read_ply_vertices(path, limit=40000):
    return read_ply_cloud(path, limit)[0]


if __name__ == "__main__":
    main()
