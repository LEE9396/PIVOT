"""삼각 메시에서 부피·도심·관성텐서를 정확히 계산한다.

왜 직접 만드나
--------------
Drake 의 CalcSpatialInertia 는 .obj 와 사면체 .vtk 만 받는다. 스캔 파이프라인이
내놓는 것은 .stl 과 .ply 다. 그리고 이 드레이크 환경에는 trimesh/open3d 가
없다. 필요한 건 부피·도심·관성 셋뿐이고, 닫힌 삼각 메시에서는 닫힌 식으로
정확히 나오므로 여기서 직접 구한다 (발산정리 + 사면체 분해).

우리 알고리즘이 메시에서 실제로 필요로 하는 것은 딱 두 가지다.

    V_i   부위 부피
    c_i   부위 도심 (균일밀도 가정이므로 무게중심과 같다)

회귀행렬 A(theta) 는 이 둘만으로 만들어진다. 모양 자체는 필요 없다.
"""

import struct
from pathlib import Path

import numpy as np

_CANONICAL = np.array([[1 / 60.0, 1 / 120.0, 1 / 120.0],
                       [1 / 120.0, 1 / 60.0, 1 / 120.0],
                       [1 / 120.0, 1 / 120.0, 1 / 60.0]])


# ---------------------------------------------------------------------------
# 읽기
# ---------------------------------------------------------------------------
def read_stl(path):
    """이진/아스키 STL. (정점 Nx3, 면 Mx3) 를 돌려준다."""
    raw = open(path, "rb").read()
    if raw[:5].lower() == b"solid" and b"facet" in raw[:2048]:
        return _read_stl_ascii(raw.decode("utf-8", "replace"))
    n_tri = struct.unpack_from("<I", raw, 80)[0]
    expected = 84 + n_tri * 50
    if len(raw) < expected:
        raise ValueError(f"STL 길이가 헤더와 안 맞는다: {path}")
    data = np.frombuffer(raw[84:expected], dtype=np.uint8).reshape(n_tri, 50)
    floats = data[:, :48].copy().view(np.float32).reshape(n_tri, 4, 3)
    tris = floats[:, 1:, :].astype(np.float64)          # 법선 빼고 정점 3개
    return _weld(tris)


def _read_stl_ascii(text):
    verts = [list(map(float, line.split()[1:4]))
             for line in text.splitlines() if line.strip().startswith("vertex")]
    tris = np.asarray(verts, dtype=np.float64).reshape(-1, 3, 3)
    return _weld(tris)


def read_ply(path):
    """ascii / binary_little_endian PLY. 정점 x,y,z 와 삼각 면만 읽는다."""
    with open(path, "rb") as handle:
        raw = handle.read()
    end = raw.find(b"end_header")
    if end < 0:
        raise ValueError(f"PLY 헤더를 못 찾았다: {path}")
    header = raw[:end].decode("ascii", "replace").splitlines()
    body = raw[raw.find(b"\n", end) + 1:]

    fmt = n_vert = n_face = None
    face_count_type, face_index_type = "uchar", "int"
    face_extra = []
    props, element = [], None
    for line in header:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "format":
            fmt = parts[1]
        elif parts[0] == "element":
            element = parts[1]
            if element == "vertex":
                n_vert = int(parts[2])
            elif element == "face":
                n_face = int(parts[2])
        elif parts[0] == "property" and element == "vertex" and parts[1] != "list":
            props.append((parts[1], parts[2]))
        elif parts[0] == "property" and element == "face":
            if parts[1] == "list":
                face_count_type, face_index_type = parts[2], parts[3]
            else:                       # 면마다 붙은 스칼라 (quality 등) — 건너뛴다
                face_extra.append(parts[1])
    if fmt == "ascii":
        return _read_ply_ascii(body.decode("ascii", "replace"), n_vert, n_face,
                               [p[1] for p in props])
    if fmt != "binary_little_endian":
        raise ValueError(f"지원하지 않는 PLY 형식: {fmt}")

    sizes = {"float": 4, "float32": 4, "double": 8, "float64": 8,
             "uchar": 1, "uint8": 1, "char": 1, "int8": 1,
             "short": 2, "ushort": 2, "int": 4, "uint": 4}
    codes = {"float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
             "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
             "short": "i2", "ushort": "u2", "int": "i4", "uint": "u4"}
    dtype = np.dtype([(name, codes[kind]) for kind, name in props])
    stride = sum(sizes[k] for k, _ in props)
    verts = np.frombuffer(body[:n_vert * stride], dtype=dtype, count=n_vert)
    vertices = np.column_stack([verts["x"], verts["y"], verts["z"]]).astype(float)

    count_dt = np.dtype("<" + codes[face_count_type])
    index_dt = np.dtype("<" + codes[face_index_type])
    faces, offset = [], n_vert * stride
    for _ in range(n_face or 0):
        count = int(np.frombuffer(body[offset:offset + count_dt.itemsize],
                                  dtype=count_dt, count=1)[0])
        offset += count_dt.itemsize
        idx = np.frombuffer(body[offset:offset + index_dt.itemsize * count],
                            dtype=index_dt, count=count)
        offset += index_dt.itemsize * count
        offset += sum(sizes[kind] for kind in face_extra)
        for k in range(1, count - 1):          # 부채꼴 삼각화
            faces.append([idx[0], idx[k], idx[k + 1]])
    return vertices, np.asarray(faces, dtype=np.int64)


def read_obj(path):
    """삼각 OBJ의 정점과 면. texture/normal 인덱스는 질량 특성에 불필요하다."""
    vertices, faces = [], []
    for line in open(path):
        if line.startswith("v "):
            vertices.append([float(v) for v in line.split()[1:4]])
        elif line.startswith("f "):
            indices = [int(v.split("/", 1)[0]) - 1 for v in line.split()[1:]]
            for k in range(1, len(indices) - 1):
                faces.append([indices[0], indices[k], indices[k + 1]])
    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=np.int64)


def _read_ply_ascii(text, n_vert, n_face, names):
    lines = [l for l in text.splitlines() if l.strip()]
    vertices = np.array([[float(v) for v in lines[i].split()[:3]]
                         for i in range(n_vert)])
    faces = []
    for line in lines[n_vert:n_vert + (n_face or 0)]:
        idx = [int(v) for v in line.split()]
        for k in range(1, idx[0] - 1):
            faces.append([idx[1], idx[1 + k], idx[2 + k]])
    return vertices, np.asarray(faces, dtype=np.int64)


def read_mesh(path):
    path = str(path)
    if path.lower().endswith(".stl"):
        return read_stl(path)
    if path.lower().endswith(".ply"):
        return read_ply(path)
    if path.lower().endswith(".obj"):
        return read_obj(path)
    raise ValueError(f"지원하지 않는 확장자: {path}")


def _weld(tris, decimals=9):
    """겹치는 정점을 합쳐 (정점, 면) 으로 만든다. STL 은 정점을 공유하지 않는다."""
    flat = tris.reshape(-1, 3)
    keys = np.round(flat, decimals)
    _, first, inverse = np.unique(keys, axis=0, return_index=True,
                                  return_inverse=True)
    return flat[first], inverse.reshape(-1, 3)


# ---------------------------------------------------------------------------
# 질량 특성
# ---------------------------------------------------------------------------
def mass_properties(vertices, faces, density=1.0):
    """닫힌 삼각 메시의 부피·도심·관성텐서(도심 기준).

    사면체 분해로 정확히 계산한다. 근사가 아니다.
    면의 감김 방향이 바깥을 향해야 부피가 양수로 나온다. 음수면 뒤집어 준다.
    """
    v = np.asarray(vertices, dtype=float)
    f = np.asarray(faces, dtype=np.int64)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]

    det = np.einsum("ij,ij->i", a, np.cross(b, c))      # 6 x 사면체 부피
    volume = det.sum() / 6.0
    if volume < 0:                                       # 감김이 반대
        a, b, c, det, volume = b, a, c, -det, -volume
    if volume <= 0:
        raise ValueError("부피가 0 이하다 — 닫힌 메시가 아닐 수 있다")

    centroid = ((det[:, None] * (a + b + c) / 4.0).sum(axis=0)
                / (6.0 * volume))

    # 원점 기준 공분산 (사면체마다 C = det * A C0 A^T)
    A = np.stack([a, b, c], axis=2)                      # (M, 3, 3) 열이 정점
    covariance = np.einsum("m,mij,jk,mlk->il", det, A, _CANONICAL, A)
    # 도심 기준으로 옮긴다
    covariance -= volume * np.outer(centroid, centroid)
    inertia = np.trace(covariance) * np.eye(3) - covariance
    return dict(volume=volume, centroid=centroid,
                inertia=density * inertia, mass=density * volume,
                aabb=(v.min(axis=0), v.max(axis=0)))


def watertight_report(vertices, faces):
    """닫힌 메시인가. 열려 있으면 부피·도심이 틀린다."""
    f = np.asarray(faces, dtype=np.int64)
    edges = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    keys = np.sort(edges, axis=1)
    _, counts = np.unique(keys, axis=0, return_counts=True)
    return dict(n_vertices=len(vertices), n_faces=len(f),
                boundary_edges=int(np.sum(counts == 1)),
                nonmanifold_edges=int(np.sum(counts > 2)),
                watertight=bool(np.all(counts == 2)))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--density", type=float, default=1000.0)
    args = ap.parse_args()
    for path in args.paths:
        v, f = read_mesh(path)
        info = watertight_report(v, f)
        try:
            props = mass_properties(v, f, args.density)
            print(f"{path}")
            print(f"  정점 {info['n_vertices']:>6}  면 {info['n_faces']:>6}"
                  f"  닫힘 {info['watertight']}"
                  f"  경계선 {info['boundary_edges']}"
                  f"  비다양체 {info['nonmanifold_edges']}")
            print(f"  부피 {1e6*props['volume']:10.2f} cm^3"
                  f"  질량@{args.density:.0f} {1000*props['mass']:8.2f} g")
            print(f"  도심 {np.round(props['centroid'], 5)} m")
            print(f"  관성 고윳값 {np.round(np.linalg.eigvalsh(props['inertia']), 8)}")
        except ValueError as exc:
            print(f"{path}: {exc}  (닫힘 {info['watertight']})")


# ---------------------------------------------------------------------------
# 볼록 조각들의 합집합 — 열린 메시를 우회하는 길
#
# 스캔이 내놓은 링크 메시는 구멍이 뚫려 있어 부피·도심을 못 믿는다. 반면
# 볼록 분해(COACD) 조각들은 하나하나가 닫혀 있다. 그래서 링크에 속한 조각들의
# **합집합** 을 재면 닫힌 값이 나온다.
#
# 조각들은 서로 조금씩 겹치므로 단순히 부피를 더하면 과대평가된다. 겹침을
# 제대로 빼려면 합집합을 실제로 구해야 하는데, 볼록다면체 합집합의 정확한
# 계산은 무겁다. 대신 격자로 채워 센다. 격자 간격 1~2 mm 면 이 크기 물체에서
# 0.1 % 수준까지 맞는다.
# ---------------------------------------------------------------------------
def convex_face_planes(vertices, faces):
    """볼록 다면체의 면 평면 (법선, 오프셋). 안쪽이면 n·x <= d."""
    v = np.asarray(vertices, dtype=float)
    f = np.asarray(faces, dtype=np.int64)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    normals = np.cross(b - a, c - a)
    lengths = np.linalg.norm(normals, axis=1)
    keep = lengths > 1e-14
    normals = normals[keep] / lengths[keep, None]
    offsets = np.einsum("ij,ij->i", normals, a[keep])
    center = v.mean(axis=0)
    outward = normals @ center - offsets < 0        # 중심이 안쪽이어야 한다
    normals[~outward] *= -1.0
    offsets[~outward] *= -1.0
    return normals, offsets


def union_properties(pieces, pitch=0.002, margin=1e-9):
    """볼록 조각 목록의 합집합 부피와 도심.

    pieces : [(정점, 면), ...] — 각각 닫힌 볼록 다면체여야 한다.
    pitch  : 격자 간격 [m]. 작을수록 정확하지만 메모리를 더 쓴다.
    """
    lows = np.min([np.asarray(v).min(axis=0) for v, _ in pieces], axis=0)
    highs = np.max([np.asarray(v).max(axis=0) for v, _ in pieces], axis=0)
    lows -= pitch
    highs += pitch
    shape = np.maximum(np.ceil((highs - lows) / pitch).astype(int), 1)
    if shape.prod() > 60_000_000:
        raise MemoryError(f"격자가 너무 큽니다 ({shape}). pitch 를 키우세요.")
    occupied = np.zeros(shape, dtype=bool)

    for vertices, faces in pieces:
        normals, offsets = convex_face_planes(vertices, faces)
        v = np.asarray(vertices, dtype=float)
        # 이 조각의 AABB 안에서만 검사한다
        i0 = np.maximum(((v.min(axis=0) - lows) / pitch).astype(int) - 1, 0)
        i1 = np.minimum(((v.max(axis=0) - lows) / pitch).astype(int) + 2, shape)
        if np.any(i1 <= i0):
            continue
        axes = [lows[k] + (np.arange(i0[k], i1[k]) + 0.5) * pitch
                for k in range(3)]
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
        flat = grid.reshape(-1, 3)
        inside = np.all(flat @ normals.T - offsets <= margin, axis=1)
        block = occupied[i0[0]:i1[0], i0[1]:i1[1], i0[2]:i1[2]]
        occupied[i0[0]:i1[0], i0[1]:i1[1], i0[2]:i1[2]] = (
            block | inside.reshape(block.shape))

    count = int(occupied.sum())
    if count == 0:
        raise ValueError("합집합이 비었습니다 — 조각이 볼록인지 확인하세요")
    index = np.argwhere(occupied)
    centers = lows + (index + 0.5) * pitch
    volume = count * pitch ** 3
    centroid = centers.mean(axis=0)

    # 관성텐서도 같은 격자로 (도심 기준, 단위밀도)
    delta = centers - centroid
    covariance = (delta.T @ delta) * pitch ** 3
    inertia = np.trace(covariance) * np.eye(3) - covariance
    return dict(volume=volume, centroid=centroid, inertia=inertia,
                aabb=(centers.min(axis=0) - pitch / 2,
                      centers.max(axis=0) + pitch / 2),
                pitch=pitch, n_voxels=count)


def vertex_normals(vertices, faces):
    """면 법선을 정점에 모아 평균낸 부드러운 법선."""
    v = np.asarray(vertices, dtype=float)
    f = np.asarray(faces, dtype=np.int64)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    face_n = np.cross(b - a, c - a)
    out = np.zeros_like(v)
    for k in range(3):
        np.add.at(out, f[:, k], face_n)
    lengths = np.linalg.norm(out, axis=1, keepdims=True)
    lengths[lengths < 1e-20] = 1.0
    return out / lengths


def write_obj(path, vertices, faces, name="mesh", colors=None, normals=None):
    """Drake 가 렌더링할 수 있는 형식으로 저장한다.

    Drake 의 Mesh 도형은 .stl 을 안 받고 .obj/.gltf/.vtk 를 받는다. 스캔이
    내놓는 것은 .stl 과 .ply 라 여기서 한 번 바꿔 준다.

    법선을 반드시 써야 한다. VTK 렌더러는 법선 없는 OBJ 를 거부하고
    ("OBJ has no normals"), 있으면 음영이 살아 실물처럼 보인다.
    normals 를 주면 그것을 쓴다 — 메시를 조각으로 쪼갤 때 원본 법선을
    물려주면 조각 경계에서 음영이 끊기지 않는다.

    colors 를 주면 정점 색을 함께 쓴다 (v x y z r g b). 다만 Drake 의
    Meshcat 은 재질에 vertexColors=false 를 박아 보내므로 **화면에서는
    무시된다** (직접 확인함). 화면 색은 색 무리로 쪼개서 조각마다 단색을
    주는 방식으로 낸다 (split_by_color). 정점 색은 다른 도구가 읽을 수
    있도록 남겨 두는 것이다.
    """
    v = np.asarray(vertices, dtype=float)
    f = np.asarray(faces, dtype=np.int64)
    n = vertex_normals(v, f) if normals is None else np.asarray(normals, float)
    lines = [f"o {name}"]
    if colors is None:
        lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in v]
    else:
        rgb = np.asarray(colors, dtype=float)
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
        lines += [f"v {x:.6f} {y:.6f} {z:.6f} {r:.4f} {g:.4f} {b:.4f}"
                  for (x, y, z), (r, g, b) in zip(v, rgb)]
    lines += [f"vn {x:.6f} {y:.6f} {z:.6f}" for x, y, z in n]
    idx = f + 1                                        # OBJ 는 1부터 센다
    lines += [f"f {a}//{a} {b}//{b} {c}//{c}" for a, b, c in idx]
    Path(path).write_text("\n".join(lines) + "\n")
    return path


def split_by_color(vertices, faces, colors, k=6, iters=40, seed=0):
    """정점 색이 비슷한 면끼리 묶어 메시를 k 조각으로 쪼갠다.

    왜 쪼개나
    ---------
    Drake 는 메시 하나에 **단색 하나**만 입힌다 (Meshcat 으로 보내는 재질에
    vertexColors=false 가 박혀 있다). 그래서 스캔 색을 화면에 내려면 색이
    비슷한 면끼리 묶어 조각으로 나누고, 조각마다 그 무리의 평균색을 주는
    수밖에 없다. 조각 수 k 를 늘릴수록 원본에 가까워지고 도형 수도 늘어난다.

    돌려주는 것: [(정점, 면, (r,g,b,1)), ...]  — 색이 밝은 순.
    면의 색은 세 정점 색의 평균으로 본다. 무리 나누기는 k-means 이고,
    시작점은 밝기 분위수로 잡아 매번 같은 결과가 나오게 했다.
    """
    v = np.asarray(vertices, dtype=float)
    f = np.asarray(faces, dtype=np.int64)
    c = np.asarray(colors, dtype=float)
    if c.max() > 1.0:
        c = c / 255.0
    face_color = c[f].mean(axis=1)                    # 면마다 대표색

    luma = face_color @ np.array([0.2126, 0.7152, 0.0722])
    centers = np.array([face_color[np.argmin(np.abs(luma - q))]
                        for q in np.quantile(luma, np.linspace(0, 1, k))])
    label = np.zeros(len(f), dtype=int)
    for _ in range(iters):
        distance = ((face_color[:, None, :] - centers[None]) ** 2).sum(axis=2)
        new_label = distance.argmin(axis=1)
        if np.array_equal(new_label, label):
            break
        label = new_label
        for j in range(k):
            if (label == j).any():
                centers[j] = face_color[label == j].mean(axis=0)

    normals = vertex_normals(v, f)
    pieces = []
    for j in np.argsort(-centers @ np.array([0.2126, 0.7152, 0.0722])):
        mask = label == j
        if not mask.any():
            continue
        used = np.unique(f[mask])
        remap = np.full(len(v), -1, dtype=np.int64)
        remap[used] = np.arange(len(used))
        rgb = tuple(float(x) for x in centers[j])
        pieces.append((v[used], remap[f[mask]], normals[used], rgb + (1.0,)))
    return pieces


def read_ply_colors(path):
    """PLY 의 정점 색 (N x 3, 0~255). 색이 없으면 None."""
    with open(path, "rb") as handle:
        raw = handle.read()
    end = raw.find(b"end_header")
    header = raw[:end].decode("ascii", "replace")
    if "property uchar red" not in header:
        return None
    vertices, faces = read_ply(path)     # 파서를 한 번 더 태우는 대신
    # 색만 다시 읽는다 (형식은 read_ply 와 동일한 규칙)
    lines = header.splitlines()
    props, element, n_vert = [], None, 0
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "element":
            element = parts[1]
            if element == "vertex":
                n_vert = int(parts[2])
        elif parts[0] == "property" and element == "vertex" and parts[1] != "list":
            props.append((parts[1], parts[2]))
    sizes = {"float": 4, "float32": 4, "double": 8, "float64": 8,
             "uchar": 1, "uint8": 1, "char": 1, "int8": 1,
             "short": 2, "ushort": 2, "int": 4, "uint": 4}
    codes = {"float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
             "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
             "short": "i2", "ushort": "u2", "int": "i4", "uint": "u4"}
    dtype = np.dtype([(name, codes[kind]) for kind, name in props])
    stride = sum(sizes[k] for k, _ in props)
    body = raw[raw.find(b"\n", end) + 1:]
    data = np.frombuffer(body[:n_vert * stride], dtype=dtype, count=n_vert)
    return np.column_stack([data["red"], data["green"], data["blue"]])
