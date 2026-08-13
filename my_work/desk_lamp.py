"""실물 스캔한 스탠드 램프를 파이프라인에 물린다.

입력 (mesh refinement 배달물, 0813 minimal_sim 판)
--------------------------------------------------
    drake/object.urdf                         링크 3개 + 회전관절 2개
    drake/meshes/link_*.stl                   링크별 닫힌 메시 (색 없음)
    geometry/parts/link_*_watertight_rgb.ply  링크별 닫힌 메시 + **정점 색**
    geometry/raw_watertight_multiview_rgb.ply 통짜 메시 (참고용)

배달물 위치는 세 곳을 이 순서로 찾는다.
    1) 환경변수 DESK_LAMP_DELIVERY
    2) 저장소 안 assets/desk_lamp_minimal_sim   <- git clone 하면 여기 있다
    3) ~/Downloads 의 옛 배달물 (구조가 다르면 알아서 맞춘다)

색을 화면에 어떻게 내나
-----------------------
스캔이 3DGS 에서 구운 색이 파트 메시의 **정점**에 들어 있다. 그런데 Drake 는
메시 하나에 단색 하나만 입힌다 — Meshcat 으로 보내는 재질에 vertexColors=false
가 박혀 나가는 것을 메시지를 뜯어 확인했다. 그래서 색이 비슷한 면끼리 묶어
파트를 여섯 조각으로 나누고, 조각마다 그 무리의 평균색을 준다
(mesh_props.split_by_color). 정점 색은 .obj 에도 함께 적어 둔다 — 다른
도구가 읽을 수 있게.

우리 알고리즘이 메시에서 필요로 하는 것은 부피 V 와 도심 c 둘뿐이다.
색은 순전히 사람이 보기 위한 것이고 추정에 아무 영향이 없다.

스캔 부피 != 재료 부피
----------------------
GT 실측과 비교하면 스캔 부피가 재료 부피의 0.6 ~ 8 배다.

    베이스   스캔 166.8  실측 290.2 cm^3   <- 바닥면이 테이블에 닿아 안 찍힘
    Arm      스캔 219.1  실측  72.9 cm^3   <- 가는 관을 겉면으로만 떠서 부풀음
    Head     스캔 441.8  실측  52.8 cm^3   <- 갓이 속이 비어 있음

이건 고장이 아니라 스캔의 본질이다. 카메라는 **겉모양**만 본다.

중요한 건 이것이 문제가 안 된다는 점이다. 알고리즘은 rho x V 가 렌치를
설명하도록 rho 를 맞춘다. V 가 2 배 크면 rho 가 절반으로 나오고 **곱인 질량은
그대로**다. 시뮬레이터에 필요한 것은 질량이지 밀도가 아니다.

그래서 이 파일이 쓰는 rho 는 '겉모양 부피로 나눈 유효 밀도'다. 물리적 밀도가
아니며, 재질을 알아맞히는 데 쓰면 안 된다. 채점도 **질량**으로 한다.

한계 (시뮬레이션으로는 검증 못 하는 것)
---------------------------------------
도심은 사정이 다르다. 속이 빈 갓은 겉모양 도심과 재료 도심이 다르다. 그
차이는 라운드를 늘려도 안 없어지는 치우침이 된다 (study_hinge.py 와 같은
구조). 얼마나 아픈지는 --com-error 로 재볼 수 있다.

실행:
  ../robot_learning/scripts/run_drake_env.sh python desk_lamp.py
"""

import argparse
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

import mesh_props as mp
from density_id_objects import Joint, ObjectSpec, Part

# 0813 배달물부터 분할 메시가 **닫혀** 있다 (경계선 0). 그전에는 구멍이
# 104~283 개라 부피·도심을 못 믿어서 볼록 조각의 합집합으로 우회했었다.
# 이제는 분할 메시에서 바로 정확히 구한다.
#
# 배달물이 두 가지 구조로 온다.
#   "minimal"     drake/object.urdf + geometry/parts/*_rgb.ply   (0813~, 색 있음)
#   "refinement"  issacimport/.../object.urdf + DATA/.../mesh/   (0812, 색 없음)
REPO_ASSETS = Path(__file__).resolve().parents[1] / "assets"


def _layout_of(root):
    root = Path(root)
    if (root / "drake" / "visuals").is_dir() and (root / "drake"
                                                  / "collisions").is_dir():
        # 0814 판: 볼록 분해와 glTF 화면 메시를 직접 준다. 우리가 만들 것이 없다.
        return "minimal_v2"
    if (root / "drake" / "object.urdf").is_file():
        return "minimal"
    if (root / "issacimport").is_dir():
        return "refinement"
    return None


def _resolve_delivery():
    """배달물 폴더를 찾는다. 환경변수 -> 저장소 -> ~/Downloads 순."""
    roots = []
    env = os.environ.get("DESK_LAMP_DELIVERY")
    if env:
        roots.append(Path(env).expanduser())
    roots.append(REPO_ASSETS / "desk_lamp_minimal_sim")
    downloads = Path.home() / "Downloads"
    if downloads.is_dir():
        roots += sorted(downloads.glob("**/desk_lamp_minimal_sim"))
        roots += sorted(downloads.glob("desk_lamp_refinement_delivery_*"),
                        reverse=True)
    for root in roots:
        layout = _layout_of(root)
        if layout:
            return root, layout
    raise FileNotFoundError(
        "램프 배달물을 못 찾았다. 다음 중 하나를 하라.\n"
        f"  1) 저장소 안에 두기: {REPO_ASSETS / 'desk_lamp_minimal_sim'}\n"
        "  2) 환경변수로 알려주기: export DESK_LAMP_DELIVERY=/경로/desk_lamp_minimal_sim\n"
        "찾아본 곳: " + ", ".join(str(r) for r in roots))


DELIVERY, LAYOUT = _resolve_delivery()
if LAYOUT == "minimal_v2":
    URDF = DELIVERY / "drake" / "object.urdf"
    PARTS = DELIVERY / "geometry" / "parts"
    VISUALS = DELIVERY / "drake" / "visuals"
    COLLISIONS = DELIVERY / "drake" / "collisions"
    SCAN = None
elif LAYOUT == "minimal":
    URDF = DELIVERY / "drake" / "object.urdf"
    PARTS = DELIVERY / "geometry" / "parts"
    VISUALS = COLLISIONS = None
    SCAN = None
else:
    VISUALS = COLLISIONS = None
    URDF = (DELIVERY
            / "issacimport/issac_sim_urdf/desk_lamp_refined_v1/urdf/object.urdf")
    PARTS = None
    SCAN = (DELIVERY / "DATA/desk_lamp_20260810_094142_gs2mesh/mesh"
            / "mesh_refinement_click_v1")

# 실측 GT — 램프를 분해해 부위별로 저울과 부피 측정.
# 스캔의 link 이름과 사람이 부르는 이름의 대응은 형상으로 확정했다.
#   link_2  AABB 282x190x88, 도심 z 가장 낮음, 뿌리에 용접  -> 베이스
#   link_3  AABB  70x256x274, 가늘고 김                     -> 연결부(Arm)
#   link_1  AABB 196x 90x148, 도심 z 가장 높음, 사슬 끝     -> Head
GROUND_TRUTH = {
    "link_2": dict(label="베이스", mass_kg=0.396, material_cm3=290.20),
    "link_3": dict(label="연결부(Arm)", mass_kg=0.082, material_cm3=72.89),
    "link_1": dict(label="Head", mass_kg=0.084, material_cm3=52.82),
}

# 색이 없는 배달물을 만났을 때만 쓰는 짐작값 (재질에서 유추).
FALLBACK_COLORS = {"link_2": (0.16, 0.17, 0.19, 1.0),   # 베이스: 무광 검정
                   "link_3": (0.62, 0.63, 0.65, 1.0),   # 연결부: 알루미늄
                   "link_1": (0.93, 0.93, 0.90, 1.0)}   # 갓: 미색

# Drake 의 Mesh/Convex 는 .obj 를 받고 스캔은 .ply/.stl 을 내므로 한 번 바꿔
# 캐시한다. 화면용은 분할 메시 하나, 충돌용은 **볼록 조각들** 이다.
#
# 충돌에 AABB 상자를 쓰면 안 된다. 램프 Arm 은 AABB 가 70x256x274 mm 인데
# 실제 재료는 그 1/3 이라, 상자로 보면 갈 수 있는 자세를 과하게 막는다.
CACHE = Path(__file__).resolve().parent / "cache_lamp_obj"

_MESH_CACHE = {}

# 스캔 색에는 촬영 조명의 색이 함께 구워져 있다. 이 배달물은 평균이
# (0.56, 0.49, 0.61) 로 보랏빛이 돈다. WHITE_BALANCE 를 켜면 물체 전체의
# 평균이 무채색이 되도록 채널을 나눠 (회색 세계 가정) 그 치우침을 뺀다.
# 밝고 어두운 차이는 그대로 남는다. 기본은 꺼짐 — 받은 자료 그대로 본다.
# 켜려면 --white-balance 또는 DESK_LAMP_WHITE_BALANCE=1 (다른 화면에서도 먹는다).
WHITE_BALANCE = os.environ.get(
    "DESK_LAMP_WHITE_BALANCE", "").lower() not in ("", "0", "false", "no")
_GAIN = None


def _raw_mesh(path):
    vertices, faces = mp.read_mesh(path)
    colors = (mp.read_ply_colors(path)
              if str(path).lower().endswith(".ply") else None)
    return vertices, faces, colors


def _mesh_path(name):
    if LAYOUT in ("minimal", "minimal_v2"):
        hits = sorted(PARTS.glob(f"{name}_*rgb.ply")) or sorted(
            PARTS.glob(f"{name}_*.ply"))
        return hits[0] if hits else DELIVERY / "drake" / "meshes" / f"{name}.stl"
    return SCAN / f"segment_mesh/{name}.ply"


def _white_balance_gain():
    """물체 전체 평균을 무채색으로 만드는 채널 이득."""
    global _GAIN
    if _GAIN is None:
        stack = [c for c in (_raw_mesh(_mesh_path(f"link_{i}"))[2]
                             for i in (1, 2, 3)) if c is not None]
        if not stack:
            _GAIN = np.ones(3)
        else:
            mean = np.vstack(stack).astype(float).mean(axis=0)
            _GAIN = mean.mean() / np.maximum(mean, 1e-6)
    return _GAIN


def part_mesh(name):
    """파트 메시 (정점, 면, 정점색 or None). 배달물 구조를 알아서 맞춘다."""
    key = (name, WHITE_BALANCE)
    if key in _MESH_CACHE:
        return _MESH_CACHE[key]
    vertices, faces, colors = _raw_mesh(_mesh_path(name))
    if colors is not None and WHITE_BALANCE:
        colors = np.clip(np.asarray(colors, float) * _white_balance_gain(),
                         0.0, 255.0)
    _MESH_CACHE[key] = (vertices, faces, colors)
    return _MESH_CACHE[key]


def part_color(name):
    """색 조각을 못 쓸 때의 대표색 RGBA (정점 색의 중앙값)."""
    colors = part_mesh(name)[2]
    if colors is None:
        return FALLBACK_COLORS[name]
    median = np.median(np.asarray(colors, dtype=float), axis=0) / 255.0
    return tuple(float(x) for x in median) + (1.0,)


def scan_color(name):
    """정점 색의 중앙값 (0~1). 색이 없으면 None. 보고용."""
    colors = part_mesh(name)[2]
    if colors is None:
        return None
    return tuple(np.median(np.asarray(colors, dtype=float), axis=0) / 255.0)


def collision_meshes(name):
    """충돌용 볼록 조각 경로들.

    minimal 배달물에는 볼록 분해가 없다. 예전 배달물에서 구워 둔 캐시가
    있으면 그것을 쓴다 — 두 배달물의 링크 메시가 정점 수·면 수·부피·도심
    까지 같은 것을 확인했다 (link_1 441.02 cm^3, link_2 165.70, link_3
    219.58 로 일치). 캐시도 없으면 링크 메시 하나를 통째로 넘기는데,
    Drake 의 Convex 가 볼록 껍질로 감싸므로 굽은 팔이 실제보다 굵어진다.
    """
    if LAYOUT == "minimal_v2":
        # 배달물이 볼록 분해를 직접 준다. 우리가 만들 것이 없다.
        pieces = sorted((COLLISIONS / name).glob("part_*.obj"))
        if pieces:
            return tuple(str(p) for p in pieces)
    cached = sorted(CACHE.glob(f"{name}_convex_*.obj"),
                    key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    if cached:
        return tuple(str(p) for p in cached)
    if LAYOUT != "minimal":
        groups = json.load(open(SCAN / "attributes/click_groups.json"))
        pieces = []
        for k in groups[f"part_{name.rsplit('_', 1)[1]}"]:
            piece = CACHE / f"{name}_convex_{k}.obj"
            if not piece.exists():
                v, f = mp.read_mesh(SCAN / f"meshes/decomposed_{k}.ply")
                mp.write_obj(piece, v, f, f"{name}_{k}")
            pieces.append(str(piece))
        return tuple(pieces)
    print(f"  [주의] {name}: 볼록 분해 조각이 없어 링크 메시 하나를 볼록"
          " 껍질로 쓴다. 충돌 검사가 실제보다 보수적이 된다.")
    hull = CACHE / f"{name}_hull.obj"
    if not hull.exists():
        vertices, faces, _ = part_mesh(name)
        mp.write_obj(hull, vertices, faces, f"{name}_hull")
    return (str(hull),)


N_COLOR_PIECES = 6      # 파트 하나를 몇 가지 색으로 나눠 그릴지


def ensure_obj(pitch=0.002, n_color=N_COLOR_PIECES):
    """스캔 메시를 Drake 가 읽는 .obj 로 바꿔 캐시하고 경로를 돌려준다.

    정점 색이 있으면 색이 비슷한 면끼리 묶어 조각 .obj 로도 굽는다.
    Drake 는 메시 하나에 단색 하나만 입히므로, 스캔 색을 화면에 내려면
    이렇게 나누는 수밖에 없다 (mesh_props.split_by_color 의 설명 참고).
    """
    CACHE.mkdir(exist_ok=True)
    out = {}
    for index in (1, 2, 3):
        name = f"link_{index}"
        if LAYOUT == "minimal_v2":
            # 배달물이 화면용 glTF 를 준다. 정점 색(COLOR_0)이 들어 있어
            # Meshcat 이 그대로 그린다 — 색 무리로 쪼갤 필요가 없다.
            hits = sorted(VISUALS.glob(f"{name}_*.gltf"))
            if hits:
                out[name] = dict(visual=str(hits[0]), pieces=(),
                                 collision=collision_meshes(name))
                continue
        vertices, faces, colors = part_mesh(name)
        tag = f"{n_color}wb" if WHITE_BALANCE else f"{n_color}"
        visual = CACHE / (f"{name}_visual_rgb.obj" if colors is not None
                          else f"{name}_visual.obj")
        if not visual.exists():
            mp.write_obj(visual, vertices, faces, name, colors=colors)

        pieces = ()
        if colors is not None and n_color > 1:
            built = []
            for j in range(n_color):
                path = CACHE / f"{name}_shade_{tag}_{j}.obj"
                meta = CACHE / f"{name}_shade_{tag}_{j}.json"
                if not (path.exists() and meta.exists()):
                    built = None
                    break
                built.append((str(path), tuple(json.load(open(meta)))))
            if built is None:
                built = []
                split = mp.split_by_color(vertices, faces, colors, k=n_color)
                for j, (v, f, n, rgba) in enumerate(split):
                    path = CACHE / f"{name}_shade_{tag}_{j}.obj"
                    meta = CACHE / f"{name}_shade_{tag}_{j}.json"
                    mp.write_obj(path, v, f, f"{name}_shade_{j}", normals=n)
                    meta.write_text(json.dumps(list(rgba)))
                    built.append((str(path), rgba))
            pieces = tuple(built)

        out[name] = dict(visual=str(visual), pieces=pieces,
                         collision=collision_meshes(name))
    return out


# ---------------------------------------------------------------------------
def read_urdf(path=URDF):
    """링크 프레임 원점(스캔 좌표)과 관절 정보를 뽑는다.

    이 URDF 는 각 링크의 메시를 '스캔 좌표 그대로' 두고, 링크 프레임만 관절
    축 위로 옮겨 놓았다. visual/origin 이 그 옮긴 양의 음수다.
    """
    root = ET.parse(path).getroot()
    frame_origin = {}
    for link in root.findall("link"):
        visual = link.find("visual/origin")
        if visual is None:
            continue
        offset = np.array([float(v) for v in visual.get("xyz").split()])
        frame_origin[link.get("name")] = -offset      # 스캔좌표 기준 프레임 원점

    joints = []
    for joint in root.findall("joint"):
        if joint.get("type") != "revolute":
            continue
        limit = joint.find("limit")
        joints.append(dict(
            name=joint.get("name"),
            parent=joint.find("parent").get("link"),
            child=joint.find("child").get("link"),
            origin=np.array([float(v)
                             for v in joint.find("origin").get("xyz").split()]),
            axis=np.array([float(v)
                           for v in joint.find("axis").get("xyz").split()]),
            limits=(float(limit.get("lower")), float(limit.get("upper")))))
    return frame_origin, joints


def part_geometry(pitch=0.002):
    """파트별 부피·도심·관성 (스캔 좌표계).

    분할 메시가 닫혀 있으면 거기서 바로 정확히 구한다. 열려 있으면
    (0812 배달물) 볼록 조각의 합집합으로 우회한다.
    """
    out = {}
    for index in (1, 2, 3):
        name = f"link_{index}"
        vertices, faces, _ = part_mesh(name)
        if mp.watertight_report(vertices, faces)["watertight"]:
            props = mp.mass_properties(vertices, faces)
            props["aabb"] = (vertices.min(axis=0), vertices.max(axis=0))
            out[name] = props
        else:
            groups = json.load(open(SCAN / "attributes/click_groups.json"))
            pieces = [mp.read_mesh(SCAN / f"meshes/decomposed_{k}.ply")
                      for k in groups[f"part_{index}"]]
            out[name] = mp.union_properties(pieces, pitch=pitch)
    return out


def grasp_width_mm(name, centroid_scan, slab_mm=25.0):
    """파지점 근처의 실제 단면 폭 [mm].

    AABB 는 굽은 팔의 바깥 상자라 실제 물어야 할 폭보다 훨씬 크다.
    도심을 지나는 얇은 판(slab) 안의 점들만 보고 폭을 잰다. 세 축 중
    가장 좁은 방향이 죠가 물기 좋은 방향이다.
    """
    vertices = part_mesh(name)[0]
    widths, axes = [], []
    for axis in range(3):
        near = np.abs(vertices[:, axis] - centroid_scan[axis]) <= slab_mm * 1e-3
        if near.sum() < 3:
            widths.append(np.ptp(vertices[:, axis]))
            axes.append(axis)
            continue
        others = [k for k in range(3) if k != axis]
        spans = [np.ptp(vertices[near][:, k]) for k in others]
        widths.append(min(spans))
        # 그 폭이 **어느 축 방향** 인지도 남긴다. 죠를 그 축에 맞춰야
        # 실제로 물린다 (안 그러면 긴 축을 오므리려 든다).
        axes.append(others[int(np.argmin(spans))])
    best = int(np.argmin(widths))
    return float(1000.0 * widths[best]), int(axes[best])


def pinch_grasp(name, min_width_mm=12.0, max_width_mm=70.0):
    """그리퍼가 **실제로 물 수 있는 자리**를 볼록 조각에서 고른다.

    도심을 파지점으로 쓰면 안 되는 경우가 있다. 굽은 팔의 도심은 재료 한가운데
    있어도 그 자리의 단면 중심은 아니어서, 죠를 오므리면 한쪽 패드가 10 mm 씩
    허공을 문다 (실제로 그랬다).

    볼록 조각 하나하나는 팔을 토막 낸 것이라 거의 원기둥이다. 그러니
      - 조각의 중심   -> 재료 한가운데
      - 가장 좁은 주축 -> 죠가 물 방향, 그 폭이 곧 필요한 개구량
      - 가장 긴 주축   -> 손목 축을 따라 뻗을 방향
    으로 삼으면 실제로 물리는 자세가 된다.

    너무 얇은 조각(모서리 파편)은 거른다. 남은 것 중 **가장 긴** 조각을
    고른다 — 팔의 몸통이고, 패드가 길이 방향으로 물기 좋다.

    돌려주는 것: dict(point, jaw_axis, long_axis, width_mm)  — 스캔 좌표계.
    """
    best = None
    for path in collision_meshes(name):
        vertices = np.array([[float(t) for t in line.split()[1:4]]
                             for line in open(path) if line.startswith("v ")])
        if len(vertices) < 4:
            continue
        center = vertices.mean(axis=0)
        _, _, axes = np.linalg.svd(vertices - center, full_matrices=False)
        spans = [(float(np.ptp((vertices - center) @ axes[k])), k)
                 for k in range(3)]
        spans.sort()
        # 폭은 **파지점 근처** 에서 잰다. 조각 전체의 최대 폭을 쓰면 패드가
        # 실제 단면보다 넓게 벌어져 물체가 죠 사이에서 논다.
        along = (vertices - center) @ axes[spans[2][1]]
        near = vertices[np.abs(along) <= 0.015]
        if len(near) < 4:
            near = vertices
        width = 1000.0 * float(np.ptp((near - center) @ axes[spans[0][1]]))
        if not (min_width_mm <= width <= max_width_mm):
            continue
        length = spans[2][0]
        if best is None or length > best["length"]:
            best = dict(point=center, jaw_axis=axes[spans[0][1]],
                        long_axis=axes[spans[2][1]], width_mm=width,
                        length=length, piece=str(path))
    return best


def rerooted_joints(joint_rows, root):
    """트리를 root 기준으로 다시 세운다.

    스캔 URDF 는 base -> link_2 -> link_3 -> link_1 사슬이라 link_2 가 뿌리다.
    로봇이 link_3(연결부)를 잡으면 link_3 가 뿌리가 되고 link_2 는 자식이
    된다. 그러면 그 관절의 방향이 뒤집힌다.

    뒤집을 때 바뀌는 것
      부모/자식   서로 바꾼다
      축          -n  (같은 물리 회전을 같은 부호의 각으로 표현하려면)
      관절 위치   원래 자식 프레임 원점이 관절 축이었으므로
                  새 부모 프레임에서는 (0,0,0),
                  새 자식(원래 부모) 프레임에서는 원래 origin
    """
    edges = {}
    for row in joint_rows:
        edges.setdefault(row["parent"], []).append((row, False))
        edges.setdefault(row["child"], []).append((row, True))

    order, out, seen = [root], [], {root}
    queue = [root]
    while queue:
        node = queue.pop(0)
        for row, reversed_edge in edges.get(node, []):
            other = row["parent"] if reversed_edge else row["child"]
            if other in seen:
                continue
            seen.add(other); order.append(other); queue.append(other)
            if reversed_edge:
                out.append(dict(name=row["name"], parent=node, child=other,
                                origin=np.zeros(3),
                                origin_in_child=row["origin"],
                                axis=-np.asarray(row["axis"]),
                                limits=row["limits"]))
            else:
                out.append(dict(name=row["name"], parent=node, child=other,
                                origin=np.asarray(row["origin"]),
                                origin_in_child=np.zeros(3),
                                axis=np.asarray(row["axis"]),
                                limits=row["limits"]))
    return order, out


# 로봇이 잡는 자리. 사슬의 뿌리인 link_2(베이스)를 잡는다.
#
#   "centroid"    베이스 도심을 파지점으로 (기본)
#   "base_frame"  URDF 링크 프레임 원점 = 스캔 원점
#
# base_frame 을 기본으로 두면 안 된다. 이 스캔에서 link_2 의 프레임 원점은
# 관절 축 위가 아니라 스캔 좌표 원점이고, 베이스 도심에서 285 mm 떨어진
# 허공이다. 거기를 파지점으로 삼으면 모든 모멘트팔이 실물과 어긋난다.
DEFAULT_GRASP = "pinch"
# 로봇이 잡는 **부위**. link_3 은 연결부(Arm) 로, 단면이 가늘어 그리퍼가
# 실제로 물 수 있는 유일한 부위다 (베이스 88 mm, Head 90 mm vs 개구 53 mm).
# 여기를 잡으면 트리가 link_3 를 뿌리로 다시 세워지고, link_2 와 link_1 이
# 각각 자식이 되는 갈래 구조가 된다.
DEFAULT_GRASP_PART = "link_3"


def build_spec(pitch=0.002, com_error_mm=0.0, seed=0, grasp_at=DEFAULT_GRASP,
               grasp_part=DEFAULT_GRASP_PART):
    """배달물을 ObjectSpec 으로 바꾼다.

    grasp_at 은 로봇이 link_2(베이스)의 어디를 잡는가다. 센서 프레임 원점이
    거기 놓이고, 모든 모멘트팔이 그 점 기준으로 계산된다.

    com_error_mm > 0 이면 **진리 쪽 도심만** 그만큼 흔든다. 속 빈 갓처럼
    겉모양 도심과 재료 도심이 다른 상황을 흉내내기 위한 것이다.
    """
    frame_origin, joint_rows = read_urdf()
    geometry = part_geometry(pitch)
    meshes = ensure_obj(pitch)
    rng = np.random.default_rng(seed)

    # 잡는 부위를 뿌리로 트리를 다시 세운다.
    order, joint_rows = rerooted_joints(joint_rows, grasp_part)

    # 센서(파지점) 원점. 잡는 부위의 도심에 둔다. URDF 링크 프레임 원점은
    # 이 스캔에서 물체 바깥 허공일 수 있어 파지점으로 쓰면 모멘트팔이 어긋난다.
    root = order[0]
    pinch = pinch_grasp(root) if grasp_at == "pinch" else None
    if pinch is None and grasp_at == "pinch":
        print(f"  [주의] {root} 에서 물 만한 볼록 조각을 못 찾아 도심을 씁니다")
    if grasp_at == "base_frame":
        sensor_origin = frame_origin.get(root, np.zeros(3))
    elif pinch is not None:
        sensor_origin = pinch["point"]
    else:
        sensor_origin = geometry[root]["centroid"]

    parts = []
    for name in order:
        info = geometry[name]
        origin = frame_origin.get(name, np.zeros(3))     # 링크 프레임 원점
        centroid = info["centroid"] - origin             # 링크 프레임 기준 도심
        if com_error_mm > 0.0:
            direction = rng.normal(size=3)
            centroid = centroid + com_error_mm * 1e-3 * (
                direction / np.linalg.norm(direction))
        size = (info["aabb"][1] - info["aabb"][0]) * 1000.0
        gt = GROUND_TRUTH[name]
        width_mm, width_axis = grasp_width_mm(name, info["centroid"])
        if pinch is not None and name == root:
            # 잡는 부위는 볼록 조각에서 잰 값을 쓴다 (실제로 물리는 자리).
            width_mm = pinch["width_mm"]
            width_axis = pinch["jaw_axis"]
            long_axis = pinch["long_axis"]
        else:
            long_axis = None
        # 메시는 스캔 좌표에 있고 몸체 프레임 원점은 도심이다.
        # 따라서 몸체 프레임 -> 메시 좌표 평행이동 = -도심(스캔 좌표).
        parts.append(Part(
            name=name,
            bbox_mm=tuple(size),
            volume_cm3=1e6 * info["volume"],
            # 유효 밀도 = 실측 질량 / 스캔 부피. 물리 밀도가 아니다.
            rho_gt=gt["mass_kg"] / info["volume"],
            bbox_center_in_link_mm=tuple(centroid * 1000.0),
            shell_centroid_in_link_mm=tuple(centroid * 1000.0),
            color=part_color(name),
            inertia_unit=info["inertia"] / info["volume"],
            grasp_width_mm=width_mm,
            grasp_axis=width_axis,
            grasp_long_axis=long_axis,
            visual_mesh=meshes[name]["visual"],
            visual_pieces=meshes[name]["pieces"],
            collision_meshes=meshes[name]["collision"],
            mesh_offset_m=tuple(-info["centroid"])))

    joints = []
    for row in joint_rows:
        joints.append(Joint(
            name=row["name"], parent=row["parent"], child=row["child"],
            origin_in_parent_link_mm=tuple(row["origin"] * 1000.0),
            origin_in_child_link_mm=tuple(row["origin_in_child"] * 1000.0),
            axis=tuple(row["axis"] / np.linalg.norm(row["axis"])),
            limits_rad=row["limits"]))

    base_centroid = np.array(parts[0].bbox_center_in_link_mm) / 1000.0
    base_frame = frame_origin.get(root, np.zeros(3))
    return ObjectSpec(
        key="desklamp",
        label="desk lamp (real scan, mesh refinement v1)",
        parts=parts, joints=joints,
        base_bbox_center_in_sensor_mm=tuple(
            (base_frame + base_centroid - sensor_origin) * 1000.0),
        notes="실물 스캔. 밀도는 겉모양 부피로 나눈 유효 밀도이며 물리 밀도가 아니다.")


# ---------------------------------------------------------------------------
def report(spec):
    print(f"{spec.label}")
    print(f"  사슬  {' -> '.join(p.name for p in spec.parts)}")
    print(f"\n  {'링크':<8}{'이름':<12}{'스캔부피':>10}{'실측부피':>10}"
          f"{'비':>6}{'실측질량':>10}{'유효밀도':>11}")
    for part in spec.parts:
        gt = GROUND_TRUTH[part.name]
        print(f"  {part.name:<8}{gt['label']:<12}"
              f"{part.volume_cm3:>9.1f}{gt['material_cm3']:>10.1f}"
              f"{part.volume_cm3/gt['material_cm3']:>6.2f}"
              f"{1000*gt['mass_kg']:>9.1f} g{part.rho_gt:>10.1f}")
    total_scan = sum(p.volume_cm3 for p in spec.parts)
    total_gt = sum(GROUND_TRUTH[p.name]["material_cm3"] for p in spec.parts)
    total_m = sum(GROUND_TRUTH[p.name]["mass_kg"] for p in spec.parts)
    print(f"  {'합계':<20}{total_scan:>9.1f}{total_gt:>10.1f}"
          f"{total_scan/total_gt:>6.2f}{1000*total_m:>9.1f} g")
    print(f"\n  관절")
    for joint in spec.joints:
        print(f"    {joint.name:<12}{joint.parent} -> {joint.child}"
              f"  축 {np.round(joint.axis, 3)}"
              f"  범위 {np.round(np.degrees(joint.limits_rad), 0)} deg")
    print(f"\n  센서(파지점) 기준 베이스 도심"
          f" {np.round(spec.base_bbox_center_in_sensor_mm, 1)} mm")


def identifiability(spec, n_pose=6, seed=0):
    """세 파트의 질량이 원리상 구분되는가."""
    import density_id_drake as alg
    import design_core as dc

    rng = np.random.default_rng(seed)
    bounds = [j.limits_rad for j in spec.joints]
    thetas = [np.array([rng.uniform(lo, hi) for lo, hi in bounds])
              for _ in range(n_pose)]
    rows = []
    weight = 1.0 / np.sqrt(np.tile(alg.R_EPS_DIAG, len(dc.CANONICAL_TRIAD)))
    for theta in thetas:
        rows.append(dc.regressor(theta, dc.CANONICAL_TRIAD) * weight[:, None])
    C = np.vstack(rows)
    scale = np.array([p.rho_gt for p in spec.parts]) * 0.01   # 1 % 흔들었을 때
    svals = np.linalg.svd(C * scale, compute_uv=False)
    return svals


def validate(spec, target=0.01, max_rounds=12, seeds=8, com_error_mm=0.0,
             pitch=0.002, verbose=True):
    """폐루프를 돌리고 **질량** 으로 채점한다.

    밀도로 채점하면 안 된다. 스캔 부피가 재료 부피와 다르므로 밀도는 유효값
    이고, 우리가 실제로 쓰는 것도 시뮬레이터에 넣을 질량이다.
    """
    import density_id_drake as alg
    import density_id_objects as obj
    import design_core as dc

    volumes = np.array([p.volume_m3 for p in spec.parts])
    gt_mass = np.array([GROUND_TRUTH[p.name]["mass_kg"] for p in spec.parts])

    rounds, mass_err, converged, claimed = [], [], [], []
    for seed in range(seeds):
        obj.set_measurement_averaging()
        if com_error_mm > 0.0:
            # 진리 쪽만 도심을 흔든다. 추정기는 스캔 도심을 그대로 믿는다.
            truth_spec = build_spec(pitch=pitch, com_error_mm=com_error_mm,
                                    seed=1000 + seed)
            obj.bind_object(truth_spec)
            truth = (alg.TRUTH_PLANT, alg.TRUTH_CTX)
            obj.bind_object(spec)
            alg.TRUTH_PLANT, alg.TRUTH_CTX = truth
        else:
            obj.bind_object(spec)
        obj.apply_weight_prior(spec, float(gt_mass.sum()))
        result = dc.closed_loop(spec, target=target, max_rounds=max_rounds,
                                seed=seed, verbose=False)
        mass = result["rho_hat"] * volumes
        rounds.append(result["rounds"])
        converged.append(result["converged"])
        claimed.append(result["worst"])
        mass_err.append(np.abs(mass - gt_mass) / gt_mass)
    mass_err = np.array(mass_err)
    if verbose:
        print(f"\n폐루프 검증  (목표 반폭 {100*target:.1f}%, "
              f"최대 {max_rounds}라운드, seed {seeds}개"
              + (f", 도심오차 {com_error_mm:.0f} mm" if com_error_mm else "")
              + ")")
        print(f"  라운드 평균 {np.mean(rounds):.1f}   "
              f"수렴 {sum(converged)}/{seeds}   "
              f"주장 반폭 {100*np.mean(claimed):.2f}%")
        print(f"  {'링크':<8}{'이름':<12}{'GT 질량':>10}"
              f"{'평균 오차':>11}{'최악 seed':>11}")
        for k, part in enumerate(spec.parts):
            gt = GROUND_TRUTH[part.name]
            print(f"  {part.name:<8}{gt['label']:<12}{1000*gt['mass_kg']:>9.1f} g"
                  f"{100*mass_err[:, k].mean():>10.2f}%"
                  f"{100*mass_err[:, k].max():>10.2f}%")
        print(f"  {'':<20}{'파트 최대':>10}"
              f"{100*mass_err.max(axis=1).mean():>10.2f}%"
              f"{100*mass_err.max():>10.2f}%")
    return mass_err, rounds, converged


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pitch-mm", type=float, default=2.0)
    ap.add_argument("--target", type=float, default=0.01)
    ap.add_argument("--max-rounds", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--com-error-mm", type=float, nargs="*", default=[0.0],
                    help="진리 도심을 흔들어 '겉모양 도심 != 재료 도심' 을 흉내")
    ap.add_argument("--white-balance", action="store_true",
                    help="스캔 색의 조명 치우침을 뺀다 (회색 세계 가정)")
    args = ap.parse_args()

    WHITE_BALANCE = args.white_balance
    spec = build_spec(pitch=args.pitch_mm / 1000.0)
    report(spec)

    import density_id_objects as obj
    obj.set_measurement_averaging()
    obj.bind_object(spec)
    svals = identifiability(spec)
    print(f"\n식별성 (각 질량을 1 % 흔들었을 때 특이값)")
    print(f"  {np.round(svals, 3)}   조건수 {svals[0]/svals[-1]:.1f}")
    print(f"  -> 마지막 값이 0 에 가까우면 그 조합은 아무리 재도 구분 안 된다")

    for com in args.com_error_mm:
        validate(spec, target=args.target, max_rounds=args.max_rounds,
                 seeds=args.seeds, com_error_mm=com,
                 pitch=args.pitch_mm / 1000.0)
