"""그리퍼를 갈아끼울 수 있게 모아 둔 곳.

    pgc140       DH PGC-140-50   개구 53 mm  (기존)
    robotiq2f85  Robotiq 2F-85   개구 78 mm

왜 바꿀 수 있어야 하나
----------------------
잡을 수 있는 물체가 개구량으로 갈린다. 스탠드 램프는 가장 가는 부위인
연결부(Arm)조차 단면이 70 mm 라 PGC 로는 못 잡는다.

    부위                최소 단면    PGC 53mm   Robotiq 78mm
    램프 연결부(Arm)      70 mm         X            O
    램프 베이스           88 mm         X            X
    램프 Head             90 mm         X            X
    3-link                44 mm         O            O
    2-link                46 mm         O            O

파지 모형
---------
이 파이프라인은 파지를 접촉으로 풀지 않는다. 물체를 파지점에 **용접**해
붙인다. 측정은 정지 상태의 중력 렌치만 쓰므로 접촉력이 필요 없기 때문이다.
그래서 그리퍼에서 실제로 쓰이는 것은 네 가지뿐이다.

    1) 충돌 형상      팔·테이블과 부딪히는지
    2) TCP 위치       파지점이 손목에서 얼마나 떨어져 있는지
    3) 질량           준정적 판정에 쓰인다
    4) 개구량         그 물체를 물 수 있는지

Robotiq 은 4절링크라 개구량과 관절각이 비례하지 않는다. 그래서 추측하지 않고
Drake 로 정기구학을 돌려 **패드 사이 실제 간격**을 재서 표로 넣었다.
(third_party/robotiq_arg85_description, 콜리전 메시의 볼록 껍질 기준)
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[1]
ROBOTIQ_ROOT = WORKSPACE / "third_party" / "robotiq_arg85_description"
ROBOTIQ_CACHE = Path(__file__).resolve().parent / "cache_robotiq_obj"

# URDF 에서 읽은 mimic 계수. finger_joint 하나가 나머지를 끌고 간다.
ROBOTIQ_MIMIC = {"finger_joint": 1.0,
                 "left_inner_knuckle_joint": 1.0,
                 "left_inner_finger_joint": -1.0,
                 "right_inner_knuckle_joint": -1.0,
                 "right_inner_finger_joint": 1.0,
                 "right_outer_knuckle_joint": -1.0}

# Drake 로 실측한 표 (measure_robotiq() 로 다시 뽑을 수 있다).
#   각도 [rad] -> 패드 사이 간격 [m], 패드 중점의 z [m]
ROBOTIQ_ANGLE = np.array([0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35,
                          0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70])
ROBOTIQ_GAP_M = np.array([0.0785, 0.0738, 0.0690, 0.0641, 0.0590, 0.0538,
                          0.0484, 0.0430, 0.0375, 0.0319, 0.0263, 0.0206,
                          0.0149, 0.0092, 0.0035])
# 패드 **면의 중심** 높이다. 예전에는 손가락 링크의 몸체 원점을 썼는데
# (0.115~0.126), 이 URDF 는 원점이 패드에서 39 mm 아래라 물체가 손가락
# 위에 떠 붙었다. 화면으로 확인하고 고친 값이다.
ROBOTIQ_TCP_Z_M = np.array([0.1365, 0.1381, 0.1397, 0.1411, 0.1424, 0.1435,
                            0.1446, 0.1455, 0.1462, 0.1469, 0.1473, 0.1477,
                            0.1479, 0.1479, 0.1478])


def robotiq_angle_for_opening(opening_m):
    """원하는 개구량을 만드는 finger_joint 각도 [rad]."""
    opening = float(np.clip(opening_m, ROBOTIQ_GAP_M.min(), ROBOTIQ_GAP_M.max()))
    return float(np.interp(opening, ROBOTIQ_GAP_M[::-1], ROBOTIQ_ANGLE[::-1]))


def robotiq_tcp_z(opening_m):
    """그 개구량에서 파지점(패드 중점)의 z [m]."""
    return float(np.interp(robotiq_angle_for_opening(opening_m),
                           ROBOTIQ_ANGLE, ROBOTIQ_TCP_Z_M))


def ensure_robotiq_obj():
    """STL 을 Drake 가 읽는 OBJ 로 바꿔 캐시한다 (법선 포함)."""
    import mesh_props as mp

    ROBOTIQ_CACHE.mkdir(exist_ok=True)
    for stl in sorted((ROBOTIQ_ROOT / "meshes").glob("*.STL")):
        target = ROBOTIQ_CACHE / (stl.stem + ".obj")
        if not target.exists():
            vertices, faces = mp.read_mesh(stl)
            mp.write_obj(target, vertices, faces, stl.stem)
    return ROBOTIQ_CACHE


def robotiq_urdf_string(opening_m, include_visuals=True, model_name="robotiq"):
    """지정한 개구량에서 굳힌 Robotiq 2F-85.

    관절을 전부 fixed 로 바꾼다. 파지가 용접으로 모형화되므로 손가락이
    움직일 필요가 없고, mimic 을 그대로 두면 파서와 IK 가 복잡해진다.
    """
    if not ROBOTIQ_ROOT.is_dir():
        raise FileNotFoundError(
            f"Robotiq 기술서를 찾지 못했습니다: {ROBOTIQ_ROOT}\n"
            "  git clone https://github.com/a-price/robotiq_arg85_description "
            f"{ROBOTIQ_ROOT}")
    cache = ensure_robotiq_obj()
    angle = robotiq_angle_for_opening(opening_m)

    root = ET.parse(ROBOTIQ_ROOT / "robots/robotiq_arg85_description.URDF").getroot()
    root.set("name", model_name)
    for mesh in root.iter("mesh"):
        stem = Path(mesh.get("filename")).stem
        # 화면은 fine, 충돌은 coarse 를 쓴다 (URDF 가 이미 그렇게 나눠 놨다)
        mesh.set("filename", "file://" + str((cache / (stem + ".obj")).resolve()))
    for link in root.findall("link"):
        if not include_visuals:
            for visual in list(link.findall("visual")):
                link.remove(visual)
    for joint in root.findall("joint"):
        name = joint.get("name")
        multiplier = ROBOTIQ_MIMIC.get(name, 0.0)
        if multiplier and joint.get("type") == "revolute":
            # 회전을 origin 의 rpy 로 구워 넣고 관절을 굳힌다.
            axis = joint.find("axis")
            direction = np.fromstring(axis.get("xyz"), sep=" ")
            direction = direction / np.linalg.norm(direction)
            origin = joint.find("origin")
            if origin is None:
                origin = ET.SubElement(joint, "origin", xyz="0 0 0")
            rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
            # 이 URDF 의 관절 축은 모두 y 축이라 pitch 하나로 표현된다.
            rpy[1] += multiplier * angle * direction[1] * -1.0
            origin.set("rpy", " ".join(f"{v:.9g}" for v in rpy))
        joint.set("type", "fixed")
        for tag in ("axis", "limit", "mimic", "dynamics"):
            for child in list(joint.findall(tag)):
                joint.remove(child)
    return ET.tostring(root, encoding="unicode")


@dataclass(frozen=True)
class GripperSpec:
    key: str
    label: str
    base_frame: str            # 마운트에 용접할 프레임 이름
    body_names: tuple          # 충돌 필터에서 '붙어 있는 것이 정상' 인 몸체들
    max_opening_m: float
    finger_joint_names: tuple = ()      # 비어 있으면 손가락이 굳어 있다는 뜻
    tcp_z_m: float = 0.125


GRIPPERS = {
    "pgc140": GripperSpec(
        key="pgc140", label="DH PGC-140-50",
        base_frame="base_link",
        body_names=("base_link", "finger1_link", "finger2_link"),
        max_opening_m=0.053,
        finger_joint_names=("finger1_joint", "finger2_joint"),
        tcp_z_m=0.125),
    "robotiq2f85": GripperSpec(
        key="robotiq2f85", label="Robotiq 2F-85",
        base_frame="robotiq_85_base_link",
        body_names=("robotiq_85_base_link",
                    "left_outer_knuckle", "left_outer_finger",
                    "left_inner_knuckle", "left_inner_finger",
                    "right_outer_knuckle", "right_outer_finger",
                    "right_inner_knuckle", "right_inner_finger"),
        max_opening_m=float(ROBOTIQ_GAP_M.max()),
        finger_joint_names=(),
        tcp_z_m=float(ROBOTIQ_TCP_Z_M[0])),
}


def _pad_face_z(plant, context, fingers, cache, band_mm=2.0):
    """두 패드가 마주보는 면의 중심 높이 [m].

    각 손가락 메시를 월드로 옮긴 뒤, 죠 축(x=0) 에 가장 가까운 정점들만
    (band_mm 안) 골라 그 평균 높이를 쓴다. 이 면이 물체에 닿는 자리다.
    """
    heights = []
    for body in fingers:
        X_WB = plant.EvalBodyPoseInWorld(context, body)
        vertices = []
        for name in ("inner_finger_coarse", "inner_finger_fine"):
            path = cache / f"{name}.obj"
            if path.is_file():
                for line in path.read_text().splitlines():
                    if line.startswith("v "):
                        vertices.append([float(v) for v in line.split()[1:4]])
                break
        if not vertices:
            return float("nan")
        world = np.array([X_WB @ np.asarray(v, dtype=float) for v in vertices])
        inner = np.abs(world[:, 0]).min()
        band = world[np.abs(world[:, 0]) <= inner + band_mm * 1e-3]
        # 평균이 아니라 **가운데**를 쓴다. 메시 정점이 위쪽에 몰려 있어
        # 평균을 쓰면 TCP 가 패드 위쪽으로 15 mm 밀린다.
        heights.append(0.5 * float(band[:, 2].min() + band[:, 2].max()))
    return float(np.mean(heights))


def measure_robotiq(n=17):
    """표를 다시 뽑는다. URDF 를 바꾸면 이걸 돌려 위 상수를 갱신한다."""
    from pydrake.multibody.parsing import Parser
    from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
    from pydrake.systems.framework import DiagramBuilder

    cache = ensure_robotiq_obj()
    root = ET.parse(ROBOTIQ_ROOT / "robots/robotiq_arg85_description.URDF").getroot()
    for mesh in root.iter("mesh"):
        stem = Path(mesh.get("filename")).stem
        mesh.set("filename", "file://" + str((cache / (stem + ".obj")).resolve()))
    for joint in root.findall("joint"):
        for mimic in list(joint.findall("mimic")):
            joint.remove(mimic)
        for limit in joint.findall("limit"):
            limit.set("lower", "-1.0"); limit.set("upper", "1.0")

    builder = DiagramBuilder()
    plant, _ = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    model = Parser(plant).AddModelsFromString(
        ET.tostring(root, encoding="unicode"), "urdf")[0]
    plant.WeldFrames(plant.world_frame(),
                     plant.GetFrameByName("robotiq_85_base_link", model))
    plant.Finalize()
    context = plant.GetMyContextFromRoot(builder.Build().CreateDefaultContext())

    left = plant.GetBodyByName("left_inner_finger", model)
    right = plant.GetBodyByName("right_inner_finger", model)
    left_ids = set(plant.GetCollisionGeometriesForBody(left))
    right_ids = set(plant.GetCollisionGeometriesForBody(right))

    rows = []
    for angle in np.linspace(0.0, 0.75, n):
        q = np.zeros(plant.num_positions())
        for name, multiplier in ROBOTIQ_MIMIC.items():
            q[plant.GetJointByName(name, model).position_start()] = multiplier * angle
        plant.SetPositions(context, q)
        query = plant.get_geometry_query_input_port().Eval(context)
        gap = np.inf
        for pair in query.ComputeSignedDistancePairwiseClosestPoints(1.0):
            if ((pair.id_A in left_ids and pair.id_B in right_ids)
                    or (pair.id_A in right_ids and pair.id_B in left_ids)):
                gap = pair.distance
        # TCP 는 **패드가 서로 마주보는 자리**다. 손가락 링크의 몸체 원점이
        # 아니다. 이 URDF 는 원점이 패드에서 39 mm 아래라, 몸체 원점을 쓰면
        # 물체가 패드보다 그만큼 안쪽에 붙어 화면에서 손가락 위에 떠 보인다.
        #
        # 가장 가까운 두 점을 쓰는 것도 안 된다. 마주보는 두 평면 사이의
        # 최근접점은 그 면 어디여도 되므로 Drake 가 모서리를 돌려주고,
        # 각도를 조금 바꾸면 값이 20 mm 씩 튄다 (실제로 그랬다).
        #
        # 그래서 패드 **면의 중심**을 직접 잰다. 패드 메시에서 맞은편을
        # 향하는 면(= 죠 축에 가장 가까운 x)에 놓인 정점들의 평균 높이다.
        z = _pad_face_z(plant, context, (left, right), cache)
        rows.append((angle, gap, z))
    return rows


if __name__ == "__main__":
    print(f"{'각도[rad]':>10}{'개구[mm]':>11}{'TCP z[mm]':>12}")
    for angle, gap, z in measure_robotiq():
        print(f"{angle:>10.4f}{1000*gap:>11.1f}{1000*z:>12.1f}")
    print("\n등록된 그리퍼")
    for spec in GRIPPERS.values():
        print(f"  {spec.key:<12}{spec.label:<18}"
              f" 개구 {1000*spec.max_opening_m:.0f} mm"
              f"  TCP z {1000*spec.tcp_z_m:.1f} mm")
