"""추정한 부위별 밀도로 sim-ready URDF 를 만든다.

이 파이프라인의 최종 산출물이다. 메시만 아는 상태에서 출발해 로봇이
물체를 들고 몇 번 재고 나면, 부위별 밀도가 확정되고 그것으로부터
시뮬레이터가 바로 쓸 수 있는 관성 정보가 나온다.

    메시(외형 부피)  ->  탐색·측정  ->  밀도  ->  질량 · 무게중심 · 관성텐서
                                                        |
                                                        v
                                                   URDF (sim ready)

균일밀도 가정이므로 각 부위의 무게중심은 외형 중심이고, 관성텐서는
직육면체 공식으로 닫힌 형태가 나온다. 값은 Drake 로 되읽어 검증한다.

실행:
    cd ~/Desktop/PIVOT/my_work
    ../robot_learning/scripts/run_drake_env.sh python export_urdf.py \
        --object 3link --out outputs/estimated_3link.urdf
"""

import argparse
import os
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path
from xml.dom import minidom

import numpy as np

import density_id_drake as alg
import density_id_objects as obj
import design_core as dc

MM = 1e-3


def box_inertia_about_com(mass, dims_m):
    """직육면체의 무게중심 기준 관성텐서 (축 정렬이라 대각)."""
    lx, ly, lz = dims_m
    return mass / 12.0 * np.diag([ly ** 2 + lz ** 2,
                                  lx ** 2 + lz ** 2,
                                  lx ** 2 + ly ** 2])


def part_inertial(part, rho, extras=()):
    """밀도 하나에서 질량·무게중심·관성텐서를 만든다.

    균일밀도이므로 무게중심은 외형 중심이고, 링크 프레임 기준 위치는
    bbox_center_in_link_mm 이다.

    extras 는 이 링크에 얹힌 별도 질량 [(질량[kg], 링크프레임 위치[m]), ...].
    URDF 의 joint 에는 질량을 실을 수 없으므로, 힌지 질량을 자식 링크의
    inertial 에 합쳐 넣는다. 이걸 빼먹으면 시뮬레이터가 힌지 없는 물체를
    돌리게 되고, 애써 추정한 값이 그만큼 틀어진다.
    """
    mass = rho * part.volume_m3
    com = np.array(part.bbox_center_in_link_mm) * MM

    if part.inertia_unit is not None:
        # 스캔 메시에서 온 부위. 도심 기준 **단위질량당** 관성텐서를 이미
        # 들고 있으므로 질량만 곱하면 된다.
        #
        # 여기서 직육면체 공식을 쓰면 안 된다. 램프 Arm 은 AABB 가
        # 70x256x274 mm 인데 실제 재료는 그 1/3 이라, 상자로 보면 관성이
        # 통째로 부풀려진다. 게다가 상자 공식은 관성적(慣性積)을 0 으로
        # 놓는데, 굽은 팔은 링크 축에 정렬돼 있지 않아 실제로 0 이 아니다.
        inertia = mass * np.asarray(part.inertia_unit, dtype=float)
    else:
        inertia = box_inertia_about_com(mass, tuple(d * MM for d in part.bbox_mm))

    if not extras:
        return mass, com, inertia

    # 링크 프레임 원점 기준으로 모아 합친 뒤 다시 무게중심 기준으로 옮긴다.
    total = mass
    moment = mass * com
    about_origin = inertia + mass * (float(com @ com) * np.eye(3)
                                     - np.outer(com, com))
    for extra_mass, position in extras:
        position = np.asarray(position, dtype=float)
        total += extra_mass
        moment += extra_mass * position
        about_origin += extra_mass * (float(position @ position) * np.eye(3)
                                      - np.outer(position, position))
    com_total = moment / total
    inertia_total = about_origin - total * (
        float(com_total @ com_total) * np.eye(3) - np.outer(com_total, com_total))
    return total, com_total, inertia_total


def hinge_extras(spec, rho_hat):
    """링크마다 얹힌 힌지 질량. 힌지는 자식 링크의 프레임 원점(핀 축)에 있다."""
    table = obj.body_table(spec)
    extras = {p.name: [] for p in spec.parts}
    joints = {j.name: j for j in spec.joints}
    for row, rho in zip(table, rho_hat):
        if row["kind"] != "hinge":
            continue
        joint = joints[row["joint"]]
        # 힌지 무게중심은 자식 링크 프레임 원점(핀 축) + 오프셋 위치에 있다.
        extras[joint.child].append(
            (rho * row["volume_m3"],
             np.array(joint.hinge_com_offset_mm) * MM))
    return extras


def build_urdf(spec, rho_hat, Sigma=None, name=None, mesh_dir=None,
               mesh_names=None):
    """추정 밀도로 URDF 트리를 만든다. Sigma 를 주면 불확실성을 주석으로 남긴다."""
    robot = ET.Element("robot", name=name or f"{spec.key}_estimated")

    std = np.sqrt(np.diag(Sigma)) if Sigma is not None else None
    header = [f"부위별 밀도를 로봇 측정으로 추정해 생성한 sim-ready 자산.",
              f"가정: 각 부위 내부 밀도 균일, 측정 중 관절 고정."]
    if std is not None:
        header.append("추정 밀도 [kg/m^3] (95% 구간):")
        for part, rho, sd in zip(spec.parts, rho_hat, std):
            header.append(f"  {part.name}: {rho:.1f} +/- {1.96 * sd:.1f}")
    robot.append(ET.Comment("\n     " + "\n     ".join(header) + "\n  "))

    extras = hinge_extras(spec, rho_hat)
    for index, (part, rho) in enumerate(zip(spec.parts, rho_hat)):
        mass, com, inertia = part_inertial(part, rho, extras[part.name])
        link = ET.SubElement(robot, "link", name=part.name)

        if mesh_dir and mesh_names and part.name in mesh_names:
            for tag in ("visual", "collision"):
                node = ET.SubElement(link, tag)
                geometry = ET.SubElement(node, "geometry")
                ET.SubElement(geometry, "mesh",
                              filename=f"{mesh_dir}/{mesh_names[part.name]}",
                              scale="0.001 0.001 0.001")
        else:
            dims = " ".join(f"{d * MM:.6f}" for d in part.bbox_mm)
            origin = " ".join(f"{v:.6f}" for v in com)
            for tag in ("visual", "collision"):
                node = ET.SubElement(link, tag)
                ET.SubElement(node, "origin", xyz=origin)
                geometry = ET.SubElement(node, "geometry")
                ET.SubElement(geometry, "box", size=dims)

        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "origin",
                      xyz=" ".join(f"{v:.6f}" for v in com))
        ET.SubElement(inertial, "mass", value=f"{mass:.6f}")
        ET.SubElement(inertial, "inertia",
                      ixx=f"{inertia[0, 0]:.9f}", ixy=f"{inertia[0, 1]:.9f}",
                      ixz=f"{inertia[0, 2]:.9f}", iyy=f"{inertia[1, 1]:.9f}",
                      iyz=f"{inertia[1, 2]:.9f}", izz=f"{inertia[2, 2]:.9f}")

    for joint in spec.joints:
        node = ET.SubElement(robot, "joint", name=joint.name, type="revolute")
        ET.SubElement(node, "parent", link=joint.parent)
        ET.SubElement(node, "child", link=joint.child)
        # URDF joint origin 은 '부모 링크 프레임 -> 자식 링크 프레임' 변환이다.
        # 관절 축이 부모 프레임 o_P, 자식 프레임 o_C 에 있으면 그 차이가 된다.
        child_origin = np.array(joint.origin_in_child_link_mm
                                if joint.origin_in_child_link_mm is not None
                                else (0.0, 0.0, 0.0))
        offset = np.array(joint.origin_in_parent_link_mm) - child_origin
        ET.SubElement(node, "origin",
                      xyz=" ".join(f"{v * MM:.6f}" for v in offset))
        ET.SubElement(node, "axis",
                      xyz=" ".join(f"{v:.1f}" for v in joint.axis))
        ET.SubElement(node, "limit",
                      lower=f"{joint.limits_rad[0]:.6f}",
                      upper=f"{joint.limits_rad[1]:.6f}",
                      effort="10", velocity="1")
    return robot


# ---------------------------------------------------------------------------
# 원본 URDF 에 물성만 채워 넣기
#
# 스캔 배달물은 이미 형상(메시·볼록분해·관절)을 제대로 들고 있고, 비어 있는
# 것은 **물성뿐**이다. 배달물의 <inertial> 은 밀도 1000 kg/m^3 로 계산한
# 자리표시자다 (질량이 부피 그대로다: link_1 441.02 cm^3 -> 0.4410195 kg).
#
# 그러니 우리가 할 일은 새 물체를 짓는 게 아니라 그 자리표시자를 추정값으로
# 바꾸는 것이다. build_urdf 로 만들면 메시가 AABB 상자로 퇴화하고, 파지
# 부위를 뿌리로 다시 세운 트리가 나가고, 배달물의 볼록분해가 통째로 버려진다.
# 시뮬레이터에 그 상자를 넘기면 램프 Arm 이 실제 재료의 3배 굵기가 된다.
# ---------------------------------------------------------------------------
def _rewrite_mesh_paths(root, source, out):
    """메시 경로가 새 위치에서도 풀리도록 고친다.

    배달물 URDF 의 filename 은 배달물 폴더 기준 상대경로다("visuals/...").
    산출물을 my_work/ 에 두면 그 상대경로가 깨진다. 배달물 폴더는 건드리지
    않는 것이 규칙이므로(AGENTS.md), 대신 경로를 산출물 기준으로 다시 쓴다.
    """
    source_dir = Path(source).resolve().parent
    out_dir = Path(out).resolve().parent
    n = 0
    for mesh in root.iter("mesh"):
        name = mesh.get("filename")
        if name is None or name.startswith(("package://", "/")):
            continue
        absolute = (source_dir / name).resolve()
        mesh.set("filename", os.path.relpath(absolute, out_dir))
        n += 1
    return n


def _set_inertial(link, mass, com, inertia):
    """링크의 <inertial> 을 갈아끼운다. 없으면 만든다.

    URDF 규약상 <inertia> 는 <origin> 프레임 기준이고 그 원점이 무게중심이다.
    균일밀도 가정이라 무게중심은 외형 도심이고, 배달물이 적어 둔 origin 과
    같아야 한다 (main 에서 실제로 대조해 확인한다).
    """
    inertial = link.find("inertial")
    if inertial is None:
        inertial = ET.SubElement(link, "inertial")
    for child in list(inertial):
        inertial.remove(child)
    ET.SubElement(inertial, "origin",
                  xyz=" ".join(f"{v:.9g}" for v in com), rpy="0 0 0")
    ET.SubElement(inertial, "mass", value=f"{mass:.9g}")
    ET.SubElement(inertial, "inertia",
                  ixx=f"{inertia[0, 0]:.9g}", ixy=f"{inertia[0, 1]:.9g}",
                  ixz=f"{inertia[0, 2]:.9g}", iyy=f"{inertia[1, 1]:.9g}",
                  iyz=f"{inertia[1, 2]:.9g}", izz=f"{inertia[2, 2]:.9g}")


def inject_into_source(source, spec, rho_hat, out, Sigma=None):
    """배달물 URDF 를 그대로 두고 <inertial> 만 추정값으로 바꾼다.

    형상·관절·트리 구조·볼록분해는 배달물 것을 **한 글자도 안 바꾼다**.
    바뀌는 것은 링크마다 질량·무게중심·관성텐서 셋뿐이다.

    돌려주는 것: 링크별 (이름, 질량, 원본대비 origin 차이[m]) 목록.
    """
    tree = ET.parse(source)
    root = tree.getroot()

    extras = hinge_extras(spec, rho_hat)
    wanted = {}
    for part, rho in zip(spec.parts, rho_hat):
        wanted[part.name] = part_inertial(part, rho, extras[part.name])

    std = np.sqrt(np.diag(Sigma)) if Sigma is not None else None
    header = ["로봇이 물체를 잡고 손목 F/T 를 재서 추정한 물성을 채워 넣었다.",
              f"원본: {Path(source).resolve()}",
              "형상·관절·충돌메시는 원본 그대로이며, 링크별 질량·무게중심·",
              "관성텐서만 바뀌었다. 가정: 부위 내부 밀도 균일.",
              "밀도는 '스캔 겉모양 부피로 나눈 유효 밀도'라 물리 밀도가 아니다."]
    if std is not None:
        header.append("추정 밀도 [kg/m^3] (95% 구간):")
        for part, rho, sd in zip(spec.parts, rho_hat, std):
            header.append(f"  {part.name}: {rho:.1f} +/- {1.96 * sd:.1f}")
    root.insert(0, ET.Comment("\n     " + "\n     ".join(header) + "\n  "))

    rows, seen = [], set()
    for link in root.iter("link"):
        name = link.get("name")
        if name not in wanted:
            continue
        mass, com, inertia = wanted[name]
        before = link.find("inertial/origin")
        moved = (np.linalg.norm(
            np.array([float(v) for v in before.get("xyz").split()]) - com)
            if before is not None and before.get("xyz") else float("nan"))
        _set_inertial(link, mass, com, inertia)
        rows.append(dict(name=name, mass=mass, origin_shift_m=moved))
        seen.add(name)

    missing = set(wanted) - seen
    if missing:
        raise KeyError(f"원본 URDF 에 없는 링크: {sorted(missing)}. "
                       "배달물과 spec 의 링크 이름이 어긋났다.")
    n_mesh = _rewrite_mesh_paths(root, source, out)
    write_urdf(root, out)
    return rows, n_mesh


def source_urdf_for(spec):
    """이 물체의 원본(배달물) URDF. 없으면 None — 그때는 build_urdf 로 짓는다.

    3-link·2-link 는 CAD 치수에서 만든 물체라 원본 URDF 가 없고, 실제로
    직육면체라 상자로 내보내도 맞다.
    """
    if spec.key != "desklamp":
        return None
    import desk_lamp
    return desk_lamp.URDF


def export(spec, rho_hat, out, Sigma=None, log=print):
    """산출물 URDF 를 만든다. 원본이 있으면 거기에 물성만 채운다.

    호출하는 쪽(dual_view, main)이 둘 다 이 함수를 쓴다. 모드 선택을 한
    군데로 모아 두어야 화면으로 돌릴 때와 숫자만 돌릴 때가 안 갈라진다.
    """
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    source = source_urdf_for(spec)
    if source is None:
        write_urdf(build_urdf(spec, rho_hat, Sigma), out)
        log(f"  URDF 저장 -> {out}  (CAD 치수에서 새로 지음)")
        return None

    rows, n_mesh = inject_into_source(source, spec, rho_hat, out, Sigma)
    log(f"  URDF 저장 -> {out}")
    log(f"    원본 {source} 의 형상·관절을 그대로 두고 물성만 채웠습니다"
        f" (링크 {len(rows)}개, 메시 경로 {n_mesh}개 재작성)")
    for row in rows:
        # origin 차이가 크면 배달물과 우리 도심 계산이 어긋났다는 뜻이다.
        note = ("" if not (row["origin_shift_m"] > 1e-6)
                else f"   [주의] 무게중심이 원본과 {1000*row['origin_shift_m']:.2f} mm 다름")
        log(f"    {row['name']:<8} 질량 {1000*row['mass']:7.2f} g{note}")
    return rows


def write_urdf(root, path):
    raw = ET.tostring(root, encoding="unicode")
    pretty = minidom.parseString(raw).toprettyxml(indent="  ")
    pretty = "\n".join(line for line in pretty.split("\n") if line.strip())
    Path(path).write_text(pretty)
    return path


# ---------------------------------------------------------------------------
def verify_urdf(path, spec, rho_hat):
    """만든 URDF 를 Drake 로 되읽어 질량·무게중심·관성이 맞는지 확인한다."""
    from pydrake.multibody.parsing import Parser
    from pydrake.multibody.plant import MultibodyPlant

    plant = MultibodyPlant(time_step=0.0)
    Parser(plant).AddModels(str(path))
    plant.Finalize()
    context = plant.CreateDefaultContext()
    plant.SetPositions(context, np.zeros(plant.num_positions()))

    rows = []
    extras = hinge_extras(spec, rho_hat)
    for part, rho in zip(spec.parts, rho_hat):
        body = plant.GetBodyByName(part.name)
        spatial = body.CalcSpatialInertiaInBodyFrame(context)
        expect_mass, expect_com, expect_inertia = part_inertial(
            part, rho, extras[part.name])
        got_inertia = (spatial.CalcRotationalInertia().CopyToFullMatrix3()
                       - spatial.get_mass() * (
                           (spatial.get_com() @ spatial.get_com()) * np.eye(3)
                           - np.outer(spatial.get_com(), spatial.get_com())))
        rows.append(dict(
            name=part.name,
            mass_err=abs(spatial.get_mass() - expect_mass),
            com_err=float(np.linalg.norm(spatial.get_com() - expect_com)),
            inertia_err=float(np.linalg.norm(got_inertia - expect_inertia)
                              / max(np.linalg.norm(expect_inertia), 1e-12)),
        ))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--object", choices=tuple(obj.OBJECTS) + ("desklamp",),
                        default="3link")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--target", type=float, default=0.01,
                        help="정지 조건: 95%% 상대 반폭")
    parser.add_argument("--max-rounds", type=int, default=25)
    parser.add_argument("--joint-range-deg", type=float, nargs=2,
                        default=(0.0, 180.0))
    parser.add_argument("--samples-per-hold", type=int,
                        default=obj.DEFAULT_SAMPLES_PER_HOLD)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prior", choices=("weight", "mesh"), default="mesh")
    # 검토 지점 스위치. 기본값은 study_*.py 로 확인한 결론이다.
    parser.add_argument("--select", choices=("continuous", "grid"),
                        default="continuous")
    parser.add_argument("--criterion", choices=dc.CRITERIA, default="D")
    parser.add_argument("--estimator", choices=("tls", "wls"), default="tls")
    parser.add_argument("--stop-rule",
                        choices=("variance", "residual", "bias"),
                        default="residual")
    parser.add_argument("--systematic", type=float, default=0.3)
    args = parser.parse_args()

    if args.object == "desklamp":
        import desk_lamp
        spec = desk_lamp.build_spec()
    else:
        spec = obj.OBJECTS[args.object]
    # 관절은 사용자가 맞춰 두면 고정된다는 전제라 힌지 토크 필터를 쓰지 않는다.
    rho_gt = obj.bind_object(spec)
    obj.set_measurement_averaging(args.samples_per_hold,
                                  obj.DEFAULT_BIAS_FRACTION)

    print(f"{spec.label}")
    if args.prior == "weight":
        # 저울에는 힌지까지 붙은 채로 올라간다.
        total = obj.assembled_mass_kg(spec, rho_gt)
        mu, _, mean_density = obj.apply_weight_prior(spec, total)
        print(f"  출발: 저울로 총무게 {1000*total:.1f} g 만 앎"
              f" -> 모든 부위를 평균 밀도 {mean_density:.0f} kg/m^3 로 시작")
    else:
        mu, Sigma0, lows, highs = obj.apply_mesh_prior(spec)
        print(f"  출발: 메시 외형 부피만 앎"
              f" -> 밀도 사전분포 {mu[0]:.0f} +/- {np.sqrt(Sigma0[0,0]):.0f}"
              f" kg/m^3 (구간 {lows[0]:.0f}~{highs[0]:.0f})")

    limits = [tuple(np.deg2rad(args.joint_range_deg))] * len(spec.joints)
    alg.JOINT_LIMITS = limits
    spec = replace(spec, joints=[replace(j, limits_rad=lim)
                                 for j, lim in zip(spec.joints, limits)])

    print(f"  설정: 후보={args.select}  기준={args.criterion}-최적"
          f"  추정기={args.estimator.upper()}  정지={args.stop_rule}")
    result = dc.closed_loop(spec, target=args.target,
                            max_rounds=args.max_rounds, seed=args.seed,
                            select=args.select, criterion=args.criterion,
                            estimator=args.estimator,
                            stop_rule=args.stop_rule,
                            systematic=args.systematic, verbose=True)
    print(f"  {result['rounds']} 라운드"
          f" (95% 상대 반폭 {100*result['worst']:.3f}%"
          f" / 목표 {100*args.target:.3f}%)"
          f"{'' if result['converged'] else '  — 목표 미달'}")

    Sigma = np.diag((result["half"] * result["rho_hat"] / 1.96) ** 2)
    out = args.out or Path("outputs") / f"estimated_{spec.key}.urdf"
    print()
    export(spec, result["rho_hat"], out, Sigma)

    extras = hinge_extras(spec, result["rho_hat"])
    print(f"\n  {'부위':<13}{'밀도':>10}{'질량':>11}{'무게중심 x':>13}{'Izz':>14}"
          f"   (힌지 질량 포함)")
    for part, rho in zip(spec.parts, result["rho_hat"]):
        mass, com, inertia = part_inertial(part, rho, extras[part.name])
        print(f"  {part.name:<13}{rho:>10.1f}{1000*mass:>9.1f} g"
              f"{1000*com[0]:>11.1f} mm{inertia[2,2]*1e6:>11.1f}e-6")
    table = obj.body_table(spec)
    for row, rho in zip(table, result["rho_hat"]):
        if row["kind"] == "hinge":
            print(f"    └ {row['name']:<11}{rho:>10.1f}"
                  f"{1000*rho*row['volume_m3']:>9.1f} g   -> "
                  f"{ {j.name: j.child for j in spec.joints}[row['joint']] } 에 합산")

    print(f"\n  Drake 로 되읽어 검증")
    for row in verify_urdf(out, spec, result["rho_hat"]):
        print(f"    {row['name']:<13} 질량오차 {row['mass_err']:.2e} kg"
              f"  무게중심오차 {row['com_err']:.2e} m"
              f"  관성 상대오차 {row['inertia_err']:.2e}")

    print(f"\n  GT 대비 (검증용, 실제 파이프라인에서는 알 수 없음)")
    for part, est, gt in zip(spec.parts, result["rho_hat"], rho_gt):
        m_est = est * part.volume_m3
        m_gt = gt * part.volume_m3
        print(f"    {part.name:<13} 밀도 {est:8.1f} / {gt:7.1f}"
              f"  질량 {1000*m_est:7.2f} / {1000*m_gt:7.2f} g"
              f"  오차 {100*abs(est-gt)/gt:5.2f}%")


if __name__ == "__main__":
    main()
