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
    cd ~/Desktop/HTD-main/my_work
    ../robot_learning/scripts/run_drake_env.sh python export_urdf.py \
        --object 3link --out estimated_3link.urdf
"""

import argparse
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
    dims_m = tuple(d * MM for d in part.bbox_mm)
    mass = rho * part.volume_m3
    com = np.array(part.bbox_center_in_link_mm) * MM
    inertia = box_inertia_about_com(mass, dims_m)

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
    parser.add_argument("--object", choices=tuple(obj.OBJECTS), default="3link")
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
    out = args.out or Path(f"estimated_{spec.key}.urdf")
    write_urdf(build_urdf(spec, result["rho_hat"], Sigma), out)
    print(f"\n  URDF 저장 -> {out}")

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
