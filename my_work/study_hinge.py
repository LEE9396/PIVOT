"""힌지 질량을 무시해도 되는가.

지금 모형은 물체를 '부위 = 균일밀도 상자' 들로만 본다. 힌지는 관절로만
들어가 있고 **질량이 0** 이다. 시뮬레이션에서는 측정용 plant 와 추정용 plant
가 똑같이 힌지를 무시하므로 오차가 안 생긴다. 하지만 실물에서는 힌지가
실제로 거기 붙어 있고, 추정기는 그걸 모른다.

그 차이가 얼마나 아픈지 잰다. 측정용 plant 에만 힌지 질량을 넣고, 추정기는
지금 그대로 둔 뒤 밀도 오차를 본다. 힌지는 관절 축 위에 붙어 있으므로 절반씩
양쪽 링크에 나눠 단다.

실행:
  ../robot_learning/scripts/run_drake_env.sh python study_hinge.py
"""

import argparse

import numpy as np
from pydrake.math import RigidTransform
from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.tree import (FixedOffsetFrame, RevoluteJoint,
                                    SpatialInertia, UnitInertia)

import angle_aware as aa
import density_id_drake as alg
import density_id_objects as obj
import design_core as dc

MM = obj.MM


def build_plant_with_hinges(spec, densities, hinge_mass_kg):
    """힌지 질량을 점질량으로 달아 만든 plant. 측정(진리) 쪽에만 쓴다."""
    plant = MultibodyPlant(time_step=0.0)
    parts = {p.name: p for p in spec.parts}

    # 각 링크 프레임에서 힌지가 붙는 위치를 모은다.
    extra = {p.name: [] for p in spec.parts}
    for joint in spec.joints:
        origin = np.array(joint.origin_in_parent_link_mm)
        on_parent = (origin - np.array(
            parts[joint.parent].bbox_center_in_link_mm)) * MM
        on_child = (-np.array(
            parts[joint.child].bbox_center_in_link_mm)) * MM
        extra[joint.parent].append((0.5 * hinge_mass_kg, on_parent))
        extra[joint.child].append((0.5 * hinge_mass_kg, on_child))

    bodies = {}
    for part, rho in zip(spec.parts, densities):
        dims = tuple(d * MM for d in part.bbox_mm)
        m_part = rho * part.volume_m3
        # 상자 본체: 무게중심이 body frame 원점, 원점 기준 관성
        inertia = m_part * np.array(
            UnitInertia.SolidBox(*dims).CopyToFullMatrix3())
        mass = m_part
        moment = np.zeros(3)
        for m_h, r in extra[part.name]:
            mass += m_h
            moment += m_h * r
            inertia += m_h * (float(r @ r) * np.eye(3) - np.outer(r, r))
        com = moment / mass
        G = inertia / mass
        bodies[part.name] = plant.AddRigidBody(
            part.name,
            SpatialInertia(mass, com,
                           UnitInertia(G[0, 0], G[1, 1], G[2, 2],
                                       G[0, 1], G[0, 2], G[1, 2])))

    plant.WeldFrames(plant.world_frame(),
                     bodies[spec.parts[0].name].body_frame(),
                     RigidTransform(
                         np.array(spec.base_bbox_center_in_sensor_mm) * MM))
    for joint in spec.joints:
        origin = np.array(joint.origin_in_parent_link_mm)
        on_parent = (origin - np.array(
            parts[joint.parent].bbox_center_in_link_mm)) * MM
        on_child = (-np.array(
            parts[joint.child].bbox_center_in_link_mm)) * MM
        plant.AddJoint(RevoluteJoint(
            joint.name,
            plant.AddFrame(FixedOffsetFrame(
                f"{joint.name}_parent", bodies[joint.parent].body_frame(),
                RigidTransform(on_parent))),
            plant.AddFrame(FixedOffsetFrame(
                f"{joint.name}_child", bodies[joint.child].body_frame(),
                RigidTransform(on_child))),
            joint.axis, damping=0.0))
    plant.Finalize()
    return plant, bodies


def run(spec, rho_gt, hinge_mass_kg, rounds, seed, rel=0.05):
    """힌지가 달린 물체를 재고, 힌지를 모르는 추정기로 푼다."""
    plant, _ = build_plant_with_hinges(spec, rho_gt, hinge_mass_kg)
    alg.TRUTH_PLANT, alg.TRUTH_CTX = plant, plant.CreateDefaultContext()

    g = dc.CANONICAL_TRIAD
    rng = np.random.default_rng(seed)
    S, rho = alg.SIGMA0.copy(), alg.MU0.copy()
    bounds = [j.limits_rad for j in spec.joints]
    blocks, hist, angles = [], [], []
    for index in range(rounds):
        th, _ = dc.continuous_best(
            bounds, lambda t: dc.utility(t, rho, S, g, "D", rel),
            n_starts=8, seed=seed + index)
        th = np.atleast_1d(th)
        sig = np.sqrt(np.diag(aa.angle_covariance(th, rel)))
        actual = th + rng.normal(0, sig)
        meas = actual + rng.normal(0, sig)
        y = alg.measure(actual, g_dirs=list(g), rng=rng)
        A = dc.regressor(meas, g)
        R = dc.effective_cov(meas, rho, g, rel)
        blocks.append((A, y, R))
        hist.append((meas, y))
        angles.append(np.degrees(th))
        S = dc.posterior(S, A, R)
        rw = dc.wls_map(blocks, alg.MU0, alg.SIGMA0, alg.RHO_BOUNDS)
        rho, _ = dc.tls_map(hist, alg.MU0, alg.SIGMA0, alg.RHO_BOUNDS, g,
                            rho_init=rw, rel_error=rel)
    half = dc.half_width(S, rho,
                         inflate=dc.residual_inflation(blocks, rho))
    return rho, half, angles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", default="3link", choices=list(obj.OBJECTS))
    ap.add_argument("--masses-g", type=float, nargs="*",
                    default=[0, 5, 10, 20, 30, 50])
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=8)
    args = ap.parse_args()

    spec = obj.OBJECTS[args.object]
    obj.set_measurement_averaging()
    rho_gt = obj.bind_object(spec)
    total = obj.assembled_mass_kg(spec)
    obj.apply_weight_prior(spec, total)
    saved = (alg.TRUTH_PLANT, alg.TRUTH_CTX)

    print(f"{spec.label}   부위 {len(spec.parts)}개   관절 {len(spec.joints)}개")
    print(f"물체 총질량 {1000*total:.1f} g   부위 질량 "
          + ", ".join(f"{1000*p.mass_kg:.0f}g" for p in spec.parts))
    print("\n추정기는 힌지를 전혀 모른다. 측정 쪽에만 힌지를 달고 오차를 본다.")
    print(f"(힌지 1개를 절반씩 양쪽 링크에 나눠 답, {len(spec.joints)}개 관절,"
          f" {args.rounds}라운드, seed {args.seeds}개)\n")

    print(f"  {'힌지1개':>8}{'총질량비':>9}" +
          "".join(f"{p.name[:11]:>13}" for p in spec.parts) +
          f"{'최대오차':>10}{'주장반폭':>10}")
    for gram in args.masses_g:
        errs, halves = [], []
        for s in range(args.seeds):
            rho, half, angles = run(spec, rho_gt, gram / 1000.0,
                                    args.rounds, 100 + s)
            errs.append(np.abs(rho - rho_gt) / rho_gt)
            halves.append(half)
        e = np.mean(errs, axis=0)
        share = 100 * len(spec.joints) * (gram / 1000.0) / total
        print(f"  {gram:>6.0f} g{share:>8.1f}%" +
              "".join(f"{100*v:>12.2f}%" for v in e) +
              f"{100*e.max():>9.2f}%"
              f"{100*np.mean([h.max() for h in halves]):>9.2f}%")

    alg.TRUTH_PLANT, alg.TRUTH_CTX = saved
    print("\n읽는 법")
    print("  '주장반폭'은 알고리즘이 스스로 말하는 불확실성이다. 실제 오차가")
    print("  그보다 크면, 알고리즘은 다 맞췄다고 믿으면서 틀린 답을 내놓는다.")
    print("  힌지 질량은 잡음이 아니라 **모형이 빠뜨린 것**이라 라운드를 늘려도")
    print("  안 줄어든다. 시뮬레이션에서는 안 보이고 실물에서만 나타난다.")


if __name__ == "__main__":
    main()
