"""density_id_objects.py 의 Drake 모델이 CAD 수치와 맞는지 검증한다.

q=0(일직선)에서 각 part의 외형 상자가 차지하는 x 구간을 뽑아,
README가 명시한 링크 간 4 mm 간격과 전체 길이를 확인한다.
"""

import numpy as np

import density_id_objects as obj

MM = 1e-3


def report(spec, expected_total_mm):
    plant, bodies = obj.build_plant(spec, [r["rho_gt"] for r in obj.body_table(spec)])
    context = plant.CreateDefaultContext()
    plant.SetPositions(context, np.zeros(plant.num_positions()))

    print(f"\n{spec.label}  (q = 0, 일직선)")
    print(f"  {'part':<13}{'외형중심 x':>12}{'x 구간 [mm]':>22}")
    spans = []
    for part in spec.parts:
        center = plant.EvalBodyPoseInWorld(context, bodies[part.name]).translation()
        half = part.bbox_mm[0] * MM / 2.0
        lo, hi = (center[0] - half) / MM, (center[0] + half) / MM
        spans.append((lo, hi))
        print(f"  {part.name:<13}{center[0] / MM:>10.2f} mm"
              f"{f'{lo:8.2f} .. {hi:8.2f}':>22}")

    print("  링크 간 간격:", end="")
    for (_, hi), (lo, _) in zip(spans[:-1], spans[1:]):
        gap = lo - hi
        mark = "OK" if abs(gap - 4.0) < 1e-6 else f"기대 4.0 과 다름"
        print(f"  {gap:.3f} mm ({mark})", end="")
    total = spans[-1][1] - spans[0][0]
    verdict = "OK" if abs(total - expected_total_mm) < 1e-6 else "불일치"
    print(f"\n  전체 펼친 길이: {total:.2f} mm (README {expected_total_mm} mm, {verdict})")

    mass = plant.CalcTotalMass(context)
    com = plant.CalcCenterOfMassPositionInWorld(context)
    expected_mass = sum(p.mass_kg for p in spec.parts)
    print(f"  총질량 {1000 * mass:.1f} g (기대 {1000 * expected_mass:.1f} g), "
          f"CoM x = {com[0] / MM:.2f} mm")


def check_derived(spec):
    """derived_quantities 를 손계산과 대조한다.

    q=0 에서 모든 상자가 같은 방향으로 정렬되므로 총질량·무게중심·관성텐서를
    직육면체 공식과 평행축 정리로 직접 계산할 수 있다.
    """
    rho = np.array([p.rho_gt for p in spec.parts], dtype=float)
    theta = np.zeros(len(spec.joints))
    got = obj.derived_quantities(spec, rho, theta)

    # 손계산 — q=0 에서 각 상자 중심의 센서 좌표계 위치
    plant, bodies = obj.build_plant(spec, rho)
    context = plant.CreateDefaultContext()
    plant.SetPositions(context, theta)
    centers = np.stack([
        plant.EvalBodyPoseInWorld(context, bodies[p.name]).translation()
        for p in spec.parts
    ])
    masses = np.array([r * p.volume_m3 for p, r in zip(spec.parts, rho)])

    mass_hand = masses.sum()
    com_hand = (masses[:, None] * centers).sum(axis=0) / mass_hand

    inertia_hand = np.zeros((3, 3))
    for part, mass, center in zip(spec.parts, masses, centers):
        lx, ly, lz = (d * MM for d in part.bbox_mm)
        own = mass / 12.0 * np.diag([ly**2 + lz**2, lx**2 + lz**2, lx**2 + ly**2])
        d = center - com_hand                       # 평행축 정리
        inertia_hand += own + mass * ((d @ d) * np.eye(3) - np.outer(d, d))

    print(f"\n{spec.label}  파생량 손계산 대조 (q = 0)")
    print(f"  총질량    Drake {1000 * got['total_mass_kg']:.4f} g"
          f"   손계산 {1000 * mass_hand:.4f} g"
          f"   차이 {1000 * abs(got['total_mass_kg'] - mass_hand):.2e} g")
    print(f"  무게중심  차이 {1000 * np.linalg.norm(got['com_m'] - com_hand):.2e} mm")
    rel = (np.linalg.norm(got["inertia_about_com"] - inertia_hand)
           / np.linalg.norm(inertia_hand))
    print(f"  관성텐서  상대차이 {rel:.2e}")
    verdict = "일치" if rel < 1e-9 else "불일치"
    print(f"  판정: {verdict}")


if __name__ == "__main__":
    # 2-link: 75 + 4 + 60 = 139 mm, 3-link: 150 + 4 + 110 + 4 + 85 = 353 mm
    report(obj.TWO_LINK, 139.0)
    report(obj.THREE_LINK, 353.0)
    check_derived(obj.TWO_LINK)
    check_derived(obj.THREE_LINK)
