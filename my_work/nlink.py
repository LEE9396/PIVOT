"""파트 수를 늘린 합성 물체 생성기.

D / A / E 중 뭘 써야 하는지는 파트가 2~3개일 때는 거의 안 갈린다. 미지수가
적으면 어느 기준으로 골라도 비슷한 자세가 나오기 때문이다. 차이는 파트가
늘어나 **공분산의 방향들이 서로 크게 달라질 때** 벌어진다.

그래서 실제 3-link (v3) 의 치수 규칙을 그대로 이어붙여 P = 2..8 짜리 물체를
만든다. 규칙은 CAD 에서 읽은 그대로다.

  - 단면 44 x 44 mm, 링크 사이 간격 4 mm
  - 자식 링크 프레임은 핀 축 위에 있고 링크는 x = +2 부터 시작
  - 힌지는 표면 장착이라 축이 중심선에서 27 mm 벗어나 있다
  - 관절 축은 z, -y, z, -y ... 로 번갈아 든다
  - 외부 부피는 bbox 의 약 98 % (리세스 공제)
  - **관절마다 실측 힌지 41 g 이 달린다** (실물 2-link / 3-link 와 같은 부품)

힌지를 왜 반드시 넣는가
-----------------------
힌지를 빼면 합성 물체가 실물과 **다른 문제**가 된다. 실물 3-link 는 미지수가
5개(부위 3 + 힌지 2)인데, 힌지 없는 합성 P=3 은 3개뿐이다. 그러면 P=2..8
스윕이 실물보다 쉬운 문제를 재게 되어 두 실험을 나란히 놓을 수 없다.

게다가 힌지를 추정 벡터에서 빼면 그 질량 몫이 부위 밀도로 흘러들어 라운드를
늘려도 안 없어지는 치우침이 된다 (3-link 에서 23 %, study_hinge.py).

힌지를 넣으면 미지수가 P + (P-1) = 2P-1 이 되고, 실물과 정확히 맞는다.
  P=2 -> 3 개 (실물 2-link 와 같음)
  P=3 -> 5 개 (실물 3-link 와 같음)
"""

import numpy as np

from density_id_objects import MEASURED_HINGE_KG, Joint, ObjectSpec, Part

CROSS_MM = 44.0
GAP_MM = 4.0
OFFSET_MM = 27.0
FILL = 0.98

_PALETTE = [
    (0.25, 0.55, 0.35, 1.0), (0.90, 0.40, 0.10, 1.0), (0.45, 0.30, 0.80, 1.0),
    (0.20, 0.55, 0.85, 1.0), (0.85, 0.75, 0.20, 1.0), (0.80, 0.30, 0.50, 1.0),
    (0.35, 0.75, 0.55, 1.0), (0.55, 0.57, 0.60, 1.0),
]


# 링크 길이를 뽑는 범위. 실물 커스텀 물체(150 / 110 / 97.5 mm)가 이 안에 든다.
LEN_MIN_MM, LEN_MAX_MM = 70.0, 150.0


def link_lengths(n_part, rng=None):
    """링크 길이 [mm].

    rng 를 주면 [LEN_MIN, LEN_MAX] 에서 균일하게 뽑는다. **스윕에는 이쪽을 쓴다.**

    왜 무작위인가
    -------------
    고정 규칙은 어느 쪽으로든 결과를 정한다.

      단조 감소(150->60)  말단 링크가 짧아져 마지막 관절의 지렛대가 사라진다.
                          이웃한 두 부위가 어떤 자세에서도 같이 움직여 안 갈린다.
                          p=4 에서 이미 link2/link3 반폭이 1.27 / 1.29 % 로 붙는다.
      등길이              반대로 유리한 모양을 고른 셈이 된다.

    범위 안에서 뽑으면 둘 다 피하고 **'전형적인 사슬'** 에 대한 결과가 된다.
    seed 마다 물체가 달라지므로 오차 막대도 생긴다 — 물체 하나에 대한 결과가
    아니게 된다.

    rng 를 안 주면 예전 규칙(단조 감소)을 그대로 돌려준다. 기존 결과를
    재현할 때 쓴다.
    """
    if rng is None:
        return [150.0] + [max(60.0, 110.0 - 12.5 * k) for k in range(n_part - 1)]
    return list(rng.uniform(LEN_MIN_MM, LEN_MAX_MM, n_part))


def densities(n_part, low=700.0, high=5200.0):
    """자릿수가 고르게 퍼지도록 로그 등간격. GT 이며 채점에만 쓴다."""
    return list(np.round(np.geomspace(high, low, n_part), 1))


def n_unknowns(n_part):
    """추정 벡터의 길이 = 부위 n_part 개 + 힌지 (n_part-1) 개 = 2*n_part - 1."""
    return 2 * n_part - 1


def round_lower_bound(n_part):
    """THEORY.md 정리 3: rank <= 1 + min(3R, n) 이므로 R >= ceil((n-1)/3).

    n = 2*n_part - 1 이므로 R >= ceil((2*n_part - 2)/3).
    P=2 -> 1, P=3 -> 2 로 실물 관측 라운드 수와 일치한다.
    """
    return int(np.ceil((n_unknowns(n_part) - 1) / 3.0))


def make_spec(n_part, rho_gt=None, limits_rad=(0.0, np.pi),
              hinge_kg=MEASURED_HINGE_KG, seed=None, lengths=None):
    """파트 n_part 개짜리 직렬 물체 사양. 관절마다 실측 힌지가 달린다.

    seed 를 주면 링크 길이를 [LEN_MIN, LEN_MAX] 에서 뽑는다 (스윕용).
    seed 도 lengths 도 안 주면 예전 단조 감소 규칙을 쓴다.
    hinge_kg=0.0 이면 힌지 없는 예전 사양 (study_hinge.py 의 대조군).
    """
    if lengths is None:
        rng = np.random.default_rng(seed) if seed is not None else None
        lengths = link_lengths(n_part, rng)
    lengths = list(lengths)
    rho_gt = list(rho_gt) if rho_gt is not None else densities(n_part)

    parts, joints = [], []
    centerline = (0.0, 0.0)          # (y, z) — 자식 프레임 기준 링크 중심선
    for index, length in enumerate(lengths):
        start = 0.0 if index == 0 else GAP_MM / 2.0
        cy, cz = centerline
        center = (start + length / 2.0, cy, cz)
        volume = FILL * length * CROSS_MM * CROSS_MM * 1e-3   # mm^3 -> cm^3
        parts.append(Part(
            name=f"link{index}" + ("_base" if index == 0 else ""),
            bbox_mm=(length, CROSS_MM, CROSS_MM),
            volume_cm3=round(volume, 2),
            rho_gt=float(rho_gt[index]),
            bbox_center_in_link_mm=center,
            shell_centroid_in_link_mm=center,
            color=_PALETTE[index % len(_PALETTE)],
            rho_empty=250.0,
            cavity_cm3=round(0.4 * volume, 1),
        ))

        if index + 1 < len(lengths):
            x_joint = start + length + GAP_MM / 2.0
            if index % 2 == 0:       # z 축 회전, 옆으로(+y) 어긋난 핀
                origin = (x_joint, cy + OFFSET_MM, cz)
                axis = (0.0, 0.0, 1.0)
                centerline = (-OFFSET_MM, 0.0)
            else:                    # -y 축 회전, 위로(+z) 어긋난 핀
                origin = (x_joint, cy, cz + OFFSET_MM)
                axis = (0.0, -1.0, 0.0)
                centerline = (0.0, -OFFSET_MM)
            joints.append(Joint(
                name=f"joint{index + 1}",
                parent=parts[index].name,
                child=f"link{index + 1}",
                origin_in_parent_link_mm=origin,
                axis=axis,
                limits_rad=tuple(limits_rad),
                hinge_mass_kg=hinge_kg,
            ))

    return ObjectSpec(
        key=f"{n_part}link",
        label=f"{n_part}-part synthetic chain",
        parts=parts,
        joints=joints,
        base_bbox_center_in_sensor_mm=parts[0].bbox_center_in_link_mm,
        notes="3-link (v3) 치수 규칙을 그대로 이어붙인 합성 물체",
    )


if __name__ == "__main__":
    print(f"{'P':>2} {'관절':>4} {'미지수':>6} {'하한R':>5} {'부위질량':>9} "
          f"{'힌지질량':>9} {'총질량':>9}")
    for n in range(2, 9):
        spec = make_spec(n)
        part_kg = sum(p.mass_kg for p in spec.parts)
        hinge_kg = sum(j.hinge_mass_kg for j in spec.joints)
        print(f"{n:>2} {len(spec.joints):>4} {n_unknowns(n):>6} "
              f"{round_lower_bound(n):>5} {1000*part_kg:>8.1f}g "
              f"{1000*hinge_kg:>8.1f}g {1000*(part_kg+hinge_kg):>8.1f}g")
