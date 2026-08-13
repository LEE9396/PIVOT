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
"""

import numpy as np

from density_id_objects import Joint, ObjectSpec, Part

CROSS_MM = 44.0
GAP_MM = 4.0
OFFSET_MM = 27.0
FILL = 0.98

_PALETTE = [
    (0.25, 0.55, 0.35, 1.0), (0.90, 0.40, 0.10, 1.0), (0.45, 0.30, 0.80, 1.0),
    (0.20, 0.55, 0.85, 1.0), (0.85, 0.75, 0.20, 1.0), (0.80, 0.30, 0.50, 1.0),
    (0.35, 0.75, 0.55, 1.0), (0.55, 0.57, 0.60, 1.0),
]


def link_lengths(n_part):
    """base 150 mm 에서 시작해 끝으로 갈수록 짧아진다 (실물 3-link 와 동일)."""
    return [150.0] + [max(60.0, 110.0 - 12.5 * k) for k in range(n_part - 1)]


def densities(n_part, low=700.0, high=5200.0):
    """자릿수가 고르게 퍼지도록 로그 등간격. GT 이며 채점에만 쓴다."""
    return list(np.round(np.geomspace(high, low, n_part), 1))


def make_spec(n_part, rho_gt=None, limits_rad=(0.0, np.pi)):
    """파트 n_part 개짜리 직렬 물체 사양."""
    lengths = link_lengths(n_part)
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
    for n in range(2, 9):
        spec = make_spec(n)
        total = sum(p.mass_kg for p in spec.parts)
        print(f"P={n}  관절 {len(spec.joints)}개  총질량 {1000*total:6.1f} g  "
              f"GT {[p.rho_gt for p in spec.parts]}")
