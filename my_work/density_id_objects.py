"""실제 2-link / 3-link 검증 물체에 대한 part별 밀도 식별 검증.

density_id_drake.py의 추정기(S1~S6)를 **그대로** 재사용하고, 대상 물체만
실제 CAD 수치로 교체한다. 알고리즘 코드는 손대지 않는다.

기하 출처
---------
2-link : 2_link_custom_object/README.md + model.py 상단 파라미터
         Parent 75x46x40 mm (138.0 cm^3), Child 60x46x40 mm (110.4 cm^3)
         Southco E6-10-208C-50 힌지 1개, 간격 4 mm, 핀 축이 상면 위 5 mm
3-link : 3_link_custom_object_v3/urdf/phantom_v3.urdf + geometry.csv
         link0 288.12 / link1 208.40 / link2 162.28 cm^3 (리세스 공제 후)
         joint1 (0.152, 0.027, 0) axis +Z, joint2 (0.114, -0.027, 0.027) axis -Y

주의: 레포의 robot_learning/assets/phantom_v3 는 이것과 다른 구버전 물체다
(link0 부피 82.8 cm^3, 관절 범위 +-2.32 rad). 여기서는 쓰지 않는다.

모델링 가정
-----------
각 part를 외부 부피 전체에 밀도가 균일한 강체로 본다. README의
"평균 밀도 GT = 질량 / 외부 부피" 정의와 같은 가정이며, 추정기가 푸는
미지수도 바로 이 평균 밀도다. 따라서 무게추의 실제 위치는 모델에 없고,
각 part의 무게중심은 외형 도심에 놓인다.

실행:
    cd ~/Desktop/PIVOT/my_work
    ../robot_learning/scripts/run_drake_env.sh python density_id_objects.py
    ../robot_learning/scripts/run_drake_env.sh python density_id_objects.py --render
"""

import argparse
from dataclasses import dataclass, field

import numpy as np
from pydrake.geometry import Box, Convex, Mesh
from pydrake.math import RigidTransform
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph, MultibodyPlant
from pydrake.multibody.tree import (
    FixedOffsetFrame,
    RevoluteJoint,
    SpatialInertia,
    UnitInertia,
)

import density_id_drake as alg

MM = 1e-3
CM3 = 1e-6


@dataclass(frozen=True)
class Hinge:
    label: str
    holding_torque_nm: float
    note: str


# 명가철물 힘조절경첩 HP-TC 계열은 PA6 플라스틱, 나사로 토크를 조절한다.
# 제조사가 토크값을 공개하지 않으므로 같은 회사 금속 토크힌지(HT-TC2626:
# 0.1~0.2 / 0.5~0.6 N·m)의 범위를 기준으로 잡고, --hinge-torque 로 바꾼다.
# 실물은 반드시 토크 게이지로 실측해야 한다.
# 힌지 본체 재질 밀도. 부피 = 질량 / 이 값 으로 환산해 '부위' 하나로 만든다.
# 정확할 필요는 없다. 밀도와 부피의 곱(=질량)만 맞으면 렌치가 같다.
HINGE_MATERIAL_DENSITY = 1150.0     # PA6 나일론
HINGE_PRIOR_REL_SIGMA = 0.02        # 저울로 잰 힌지 질량의 상대 불확실성 2 %
MEASURED_HINGE_KG = 0.041           # 명가철물 토크힌지 실측 (저울)

MG_PLASTIC_DEFAULT_NM = 0.15
HINGES = {
    "mg_plastic": Hinge("명가철물 힘조절경첩 HP-TC (PA6)", MG_PLASTIC_DEFAULT_NM,
                        "나사 조절식, 토크 미공개 — 실측 필요"),
    "southco": Hinge("Southco E6-10-208C-50", 0.9, "알루미늄, 공장 프리셋"),
}

# 관절이 "고정된다"고 보기 위한 안전배수. 토크 편차와 PA6 크리프를 감안.
DEFAULT_SAFETY = 1.5

# AFT200 은 1000 Hz 로 샘플링한다(실험실 설정). 정지 자세에서 잠시 멈춰
# 여러 샘플을 평균내면 백색 노이즈가 sqrt(N) 로 줄어든다. 다만 센서 바이어스
# 드리프트처럼 평균으로 사라지지 않는 성분이 남으므로, 그 비율을 따로 둔다.
DEFAULT_SAMPLES_PER_HOLD = 1000     # 1000 Hz x 1 초 정지
DEFAULT_BIAS_FRACTION = 0.05        # 평균으로 줄지 않는 성분의 비율

_BASE_SIGMA_F = alg.SIGMA_F
_BASE_SIGMA_T = alg.SIGMA_T


def set_measurement_averaging(n_samples=DEFAULT_SAMPLES_PER_HOLD,
                              bias_fraction=DEFAULT_BIAS_FRACTION):
    """정지 측정에서 N 샘플을 평균낸 것으로 노이즈 모델을 바꾼다.

    sigma_eff = sigma * sqrt(bias^2 + (1 - bias^2) / N)

    N -> 무한대 여도 sigma * bias 아래로는 내려가지 않는다. 이것이 실제
    F/T 센서에서 tare 후에도 남는 바닥이다. 추정기의 백색화 가중치도 함께
    바꿔야 추정량이 개선된 정밀도를 실제로 활용한다.
    """
    scale = np.sqrt(bias_fraction ** 2
                    + (1.0 - bias_fraction ** 2) / max(n_samples, 1))
    alg.SIGMA_F = _BASE_SIGMA_F * scale
    alg.SIGMA_T = _BASE_SIGMA_T * scale
    alg.R_EPS_DIAG = np.array([alg.SIGMA_F ** 2] * 3 + [alg.SIGMA_T ** 2] * 3)
    alg.R_STACK_DIAG = np.tile(alg.R_EPS_DIAG, len(alg.G_DIRS))
    alg.W_HALF = 1.0 / np.sqrt(alg.R_STACK_DIAG)
    return dict(n_samples=n_samples, bias_fraction=bias_fraction,
                sigma_f_n=alg.SIGMA_F, sigma_t_nm=alg.SIGMA_T)


# ---------------------------------------------------------------------------
# 사전분포를 물리적으로 정한다.
#
# 기존 사전분포(모든 부위 1000 +- 3000)는 근거 없이 넓게 잡은 값이었다.
# 설계 문서에는 실제로 두 가지 한계가 적혀 있다.
#   하한: 추 없이 출력만 한 상태의 평균 밀도 (rho_empty)
#   상한: 내부 공동을 채울 수 있는 가장 무거운 재료로 가득 채운 상태
# 이 구간을 균등분포로 보면 평균과 표준편차가 바로 나온다.
# ---------------------------------------------------------------------------
FILL_MATERIALS = {          # kg/m^3
    "lead": 11340.0,
    "steel": 7850.0,
    "sand": 1600.0,
    "none": 0.0,
}


def density_upper_bound(part, fill_density):
    """공동을 fill 재료로 가득 채웠을 때의 평균 밀도."""
    shell_mass = part.rho_empty * part.volume_m3
    fill_mass = part.cavity_cm3 * CM3 * fill_density
    return (shell_mass + fill_mass) / part.volume_m3


def physical_prior(spec, fill="lead"):
    """[빈 출력물, 가득 채운 상태] 균등분포에서 사전 평균과 공분산을 만든다."""
    fill_density = FILL_MATERIALS[fill] if isinstance(fill, str) else float(fill)
    lows, highs = [], []
    for part in spec.parts:
        lows.append(part.rho_empty)
        highs.append(density_upper_bound(part, fill_density))
    lows, highs = np.array(lows), np.array(highs)
    mu = 0.5 * (lows + highs)
    sigma = (highs - lows) / np.sqrt(12.0)     # 균등분포의 표준편차
    return mu, np.diag(sigma ** 2), lows, highs


# 순수 메시만 있을 때의 사전분포.
#
# 실제 파이프라인의 출발점은 CAD 도면이 아니라 스캔·재구성된 메시다. 그때
# 아는 것은 외형 부피뿐이고, 속이 비었는지 꽉 찼는지는 모른다. 그래서
# 넓은 재료 구간을 그대로 불확실성으로 삼는다. 탐색이 진행되며 이 값이
# 실제 물리량으로 대체되는 것이 이 연구의 목적이다.
MESH_PRIOR_RANGE = (150.0, 8000.0)   # 속 빈 플라스틱 ~ 금속 충전


def mesh_prior(spec, density_range=MESH_PRIOR_RANGE,
               hinge_rel_sigma=HINGE_PRIOR_REL_SIGMA):
    """메시 외형 부피만 아는 상태의 사전분포. 부위마다 동일하게 넓다.

    힌지만은 예외다. 저울로 따로 재서 알고 있으므로 좁게 준다.
    """
    low, high = density_range
    table = body_table(spec)
    sigma = (high - low) / np.sqrt(12.0)      # 균등분포
    mu, var, lows, highs = [], [], [], []
    for row in table:
        if row["kind"] == "part":
            mu.append(0.5 * (low + high)); var.append(sigma ** 2)
            lows.append(low); highs.append(high)
        else:
            mu.append(row["rho_gt"])
            var.append((hinge_rel_sigma * row["rho_gt"]) ** 2)
            lows.append(0.5 * row["rho_gt"]); highs.append(2.0 * row["rho_gt"])
    return (np.array(mu), np.diag(var), np.array(lows), np.array(highs))


def apply_mesh_prior(spec, density_range=MESH_PRIOR_RANGE):
    """추정기를 '메시만 아는 상태'로 초기화한다."""
    mu, Sigma, lows, highs = mesh_prior(spec, density_range)
    alg.MU0 = mu
    alg.SIGMA0 = Sigma
    alg.RHO_BOUNDS = (float(lows.min()), float(highs.max()))
    return mu, Sigma, lows, highs


# 저울로 총무게만 잰 상태의 사전분포.
#
# 실험 절차상 가장 자연스러운 출발점이다. 물체 전체를 저울에 올리면
# 총질량 M 을 정확히 알 수 있고, 외형 부피 합 V_total 도 메시에서 나온다.
# 그러면 평균 밀도 M / V_total 이 나오지만, **부위별로 어떻게 나뉘는지는
# 전혀 모른다**. 그것이 바로 탐색이 풀어야 할 문제다.
#
# 이 정보를 공분산으로 옮기면 이렇게 된다.
#   - 총질량을 바꾸는 방향(부피 벡터 V 방향)  -> 거의 확실하다 (저울 정확도)
#   - 그와 직교하는 방향(질량 재분배)         -> 전혀 모른다
DEFAULT_SCALE_REL_ERROR = 0.002     # 저울 상대오차 0.2 %


def weight_prior(spec, total_mass_kg, scale_rel_error=DEFAULT_SCALE_REL_ERROR,
                 redistribution_sigma=None,
                 hinge_rel_sigma=HINGE_PRIOR_REL_SIGMA):
    """총무게만 아는 상태의 사전분포.

    반환한 공분산은 등방이 아니다. 총질량 방향은 좁고 재분배 방향은 넓다.

    힌지는 다르게 다룬다. 저울로 따로 재서 질량을 알고 있으므로 좁은 폭을
    준다 (hinge_rel_sigma, 기본 2 %). 그러면 미지수가 사실상 안 늘어나
    조건수가 그대로 유지된다. 힌지 질량이 미덥지 않으면 이 값을 키우면
    되고, 그만큼 자세가 더 필요해진다.

      hinge_rel_sigma  조건수(2자세)   성격
        0.02 (기본)        21          저울로 잰 값 그대로 씀
        0.50               ~150        사실상 미지수로 품
    """
    table = body_table(spec)
    n_part = len(spec.parts)
    part_volumes = np.array([row["volume_m3"] for row in table[:n_part]])
    hinge_mass = float(sum(j.hinge_mass_kg for j, _, _ in hinge_bodies(spec)))

    # 힌지 질량은 이미 아는 값이므로 저울 총무게에서 빼고 나눈다.
    mean_density = max(total_mass_kg - hinge_mass, 1e-9) / part_volumes.sum()
    mu = np.array([mean_density if row["kind"] == "part" else row["rho_gt"]
                   for row in table])

    if redistribution_sigma is None:
        # 부위 밀도가 평균의 절반~두 배 사이에 있다고 보는 정도의 폭
        redistribution_sigma = mean_density

    # 총질량을 바꾸는 방향 (밀도 공간에서 부피 벡터). 부위들 사이에서만 잡는다.
    direction = part_volumes / np.linalg.norm(part_volumes)
    along = (mean_density * scale_rel_error) ** 2
    across = redistribution_sigma ** 2
    projector = np.outer(direction, direction)
    block = across * (np.eye(n_part) - projector) + along * projector

    Sigma = np.zeros((len(table), len(table)))
    Sigma[:n_part, :n_part] = block
    for index in range(n_part, len(table)):
        Sigma[index, index] = (hinge_rel_sigma * table[index]["rho_gt"]) ** 2
    return mu, Sigma, mean_density


def assembled_mass_kg(spec, densities=None):
    """저울에 올렸을 때 실제로 읽히는 총질량. 힌지도 같이 달려 있다."""
    table = body_table(spec)
    if densities is None:
        densities = [row["rho_gt"] for row in table]
    return float(sum(row["volume_m3"] * rho
                     for row, rho in zip(table, densities)))


def apply_weight_prior(spec, total_mass_kg, **kwargs):
    """추정기를 '저울로 총무게만 잰 상태'로 초기화한다."""
    mu, Sigma, mean_density = weight_prior(spec, total_mass_kg, **kwargs)
    alg.MU0 = mu
    alg.SIGMA0 = Sigma
    alg.RHO_BOUNDS = (50.0, 20000.0)
    return mu, Sigma, mean_density


def apply_physical_prior(spec, fill="lead"):
    """추정기의 사전분포와 박스 제약을 물리적 한계로 교체한다."""
    mu, Sigma, lows, highs = physical_prior(spec, fill)
    alg.MU0 = mu
    alg.SIGMA0 = Sigma
    alg.RHO_BOUNDS = (float(lows.min()), float(highs.max()))
    return mu, Sigma, lows, highs


@dataclass(frozen=True)
class Part:
    name: str
    bbox_mm: tuple            # 외형 (x, y, z)
    volume_cm3: float         # 외부 부피 GT (리세스 공제 후)
    rho_gt: float             # 이번 검증에서 부여한 part별 밀도 GT [kg/m^3]
    bbox_center_in_link_mm: tuple    # 링크 프레임 기준 외형 중심 = body frame 원점
    shell_centroid_in_link_mm: tuple  # geometry.csv 의 실제 셸 도심
    color: tuple
    rho_empty: float = 0.0    # 추 없이 출력만 한 상태의 평균 밀도 (물리적 하한)
    cavity_cm3: float = 0.0   # 추를 넣을 수 있는 내부 공동 부피
    # 메시에서 온 부위일 때: 도심 기준 단위질량당 관성텐서 [m^2] (3x3).
    # None 이면 bbox 를 직육면체로 보고 계산한다. 준정적 렌치에는 영향이
    # 없고(질량과 위치만 쓴다), URDF 로 내보낼 때만 쓰인다.
    inertia_unit: object = None
    # 스캔 메시를 쓰는 부위의 화면·충돌 형상.
    #
    #   visual_mesh      화면에 그릴 .obj 하나
    #   collision_meshes 충돌 판정용 **볼록** .obj 조각들
    #   mesh_offset_m    몸체 프레임(=도심) 에서 메시 좌표계로 가는 평행이동
    #
    # 없으면 bbox 를 직육면체로 그리고 충돌도 그 상자로 본다. 3-link 처럼
    # 실제로 직육면체인 물체는 그래도 되지만, 스탠드 램프처럼 가는 팔이
    # 휘어 있는 물체는 AABB 가 실물보다 훨씬 뚱뚱해서 충돌이 과하게 잡힌다.
    # (램프 Arm 은 AABB 가 70x256x274 mm 지만 실제 재료 부피는 그 1/3 이다)
    visual_mesh: str = None
    collision_meshes: tuple = ()
    mesh_offset_m: tuple = (0.0, 0.0, 0.0)
    # 스캔 색을 화면에 내기 위한 조각들: ((.obj 경로, (r,g,b,a)), ...).
    # 비어 있지 않으면 visual_mesh 대신 이 조각들을 그린다. Drake 는 메시
    # 하나에 단색 하나만 입히므로, 색이 비슷한 면끼리 묶어 나눈 것이다.
    visual_pieces: tuple = ()
    # 그리퍼 죠가 실제로 물어야 하는 단면 [mm]. None 이면 AABB 로 짐작한다.
    #
    # AABB 로 짐작하면 안 되는 경우가 있다. 스탠드 램프의 연결부는 굽은
    # 팔이라 AABB 가 70x256x274 mm 지만, 파지점 근처의 실제 단면은 그보다
    # 훨씬 얇다. AABB 를 쓰면 "개구 256 mm 가 필요하다" 는 엉뚱한 값이 나온다.
    grasp_width_mm: float = None
    # 그 폭이 **물체 좌표계의 어느 축** 방향인가 (0=x, 1=y, 2=z).
    # 죠는 이 축에 맞춰 오므린다. None 이면 AABB 에서 가장 좁은 축을 쓴다.
    grasp_axis: int = None

    @property
    def volume_m3(self):
        return self.volume_cm3 * CM3

    @property
    def mass_kg(self):
        return self.rho_gt * self.volume_m3


@dataclass(frozen=True)
class Joint:
    name: str
    parent: str
    child: str
    origin_in_parent_link_mm: tuple   # URDF joint origin (부모 링크 프레임)
    axis: tuple
    limits_rad: tuple
    # 자식 링크 프레임에서 본 관절 축 위치. None 이면 (0,0,0) — 자식 프레임
    # 원점이 곧 관절 축이라는 뜻이고, 스캔이 내놓는 URDF 는 보통 그렇다.
    #
    # 트리를 다른 링크 기준으로 다시 세울 때 필요하다. 예를 들어 사슬
    # base->link_2->link_3->link_1 에서 link_3 을 잡으면 link_2 가 자식이
    # 되는데, link_2 의 프레임 원점은 관절 축 위가 아니다. 그때 이 값이
    # 관절 축이 link_2 프레임 어디인지를 알려준다.
    origin_in_child_link_mm: tuple = None
    # 힌지 금속/플라스틱 부품의 실측 질량. 0 이면 힌지를 무시한다.
    #
    # 무시하면 안 된다. 힌지 41 g 짜리를 빼놓고 풀면 link1_elbow 밀도가
    # 29 % 틀리는데, 알고리즘은 2.9 % 라고 주장한다 (study_hinge.py).
    # 힌지 질량은 잡음이 아니라 모형이 빠뜨린 것이라 라운드를 늘려도
    # 안 없어지고, 시뮬레이션에서는 양쪽 plant 가 똑같이 무시하므로
    # 아예 보이지도 않는다.
    #
    # 힌지는 핀 축 위에 붙어 있으므로 부모/자식 어느 쪽에 달아도 중력
    # 렌치가 같다. 여기서는 자식 링크에 단다.
    hinge_mass_kg: float = 0.0
    hinge_density: float = HINGE_MATERIAL_DENSITY
    # 힌지 무게중심이 핀 축에서 얼마나 벗어나 있는가 (자식 링크 프레임, mm).
    #
    # 기본 (0,0,0) 은 "핀 축 위" 라는 뜻인데, 실제 힌지는 날개가 한쪽으로
    # 뻗어 있어 도심이 벗어나 있다. 이게 왜 중요한지:
    #
    #   도심 오프셋 5 mm  -> 토크 오차 2.0 mN·m = 센서 잡음의 11 배
    #              10 mm  -> 4.0 mN·m = 23 배
    #
    # 질량은 저울로 5 초면 재지만 위치는 재기 어렵다. 두 가지 방법이 있다.
    #   (a) 메시 스캔에서 힌지를 따로 분할해 도심을 읽는다
    #   (b) 힌지만 떼어 두 자세로 저울에 올려 도심을 역산한다
    # (b) 가 더 정확하다. 토크힌지 안에는 금속 핀과 스프링이 들어 있어
    # 균일밀도가 아니고, 메시 도심(=기하 중심)과 무게중심이 다르기 때문이다.
    hinge_com_offset_mm: tuple = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class ObjectSpec:
    key: str
    label: str
    parts: list
    joints: list
    base_bbox_center_in_sensor_mm: tuple  # 센서 원점 기준 base 외형 중심
    notes: str = ""
    camera: tuple = field(default=((0.45, -0.55, 0.30), (0.10, 0.0, 0.0)))


# ---------------------------------------------------------------------------
# 2-link (v2)
# ---------------------------------------------------------------------------
# 센서 원점 = Parent의 -x 끝면 중심(파지점). Parent 링크 프레임과 일치.
# 핀 축: x = 75 + 4/2 = 77 mm (간격 4 mm의 중앙), z = 20 + 5 = 25 mm, 축은 y 평행.
# Child 링크 프레임은 핀 축 위. Child는 x = +2 .. +62 를 차지하므로 중심은 +32.
# q=0 이 일직선, +q 가 Child를 +z 쪽으로 접는 방향 -> axis = -y.
# 2-link 는 CAD 도심 표가 제공되지 않아 셸 도심을 외형 중심과 같게 둔다.
TWO_LINK = ObjectSpec(
    key="2link",
    label="2-link custom object (v2)",
    parts=[
        Part("parent", (75.0, 46.0, 40.0), 138.0, 1800.0,
             (37.5, 0.0, 0.0), (37.5, 0.0, 0.0), (0.30, 0.45, 0.80, 1.0), 515.0, 94.0),
        Part("child", (60.0, 46.0, 40.0), 110.4, 3600.0,
             (32.0, 0.0, -25.0), (32.0, 0.0, -25.0), (0.90, 0.55, 0.15, 1.0), 553.0, 73.0),
    ],
    joints=[
        Joint("hinge", "parent", "child", (77.0, 0.0, 25.0), (0.0, -1.0, 0.0),
              (0.0, np.pi), hinge_mass_kg=MEASURED_HINGE_KG),
    ],
    base_bbox_center_in_sensor_mm=(37.5, 0.0, 0.0),
    notes="Parent를 파지, Child가 힌지로 매달림",
    camera=((0.34, -0.40, 0.24), (0.09, 0.0, 0.02)),
)

# ---------------------------------------------------------------------------
# 3-link (v3) — URDF 값 그대로
# ---------------------------------------------------------------------------
# 링크 프레임: link0 = -X 끝면 중심, link1/link2 = 각 관절 핀 축 위.
# 표면 장착 힌지라 핀 축이 링크 중심선에서 27 mm 벗어나 있고, 링크 사이는
# 4 mm 떠 있다. 따라서 자식 링크는 자기 프레임에서 x = +2 부터 시작한다.
#   link0 : x 0..150,   중심선 y=0,   z=0    -> 외형 중심 (75, 0, 0)
#   link1 : x +2..+112, 중심선 y=-27, z=0    -> 외형 중심 (57, -27, 0)
#   link2 : x +2..+87,  중심선 y=0,   z=-27  -> 외형 중심 (44.5, 0, -27)
# 셸 도심(geometry.csv)은 뚜껑·힌지 패드 때문에 외형 중심과 다르다.
THREE_LINK = ObjectSpec(
    key="3link",
    label="3-link custom object (v3)",
    parts=[
        Part("link0_base", (150.0, 44.0, 44.0), 288.12, 5200.0,
             (75.0, 0.0, 0.0), (92.42, -0.50, 3.13), (0.25, 0.55, 0.35, 1.0), 239.0, 176.0),
        Part("link1_elbow", (110.0, 44.0, 44.0), 208.40, 1400.0,
             (57.0, -27.0, 0.0), (57.00, -27.43, 0.94), (0.90, 0.40, 0.10, 1.0), 356.0, 87.0),
        Part("link2_tip", (85.0, 44.0, 44.0), 162.28, 700.0,
             (44.5, 0.0, -27.0), (34.71, 0.0, -25.68), (0.45, 0.30, 0.80, 1.0), 312.0, 82.0),
    ],
    joints=[
        Joint("joint1", "link0_base", "link1_elbow", (152.0, 27.0, 0.0),
              (0.0, 0.0, 1.0), (0.0, np.pi), hinge_mass_kg=MEASURED_HINGE_KG),
        Joint("joint2", "link1_elbow", "link2_tip", (114.0, -27.0, 27.0),
              (0.0, -1.0, 0.0), (0.0, np.pi), hinge_mass_kg=MEASURED_HINGE_KG),
    ],
    base_bbox_center_in_sensor_mm=(75.0, 0.0, 0.0),
    notes="link0을 파지, joint1은 좌우(Z축) / joint2는 상하(Y축)",
    camera=((0.42, -0.62, 0.34), (0.13, -0.03, -0.02)),
)

OBJECTS = {spec.key: spec for spec in (TWO_LINK, THREE_LINK)}


# ---------------------------------------------------------------------------
# 힌지를 '부위' 로 취급하기
#
# 힌지는 관절 축 위에 붙은 덩어리다. 지금까지는 질량 0 으로 뒀는데, 실측
# 41 g 이면 무시할 수 없다 (study_hinge.py). 그렇다고 특별 취급할 이유도 없다.
# 부피 = 질량 / 재질밀도 로 환산해 **부위 하나** 로 넣으면, 회귀행렬·측정·
# 추정·URDF 가 전부 그대로 돌아간다. 밀도 x 부피 = 질량 만 맞으면 되기 때문.
#
# 알려진 값으로 쓸지 미지수로 풀지는 **사전분포의 폭** 하나로 갈린다.
#   좁게 (기본)  : 저울로 쟀으니 그 값으로 고정. 미지수가 안 늘어난다.
#   넓게         : 접착제·나사까지 얼마인지 모를 때. 자세가 하나 더 필요하다.
# ---------------------------------------------------------------------------
def hinge_bodies(spec):
    """질량이 주어진 힌지만 골라 (관절, 부피[m^3], 밀도) 로 돌려준다."""
    out = []
    for joint in spec.joints:
        if joint.hinge_mass_kg > 0.0:
            out.append((joint, joint.hinge_mass_kg / joint.hinge_density,
                        joint.hinge_density))
    return out


def body_table(spec):
    """추정 대상 전체. 부위 먼저, 힌지 나중. 순서가 rho 벡터의 순서다."""
    rows = [dict(name=p.name, volume_m3=p.volume_m3, rho_gt=p.rho_gt,
                 kind="part") for p in spec.parts]
    for joint, volume, density in hinge_bodies(spec):
        rows.append(dict(name=f"{joint.name}_hinge", volume_m3=volume,
                         rho_gt=density, kind="hinge", joint=joint.name))
    return rows


HINGE_PLATE_MM = 44.0               # 힌지 날개가 덮는 면 (링크 단면과 같다)


def hinge_dims_m(volume_m3):
    """시각화·관성용 판 모양 치수. 준정적 렌치에는 영향이 없다.

    렌치는 질량과 위치만으로 정해지므로 모양은 자유다. 화면에서 힌지처럼
    보이도록 링크 단면과 같은 넓이의 얇은 판으로 둔다.
    """
    face = HINGE_PLATE_MM * MM
    return (face, face, float(volume_m3) / (face * face))


def register_part_visual(plant, body, part, dims_m, prefix=""):
    """부위 하나의 화면 형상을 등록한다. 세 화면이 모두 이 함수를 쓴다.

    우선순위는 색 조각 -> 메시 하나 -> AABB 상자다. 색 조각은 스캔의 정점
    색을 무리지어 나눈 것으로, Drake 가 메시 하나에 단색 하나만 입히기
    때문에 필요하다 (mesh_props.split_by_color).
    """
    pose = RigidTransform(np.array(part.mesh_offset_m))
    if getattr(part, "visual_pieces", ()):
        for index, (path, rgba) in enumerate(part.visual_pieces):
            plant.RegisterVisualGeometry(
                body, pose, Mesh(path, 1.0),
                f"{prefix}{part.name}_visual_{index}", rgba)
    elif part.visual_mesh:
        plant.RegisterVisualGeometry(
            body, pose, Mesh(part.visual_mesh, 1.0),
            f"{prefix}{part.name}_visual", part.color)
    else:
        plant.RegisterVisualGeometry(
            body, RigidTransform(), Box(*dims_m),
            f"{prefix}{part.name}_visual", part.color)


# ---------------------------------------------------------------------------
# Drake plant 생성 (body frame = 각 part의 외형 도심)
# ---------------------------------------------------------------------------
def build_plant(spec, densities, builder=None, shell_com=False):
    """body frame 원점 = 각 part의 외형(bbox) 중심.

    shell_com=True 면 무게중심만 geometry.csv 의 셸 도심으로 옮긴다. 균일밀도
    가정이 틀렸을 때 추정이 얼마나 망가지는지 보기 위한 옵션이며, 추정기 쪽
    plant 에는 절대 쓰지 않는다.
    """
    if builder is None:
        plant = MultibodyPlant(time_step=0.0)
        scene_graph = None
    else:
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)

    parts = {p.name: p for p in spec.parts}
    bodies = {}
    for part, rho in zip(spec.parts, densities):
        mass = rho * part.volume_m3
        dims_m = tuple(d * MM for d in part.bbox_mm)
        com = np.zeros(3)
        if shell_com:
            com = (np.array(part.shell_centroid_in_link_mm)
                   - np.array(part.bbox_center_in_link_mm)) * MM
        # UnitInertia 는 body frame 원점 기준이어야 하므로 무게중심에서 옮긴다.
        unit_inertia = UnitInertia.SolidBox(*dims_m).ShiftFromCenterOfMass(-com)
        bodies[part.name] = plant.AddRigidBody(
            part.name, SpatialInertia(mass, com, unit_inertia)
        )
        if scene_graph is not None:
            register_part_visual(plant, bodies[part.name], part, dims_m)

    # base를 센서 프레임에 고정
    plant.WeldFrames(
        plant.world_frame(),
        bodies[spec.parts[0].name].body_frame(),
        RigidTransform(np.array(spec.base_bbox_center_in_sensor_mm) * MM),
    )

    for joint in spec.joints:
        origin = np.array(joint.origin_in_parent_link_mm)
        # URDF 값은 링크 프레임 기준이므로 외형 중심 기준으로 옮긴다.
        on_parent = (origin
                     - np.array(parts[joint.parent].bbox_center_in_link_mm)) * MM
        child_origin = np.array(joint.origin_in_child_link_mm
                                if joint.origin_in_child_link_mm is not None
                                else (0.0, 0.0, 0.0))
        on_child = (child_origin
                    - np.array(parts[joint.child].bbox_center_in_link_mm)) * MM
        plant.AddJoint(
            RevoluteJoint(
                joint.name,
                plant.AddFrame(
                    FixedOffsetFrame(
                        f"{joint.name}_parent",
                        bodies[joint.parent].body_frame(),
                        RigidTransform(on_parent),
                    )
                ),
                plant.AddFrame(
                    FixedOffsetFrame(
                        f"{joint.name}_child",
                        bodies[joint.child].body_frame(),
                        RigidTransform(on_child),
                    )
                ),
                joint.axis,
                damping=0.0,
            )
        )
    # 힌지: 핀 축 위에 붙은 부위 하나. 자식 링크에 용접한다.
    # (핀 축 위라 부모에 달아도 중력 렌치는 같다.)
    n_part = len(spec.parts)
    for index, (joint, volume, _) in enumerate(hinge_bodies(spec)):
        rho = densities[n_part + index]
        dims = hinge_dims_m(volume)
        name = f"{joint.name}_hinge"
        bodies[name] = plant.AddRigidBody(
            name, SpatialInertia(rho * volume, np.zeros(3),
                                 UnitInertia.SolidBox(*dims)))
        if scene_graph is not None:
            plant.RegisterVisualGeometry(bodies[name], RigidTransform(),
                                         Box(*dims), f"{name}_visual",
                                         (0.15, 0.15, 0.18, 1.0))
        on_child = ((-np.array(parts[joint.child].bbox_center_in_link_mm)
                     + np.array(joint.hinge_com_offset_mm)) * MM)
        plant.WeldFrames(
            plant.AddFrame(FixedOffsetFrame(
                f"{name}_mount", bodies[joint.child].body_frame(),
                RigidTransform(on_child))),
            bodies[name].body_frame())

    plant.Finalize()
    return plant, bodies


# ---------------------------------------------------------------------------
# 추정기(density_id_drake)의 대상 물체를 이 물체로 갈아끼운다.
# 추정기 함수 자체는 전혀 수정하지 않는다.
# ---------------------------------------------------------------------------
def bind_object(spec, shell_com=False, hinge=None, safety=DEFAULT_SAFETY,
                density_scale=1.0):
    """density_scale 은 하드웨어 한계에 맞춰 밀도 GT 를 통째로 줄일 때 쓴다.
    base(파지되는 part)는 관절 토크를 받지 않으므로 배율에서 제외한다."""
    table = body_table(spec)
    rho_gt = np.array([row["rho_gt"] for row in table], dtype=float)
    if density_scale != 1.0:
        for index, row in enumerate(table):
            if index and row["kind"] == "part":
                rho_gt[index] *= density_scale
    volumes = np.array([row["volume_m3"] for row in table])

    # 진리(측정) plant 에만 셸 도심을 반영할 수 있다. 추정기 쪽은 항상 균일밀도.
    truth_plant, truth_bodies = build_plant(spec, rho_gt, shell_com=shell_com)
    kin_plant, kin_bodies = build_plant(spec, np.ones(len(table)))

    alg.PARTS = ([(p.name, tuple(d * MM for d in p.bbox_mm), p.rho_gt)
                  for p in spec.parts]
                 + [(f"{j.name}_hinge", hinge_dims_m(v), d)
                    for j, v, d in hinge_bodies(spec)])
    alg.VOLUMES = volumes
    alg.TRUE_RHO = rho_gt
    alg.P = len(table)
    alg.JOINT_LIMITS = [j.limits_rad for j in spec.joints]
    alg.TRUTH_PLANT, alg.TRUTH_BODIES = truth_plant, truth_bodies
    alg.TRUTH_CTX = truth_plant.CreateDefaultContext()
    alg.KIN_PLANT, alg.KIN_BODIES = kin_plant, kin_bodies
    alg.KIN_CTX = kin_plant.CreateDefaultContext()
    alg.MU0 = np.full(alg.P, 1000.0)
    alg.SIGMA0 = np.diag(np.full(alg.P, 3000.0**2))

    # S3 feasibility 훅: 관절이 고정되는 자세만 후보로 남긴다.
    if hinge is None:
        alg.is_feasible = lambda theta: True
    else:
        alg.is_feasible = make_is_feasible(spec, hinge, safety, rho_gt)
    return rho_gt


# ---------------------------------------------------------------------------
# 힌지 정지토크 대비 실현 가능성 확인 (밀도 GT가 실제로 만들 수 있는 값인가)
# ---------------------------------------------------------------------------
def max_joint_torques(spec, thetas, densities=None):
    """자세별로 각 관절이 받는 중력토크의 최댓값(중력 3방향에 대해).

    측정 1회는 중력 3방향을 모두 쓰므로, 그 자세를 쓰려면 세 방향 전부에서
    관절이 버텨야 한다. 따라서 방향에 대해 max 를 취한다.
    """
    if densities is None:
        densities = [row["rho_gt"] for row in body_table(spec)]
    plant, _ = build_plant(spec, densities)
    context = plant.CreateDefaultContext()
    result = np.zeros((len(thetas), len(spec.joints)))
    for g_hat in alg.G_DIRS:
        plant.mutable_gravity_field().set_gravity_vector(alg.G_ACC * g_hat)
        for index, theta in enumerate(thetas):
            plant.SetPositions(context, np.asarray(theta))
            result[index] = np.maximum(
                result[index], np.abs(plant.CalcGravityGeneralizedForces(context))
            )
    return result


def make_is_feasible(spec, hinge, safety=DEFAULT_SAFETY, densities=None):
    """알고리즘 S3 의 feasibility 훅. 관절이 고정되는 자세만 통과시킨다."""
    limit = hinge.holding_torque_nm / safety
    if densities is None:
        densities = [row["rho_gt"] for row in body_table(spec)]
    plant, _ = build_plant(spec, densities)
    context = plant.CreateDefaultContext()

    def is_feasible(theta):
        for g_hat in alg.G_DIRS:
            plant.mutable_gravity_field().set_gravity_vector(alg.G_ACC * g_hat)
            plant.SetPositions(context, np.asarray(theta))
            if np.any(np.abs(plant.CalcGravityGeneralizedForces(context)) > limit):
                return False
        return True

    return is_feasible


def full_grid(spec, n_per_joint=7):
    axes = [np.linspace(lo, hi, n_per_joint)
            for lo, hi in (j.limits_rad for j in spec.joints)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(
        -1, len(spec.joints)
    )


def required_torque(spec, densities=None, n_per_joint=7):
    """전체 후보 자세를 모두 쓰려면 필요한 최소 힌지 토크 [N·m]."""
    grid = full_grid(spec, n_per_joint)
    return max_joint_torques(spec, grid, densities).max(axis=0)


def max_feasible_density_scale(spec, hinge, safety=DEFAULT_SAFETY,
                               n_per_joint=7):
    """모든 후보 자세를 쓸 수 있게 하려면 하류 part 밀도를 몇 배로 줄여야 하는가.

    중력토크는 하류 질량에 선형이므로, 관절별 여유의 최솟값이 그대로 배율이다.
    """
    needed = required_torque(spec, n_per_joint=n_per_joint)
    limit = hinge.holding_torque_nm / safety
    return float(np.min(limit / needed))


def hinge_load_check(spec, hinge, densities=None, safety=DEFAULT_SAFETY, steps=13):
    """각 관절이 받는 중력 토크의 최악값을 Drake로 직접 계산한다.

    손으로 레버암을 근사하지 않고, 관절각 격자 x 3개 중력 방향을 전수 스윕하며
    CalcGravityGeneralizedForces 로 관절 토크를 그대로 읽는다. 물체를 로봇이
    다시 세우므로 중력 방향은 물체 기준으로 바뀐다.
    """
    if densities is None:
        densities = [row["rho_gt"] for row in body_table(spec)]
    densities = np.asarray(densities, dtype=float)
    plant, _ = build_plant(spec, densities)
    context = plant.CreateDefaultContext()
    axes = [np.linspace(lo, hi, steps) for lo, hi in
            (j.limits_rad for j in spec.joints)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(
        -1, len(spec.joints)
    )

    worst = np.zeros(len(spec.joints))
    worst_at = [None] * len(spec.joints)
    for g_hat in alg.G_DIRS:
        plant.mutable_gravity_field().set_gravity_vector(alg.G_ACC * g_hat)
        for theta in grid:
            plant.SetPositions(context, theta)
            tau = np.abs(plant.CalcGravityGeneralizedForces(context))
            for index in range(len(spec.joints)):
                if tau[index] > worst[index]:
                    worst[index] = tau[index]
                    worst_at[index] = (theta.copy(), np.array(g_hat))

    mass_of = {p.name: rho * p.volume_m3
               for p, rho in zip(spec.parts, densities)}
    rows = []
    for index, joint in enumerate(spec.joints):
        downstream = [j.child for j in spec.joints[index:]]
        rows.append(
            dict(
                joint=joint.name,
                downstream_mass_g=1000.0 * sum(mass_of[n] for n in downstream),
                torque_nm=float(worst[index]),
                margin=(hinge.holding_torque_nm / worst[index]
                        if worst[index] > 0 else np.inf),
                theta_deg=np.degrees(worst_at[index][0]),
            )
        )
    return rows


# ---------------------------------------------------------------------------
def torque_sweep(spec, safety=DEFAULT_SAFETY,
                 torques_nm=(0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.6, 0.9)):
    """힌지 유지토크에 따라 (a) 쓸 수 있는 자세 수와 (b) 허용 밀도가 어떻게 변하는가.

    토크값이 공개되지 않은 힌지라, 실측값이 나오면 이 표에서 바로 읽으면 된다.
    """
    print(f"\n[토크 스윕] {spec.label} — 안전배수 {safety}")
    grid = full_grid(spec)
    needed = required_torque(spec)
    print(f"  현재 밀도 GT 로 전체 자세를 쓰려면 관절별 최소"
          f" {np.round(needed, 3)} N·m 필요")
    empty = " / ".join(f"{p.rho_empty:.0f}" for p in spec.parts[1:])
    print(f"  하류 part 의 빈 출력물 밀도(물리적 하한): {empty} kg/m^3")

    # 하류를 추 없이 비워둔 상태가 물리적으로 가장 가벼운 구성이다.
    # 이 구성조차 버티지 못하면 어떤 밀도를 골라도 이 힌지로는 실험이 불가능하다.
    lightest = [spec.parts[0].rho_gt] + [p.rho_empty for p in spec.parts[1:]]
    floor = required_torque(spec, densities=lightest).max()
    print(f"  하류를 전부 비운 최경량 구성에서도 {floor:.3f} N·m 필요"
          f"  -> 유지토크가 {floor * safety:.3f} N·m 미만이면 실험 자체가 불가")
    print("  '현재GT 자세' = 지금 밀도 그대로 썼을 때 고정되는 자세 수")
    print("  '배율/상한'   = 전체 자세를 쓰려면 하류 밀도를 얼마로 낮춰야 하는가")
    print(f"  {'유지토크':>9}{'현재GT 자세':>12}{'배율':>9}"
          f"{'하류 밀도 상한':>18}   판정")
    for tau in torques_nm:
        hinge = Hinge("sweep", tau, "")
        is_feasible = make_is_feasible(spec, hinge, safety)
        count = sum(1 for theta in grid if is_feasible(theta))
        scale = max_feasible_density_scale(spec, hinge, safety)
        caps = [p.rho_gt * scale for p in spec.parts[1:]]
        # 상한이 빈 출력물 밀도보다 낮으면 추를 빼도 만들 수 없다.
        impossible = [p.name for p, c in zip(spec.parts[1:], caps)
                      if c < p.rho_empty]
        verdict = ("불가: " + ", ".join(impossible) if impossible
                   else "사용가능" if count else "자세 없음")
        text = " / ".join(f"{c:.0f}" for c in caps)
        print(f"  {tau:>7.2f} N·m{count:>6}/{len(grid):<4}"
              f"{scale:>9.3f}x{text:>18}   {verdict}")
    return needed


# ---------------------------------------------------------------------------
# 추정한 밀도로부터 질량 · 무게중심 · 관성모멘트를 계산한다.
#
# 밀도만으로는 로봇이 물체를 다룰 수 없다. 실제로 필요한 것은 질량(들 수 있는가),
# 무게중심(어디를 잡아야 하는가), 관성모멘트(얼마나 빨리 돌릴 수 있는가)다.
# 균일밀도 가정 아래에서는 셋 다 밀도의 선형함수이므로, 추정한 밀도를 그대로
# Drake plant 에 넣고 Drake 가 합성하게 하면 된다.
# ---------------------------------------------------------------------------
def derived_quantities(spec, densities, theta):
    """밀도 -> (부위별 질량·관성, 물체 전체 질량·CoM·관성텐서).

    물체 전체 값은 관절각 theta 에 의존한다. 관성텐서는 물체의 무게중심을
    기준으로, 센서 좌표계 축으로 표현한다.
    """
    plant, bodies = build_plant(spec, densities)
    context = plant.CreateDefaultContext()
    plant.SetPositions(context, np.atleast_1d(np.asarray(theta, dtype=float)))

    per_part = []
    for part, rho in zip(spec.parts, densities):
        body = bodies[part.name]
        spatial = body.CalcSpatialInertiaInBodyFrame(context)
        rotational = spatial.CalcRotationalInertia().CopyToFullMatrix3()
        per_part.append(
            dict(
                name=part.name,
                mass_kg=rho * part.volume_m3,
                # body frame 원점이 곧 외형 중심이므로 이 값은 도심 기준 관성이다.
                inertia_diag=np.diag(rotational).copy(),
            )
        )

    total = plant.CalcSpatialInertia(
        context, plant.world_frame(), [b.index() for b in bodies.values()]
    )
    com = np.array(total.get_com())
    # 원점 기준 -> 무게중심 기준으로 옮긴다 (평행축 정리).
    about_origin = total.CalcRotationalInertia().CopyToFullMatrix3()
    mass = total.get_mass()
    shift = mass * ((com @ com) * np.eye(3) - np.outer(com, com))
    about_com = about_origin - shift

    return dict(
        per_part=per_part,
        total_mass_kg=float(mass),
        com_m=com,
        inertia_about_com=about_com,
    )


def report_derived(spec, rho_gt, rho_hats, theta, label):
    """추정 밀도로 계산한 파생량과 그 오차. rho_hats 는 시드별 추정 리스트."""
    truth = derived_quantities(spec, rho_gt, theta)
    estimates = [derived_quantities(spec, rho, theta) for rho in rho_hats]

    n = len(estimates)
    print(f"\n[파생량] {label}  (관절각 {np.round(np.degrees(theta), 0)} deg)")
    print(f"  오차는 시드 {n}회 각각의 오차를 평균낸 값이다."
          f" 즉 실험 1회에서 기대되는 오차.")

    # --- 부위별 질량과 관성모멘트 ---
    # 균일밀도 가정에서 관성은 질량 x (고정된 형상 관성)이므로 상대오차가
    # 질량 상대오차와 정확히 같다. 따라서 오차는 한 번만 싣는다.
    print(f"\n  {'부위':<13}{'질량 GT':>10}{'질량 오차':>16}"
          f"{'관성 GT (Ixx, Iyy, Izz)':>34}")
    for index, part in enumerate(spec.parts):
        m_gt = truth["per_part"][index]["mass_kg"]
        m_hat = np.array([e["per_part"][index]["mass_kg"] for e in estimates])
        errs = 100 * np.abs(m_hat - m_gt) / m_gt
        i_gt = truth["per_part"][index]["inertia_diag"]
        diag = ", ".join(f"{v * 1e6:.1f}" for v in i_gt)
        print(f"  {part.name:<13}{1000 * m_gt:>8.1f} g"
              f"{errs.mean():>9.2f} +/-{errs.std():>4.2f}%{diag:>34}")
    print(f"  {'':13}{'':10}{'':16}{'단위 1e-6 kg·m^2':>34}")
    print(f"  관성모멘트의 상대오차는 질량 상대오차와 동일하다"
          f" (I = m x 형상관성, 형상은 이미 앎).")

    # --- 물체 전체 ---
    m_gt = truth["total_mass_kg"]
    m_hat = np.array([e["total_mass_kg"] for e in estimates])
    c_gt = truth["com_m"]
    c_hat = np.array([e["com_m"] for e in estimates])
    i_gt = truth["inertia_about_com"]
    i_hat = np.array([e["inertia_about_com"] for e in estimates])

    mass_errs = 100 * np.abs(m_hat - m_gt) / m_gt
    com_errs = 1000 * np.linalg.norm(c_hat - c_gt, axis=1)
    inertia_errs = 100 * np.array([
        np.linalg.norm(i - i_gt) / np.linalg.norm(i_gt) for i in i_hat
    ])

    print(f"\n  물체 전체 (부위를 합성한 값 — 로봇이 실제로 쓰는 양)")
    print(f"    총질량     GT {1000 * m_gt:>8.1f} g"
          f"    오차 {mass_errs.mean():>6.3f} +/-{mass_errs.std():.3f}%")
    print(f"    무게중심   GT ({', '.join(f'{v * 1000:6.1f}' for v in c_gt)}) mm"
          f"    오차 {com_errs.mean():>6.3f} +/-{com_errs.std():.3f} mm"
          f"  (최대 {com_errs.max():.3f})")
    print(f"    관성텐서   GT 대각 ({', '.join(f'{v * 1e6:7.1f}' for v in np.diag(i_gt))})"
          f" 1e-6 kg·m^2")
    print(f"    (무게중심 기준)      오차 {inertia_errs.mean():>6.3f}"
          f" +/-{inertia_errs.std():.3f}%   (Frobenius 노름)")
    return dict(truth=truth, estimates=estimates,
                mass_errs=mass_errs, com_errs=com_errs,
                inertia_errs=inertia_errs)


def com_mismatch_study(spec, n_rounds=6, n_seeds=10, hinge=None,
                       safety=DEFAULT_SAFETY, density_scale=1.0):
    """무게중심이 외형 중심이 아닐 때 추정이 얼마나 틀어지는가.

    이 실험의 전제는 **밀도가 균일**하다는 것이므로 이것은 기본 경로가
    아니라 전제가 깨졌을 때를 보는 강건성 점검이다. --com-mismatch 로 켠다.

    진리 plant 의 무게중심만 geometry.csv 의 셸 도심으로 옮기고, 추정기는
    균일밀도(외형 중심)를 그대로 믿게 둔다. 추를 넣기 전 빈 출력물의
    무게중심 편차에 해당한다.
    """
    rho_gt = bind_object(spec, shell_com=True, hinge=hinge, safety=safety,
                         density_scale=density_scale)
    finals = np.array([
        alg.run_loop(alg.select_active, n_rounds, seed=s)[-1]["rho"]
        for s in range(n_seeds)
    ])
    rho_mean = finals.mean(axis=0)
    offsets = [
        np.linalg.norm(np.array(p.shell_centroid_in_link_mm)
                       - np.array(p.bbox_center_in_link_mm))
        for p in spec.parts
    ]
    print("\n[모델 불일치] 무게중심을 셸 도심으로 옮겼을 때 (균일밀도 가정 위반)")
    print(f"  {'part':<13}{'CoM 편차':>10}{'GT':>9}{'추정평균':>11}{'오차':>9}")
    for index, part in enumerate(spec.parts):
        error = 100 * abs(rho_mean[index] - rho_gt[index]) / rho_gt[index]
        print(f"  {part.name:<13}{offsets[index]:>7.1f} mm{rho_gt[index]:>9.0f}"
              f"{rho_mean[index]:>11.1f}{error:>8.2f}%")
    max_rel = np.max(np.abs(rho_mean - rho_gt) / rho_gt)
    print(f"  최대 상대오차 {100 * max_rel:.2f}%")
    bind_object(spec, hinge=hinge, safety=safety, density_scale=density_scale)
    return max_rel


def validate(spec, hinge, n_rounds=6, n_random_seeds=10,
             safety=DEFAULT_SAFETY, density_scale=1.0):
    rho_gt = bind_object(spec, hinge=hinge, safety=safety,
                         density_scale=density_scale)
    print("=" * 74)
    print(f"{spec.label}  —  {spec.notes}")
    print(f"힌지: {hinge.label}, 유지토크 {hinge.holding_torque_nm} N·m"
          f" ({hinge.note}), 안전배수 {safety}")
    print("=" * 74)

    scaled = "" if density_scale == 1.0 else f"  (하류 밀도 x{density_scale:.3f})"
    print(f"\n[GT] part별로 서로 다른 밀도를 부여{scaled}")
    print(f"  {'part':<13}{'외부부피':>11}{'밀도GT':>11}{'질량':>10}"
          f"{'빈출력물':>10}")
    impossible = []
    for part, rho in zip(spec.parts, rho_gt):
        mass = rho * part.volume_m3
        flag = ""
        if part.rho_empty and rho < part.rho_empty:
            flag = "  <- 빈 출력물보다 낮음(제작 불가)"
            impossible.append(part.name)
        print(f"  {part.name:<13}{part.volume_cm3:>9.2f} cm3"
              f"{rho:>9.0f}  {1000 * mass:>7.1f} g{part.rho_empty:>9.0f}{flag}")
    total = sum(r * p.volume_m3 for p, r in zip(spec.parts, rho_gt))
    print(f"  {'합계':<13}{'':>11}{'':>11}{1000 * total:>7.1f} g")
    if impossible:
        print(f"  경고: {', '.join(impossible)} 는 추를 다 빼도 이 밀도를 만들 수 없다.")

    print(f"\n[하드웨어] 유지토크 {hinge.holding_torque_nm} N·m 대비"
          f" (Drake 중력토크 전수 스윕)")
    for row in hinge_load_check(spec, hinge, densities=rho_gt, safety=safety):
        verdict = ("고정됨" if row["margin"] >= safety else
                   "여유부족" if row["margin"] >= 1.0 else "미끄러짐")
        print(f"  {row['joint']:<10} 하류질량 {row['downstream_mass_g']:>6.1f} g"
              f"   최악토크 {row['torque_nm']:.3f} N·m"
              f" @ q={np.round(row['theta_deg'], 0)}"
              f"   여유 {row['margin']:>5.1f}x  {verdict}")

    grid = full_grid(spec)
    candidates = alg.candidate_grid()
    print(f"\n[사용 가능한 자세] 관절이 고정되는 자세만 측정에 쓴다")
    print(f"  전체 후보 {len(grid)}개 중 {len(candidates)}개 통과"
          f"  ({100 * len(candidates) / len(grid):.0f}%)")
    if not candidates:
        print("  통과한 자세가 없다. 이 밀도 GT 는 이 힌지로 실험할 수 없다.")
        return None
    torques = max_joint_torques(spec, candidates, rho_gt)
    print(f"  통과 자세의 관절토크 최대 {torques.max():.3f} N·m"
          f" (한계 {hinge.holding_torque_nm / safety:.3f} N·m)")

    svals = alg.structural_identifiability(candidates)
    rank = int(np.sum(svals > 1e-6 * svals[0]))
    print(f"\n[식별성] 후보자세 {len(candidates)}개, 특이값 "
          f"{np.array2string(svals, precision=3)}")
    print(f"          rank {rank}/{alg.P}, 조건수 {svals[0] / svals[-1]:.1f}")

    # 두 전략을 같은 seed 집합으로 돌려야 공정한 비교가 된다.
    seeds = range(n_random_seeds)
    hists_a = [alg.run_loop(alg.select_active, n_rounds, seed=s) for s in seeds]
    hists_r = [alg.run_loop(alg.select_random, n_rounds, seed=s) for s in seeds]
    rmse_a = np.array([[h["rmse"] for h in hist] for hist in hists_a])
    rmse_r = np.array([[h["rmse"] for h in hist] for hist in hists_r])

    print(f"\n[수렴] 같은 노이즈 seed {n_random_seeds}개에 대한 평균 +/- 표준편차")
    print(f"  {'round':>5} | {'active(D-optimal) RMSE':>24}"
          f" | {'random RMSE':>20} | {'선택 자세 [deg]':>18}")
    for index in range(n_rounds):
        deg = np.round(np.degrees(hists_a[0][index]["theta"]), 0)
        print(f"  {index + 1:>5} | {rmse_a[:, index].mean():>13.1f} +/-"
              f" {rmse_a[:, index].std():>6.1f}"
              f" | {rmse_r[:, index].mean():>11.1f} +/-"
              f" {rmse_r[:, index].std():>5.1f} | {str(deg):>18}")

    final_a = np.array([h[-1]["rho"] for h in hists_a])
    rho_mean, rho_std = final_a.mean(axis=0), final_a.std(axis=0)
    print("\n[최종 추정] active, seed 10개 평균")
    print(f"  {'part':<13}{'GT':>9}{'추정평균':>11}{'표준편차':>10}{'오차':>9}")
    for index, part in enumerate(spec.parts):
        error = 100 * abs(rho_mean[index] - rho_gt[index]) / rho_gt[index]
        print(f"  {part.name:<13}{rho_gt[index]:>9.0f}{rho_mean[index]:>11.1f}"
              f"{rho_std[index]:>10.1f}{error:>8.2f}%")
    max_rel = np.max(np.abs(rho_mean - rho_gt) / rho_gt)
    print(f"  최대 상대오차 {100 * max_rel:.2f}%,  "
          f"RMSE {rmse_a[:, -1].mean():.1f} +/- {rmse_a[:, -1].std():.1f} kg/m^3")

    # 추정한 밀도로 질량·무게중심·관성모멘트까지 계산하고 오차를 본다.
    # 물체 전체 값은 자세에 따라 달라지므로 두 자세에서 확인한다.
    flat = np.zeros(len(spec.joints))
    folded = np.array(candidates[int(np.argmax(
        [np.sum(np.abs(th)) for th in candidates]))])
    report_derived(spec, rho_gt, final_a, flat, "펼친 자세")
    report_derived(spec, rho_gt, final_a, folded, "가장 접힌 자세")

    return dict(spec=spec, rmse_active=rmse_a, rmse_random=rmse_r, rho_gt=rho_gt)


def plot(results, path="density_id_objects.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(results), figsize=(11, 4), squeeze=False)
    for ax, result in zip(axes[0], results):
        rounds = np.arange(1, result["rmse_active"].shape[1] + 1)
        for label, data, style, color in (
            ("active (D-optimal)", result["rmse_active"], "o-", "C0"),
            ("random", result["rmse_random"], "s--", "C1"),
        ):
            mean, std = data.mean(axis=0), data.std(axis=0)
            ax.semilogy(rounds, mean, style, color=color, label=label)
            ax.fill_between(rounds, mean - std, mean + std, alpha=0.15,
                            color=color)
        ax.set_title(result["spec"].label, fontsize=10)
        ax.set_xlabel("measurement round")
        ax.set_ylabel("density RMSE [kg/m$^3$]")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"\n그래프 저장 -> {path}")


def render(spec, out_dir, hinge=None, safety=DEFAULT_SAFETY, density_scale=1.0):
    """선택된 자세를 PNG로 저장해 기구학이 맞는지 눈으로 확인한다."""
    from pathlib import Path

    from PIL import Image
    from pydrake.geometry import (
        ClippingRange,
        DepthRange,
        DepthRenderCamera,
        MakeRenderEngineVtk,
        RenderCameraCore,
        RenderEngineVtkParams,
    )
    from pydrake.systems.framework import DiagramBuilder
    from pydrake.systems.sensors import CameraInfo, RgbdSensor

    from view_density_id import look_at

    rho_gt = bind_object(spec, hinge=hinge, safety=safety,
                         density_scale=density_scale)
    thetas = [h["theta"] for h in alg.run_loop(alg.select_active, 6, seed=1)]

    builder = DiagramBuilder()
    plant, _ = build_plant(spec, rho_gt, builder=builder)
    scene_graph = builder.GetSubsystemByName("scene_graph")
    scene_graph.AddRenderer("vtk", MakeRenderEngineVtk(RenderEngineVtkParams()))
    eye, target = spec.camera
    camera = builder.AddSystem(
        RgbdSensor(
            scene_graph.world_frame_id(),
            look_at(np.array(eye), np.array(target)),
            DepthRenderCamera(
                RenderCameraCore("vtk", CameraInfo(800, 600, np.pi / 4),
                                 ClippingRange(0.02, 10.0), RigidTransform()),
                DepthRange(0.02, 10.0),
            ),
        )
    )
    builder.Connect(scene_graph.get_query_output_port(),
                    camera.query_object_input_port())
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyMutableContextFromRoot(context)
    camera_context = camera.GetMyContextFromRoot(context)

    out = Path(out_dir) / spec.key
    out.mkdir(parents=True, exist_ok=True)
    for index, theta in enumerate(thetas):
        plant.SetPositions(plant_context, theta)
        image = camera.color_image_output_port().Eval(camera_context)
        deg = "_".join(f"{np.degrees(v):+.0f}" for v in theta)
        path = out / f"round{index + 1}_q{deg}.png"
        Image.fromarray(image.data[:, :, :3]).save(path)
        print(f"  저장 {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--object", choices=(*OBJECTS, "both"), default="both")
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--hinge", choices=tuple(HINGES), default="mg_plastic")
    parser.add_argument("--hinge-torque", type=float, default=None,
                        help="유지토크 [N·m]. 실측값이 있으면 여기에 넣는다")
    parser.add_argument("--safety", type=float, default=DEFAULT_SAFETY)
    parser.add_argument("--auto-scale", action="store_true",
                        help="힌지가 버틸 수 있도록 하류 part 밀도를 자동으로 낮춘다")
    parser.add_argument("--sweep", action="store_true",
                        help="유지토크별 사용가능 자세/허용밀도 표만 출력")
    parser.add_argument("--com-mismatch", action="store_true",
                        help="균일밀도 가정이 깨졌을 때의 영향까지 본다"
                             " (실험 전제는 균일밀도이므로 기본은 끔)")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--out-dir", default="frames_objects")
    args = parser.parse_args()

    hinge = HINGES[args.hinge]
    if args.hinge_torque is not None:
        hinge = Hinge(hinge.label, args.hinge_torque, hinge.note)

    keys = list(OBJECTS) if args.object == "both" else [args.object]

    if args.sweep:
        for key in keys:
            torque_sweep(OBJECTS[key], safety=args.safety)
        return

    results = []
    for key in keys:
        spec = OBJECTS[key]
        scale = 1.0
        if args.auto_scale:
            scale = min(1.0, max_feasible_density_scale(spec, hinge, args.safety))
        result = validate(spec, hinge, n_rounds=args.rounds,
                          safety=args.safety, density_scale=scale)
        if result is None:
            torque_sweep(spec, safety=args.safety)
            print()
            continue
        results.append(result)
        if args.com_mismatch:
            com_mismatch_study(spec, n_rounds=args.rounds, hinge=hinge,
                               safety=args.safety, density_scale=scale)
        print()
    if results:
        plot(results)

    if args.render:
        for key in keys:
            spec = OBJECTS[key]
            scale = (min(1.0, max_feasible_density_scale(spec, hinge, args.safety))
                     if args.auto_scale else 1.0)
            print(f"\n[렌더링] {key}")
            render(spec, args.out_dir, hinge=hinge, safety=args.safety,
                   density_scale=scale)


if __name__ == "__main__":
    main()
