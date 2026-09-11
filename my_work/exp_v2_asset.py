"""자산이 '질량이 맞다' 를 넘어 '거동이 맞다' 까지 가는가 — 시뮬 대 시뮬 대리실험.

왜 이 실험이 필요한가
--------------------
표 하나로 "part 질량 상대오차 0.2 %" 를 보이면 심사자는 바로 되묻는다.
"질량이 맞으면 시뮬레이션이 실제로 그렇게 움직이는가?" 질량 오차는 스칼라고,
거동은 궤적이다. 둘이 같은 말이 아니다 — 관성텐서, 무게중심, 링크 사이
질량비가 전부 궤적에 들어간다. 그래서 **추정한 밀도를 그대로 꽂은 plant 를
직접 굴려서** 기준 plant 의 궤적을 얼마나 따라가는지 재는 자리가 필요하다.

정직함에 대한 경계 (읽는 사람이 오해하지 않도록)
----------------------------------------------
1. 이것은 **시뮬 대 시뮬** 이다. 실물 로봇을 아직 못 찍는다. 기준(reference)은
   실제 세계가 아니라 GT 밀도를 넣은 Drake plant 다. 따라서 이 실험이 보이는
   것은 "밀도 오차가 궤적 오차로 얼마나 번지는가" 뿐이고, 실물 검증이 아니다.
   결과 표·JSON·그림 어디에도 real-world 라고 쓰지 않는다.
2. 접촉·마찰·관절감쇠는 우리 방법이 식별하지 **않는** 값이다. 그래서 모든
   variant 에 **똑같은 값**을 박아 둔다. 유일한 차이는 부위별 밀도다.
   (그렇게 안 하면 "우리가 이겼다" 가 접촉 파라미터 튜닝의 결과가 된다.)
3. 시나리오를 우리에게 유리하게 고르지 않는다. 세 시나리오를 미리 정해 두고
   전부 보고한다. 우리가 못 이기면 못 이겼다고 그대로 쓴다.

비교 대상 (variant)
------------------
  gt       GT 밀도. 다른 모두를 채점하는 **기준**. 궤적 오차 0 (정의상).
  ours     닫힌 루프가 추정한 밀도. figures/v2/main_real.json 의 seed 중앙값.
  uniform  총질량/총부피 균질 가정 ("강체 덩어리" 대조군).
  vision-* v1 논문 표(table_main_v2.tex)의 part 질량 상대오차[%]를 되살린 것.
           implied_mass = gt_mass * (1 + err/100), rho = implied_mass / V.

시나리오 (셋 다 동일 초기조건, 무구동)
------------------------------------
  (a) RELEASE  뿌리를 공중에 고정하고 관절만 놓아 중력으로 처지게 한다.
               접촉이 아예 없다 -> 순수하게 관성 파라미터만 본다.
               실물에서 가장 싸고 재현 가능한 촬영이 바로 이것이다.
  (b) DROP     자유 부유 상태로 탁자 위에 떨어뜨려 안착시킨다.
  (c) PUSH     탁자 위에 놓고 한 부위에 고정 크기의 옆방향 힘을 잠깐 준다.

실행:
    cd ~/Desktop/PIVOT/my_work
    ../robot_learning/scripts/run_drake_env.sh python exp_v2_asset.py
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from pydrake.geometry import (
    AddContactMaterial,
    Box,
    CollisionFilterDeclaration,
    Convex,
    GeometrySet,
    HalfSpace,
    ProximityProperties,
)
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.multibody.plant import (
    AddMultibodyPlantSceneGraph,
    ContactModel,
    CoulombFriction,
    DiscreteContactApproximation,
    ExternallyAppliedSpatialForce,
)
from pydrake.multibody.math import SpatialForce
from pydrake.multibody.tree import (
    FixedOffsetFrame,
    RevoluteJoint,
    SpatialInertia,
    UnitInertia,
)
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder

import density_id_objects as obj

MM = 1e-3
CM3 = 1e-6
G = 9.81

OBJECT_KEYS = ("2link", "3link", "desklamp")

# ---------------------------------------------------------------------------
# 모든 variant 가 공유하는 물리 설정.
#
# 이 값들은 우리 방법이 식별하는 대상이 **아니다**. 밀도만 갈아끼우고 나머지는
# 한 글자도 안 바꾼다는 것을 코드로 못박아 두려고 상수로 뺐다.
# ---------------------------------------------------------------------------
TIME_STEP = 1e-3            # 이산 plant. SAP 로 고정.
SAMPLE_DT = 5e-3            # 궤적 기록 간격 (200 Hz)
JOINT_DAMPING = 0.01        # N·m·s/rad. 힌지 점성 — 모든 variant 동일.
POINT_STIFFNESS = 1.0e5     # N/m. 침투허용치에서 유도하면 질량에 따라 달라져
                            # variant 마다 접촉이 달라진다. 그래서 명시 고정.
HC_DISSIPATION = 20.0       # s/m
MU_STATIC, MU_DYNAMIC = 0.9, 0.8

T_RELEASE = 3.0
T_DROP = 1.5
T_PUSH = 1.5
DROP_CLEARANCE = 0.10       # m. 탁자 위 낙하 시작 높이 (최저점 기준)
REST_CLEARANCE = 1e-3       # m. PUSH 초기 배치 (거의 닿아 있음)
PUSH_START, PUSH_END = 0.30, 0.50    # s
PUSH_G = 1.5                # 힘 = PUSH_G x (GT 총질량) x g. 모든 variant 동일.

N_IC = 5                    # 시나리오당 초기조건 개수
IC_SEED = 20260830

SCENARIOS = ("release", "drop", "push")

# ---------------------------------------------------------------------------
# v1 논문 표(table_main_v2.tex)에서 얼려 온 vision baseline.
#
# 표는 **part 질량 상대오차 [%]** 만 준다. 부호는 안 준다 (전부 양수 표기).
# 그래서 여기서는 과대추정(+)으로 되살린다 — 부호를 모른다는 사실을 결과에
# 같이 적는다. 표의 GT 질량(2link/3link 는 ballast 넣기 전 값)과 이 저장소
# spec 의 GT 질량이 다르므로, 절대질량이 아니라 **상대오차만** 옮긴다:
#     implied_mass = spec_gt_mass * (1 + err/100)
# PhysX-Omni 는 2link/3link 에서 N/A 라 애초에 안 쓴다 (지시대로 SiPhy·PUGS).
#
# 표 행 -> 이 저장소 body 이름 대응
#   Stand lamp: Base=link_2, Support=link_3, Head=link_1
#   2-link    : Long=parent, Short=child, Hinge=hinge_hinge
#   3-link    : Link0/1/2 = link0_base/link1_elbow/link2_tip,
#               Hinges = joint1_hinge & joint2_hinge (한 행이 힌지 둘을 덮는다)
# ---------------------------------------------------------------------------
VISION_ERR_PCT = {
    "desklamp": {
        "PUGS":  {"link_2": 164.1, "link_3": 10.7, "link_1": 36.4},
        "SiPhy": {"link_2": 38.3,  "link_3": 27.3, "link_1": 48.5},
    },
    "2link": {
        "PUGS":  {"parent": 7887.4, "child": 7323.6, "hinge_hinge": 423.8},
        "SiPhy": {"parent": 593.4,  "child": 544.5,  "hinge_hinge": 54.5},
    },
    "3link": {
        "PUGS":  {"link0_base": 4312.5, "link1_elbow": 3369.1,
                  "link2_tip": 3501.8,
                  "joint1_hinge": 178.2, "joint2_hinge": 178.2},
        "SiPhy": {"link0_base": 349.6, "link1_elbow": 253.5,
                  "link2_tip": 267.0,
                  "joint1_hinge": 71.7, "joint2_hinge": 71.7},
    },
}

VARIANTS = ("ours", "uniform", "vision-SiPhy", "vision-PUGS")
COLORS = {
    "gt": "#000000",
    "ours": "#0072B2",
    "uniform": "#E69F00",
    "vision-SiPhy": "#009E73",
    "vision-PUGS": "#D55E00",
}

MAIN_REAL_JSON = Path("figures/v2/main_real.json")
OUT_JSON = Path("figures/v2/asset_validation.json")
OUT_PNG = Path("figures/v2/fig_asset.png")


# ---------------------------------------------------------------------------
# 밀도 벡터 만들기
# ---------------------------------------------------------------------------
def gt_densities(spec):
    return np.array([row["rho_gt"] for row in obj.body_table(spec)], dtype=float)


def uniform_densities(spec):
    """균질 가정: 저울에 읽힌 총질량 / 총부피. GT 를 훔치지 않는다
    (총질량과 외형 부피는 우리도 아는 값이다). study_baselines 와 같은 정의."""
    table = obj.body_table(spec)
    volumes = np.array([row["volume_m3"] for row in table])
    total = obj.assembled_mass_kg(spec)
    return np.full(len(table), total / float(volumes.sum()))


def vision_densities(spec, key, method):
    """v1 표의 질량 상대오차를 밀도로 되살린다. 없는 body 는 None 을 돌려준다."""
    errs = VISION_ERR_PCT.get(key, {}).get(method)
    if errs is None:
        return None
    rho = []
    for row in obj.body_table(spec):
        if row["name"] not in errs:
            return None                     # 표에 칸이 없다 -> 이 variant 는 뺀다
        rho.append(row["rho_gt"] * (1.0 + errs[row["name"]] / 100.0))
    return np.array(rho, dtype=float)


def ours_densities(spec, key, log=print):
    """closed loop 가 추정한 밀도. 먼저 얼려 둔 main_real.json 을 본다.

    seed 8 개의 **중앙값**을 대표값으로 쓴다. seed 하나를 고르면 운이 섞이고,
    평균은 이상치에 끌린다. 파일이 없으면 여기서 같은 설정으로 직접 돌린다.
    """
    if MAIN_REAL_JSON.exists():
        blob = json.loads(MAIN_REAL_JSON.read_text())
        cell = blob.get("data", {}).get(key, {})
        runs = cell.get("per_seed", {}).get("ours")
        names = [b["name"] for b in cell.get("bodies", [])]
        want = [row["name"] for row in obj.body_table(spec)]
        if runs and names == want:
            rho = np.median(np.array([r["rho_hat"] for r in runs], float), axis=0)
            log(f"    ours <- {MAIN_REAL_JSON} (seed {len(runs)}개 중앙값,"
                f" 라운드 {[r['rounds'] for r in runs]})")
            return rho, dict(source=str(MAIN_REAL_JSON), n_seed=len(runs),
                             rounds=[r["rounds"] for r in runs])
        log(f"    [주의] {MAIN_REAL_JSON} 에 '{key}/ours' 가 맞지 않는다"
            " -> 직접 돌린다")
    return _run_closed_loop(spec, log=log)


def _run_closed_loop(spec, seeds=8, rel_error=0.05, max_rounds=14, log=print):
    """main_real.json 이 없을 때의 대체 경로. 지시된 설정 그대로."""
    import design_core as dc

    n_part = len(spec.parts)
    target = 0.005 * n_part
    log(f"    ours <- design_core.closed_loop 직접 실행"
        f" (seed 0..{seeds-1}, rel_error={rel_error}, max_rounds={max_rounds},"
        f" target={target:.4f}, estimator=tls)")
    rows, rounds = [], []
    for seed in range(seeds):
        obj.set_measurement_averaging()
        obj.bind_object(spec)
        obj.apply_weight_prior(spec, obj.assembled_mass_kg(spec))
        out = dc.closed_loop(spec, target=target, max_rounds=max_rounds,
                             seed=100 * seed, rel_error=rel_error,
                             estimator="tls", n_starts=6)
        rows.append(out["rho_hat"])
        rounds.append(int(out["rounds"]))
    rho = np.median(np.array(rows, float), axis=0)
    return rho, dict(source="design_core.closed_loop", n_seed=seeds,
                     rounds=rounds)


# ---------------------------------------------------------------------------
# plant 만들기
#
# density_id_objects.build_plant 을 그대로 못 쓴다. 그쪽은 (1) 뿌리를 항상
# 월드에 용접하고 (2) 충돌 형상이 없고 (3) 관절 감쇠가 0 이다. 여기서는
# 자유낙하와 접촉이 필요하다. 그래서 같은 spec 을 읽되 plant 는 새로 짓는다.
# 관성/프레임 규약(body frame 원점 = 외형 도심, 힌지는 자식 링크에 용접)은
# build_plant 과 한 글자도 다르지 않게 맞춘다.
# ---------------------------------------------------------------------------
def _unit_inertia(part, dims_m):
    """스캔 부위는 메시에서 잰 단위질량 관성을 쓴다. 없으면 직육면체."""
    matrix = getattr(part, "inertia_unit", None)
    if matrix is not None:
        m = np.asarray(matrix, dtype=float)
        try:
            unit = UnitInertia(m[0, 0], m[1, 1], m[2, 2],
                               m[0, 1], m[0, 2], m[1, 2])
            if unit.CouldBePhysicallyValid():
                return unit
        except Exception:
            pass
    return UnitInertia.SolidBox(*dims_m)


def _contact_props():
    """모든 variant·모든 물체가 공유하는 접촉 물성. 질량과 무관하게 고정."""
    props = ProximityProperties()
    AddContactMaterial(HC_DISSIPATION, POINT_STIFFNESS,
                       CoulombFriction(MU_STATIC, MU_DYNAMIC), props)
    return props


def build_sim_plant(spec, densities, floating, contact):
    """(diagram, plant, bodies, joints) 를 돌려준다.

    floating=False 면 뿌리 부위를 월드에 용접한다 (RELEASE).
    contact=True  면 부위마다 충돌 형상과 탁자(반평면)를 넣는다 (DROP/PUSH).
                  부위끼리의 자기충돌은 걸러 낸다 — 인접 링크가 접힐 때
                  서로 파고들어 생기는 잡음을 없애기 위해서고, 이것도 모든
                  variant 에 똑같이 적용된다.
    """
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder,
                                                     time_step=TIME_STEP)
    plant.set_contact_model(ContactModel.kPoint)
    plant.set_discrete_contact_approximation(DiscreteContactApproximation.kSap)
    props = _contact_props()

    parts = {p.name: p for p in spec.parts}
    bodies, own_geoms = {}, []
    for part, rho in zip(spec.parts, densities):
        dims_m = tuple(d * MM for d in part.bbox_mm)
        body = plant.AddRigidBody(
            part.name,
            SpatialInertia(rho * part.volume_m3, np.zeros(3),
                           _unit_inertia(part, dims_m)))
        bodies[part.name] = body
        if contact:
            X_mesh = RigidTransform(np.array(part.mesh_offset_m))
            if part.collision_meshes:
                for index, path in enumerate(part.collision_meshes):
                    plant.RegisterCollisionGeometry(
                        body, X_mesh, Convex(path, 1.0),
                        f"{part.name}_col_{index}", props)
            else:
                plant.RegisterCollisionGeometry(
                    body, RigidTransform(), Box(*dims_m),
                    f"{part.name}_col", props)

    root = spec.parts[0].name
    if not floating:
        plant.WeldFrames(plant.world_frame(), bodies[root].body_frame(),
                         RigidTransform())

    joints = []
    for joint in spec.joints:
        origin = np.array(joint.origin_in_parent_link_mm)
        on_parent = (origin
                     - np.array(parts[joint.parent].bbox_center_in_link_mm)) * MM
        child_origin = np.array(joint.origin_in_child_link_mm
                                if joint.origin_in_child_link_mm is not None
                                else (0.0, 0.0, 0.0))
        on_child = (child_origin
                    - np.array(parts[joint.child].bbox_center_in_link_mm)) * MM
        lo, hi = joint.limits_rad
        joints.append(plant.AddJoint(RevoluteJoint(
            joint.name,
            plant.AddFrame(FixedOffsetFrame(
                f"{joint.name}_parent", bodies[joint.parent].body_frame(),
                RigidTransform(on_parent))),
            plant.AddFrame(FixedOffsetFrame(
                f"{joint.name}_child", bodies[joint.child].body_frame(),
                RigidTransform(on_child))),
            joint.axis, float(lo), float(hi), JOINT_DAMPING)))

    # 힌지 덩어리. build_plant 과 같은 자리에 같은 방식으로 용접한다.
    n_part = len(spec.parts)
    for index, (joint, volume, _) in enumerate(obj.hinge_bodies(spec)):
        rho = densities[n_part + index]
        dims = obj.hinge_dims_m(volume)
        name = f"{joint.name}_hinge"
        body = plant.AddRigidBody(
            name, SpatialInertia(rho * volume, np.zeros(3),
                                 UnitInertia.SolidBox(*dims)))
        bodies[name] = body
        on_child = ((-np.array(parts[joint.child].bbox_center_in_link_mm)
                     + np.array(joint.hinge_com_offset_mm)) * MM)
        plant.WeldFrames(
            plant.AddFrame(FixedOffsetFrame(
                f"{name}_mount", bodies[joint.child].body_frame(),
                RigidTransform(on_child))),
            body.body_frame())

    if contact:
        plant.RegisterCollisionGeometry(
            plant.world_body(), RigidTransform(), HalfSpace(), "table", props)

    plant.Finalize()

    if contact:
        for part in spec.parts:
            own_geoms += list(plant.GetCollisionGeometriesForBody(
                bodies[part.name]))
        scene_graph.collision_filter_manager().Apply(
            CollisionFilterDeclaration().ExcludeWithin(GeometrySet(own_geoms)))

    diagram = builder.Build()
    return diagram, plant, bodies, joints


# ---------------------------------------------------------------------------
# 초기조건: 기하만으로 정한다 (variant 마다 같아야 하므로)
# ---------------------------------------------------------------------------
_OBJ_VERTS = {}


def _obj_vertices(path):
    if path not in _OBJ_VERTS:
        rows = []
        for line in Path(path).read_text().splitlines():
            if line.startswith("v "):
                rows.append([float(x) for x in line.split()[1:4]])
        _OBJ_VERTS[path] = np.array(rows, dtype=float)
    return _OBJ_VERTS[path]


def part_points_body(part):
    """부위의 충돌 형상 점들을 body frame 좌표로. 최저점 계산용."""
    if part.collision_meshes:
        offset = np.array(part.mesh_offset_m)
        return np.vstack([_obj_vertices(p) + offset
                          for p in part.collision_meshes])
    half = 0.5 * np.array(part.bbox_mm) * MM
    signs = np.array([[sx, sy, sz] for sx in (-1, 1)
                      for sy in (-1, 1) for sz in (-1, 1)], dtype=float)
    return signs * half


def lowest_z(plant, context, spec, bodies):
    z = np.inf
    for part in spec.parts:
        X_WB = plant.EvalBodyPoseInWorld(context, bodies[part.name])
        pts = X_WB @ part_points_body(part).T
        z = min(z, float(pts[2].min()))
    return z


def make_ics(spec, n_ic=N_IC, seed=IC_SEED):
    """시나리오와 무관한 자세 초기조건 n_ic 개. 모든 variant 가 이것을 공유한다.

    뿌리 자세(rpy)를 같이 흔드는 이유: 3link 의 joint1 은 축이 +Z 라 뿌리를
    똑바로 세워 두면 중력 토크가 정확히 0 이다. 그러면 RELEASE 가 그 관절에
    대해 아무 정보도 주지 않는다. 실물에서도 로봇은 물체를 기울여 든다.
    """
    rng = np.random.default_rng(seed)
    ics = []
    for _ in range(n_ic):
        rpy = rng.uniform(-1.0, 1.0, 3) * np.array([np.pi / 3, np.pi / 3, np.pi])
        q0 = []
        for joint in spec.joints:
            lo, hi = joint.limits_rad
            q0.append(rng.uniform(lo + 0.15 * (hi - lo), hi - 0.15 * (hi - lo)))
        ics.append(dict(rpy=rpy.tolist(), q0=[float(v) for v in q0],
                        push_dir=float(rng.uniform(-np.pi, np.pi))))
    return ics


# ---------------------------------------------------------------------------
# 시나리오 실행
# ---------------------------------------------------------------------------
def _set_config(plant, context, spec, bodies, joints, ic, X_WRoot, floating):
    if floating:
        plant.SetFreeBodyPose(context, bodies[spec.parts[0].name], X_WRoot)
    for joint, angle in zip(joints, ic["q0"]):
        joint.set_angle(context, angle)
    plant.SetVelocities(context, np.zeros(plant.num_velocities()))


def _record(plant, context, spec, bodies, joints):
    pos = np.array([plant.EvalBodyPoseInWorld(context,
                                              bodies[p.name]).translation()
                    for p in spec.parts])
    ang = np.array([j.get_angle(context) for j in joints])
    return pos, ang


def simulate(spec, densities, scenario, ic, root_pose, push_force_N,
             push_body):
    """한 variant · 한 초기조건 · 한 시나리오를 굴리고 궤적을 돌려준다."""
    floating = scenario in ("drop", "push")
    contact = floating
    duration = {"release": T_RELEASE, "drop": T_DROP, "push": T_PUSH}[scenario]

    diagram, plant, bodies, joints = build_sim_plant(spec, densities,
                                                     floating, contact)
    sim = Simulator(diagram)
    context = sim.get_mutable_context()
    plant_ctx = plant.GetMyMutableContextFromRoot(context)
    _set_config(plant, plant_ctx, spec, bodies, joints, ic, root_pose, floating)

    force_port = plant.get_applied_spatial_force_input_port()
    force_port.FixValue(plant_ctx, [])
    sim.Initialize()

    times = np.arange(0.0, duration + 0.5 * SAMPLE_DT, SAMPLE_DT)
    if scenario == "push":
        # 옆방향 힘을 켜고 끄는 시각도 표본에 들어가도록 섞어 넣는다.
        times = np.unique(np.concatenate([times, [PUSH_START, PUSH_END]]))
    pos_log, ang_log, pushed = [], [], False
    for t in times:
        if scenario == "push":
            want = PUSH_START <= t < PUSH_END
            if want != pushed:
                force_port.FixValue(plant_ctx,
                                    _push_wrench(plant, bodies, push_body,
                                                 push_force_N) if want else [])
                pushed = want
        if t > 0.0:
            sim.AdvanceTo(float(t))
        pos, ang = _record(plant, plant_ctx, spec, bodies, joints)
        pos_log.append(pos)
        ang_log.append(ang)
    return dict(t=times, pos=np.array(pos_log), ang=np.array(ang_log))


def _push_wrench(plant, bodies, body_name, force_W):
    f = ExternallyAppliedSpatialForce()
    f.body_index = bodies[body_name].index()
    f.p_BoBq_B = np.zeros(3)
    f.F_Bq_W = SpatialForce(np.zeros(3), np.asarray(force_W, dtype=float))
    return [f]


def scenario_setup(spec, densities_gt, scenario, ic):
    """기하만으로 정해지는 초기 배치. GT plant 한 번만 써서 계산하고
    모든 variant 가 그대로 쓴다 (형상은 variant 마다 완전히 같다)."""
    R = RollPitchYaw(np.array(ic["rpy"])).ToRotationMatrix()
    if scenario == "release":
        return RigidTransform(R, np.zeros(3)), None
    diagram, plant, bodies, joints = build_sim_plant(spec, densities_gt,
                                                     True, False)
    ctx = plant.CreateDefaultContext()
    _set_config(plant, ctx, spec, bodies, joints, ic,
                RigidTransform(R, np.zeros(3)), True)
    low = lowest_z(plant, ctx, spec, bodies)
    lift = (DROP_CLEARANCE if scenario == "drop" else REST_CLEARANCE) - low
    return RigidTransform(R, np.array([0.0, 0.0, lift])), None


# ---------------------------------------------------------------------------
# 채점
# ---------------------------------------------------------------------------
def score(ref, run):
    """기준 궤적 대비 오차. 길이·시각은 같은 격자라 그대로 뺀다."""
    d = run["pos"] - ref["pos"]                      # (T, P, 3)
    per_part_rmse_mm = 1e3 * np.sqrt((d ** 2).sum(axis=2).mean(axis=0))
    final_pos_mm = 1e3 * np.linalg.norm(d[-1], axis=1)
    final_ang_deg = np.degrees(np.abs(run["ang"][-1] - ref["ang"][-1]))
    return dict(
        traj_rmse_mm=per_part_rmse_mm.tolist(),
        traj_rmse_mm_max=float(per_part_rmse_mm.max()),
        traj_rmse_mm_mean=float(per_part_rmse_mm.mean()),
        final_pos_mm=final_pos_mm.tolist(),
        final_pos_mm_max=float(final_pos_mm.max()),
        final_ang_deg=final_ang_deg.tolist(),
        final_ang_deg_max=float(final_ang_deg.max()) if len(final_ang_deg) else 0.0,
    )


def med_iqr(values):
    a = np.asarray(values, dtype=float)
    return (float(np.median(a)), float(np.percentile(a, 25)),
            float(np.percentile(a, 75)))


# ---------------------------------------------------------------------------
def run_object(key, log=print):
    spec = obj.get_spec(key)
    table = obj.body_table(spec)
    names = [row["name"] for row in table]
    rho_gt = gt_densities(spec)
    volumes = np.array([row["volume_m3"] for row in table])

    log(f"\n[{key}] {spec.label}")
    log(f"  부위 {names}")
    log(f"  GT 총질량 {1e3*float((rho_gt*volumes).sum()):.1f} g")

    rho = {"gt": rho_gt}
    meta = {"gt": dict(source="ground truth (reference)")}

    ours, ours_meta = ours_densities(spec, key, log=log)
    rho["ours"], meta["ours"] = ours, ours_meta
    rho["uniform"] = uniform_densities(spec)
    meta["uniform"] = dict(source="total scale mass / total mesh volume")
    skipped = []
    for method in ("SiPhy", "PUGS"):
        v = vision_densities(spec, key, method)
        name = f"vision-{method}"
        if v is None:
            skipped.append(name)
            continue
        rho[name], meta[name] = v, dict(
            source="table_main_v2.tex (v1 frozen), signs unknown -> taken as +")
    for name in skipped:
        log(f"  [건너뜀] {name}: table_main_v2.tex 에 이 물체의 칸이 없다 (N/A)")

    log(f"  {'variant':<14}{'질량비 오차 [%] (부위별)':<44}{'총질량 [g]':>11}")
    for name in ("gt",) + tuple(v for v in VARIANTS if v in rho):
        err = 100.0 * np.abs(rho[name] - rho_gt) / rho_gt
        cell = " ".join(f"{e:8.2f}" for e in err)
        log(f"  {name:<14}{cell:<44}{1e3*float((rho[name]*volumes).sum()):>11.1f}")

    ics = make_ics(spec)
    push_body = spec.parts[-1].name
    push_mag = PUSH_G * float((rho_gt * volumes).sum()) * G

    runs, series = [], {}
    for scenario in SCENARIOS:
        for index, ic in enumerate(ics):
            root_pose, _ = scenario_setup(spec, rho_gt, scenario, ic)
            phi = ic["push_dir"]
            force = push_mag * np.array([np.cos(phi), np.sin(phi), 0.0])
            traj = {}
            for name, densities in rho.items():
                traj[name] = simulate(spec, densities, scenario, ic,
                                      root_pose, force, push_body)
            for name in rho:
                if name == "gt":
                    continue
                runs.append(dict(object=key, scenario=scenario, ic=index,
                                 variant=name, **score(traj["gt"], traj[name])))
            if index == 0:
                tip = len(spec.parts) - 1
                series[scenario] = dict(
                    t=traj["gt"]["t"].tolist(),
                    tip=spec.parts[tip].name,
                    disp_mm={n: (1e3 * np.linalg.norm(
                        traj[n]["pos"][:, tip] - traj[n]["pos"][0, tip],
                        axis=1)).tolist() for n in traj})
            log(f"    {scenario:<8} IC{index}  "
                + "  ".join(
                    f"{n}={score(traj['gt'], traj[n])['traj_rmse_mm_max']:.1f}mm"
                    for n in rho if n != "gt"))
    return dict(key=key, label=spec.label, bodies=names,
                rho={k: v.tolist() for k, v in rho.items()},
                rho_gt=rho_gt.tolist(), volumes=volumes.tolist(),
                meta=meta, skipped=skipped, ics=ics,
                push_body=push_body, push_force_N=push_mag,
                runs=runs, series=series)


# ---------------------------------------------------------------------------
def print_table(results, log=print):
    log("\n" + "=" * 96)
    log("자산 거동 충실도 — SIM-vs-SIM 대리실험 (실물 검증 아님)")
    log("기준(reference) = GT 밀도를 넣은 Drake plant. 접촉·마찰·관절감쇠는")
    log("모든 variant 에서 동일 (우리 방법이 식별하는 값이 아니다).")
    log("표기: 중앙값 [25%, 75%], 초기조건 5개 기준. 각 값은 부위 최댓값.")
    log("=" * 96)
    head = (f"{'object':<10}{'scenario':<9}{'variant':<15}"
            f"{'traj RMSE [mm]':>26}{'final pos [mm]':>26}{'final ang [deg]':>24}")
    for cell in results:
        log(f"\n{cell['label']}")
        log(head)
        for scenario in SCENARIOS:
            for variant in VARIANTS:
                rows = [r for r in cell["runs"]
                        if r["scenario"] == scenario and r["variant"] == variant]
                if not rows:
                    continue
                a = med_iqr([r["traj_rmse_mm_max"] for r in rows])
                b = med_iqr([r["final_pos_mm_max"] for r in rows])
                c = med_iqr([r["final_ang_deg_max"] for r in rows])
                log(f"{cell['key']:<10}{scenario:<9}{variant:<15}"
                    f"{a[0]:>10.2f} [{a[1]:6.2f},{a[2]:7.2f}]"
                    f"{b[0]:>10.2f} [{b[1]:6.2f},{b[2]:7.2f}]"
                    f"{c[0]:>9.2f} [{c[1]:5.2f},{c[2]:6.2f}]")


def verdict(results, log=print):
    log("\n" + "=" * 96)
    log("기대한 순서(ours < uniform, ours < vision-*)가 실제로 성립했는가")
    log("=" * 96)
    lines = []
    n_ok = n_total = 0
    for cell in results:
        for scenario in SCENARIOS:
            def med(variant, field="traj_rmse_mm_max"):
                rows = [r[field] for r in cell["runs"]
                        if r["scenario"] == scenario and r["variant"] == variant]
                return float(np.median(rows)) if rows else None
            ours = med("ours")
            if ours is None:
                continue
            for variant in VARIANTS:
                if variant == "ours":
                    continue
                other = med(variant)
                if other is None:
                    continue
                n_total += 1
                ok = ours < other
                n_ok += int(ok)
                ratio = other / ours if ours > 0 else np.inf
                lines.append(f"  {cell['key']:<10}{scenario:<9}"
                             f"ours {ours:9.3f} vs {variant:<14}{other:9.3f} mm"
                             f"   {'OK' if ok else '역전':<5} (x{ratio:.1f})")
    for line in lines:
        log(line)
    log(f"\n  성립 {n_ok}/{n_total} 쌍 (궤적 RMSE 중앙값 기준)")
    return n_ok, n_total


def caveats(results, log=print):
    """이 실험이 **보여주지 못하는 것**을 스스로 적는다.

    표만 보면 "27/27 전승" 으로 읽히기 쉬운데, 그중 일부는 실력이 아니라
    구조 때문에 그렇게 나온다. 심사자가 먼저 찾아낼 구멍을 우리가 먼저 쓴다.
    """
    lines = [
        "이것은 시뮬 대 시뮬이다. 기준은 GT 밀도를 넣은 Drake plant 이고 실물이"
        " 아니다. 여기서 나온 mm 수치는 '밀도 오차가 궤적 오차로 얼마나"
        " 번지는가' 이지 sim-to-real 격차가 아니다.",
        "형상·접촉강성·소산·마찰·관절감쇠는 네 variant 에서 완전히 같다."
        " 우리 방법은 이 값들을 식별하지 않으므로, 실물에서는 이것들이"
        " 따로 틀린 만큼 오차가 더 붙는다. 이 실험은 그 몫을 아예 못 본다.",
        "vision baseline 밀도는 v1 표(table_main_v2.tex)의 질량 상대오차"
        " **크기**만 되살린 것이다. 표에 부호가 없어 과대추정(+)으로 놓았다."
        " 부호가 반대면 궤적 오차의 크기도 달라진다.",
        "2link/3link 의 표 GT 질량은 ballast 넣기 전 예비값이라, 절대질량이"
        " 아니라 상대오차만 옮겼다.",
        "PhysX-Omni 는 2link/3link 에서 N/A 라 전부 빼고 SiPhy·PUGS 만 썼다.",
        "PUSH 는 접촉을 거치므로 작은 차이가 크게 증폭된다. ours 도 수십 mm"
        " 가 난다. 절대값이 아니라 **순서**로만 읽어야 한다.",
        "RELEASE 에서 관절은 대개 기계적 한계에 가서 멈춘다. 그래서 final"
        " joint-angle 오차가 작게 나오는 것은 잘 맞춰서가 아니라 한계가"
        " 차이를 먹어서일 수 있다. 정보는 과도구간(traj RMSE)에 있다.",
        "2link 의 RELEASE 는 원리상 밀도를 거의 못 본다: 뿌리를 고정하면"
        " 남는 자유도가 하나뿐이고, 단일 진자의 운동은 그 링크 밀도를 통째로"
        " 배율해도 변하지 않는다(m 과 I 가 같이 커진다). 표의 sub-mm 값은"
        " 실력이 아니라 이 불변성의 결과다.",
    ]
    log("\n" + "=" * 96)
    log("이 실험이 보이지 못하는 것 (읽는 사람이 먼저 물을 것들)")
    log("=" * 96)
    for line in lines:
        log(f"  - {line}")

    # 데이터로 확인: 모든 variant 가 다 작으면 그 칸은 판별력이 없는 칸이다.
    weak = []
    for cell in results:
        for scenario in SCENARIOS:
            vals = [np.median([r["traj_rmse_mm_max"] for r in cell["runs"]
                               if r["scenario"] == scenario
                               and r["variant"] == v])
                    for v in VARIANTS
                    if any(r["scenario"] == scenario and r["variant"] == v
                           for r in cell["runs"])]
            if vals and max(vals) < 1.0:
                weak.append((cell["key"], scenario, float(max(vals))))
    if weak:
        log("\n  판별력이 사실상 없는 칸 (모든 variant 의 중앙 RMSE < 1 mm):")
        for key, scenario, worst in weak:
            log(f"    {key:<10}{scenario:<9}최악 variant 도 {worst:.2f} mm")
    else:
        log("\n  모든 (물체, 시나리오) 칸에서 최악 variant 가 1 mm 이상 벌어졌다.")
    return lines, weak


def make_figure(results, path=OUT_PNG, log=print):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows, cols = len(SCENARIOS), len(results)
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 2.9 * rows),
                             squeeze=False)
    for j, cell in enumerate(results):
        for i, scenario in enumerate(SCENARIOS):
            ax = axes[i][j]
            s = cell["series"].get(scenario)
            if s is None:
                ax.axis("off")
                continue
            t = np.array(s["t"])
            # gt 를 굵게 **아래**에 깔고 ours 를 가늘게 그 위에 올린다.
            # 반대로 그리면 ours 가 gt 에 완전히 가려서 "따라간다" 는 것이
            # 그림에서 안 보인다 (그게 바로 이 그림이 보여야 할 것이다).
            order = {"gt": 2, "ours": 6}
            for name, disp in s["disp_mm"].items():
                ax.plot(t, disp, color=COLORS.get(name, "0.5"),
                        lw=3.4 if name == "gt" else 1.5,
                        ls="-" if name in ("gt", "ours") else "--",
                        alpha=1.0 if name in ("gt", "ours") else 0.9,
                        zorder=order.get(name, 4),
                        label=name if (i == 0 and j == 0) else None)
            meds = {}
            for variant in VARIANTS:
                vals = [r["traj_rmse_mm_max"] for r in cell["runs"]
                        if r["scenario"] == scenario and r["variant"] == variant]
                if vals:
                    meds[variant] = float(np.median(vals))
            # 수치는 제목 아래 줄에 둔다. 축 안에 넣으면 궤적을 가린다.
            sub = ""
            if meds:
                best = min((v for k, v in meds.items() if k != "ours"),
                           default=np.nan)
                sub = (f"\nRMSE med: ours {meds.get('ours', np.nan):.2f} / "
                       f"best baseline {best:.2f} mm")
            ax.set_title(f"{cell['key']} — {scenario} (tip {s['tip']}){sub}",
                         fontsize=8.5)
            ax.grid(alpha=0.25)
            if i == rows - 1:
                ax.set_xlabel("time [s]")
            if j == 0:
                ax.set_ylabel("tip displacement [mm]")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               frameon=False, fontsize=9)
    fig.suptitle("Asset dynamic fidelity — SIMULATION-ONLY surrogate "
                 "(sim-vs-sim; reference = GT-density plant, black).\n"
                 "Identical geometry, contact, friction and joint damping "
                 "across variants; only per-part density differs. "
                 "Initial condition #0 of 5.", fontsize=10)
    fig.tight_layout(rect=(0, 0.045, 1, 0.93))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    log(f"  그림 -> {path}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", nargs="+", default=list(OBJECT_KEYS))
    ap.add_argument("--json", default=str(OUT_JSON))
    ap.add_argument("--png", default=str(OUT_PNG))
    args = ap.parse_args()

    t0 = time.perf_counter()
    print(__doc__.split("실행:")[0].strip()[:0] or "", end="")
    print("자산 거동 충실도 실험 — SIMULATION-ONLY 대리실험 (sim-vs-sim).")
    print("  기준은 GT 밀도 plant 이고 실물이 아니다. 실물 검증이 아니다.")
    print(f"  공유 물리: dt={TIME_STEP}s, SAP, point contact k={POINT_STIFFNESS:g} N/m,"
          f" d={HC_DISSIPATION:g} s/m, mu=({MU_STATIC},{MU_DYNAMIC}),"
          f" 관절감쇠 {JOINT_DAMPING} N·m·s/rad")
    print(f"  초기조건 {N_IC}개/시나리오, 시나리오 {SCENARIOS}", flush=True)

    results = [run_object(key) for key in args.objects]
    print_table(results)
    n_ok, n_total = verdict(results)
    caveat_lines, weak_cells = caveats(results)

    blob = dict(
        kind="simulation-only surrogate (sim-vs-sim); NOT a real-world validation",
        reference="Drake plant built with ground-truth per-part densities",
        note=("Geometry, contact stiffness/dissipation/friction and joint damping "
              "are identical across variants; the only difference is per-part "
              "density. Contact and friction are NOT identified by our method."),
        vision_baseline_note=("vision-* densities are reconstructed from the "
                              "frozen v1 table tables/table_main_v2.tex: "
                              "implied_mass = gt_mass*(1+err/100). The table "
                              "reports error magnitudes only, so the sign is "
                              "unknown and taken as positive."),
        settings=dict(time_step=TIME_STEP, sample_dt=SAMPLE_DT,
                      joint_damping=JOINT_DAMPING,
                      point_stiffness=POINT_STIFFNESS,
                      hc_dissipation=HC_DISSIPATION,
                      mu=[MU_STATIC, MU_DYNAMIC],
                      T=dict(release=T_RELEASE, drop=T_DROP, push=T_PUSH),
                      drop_clearance_m=DROP_CLEARANCE,
                      rest_clearance_m=REST_CLEARANCE,
                      push_window_s=[PUSH_START, PUSH_END], push_g=PUSH_G,
                      n_ic=N_IC, ic_seed=IC_SEED, scenarios=list(SCENARIOS)),
        ordering_holds=dict(n_ok=n_ok, n_total=n_total),
        caveats=caveat_lines,
        low_information_cells=[dict(object=k, scenario=s, worst_med_rmse_mm=w)
                               for k, s, w in weak_cells],
        objects=results)
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(blob, indent=1, default=float))
    print(f"\n  JSON -> {out}")
    make_figure(results, args.png)
    print(f"  걸린 시간 {time.perf_counter()-t0:.1f} s")


if __name__ == "__main__":
    main()
