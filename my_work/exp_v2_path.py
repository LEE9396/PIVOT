"""기여 2 (Multi-Topology Measurement Path Design) 정량 표를 만드는 실험.

무엇을 주장하려는 것인가
------------------------
한 물체 관절각(configuration)마다 로봇은 중력방향이 정규직교 삼항쌍
(canonical orthonormal triad, design_core.CANONICAL_TRIAD) 을 이루는 **세**
측정 자세를 만들어야 한다. 그리고 그 자세들 **사이를 실제로 이동**해야 한다.

여기서 흔히 하는 오해가 하나 있다. "IK 가 세 자세를 다 풀었으니 이제 그
사이는 관절각을 선형보간하면 된다"는 것이다. IK 는 **점** 하나가 충돌 없음을
보장할 뿐이고, 두 점을 잇는 직선 위는 아무도 검사하지 않았다. 저장소 문서
(PIPELINE.md, path_planning.py) 는 실제로 그렇게 하다가 물체 끝 링크가
테이블을 62 mm 관통하는 구간을 만났다고 적고 있다.

이 스크립트는 그 주장을 **같은 판정 기준으로** 숫자로 만든다.

  직선 보간  : 두 IK 해 사이를 관절공간에서 그냥 선형보간 (100 경유점 이상)
  RRT-Connect: path_planning.ArmPathPlanner 가 찾은 충돌 없는 경로

둘을 **같은 술어**로 채점한다 — 모든 충돌쌍의 최소거리가 문턱
(robot_scene.MIN_DISTANCE_M = 10 mm) 이상인가. 이것은 IK 가 쓰는 제약이자
RRT 가 간선 검사에 쓰는 제약과 같은 것이라, 두 경로가 같은 잣대로 비교된다.

왜 자세를 이렇게 고르는가
-------------------------
관절각 후보는 파이프라인(dual_view.prepare)과 **같은 격자**를 쓴다 —
관절당 5칸 (dual_view 의 --steps 기본값). 전수를 다 쓰는 이유는 두 가지다.

  (1) D-최적 선택이 고른 자세만 보면 "고른 자세가 나쁜 것 아니냐"는 반문에
      답할 수 없다. 격자 전체를 재고 정보이득을 같이 적으면, 정보이득이
      큰 자세가 곧 기하적으로 까다로운 자세라는 것을 표로 보일 수 있다.
  (2) 도달 실패(IK 실패)도 결과다. 격자 전체를 훑어야 실패율이 의미를 갖는다.

왜 이 자세쌍들인가
------------------
한 라운드에서 로봇이 실제로 하는 이동은 다음과 같다.

    시작자세(home) -> m0(-z) -> m1(+x) -> m2(+y) -> 시작자세

  triad  구간 : m0->m1, m1->m2, m2->m0  (측정 자세 사이의 순서쌍. 본 실험의 주대상)
  transit 구간: home->m0, m2->home      (파이프라인이 실제로 도는 왕복 구간)

둘을 따로 집계한다. 섞으면 어느 쪽이 문제인지 표에서 사라진다.

실행:
    cd ~/Desktop/PIVOT/my_work
    ../robot_learning/scripts/run_drake_env.sh python exp_v2_path.py
    ../robot_learning/scripts/run_drake_env.sh python exp_v2_path.py --objects 3link
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import density_id_drake as alg
import density_id_objects as obj
import design_core as dc
import path_planning as pp
import robot_scene as rs

OUT_DIR = Path("figures/v2")
JSON_PATH = OUT_DIR / "path_compare.json"
FIG_PATH = OUT_DIR / "fig_path.png"

# 채점 해상도. RRT 는 간선을 0.05 rad 마다 검사하지만, 채점은 그보다 훨씬
# 촘촘하게 한다 — 계획기가 놓친 관통이 있으면 그것도 결과이기 때문이다.
# 두 경로 모두 같은 해상도로 재야 비교가 성립한다.
SCORE_STEP_RAD = 0.005
MIN_SAMPLES = 100                # 과제가 요구한 "직선 100 경유점" 하한
CLEARANCE_QUERY_M = 0.05         # 최소거리 질의 반경. 이보다 멀면 '충분히 멂'
GRAVITY_LABELS = ("-z", "+x", "+y")
DEFAULT_STEPS = 5                # dual_view 의 --steps 기본값과 같게 둔다


# ---------------------------------------------------------------------------
# 물체 준비 — dual_view.prepare 와 같은 순서로 묶는다.
# ---------------------------------------------------------------------------
def build_object(key, gripper="robotiq2f85"):
    """dual_view 가 쓰는 것과 같은 spec/사전분포/PoseChecker 를 만든다.

    desklamp 만 예외다. density_id_objects.get_spec("desklamp") 은
    desk_lamp.DEFAULT_GRASP("pinch") 로 spec 을 만드는데, 그 파지점에서는
    IK 가 격자 전체에서 실패한다(직접 확인). 실제 파이프라인인 dual_view 는
    --grasp 기본값이 "centroid" 이므로 여기서도 centroid 파지를 쓴다.
    """
    if key == "desklamp":
        import desk_lamp as lamp
        spec = lamp.build_spec(grasp_at="centroid", grasp_part="link_3")
    else:
        spec = obj.OBJECTS[key]

    # 힌지 토크 필터는 쓰지 않는다 (dual_view 의 --hinge-torque 기본값이 없음
    # = 관절이 손으로 맞춰 두면 고정된다는 이 연구의 가정).
    rho_gt = obj.bind_object(spec, hinge=None)
    limits = rs.parse_joint_range(spec, None)
    alg.JOINT_LIMITS = list(limits)
    # 정보이득을 파이프라인과 같은 사전분포에서 재려면 저울 사전분포를 건다.
    total = obj.assembled_mass_kg(spec, rho_gt)
    obj.apply_weight_prior(spec, total)
    checker = rs.PoseChecker(spec, densities=rho_gt, joint_limits_rad=limits,
                             min_distance_m=rs.MIN_DISTANCE_M, gripper=gripper)
    return spec, limits, rho_gt, checker


def config_grid(limits, n_per_joint=None):
    """구동범위 균등 격자. dual_view 의 --steps 기본값(5)과 같게 둔다."""
    if n_per_joint is None:
        n_per_joint = DEFAULT_STEPS
    axes = [np.linspace(lo, hi, n_per_joint) for lo, hi in limits]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(
        -1, len(limits))


def regressor_quality(theta):
    """이 자세가 밀도 추정에 얼마나 유용한가.

    조건수는 백색화한 회귀행렬 A_w = R^{-1/2} A 로 잰다. 백색화를 안 하면
    힘 행(N)과 토크 행(N·m)의 단위가 달라 조건수가 단위 선택의 함수가 된다.
    정보이득은 저장소의 density_id_drake.info_gain 을 그대로 쓴다.
    """
    A = dc.regressor(theta, dc.CANONICAL_TRIAD)
    weight = 1.0 / np.sqrt(np.tile(alg.R_EPS_DIAG, len(dc.CANONICAL_TRIAD)))
    svals = np.linalg.svd(A * weight[:, None], compute_uv=False)
    smin = float(svals[-1])
    # 한 자세만으로는 관측되는 양이 (총질량 1개 + 1차 모멘트 3개) = 4개뿐이다.
    # 미지수가 그보다 많은 물체(3link: 부위 3 + 힌지 2 = 5)는 자세 하나로는
    # 구조적으로 계급부족이라 조건수가 1e17 규모로 뜬다. 이것은 버그가 아니라
    # "자세를 여러 개 써야 한다"는 이 연구의 전제 그 자체다. 그래서 조건수와
    # 함께 수치계급도 같이 적는다.
    rank = int(np.sum(svals > svals[0] * 1e-10))
    return dict(
        info_gain=float(alg.info_gain(A, alg.SIGMA0)),
        cond=float(svals[0] / smin) if smin > 0 else float("inf"),
        sigma_min=smin, rank=rank, n_unknowns=int(A.shape[1]),
    )


# ---------------------------------------------------------------------------
# 경로 채점 — 직선이든 RRT 든 **이 함수 하나로만** 잰다.
# ---------------------------------------------------------------------------
class PathScorer:
    """계획기와 같은 plant/context/술어를 쓰되, 얼마나 파고들었는지까지 잰다.

    ArmPathPlanner.valid() 는 참/거짓만 돌려준다. 표에는 '얼마나 나빴나'가
    필요하므로 최소거리 값 자체와 어느 두 물체였는지를 같이 남긴다.
    """

    def __init__(self, planner, checker, spec):
        self.planner = planner
        self.plant = planner.plant
        self.context = planner.context
        self.query_port = planner.query_port
        self.min_distance_m = planner.min_distance_m
        self.lower, self.upper = planner.lower, planner.upper
        # 물체 끝 링크. 테이블을 뚫는 것은 대개 이 부위다 (그림용 궤적).
        self.tip_body = self.plant.GetBodyByName(spec.parts[-1].name,
                                                 checker.payload)

    def sample(self, nodes):
        """경로를 호길이 기준으로 균등 분할한다. 두 방식에 같은 해상도를 준다."""
        nodes = [np.asarray(p, dtype=float) for p in nodes]
        length = float(sum(np.linalg.norm(b - a)
                           for a, b in zip(nodes[:-1], nodes[1:])))
        count = max(MIN_SAMPLES, int(np.ceil(length / SCORE_STEP_RAD)) + 1)
        return pp.ArmPathPlanner.resample(nodes, count), length

    def score(self, nodes, keep_series=False):
        waypoints, length = self.sample(nodes)
        clearances = np.empty(len(waypoints))
        tips = np.empty((len(waypoints), 3))
        worst_distance, worst_pair, worst_index = np.inf, None, -1
        n_bad, n_limit = 0, 0

        for index, arm_q in enumerate(waypoints):
            if np.any(arm_q < self.lower) or np.any(arm_q > self.upper):
                n_limit += 1
            self.plant.SetPositions(self.context, self.planner.full_q(arm_q))
            query = self.query_port.Eval(self.context)
            pairs = query.ComputeSignedDistancePairwiseClosestPoints(
                CLEARANCE_QUERY_M)
            if pairs:
                worst = min(pairs, key=lambda pair: pair.distance)
                distance = float(worst.distance)
            else:
                # 질의 반경 안에 아무 쌍도 없다 = 충분히 멀다.
                distance, worst = CLEARANCE_QUERY_M, None
            clearances[index] = distance
            tips[index] = self.plant.EvalBodyPoseInWorld(
                self.context, self.tip_body).translation()
            # 술어는 계획기와 똑같다: 모든 쌍의 거리 >= 문턱.
            if distance < self.min_distance_m:
                n_bad += 1
            if distance < worst_distance:
                worst_distance, worst_index = distance, index
                if worst is not None:
                    inspector = query.inspector()
                    worst_pair = tuple(
                        self.plant.GetBodyFromFrameId(
                            inspector.GetFrameId(gid)).name()
                        for gid in (worst.id_A, worst.id_B))

        result = dict(
            n_waypoints=len(waypoints),
            length_rad=length,
            in_collision=bool(n_bad > 0),
            n_bad_waypoints=int(n_bad),
            bad_fraction=float(n_bad / len(waypoints)),
            min_clearance_mm=float(1000.0 * worst_distance),
            # 문턱 아래로 얼마나 내려갔나 (안전여유 침범량)
            violation_mm=float(max(0.0, 1000.0 * (self.min_distance_m
                                                  - worst_distance))),
            # 실제로 형상이 겹친 깊이 (음수 거리)
            penetration_mm=float(max(0.0, -1000.0 * worst_distance)),
            worst_pair=list(worst_pair) if worst_pair else None,
            worst_at=float(worst_index / max(len(waypoints) - 1, 1)),
            n_limit_violations=int(n_limit),
        )
        if keep_series:
            result["series"] = dict(
                clearance_mm=(1000.0 * clearances).tolist(),
                tip_xyz=tips.tolist(),
                waypoints=[w.tolist() for w in waypoints],
            )
        return result


# ---------------------------------------------------------------------------
# 한 물체를 통째로 돌린다.
# ---------------------------------------------------------------------------
def run_object(key, gripper="robotiq2f85", n_per_joint=None, plan_iters=20000,
               seed=0, with_transit=True):
    print(f"\n{'=' * 78}\n[{key}] 준비")
    spec, limits, rho_gt, checker = build_object(key, gripper)
    print(f"  {spec.label}  부위 {len(spec.parts)}개, 관절 {len(spec.joints)}개")
    for joint, (lo, hi) in zip(spec.joints, limits):
        print(f"    {joint.name}: {np.degrees(lo):.0f} ~ {np.degrees(hi):.0f} deg")

    grid = config_grid(limits, n_per_joint)
    arm_index = [joint.position_start() for joint in checker.arm_joints]
    planner = pp.ArmPathPlanner(
        checker.plant, checker.context, checker.arm_joints,
        rs.MIN_DISTANCE_M, checker.plant.GetPositions(checker.context),
        seed=seed)
    scorer = PathScorer(planner, checker, spec)

    # --- 1단계: 도달성. 세 방향 IK 를 먼저 다 풀어 둔다 -------------------
    # 시작 자세 탐색(find_starting_pose)은 **쓸 수 있는 자세들** 위에서만
    # 뜻이 있다. dual_view.prepare 도 도달 가능한 자세를 추린 뒤에 부른다.
    # 애초에 IK 가 안 풀리는 자세까지 넣으면 시작 자세가 늘 실패한다.
    configs, solved = [], []
    for theta in grid:
        quality = regressor_quality(theta)
        # IK 는 앞 자세의 해를 물려받지 않도록 매번 초기화한다
        # (robot_scene.solutions_for 와 같은 규칙).
        checker._last_solution = None
        solutions, ik_time = [], []
        for g_hat in dc.CANONICAL_TRIAD:
            t0 = time.perf_counter()
            solutions.append(checker.solve_robust(theta, g_hat))
            ik_time.append(time.perf_counter() - t0)
        n_ik_fail = sum(1 for s in solutions if s is None)

        entry = dict(
            theta_deg=[float(v) for v in np.degrees(theta)],
            info_gain=quality["info_gain"], cond=quality["cond"],
            sigma_min=quality["sigma_min"],
            rank=quality["rank"], n_unknowns=quality["n_unknowns"],
            ik_ok=[s is not None for s in solutions],
            ik_fail=int(n_ik_fail),
            ik_time_s=[float(t) for t in ik_time],
            pairs=[],
        )
        configs.append(entry)
        if n_ik_fail:
            print(f"  theta={np.round(np.degrees(theta), 0)}  IK 실패 "
                  f"{n_ik_fail}/3 ({[GRAVITY_LABELS[i] for i, s in enumerate(solutions) if s is None]})"
                  " — 이 자세는 경로 비교 대상이 아니다")
        else:
            solved.append((theta, entry, solutions))

    # --- 2단계: 시작 자세(home). 파이프라인은 여기서 출발해 측정 자세로 간다 ---
    home_full, presentation = (None, None)
    if with_transit and solved:
        home_full, presentation = rs.find_starting_pose(
            checker, [theta for theta, _, _ in solved])
        if home_full is None:
            print("  [주의] 쓸 수 있는 자세 전 구간에서 안전한 시작 자세를 "
                  "못 찾음 — transit 구간은 건너뛴다")
        else:
            print(f"  시작 자세 제시 위치 {np.round(presentation, 3)} m")

    # --- 3단계: 자세쌍마다 직선 보간과 RRT 를 같은 잣대로 채점 --------------
    for theta, entry, solutions in solved:
        quality = dict(info_gain=entry["info_gain"], cond=entry["cond"])
        # 계획기가 보는 고정 자세(물체 관절각, 손가락)를 이 자세로 맞춘다.
        planner.set_fixed(np.asarray(solutions[0]))
        arms = [np.asarray(s)[arm_index] for s in solutions]

        legs = [("triad", i, (i + 1) % 3, arms[i], arms[(i + 1) % 3],
                 f"m{i}({GRAVITY_LABELS[i]})->m{(i+1)%3}({GRAVITY_LABELS[(i+1)%3]})")
                for i in range(3)]
        if home_full is not None:
            home_arm = np.asarray(home_full)[arm_index]
            legs.append(("transit", -1, 0, home_arm, arms[0],
                         f"home->m0({GRAVITY_LABELS[0]})"))
            legs.append(("transit", 2, -1, arms[2], home_arm,
                         f"m2({GRAVITY_LABELS[2]})->home"))

        for kind, i_from, i_to, q_from, q_to, label in legs:
            # (1) 직선 보간 — 두 끝점만 잇는다. 아무 검사도 하지 않는다.
            t0 = time.perf_counter()
            straight_nodes = [q_from, q_to]
            straight_time = time.perf_counter() - t0
            straight = scorer.score(straight_nodes)
            straight["plan_time_s"] = float(straight_time)
            straight["planned"] = True
            straight["n_nodes"] = 2

            # (2) RRT-Connect — 같은 술어로 간선을 검사하며 찾는다.
            planner.checks = 0
            t0 = time.perf_counter()
            nodes = planner.plan(q_from, q_to, max_iters=plan_iters)
            rrt_time = time.perf_counter() - t0
            if nodes is None:
                rrt = dict(planned=False, plan_time_s=float(rrt_time),
                           n_nodes=0, collision_checks=int(planner.checks))
            else:
                rrt = scorer.score(nodes)
                rrt["planned"] = True
                rrt["plan_time_s"] = float(rrt_time)
                rrt["n_nodes"] = len(nodes)
                rrt["collision_checks"] = int(planner.checks)

            entry["pairs"].append(dict(
                kind=kind, label=label, i_from=i_from, i_to=i_to,
                delta_max_deg=float(np.degrees(np.abs(q_to - q_from).max())),
                q_from=q_from.tolist(), q_to=q_to.tolist(),
                straight=straight, rrt=rrt))

        flags = "".join("X" if p["straight"]["in_collision"] else "."
                        for p in entry["pairs"])
        pen = max(p["straight"]["penetration_mm"] for p in entry["pairs"])
        print(f"  theta={np.round(np.degrees(theta), 0)}  dI={quality['info_gain']:6.2f}"
              f"  cond={quality['cond']:9.2e}  직선충돌[{flags}]"
              f"  최대관통 {pen:5.1f} mm")

    return dict(object=key, label=spec.label,
                joint_names=[j.name for j in spec.joints],
                limits_deg=[[float(np.degrees(lo)), float(np.degrees(hi))]
                            for lo, hi in limits],
                part_names=[p.name for p in spec.parts],
                gripper=gripper,
                n_per_joint=int(n_per_joint or DEFAULT_STEPS),
                n_configs=len(grid),
                min_distance_mm=1000.0 * rs.MIN_DISTANCE_M,
                home_found=home_full is not None,
                configs=configs), checker, planner, scorer, spec, arm_index


# ---------------------------------------------------------------------------
# 집계 + 표
# ---------------------------------------------------------------------------
def aggregate(record, kind=None):
    pairs = [p for c in record["configs"] for p in c["pairs"]
             if kind is None or p["kind"] == kind]
    out = dict(n_pairs=len(pairs))
    if not pairs:
        return out
    for name in ("straight", "rrt"):
        planned = [p[name] for p in pairs if p[name]["planned"]]
        collided = [d for d in planned if d["in_collision"]]
        out[name] = dict(
            n_planned=len(planned),
            plan_fail_rate=float(1.0 - len(planned) / len(pairs)),
            collision_rate=float(len(collided) / len(planned)) if planned else float("nan"),
            max_penetration_mm=float(max([d["penetration_mm"] for d in planned], default=0.0)),
            max_violation_mm=float(max([d["violation_mm"] for d in planned], default=0.0)),
            mean_min_clearance_mm=float(np.mean([d["min_clearance_mm"] for d in planned])) if planned else float("nan"),
            limit_violations=int(sum(d["n_limit_violations"] for d in planned)),
            mean_length_rad=float(np.mean([d["length_rad"] for d in planned])) if planned else float("nan"),
            mean_plan_time_s=float(np.mean([d["plan_time_s"] for d in planned])) if planned else float("nan"),
            max_plan_time_s=float(max([d["plan_time_s"] for d in planned], default=0.0)),
        )
    return out


def ik_stats(record):
    n_config = len(record["configs"])
    n_dir = 3 * n_config
    n_fail = sum(c["ik_fail"] for c in record["configs"])
    n_config_fail = sum(1 for c in record["configs"] if c["ik_fail"])
    return dict(n_config=n_config, n_dir=n_dir, dir_fail=n_fail,
                dir_fail_rate=n_fail / n_dir if n_dir else float("nan"),
                config_fail=n_config_fail,
                config_fail_rate=n_config_fail / n_config if n_config else float("nan"))


def print_tables(records):
    print("\n" + "=" * 118)
    print("표 1. 자세 도달성 (canonical triad 세 방향 모두 IK 가 풀려야 그 자세를 쓸 수 있다)")
    print("-" * 118)
    print(f"{'object':<10} {'configs':>8} {'triad-IK 실패 자세':>18} "
          f"{'방향별 IK 실패':>16} {'쓸 수 있는 자세':>16}")
    for record in records:
        stats = ik_stats(record)
        print(f"{record['object']:<10} {stats['n_config']:>8} "
              f"{stats['config_fail']:>10}/{stats['n_config']:<7} "
              f"{stats['dir_fail']:>8}/{stats['n_dir']:<7} "
              f"{100*(1-stats['config_fail_rate']):>14.0f}%")

    for kind, title in (("triad", "측정 자세 사이 (m_i -> m_{i+1}, 순환)"),
                        ("transit", "시작 자세 왕복 (home -> m0, m2 -> home)")):
        rows = [(r, aggregate(r, kind)) for r in records]
        rows = [(r, a) for r, a in rows if a["n_pairs"]]
        if not rows:
            continue
        print("\n" + "=" * 118)
        print(f"표 2{'a' if kind == 'triad' else 'b'}. 경로 비교 — {title}"
              f"   (판정: 모든 충돌쌍 최소거리 >= {rows[0][0]['min_distance_mm']:.0f} mm)")
        print("-" * 118)
        print(f"{'object':<10} {'방식':<10} {'쌍':>4} {'충돌쌍비율':>10} "
              f"{'최대관통':>9} {'문턱침범':>9} {'평균여유':>9} {'한계위반':>8} "
              f"{'경로길이':>9} {'계획시간':>10} {'계획실패':>9}")
        print(f"{'':<10} {'':<10} {'':>4} {'[%]':>10} {'[mm]':>9} {'[mm]':>9} "
              f"{'[mm]':>9} {'[점]':>8} {'[rad]':>9} {'[s]':>10} {'[%]':>9}")
        for record, agg in rows:
            for name, label in (("straight", "직선보간"), ("rrt", "RRT-Connect")):
                data = agg[name]
                print(f"{record['object'] if name == 'straight' else '':<10} "
                      f"{label:<10} {agg['n_pairs']:>4} "
                      f"{100*data['collision_rate']:>10.1f} "
                      f"{data['max_penetration_mm']:>9.1f} "
                      f"{data['max_violation_mm']:>9.1f} "
                      f"{data['mean_min_clearance_mm']:>9.1f} "
                      f"{data['limit_violations']:>8} "
                      f"{data['mean_length_rad']:>9.2f} "
                      f"{data['mean_plan_time_s']:>10.3f} "
                      f"{100*data['plan_fail_rate']:>9.1f}")

    print("\n" + "=" * 118)
    print("표 3. 자세별 상세 — 정보이득이 큰 자세가 기하적으로도 까다로운가")
    print("-" * 118)
    print("(cond = 백색화한 회귀행렬의 조건수. 한 자세로 관측되는 양은 총질량 1 +"
          " 1차모멘트 3 = 4개뿐이라,")
    print(" 미지수가 그보다 많은 물체는 자세 하나로는 계급부족이다 — rank 열이"
          " 그 사실을 보여 준다.)")
    print(f"{'object':<9} {'theta [deg]':>16} {'dI':>7} {'cond':>10} {'rank':>6} "
          f"{'IK':>4} {'직선 충돌쌍':>12} {'직선 최대관통':>13} {'RRT 충돌쌍':>11} "
          f"{'RRT 최대계획시간':>16}")
    for record in records:
        order = sorted(range(len(record["configs"])),
                       key=lambda i: -record["configs"][i]["info_gain"])
        for order_rank, index in enumerate(order):
            config = record["configs"][index]
            theta = np.round(config["theta_deg"], 0)
            if not config["pairs"]:
                print(f"{record['object'] if order_rank == 0 else '':<9} "
                      f"{str(theta):>16} {config['info_gain']:>7.2f} "
                      f"{config['cond']:>10.2e} "
                      f"{config['rank']:>3}/{config['n_unknowns']:<2} "
                      f"{3-config['ik_fail']:>2}/3 "
                      f"{'— (IK 실패로 경로 없음)':>40}")
                continue
            straight = [p["straight"] for p in config["pairs"]]
            rrt = [p["rrt"] for p in config["pairs"]]
            n_s = sum(1 for d in straight if d["in_collision"])
            n_r = sum(1 for d in rrt if d["planned"] and d["in_collision"])
            n_r += sum(1 for d in rrt if not d["planned"])
            print(f"{record['object'] if order_rank == 0 else '':<9} "
                  f"{str(theta):>16} {config['info_gain']:>7.2f} "
                  f"{config['cond']:>10.2e} "
                  f"{config['rank']:>3}/{config['n_unknowns']:<2} "
                  f"{3-config['ik_fail']:>2}/3 "
                  f"{n_s:>6}/{len(straight):<5} "
                  f"{max(d['penetration_mm'] for d in straight):>13.1f} "
                  f"{n_r:>5}/{len(rrt):<5} "
                  f"{max(d['plan_time_s'] for d in rrt):>16.2f}")


# ---------------------------------------------------------------------------
# 그림
# ---------------------------------------------------------------------------
def highlight_config(record):
    """그림에 쓸 대표 자세.

    그림의 목적은 '직선 보간이 어디서 어떻게 깨지는가'를 보이는 것이다.
    그래서 **측정 자세 사이 직선 보간이 실제로 충돌하는 자세들 중** 정보이득이
    가장 큰 것을 고른다 — 즉 파이프라인이 실제로 고를 만한 자세다. 충돌하는
    자세가 하나도 없으면 그냥 정보이득 최대 자세를 그린다(그 경우에는
    "직선 보간도 통과했다"가 그림이 말하는 바다).
    """
    usable = [c for c in record["configs"] if c["pairs"]]
    if not usable:
        return None
    broken = [c for c in usable
              if any(p["straight"]["in_collision"] for p in c["pairs"]
                     if p["kind"] == "triad")]
    return max(broken or usable, key=lambda c: c["info_gain"])


def make_figure(records, series_store, path=FIG_PATH):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in records if highlight_config(r) is not None]
    if not rows:
        print("그릴 자세가 없다 — 그림을 건너뛴다")
        return
    fig, axes = plt.subplots(len(rows), 3, figsize=(16.5, 4.1 * len(rows)),
                             squeeze=False)
    table_top = rs.TABLE_TOP_Z_M

    for row, record in enumerate(rows):
        config = highlight_config(record)
        key = (record["object"], tuple(config["theta_deg"]))
        series = series_store.get(key, {})
        triad = [p for p in config["pairs"] if p["kind"] == "triad"]
        theta_label = ", ".join(f"{v:.0f}" for v in config["theta_deg"])

        # --- (a) 관절공간: 세 측정 자세와 두 경로 ---------------------------
        # 6차원을 그대로 그릴 수 없으므로, 이 자세의 모든 경유점을 모아
        # 주성분 2개로 편다. 직선이 정말 '직선'으로 보이는 좌표계다.
        ax = axes[row][0]
        cloud = []
        for pair in triad:
            for name in ("straight", "rrt"):
                data = series.get((pair["label"], name))
                if data is not None:
                    cloud.append(np.asarray(data["waypoints"]))
        cloud = np.vstack(cloud)
        center = cloud.mean(axis=0)
        basis = np.linalg.svd(cloud - center, full_matrices=False)[2][:2]
        project = lambda q: (np.atleast_2d(q) - center) @ basis.T

        marked = False        # 충돌 표시 범례는 실제로 충돌이 난 첫 구간에만 단다
        for index, pair in enumerate(triad):
            color = f"C{index}"
            straight = series.get((pair["label"], "straight"))
            if straight is not None:
                points = project(np.asarray(straight["waypoints"]))
                bad = np.asarray(straight["clearance_mm"]) < record["min_distance_mm"]
                ax.plot(points[:, 0], points[:, 1], "--", color=color, lw=1.4,
                        alpha=0.9,
                        label="straight-line" if index == 0 else None)
                if bad.any():
                    ax.plot(points[bad, 0], points[bad, 1], "x", color="crimson",
                            ms=5, mew=1.4,
                            label=None if marked else "straight-line in collision")
                    marked = True
            rrt = series.get((pair["label"], "rrt"))
            if rrt is not None:
                points = project(np.asarray(rrt["waypoints"]))
                ax.plot(points[:, 0], points[:, 1], "-", color=color, lw=1.8,
                        label="RRT-Connect" if index == 0 else None)
        poses = project(np.array([p["q_from"] for p in triad]))
        ax.plot(poses[:, 0], poses[:, 1], "ko", ms=9, zorder=5,
                label="measurement pose")
        for index, point in enumerate(poses):
            ax.annotate(f" m{index} ({GRAVITY_LABELS[index]})", point,
                        fontsize=9, weight="bold")
        ax.set_xlabel("arm joint space, PC1 [rad]")
        ax.set_ylabel("PC2 [rad]")
        ax.set_title(f"{record['object']}  theta = [{theta_label}] deg\n"
                     f"joint-space paths (dI = {config['info_gain']:.2f})",
                     fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="best")

        # --- (b) 최소거리 -------------------------------------------------
        ax = axes[row][1]
        for index, pair in enumerate(triad):
            color = f"C{index}"
            for name, style in (("straight", "--"), ("rrt", "-")):
                data = series.get((pair["label"], name))
                if data is None:
                    continue
                values = np.asarray(data["clearance_mm"])
                ax.plot(np.linspace(0, 1, len(values)), values, style,
                        color=color, lw=1.6,
                        label=f"{pair['label']} {'straight' if name == 'straight' else 'RRT'}")
        ax.axhline(record["min_distance_mm"], color="k", lw=1.0,
                   label=f"threshold {record['min_distance_mm']:.0f} mm")
        ax.axhline(0.0, color="crimson", lw=1.0, ls=":", label="contact (0 mm)")
        ax.set_xlabel("normalized path parameter")
        ax.set_ylabel("min pairwise signed distance [mm]")
        ax.set_title("collision clearance along the path\n"
                     "(dashed = straight-line, solid = RRT)", fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6, ncol=2, loc="lower right")

        # --- (c) 작업공간: 끝 링크가 테이블을 뚫는가 ------------------------
        ax = axes[row][2]
        tips = np.vstack([np.asarray(series[(p["label"], name)]["tip_xyz"])
                          for p in triad for name in ("straight", "rrt")
                          if (p["label"], name) in series])
        axis = int(np.argmax(tips[:, :2].std(axis=0)))    # x 와 y 중 더 움직이는 쪽
        axis_name = "xy"[axis]
        for index, pair in enumerate(triad):
            color = f"C{index}"
            for name, style in (("straight", "--"), ("rrt", "-")):
                data = series.get((pair["label"], name))
                if data is None:
                    continue
                points = np.asarray(data["tip_xyz"])
                ax.plot(points[:, axis], points[:, 2], style, color=color, lw=1.5)
                if name == "straight":
                    bad = np.asarray(data["clearance_mm"]) < record["min_distance_mm"]
                    if bad.any():
                        ax.plot(points[bad, axis], points[bad, 2], "x",
                                color="crimson", ms=5, mew=1.4)
        starts = np.array([series[(p["label"], "straight")]["tip_xyz"][0]
                           for p in triad if (p["label"], "straight") in series])
        ax.plot(starts[:, axis], starts[:, 2], "ko", ms=8, zorder=5)
        span = ax.get_xlim()
        ax.axhspan(table_top - 0.05, table_top, xmin=0, xmax=1,
                   color="0.75", zorder=0)
        ax.axhline(table_top, color="k", lw=1.2)
        ax.text(span[0], table_top, " table top", va="bottom", fontsize=8)
        ax.set_xlabel(f"world {axis_name} [m]")
        ax.set_ylabel("world z [m]")
        ax.set_title(f"end link ({record['part_names'][-1]}) trace\n"
                     "x marks straight-line collision", fontsize=10)
        ax.grid(alpha=0.3)

    fig.suptitle("Contribution 2: straight-line joint interpolation vs "
                 "RRT-Connect between the three measurement poses\n"
                 "(one row per object; the configuration shown is the most "
                 "informative one at which straight-line interpolation "
                 "collides)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    print(f"\n그림 -> {path}")


def worst_triad_leg(config):
    """대표 자세 안에서 '가장 나쁜' triad 구간 하나를 고른다.

    페이지 예산 때문에 물체당 legs 3개를 다 그릴 수 없다. 그래서 직선 보간이
    가장 깊이 관통하는 구간(=충돌이 가장 뚜렷한 구간)을 하나만 골라, 그
    구간의 직선 대 RRT 곡선만 보인다. 관통이 하나도 없으면(그 격자에서는
    이 자세가 우연히 안전한 경우) 최소여유가 가장 작은 구간을 대신 쓴다.
    """
    triad = [p for p in config["pairs"] if p["kind"] == "triad"]
    if not triad:
        return None
    colliding = [p for p in triad if p["straight"]["in_collision"]]
    pool = colliding or triad
    return max(pool, key=lambda p: p["straight"]["penetration_mm"]) if colliding \
        else min(pool, key=lambda p: p["straight"]["min_clearance_mm"])


def make_figure_compact(records, series_store, summary, path=FIG_PATH,
                        min_distance_mm=None):
    """한 줄 3칸 요약 그림 (지면 예산용). 물체당 대표 구간 하나만 보인다.

    메시지는 하나다: 관절공간 직선 보간은 (RRT-Connect 는 피하는) 장애물을
    관통한다. 그래서 물체당 곡선은 딱 두 개 — 직선(점선, 0 mm 아래로
    내려가면 충돌) 과 RRT(실선, 항상 문턱 위) — 로 줄인다. 범례는 그림
    전체에 하나만 둔다.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [(r, highlight_config(r)) for r in records]
    rows = [(r, c) for r, c in rows if c is not None]
    if not rows:
        print("그릴 자세가 없다 — 그림을 건너뛴다")
        return

    fig, axes = plt.subplots(1, len(rows), figsize=(10, 3.0), dpi=200,
                             squeeze=False)
    axes = axes[0]
    threshold_mm = min_distance_mm if min_distance_mm is not None \
        else rows[0][0]["min_distance_mm"]
    handles_labels = None

    for ax, (record, config) in zip(axes, rows):
        key = (record["object"], tuple(config["theta_deg"]))
        series = series_store.get(key, {})
        leg = worst_triad_leg(config)
        rate = summary.get(record["object"], {}).get("triad", {}) \
                      .get("straight", {}).get("collision_rate", float("nan"))

        straight = series.get((leg["label"], "straight")) if leg else None
        rrt = series.get((leg["label"], "rrt")) if leg else None

        if straight is not None:
            values = np.asarray(straight["clearance_mm"])
            s = np.linspace(0, 1, len(values))
            ax.plot(s, values, ls="--", color="0.15", lw=1.6,
                    label="straight-line")
            bad = values < threshold_mm
            if bad.any():
                ax.plot(s[bad], values[bad], "x", color="crimson", ms=5,
                        mew=1.5, label="collision (straight)")
        if rrt is not None:
            values = np.asarray(rrt["clearance_mm"])
            s = np.linspace(0, 1, len(values))
            ax.plot(s, values, ls="-", marker="o", markevery=max(len(s)//8, 1),
                    ms=3.5, color="0.15", lw=1.6, label="RRT-Connect")

        ax.axhline(threshold_mm, color="0.4", lw=1.0, ls=":",
                   label=f"{threshold_mm:.0f} mm threshold")
        ax.axhline(0.0, color="crimson", lw=1.0, ls="-.", label="contact (0 mm)")

        leg_desc = leg["label"] if leg else "?"
        ax.set_title(f"{record['object']}  (straight-line collision "
                     f"rate {100*rate:.0f}%)\nsegment {leg_desc}", fontsize=9)
        ax.set_xlabel("normalized path parameter", fontsize=8)
        if ax is axes[0]:
            ax.set_ylabel("signed min. distance [mm]", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3, lw=0.5)
        if handles_labels is None:
            handles_labels = ax.get_legend_handles_labels()

    if handles_labels is not None:
        handles, labels = handles_labels
        fig.legend(handles, labels, loc="lower center", ncol=len(labels),
                  frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Straight-line joint interpolation vs. RRT-Connect between "
                "measurement poses (worst triad segment per object)",
                fontsize=9.5, y=0.99)
    fig.subplots_adjust(top=0.76, bottom=0.24, left=0.055, right=0.985,
                        wspace=0.28)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    print(f"\n그림(요약) -> {path}")


# ---------------------------------------------------------------------------
def rescore_for_figure(key, record, checker, planner, scorer, arm_index):
    """대표 자세만 경유점 시계열을 다시 받아 온다 (JSON 을 부풀리지 않으려고)."""
    config = highlight_config(record)
    store = {}
    if config is None:
        return store
    theta = np.deg2rad(config["theta_deg"])
    checker._last_solution = None
    solutions = [checker.solve_robust(theta, g) for g in dc.CANONICAL_TRIAD]
    if any(s is None for s in solutions):
        return store
    planner.set_fixed(np.asarray(solutions[0]))
    for pair in config["pairs"]:
        if pair["kind"] != "triad":
            continue
        q_from = np.asarray(pair["q_from"])
        q_to = np.asarray(pair["q_to"])
        straight = scorer.score([q_from, q_to], keep_series=True)
        store[(pair["label"], "straight")] = straight["series"]
        nodes = planner.plan(q_from, q_to)
        if nodes is not None:
            rrt = scorer.score(nodes, keep_series=True)
            store[(pair["label"], "rrt")] = rrt["series"]
    return store


def replot(objects, gripper="robotiq2f85", seed=0, json_path=JSON_PATH,
          fig_path=FIG_PATH):
    """기존 path_compare.json 만으로 압축 그림을 다시 그린다.

    전수 격자(IK + RRT + 직선)는 이미 JSON 에 다 있으니 다시 돌리지 않는다.
    다만 그 JSON 은 대표 자세의 경유점별 시계열(clearance curve)을 담고
    있지 않다(그림 부풀리기를 피하려고 원래도 저장 안 함) — 그래서 이미
    JSON 이 골라 둔 **바로 그 대표 자세** 하나에 대해서만, RRT 를 다시 풀고
    직선을 다시 채점해 곡선을 얻는다. 이는 논문이 말하는 "비싼 실험"
    (자세 격자 전체를 IK+RRT 로 훑는 것) 이 아니라, 그림 하나 그리는 데 드는
    자세 3개짜리 재계산이다.
    """
    data = json.loads(Path(json_path).read_text())
    records = [r for r in data["objects"] if r["object"] in objects]
    if not records:
        print(f"[replot] {json_path} 에 요청한 물체가 없다: {objects}")
        return
    series_store = {}
    for record in records:
        key = record["object"]
        config = highlight_config(record)
        if config is None:
            print(f"[replot] {key}: 그릴 자세가 없다 — 건너뜀")
            continue
        spec, limits, rho_gt, checker = build_object(key, gripper)
        arm_index = [joint.position_start() for joint in checker.arm_joints]
        planner = pp.ArmPathPlanner(
            checker.plant, checker.context, checker.arm_joints,
            rs.MIN_DISTANCE_M, checker.plant.GetPositions(checker.context),
            seed=seed)
        scorer = PathScorer(planner, checker, spec)
        series_store[(key, tuple(config["theta_deg"]))] = rescore_for_figure(
            key, record, checker, planner, scorer, arm_index)
        print(f"[replot] {key}: 대표 자세 재계산 완료")
    cache_path = OUT_DIR / "_replot_series_cache.json"
    cache_path.write_text(json.dumps(
        dict(records=records,
             series=[[list(outer_key), [[list(inner_key), inner_val]
                                        for inner_key, inner_val in store.items()]]
                    for outer_key, store in series_store.items()]),
        default=float))
    make_figure_compact(records, series_store, data["summary"], path=fig_path,
                        min_distance_mm=data["min_distance_mm"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--objects", nargs="+",
                        default=["2link", "3link", "desklamp"])
    parser.add_argument("--gripper", default="robotiq2f85")
    parser.add_argument("--steps", type=int, default=None,
                        help=f"관절당 격자 칸 수 (기본 {DEFAULT_STEPS} — "
                             "dual_view 의 --steps 기본값과 같다)")
    parser.add_argument("--plan-iters", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-transit", action="store_true",
                        help="home 왕복 구간을 빼고 측정 자세 사이만 본다")
    parser.add_argument("--replot", action="store_true",
                        help="전수 실험을 다시 돌리지 않고 기존 JSON에서 "
                             "압축 그림(fig_path.png)만 다시 그린다")
    args = parser.parse_args()

    if args.replot:
        replot(args.objects, gripper=args.gripper, seed=args.seed)
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records, series_store = [], {}
    started = time.time()
    for key in args.objects:
        record, checker, planner, scorer, spec, arm_index = run_object(
            key, gripper=args.gripper, n_per_joint=args.steps,
            plan_iters=args.plan_iters, seed=args.seed,
            with_transit=not args.no_transit)
        records.append(record)
        config = highlight_config(record)
        if config is not None:
            series_store[(key, tuple(config["theta_deg"]))] = \
                rescore_for_figure(key, record, checker, planner, scorer,
                                   arm_index)

    print_tables(records)

    payload = dict(
        experiment="v2_path: straight-line joint interpolation vs RRT-Connect",
        min_distance_mm=1000.0 * rs.MIN_DISTANCE_M,
        score_step_rad=SCORE_STEP_RAD, min_samples=MIN_SAMPLES,
        clearance_query_mm=1000.0 * CLEARANCE_QUERY_M,
        gravity_triad=dc.CANONICAL_TRIAD.tolist(),
        gravity_labels=list(GRAVITY_LABELS),
        seed=args.seed, plan_iters=args.plan_iters,
        wall_time_s=time.time() - started,
        objects=records,
        summary={r["object"]: dict(ik=ik_stats(r),
                                   triad=aggregate(r, "triad"),
                                   transit=aggregate(r, "transit"),
                                   all=aggregate(r))
                 for r in records},
    )
    JSON_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\n자료 -> {JSON_PATH}")

    make_figure(records, series_store)
    print(f"총 소요 {time.time() - started:.0f} s")


if __name__ == "__main__":
    main()
