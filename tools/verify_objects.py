#!/usr/bin/env python
"""세 물체가 실제로 돌아가는지 시뮬레이션에서 확인한다.

무엇을 보나
-----------
실물 실험이 시작조차 못 했던 이유는 **자세를 하나도 못 찾는 것** 이었다.
그래서 물체마다 아래를 순서대로 본다. 앞이 막히면 뒤는 볼 필요가 없다.

  1) 스펙      부위 부피·볼록 조각 수·파지 단면
  2) 파지      그리퍼 개구 안에 들어오나
  3) 도달·충돌  각도 격자마다 중력 3방향 IK 가 풀리나 (여유 10 mm)
  4) 경로      start -> g1 -> g2 -> g3 -> start 사슬이 이어지나
  5) 최소 간격  실제로 몇 mm 남나 — 이 값이 실물의 여유를 정한다

실행
    $R python tools/verify_objects.py                 # 세 물체 전부
    $R python tools/verify_objects.py --object 2link  # 하나만
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "my_work"))

import density_id_objects as obj                                  # noqa: E402
import robot_scene as rs                                          # noqa: E402
import path_planning as pp                                        # noqa: E402
import density_id_drake as alg                                    # noqa: E402


def spec_of(key):
    """물체 이름 -> ObjectSpec.

    2link / 3link 는 density_id_objects 에 상수로 있다. 접근자 이름이
    저장소 판마다 다르므로(get_spec 이 있는 판과 OBJECTS 만 있는 판) 둘 다 본다.
    """
    if key == "desklamp":
        import desk_lamp
        return desk_lamp.build_spec()
    if hasattr(obj, "get_spec"):
        return obj.get_spec(key)
    return obj.OBJECTS[key]


def clearance_at(checker, arm_q, theta):
    q = np.asarray(arm_q).copy()
    for joint, value in zip(checker.object_joints, np.atleast_1d(theta)):
        q[joint.position_start()] = value
    checker.plant.SetPositions(checker.context, q)
    query = checker.plant.get_geometry_query_input_port().Eval(checker.context)
    pairs = query.ComputeSignedDistancePairwiseClosestPoints(0.05)
    if not pairs:
        return float("inf"), ""
    inspector = query.inspector()
    worst = min(pairs, key=lambda p: p.distance)
    name = (f"{inspector.GetName(worst.id_A).split('::')[-1]} <-> "
            f"{inspector.GetName(worst.id_B).split('::')[-1]}")
    return float(worst.distance), name


def check(key, steps=3, min_distance_m=rs.MIN_DISTANCE_M, plan=True):
    out = dict(object=key)
    started = time.time()
    spec = spec_of(key)
    out["parts"] = [dict(name=p.name, volume_cm3=round(p.volume_cm3, 1),
                         pieces=len(p.collision_meshes)) for p in spec.parts]
    out["chain"] = " -> ".join(p.name for p in spec.parts)
    import grippers as gr
    gripper = gr.GRIPPERS["robotiq2f85"]
    jaw_mm = 1000.0 * rs.jaw_dimension_m(spec)
    out["jaw_mm"] = round(jaw_mm, 1)
    out["jaw_fits"] = bool(jaw_mm <= 1000.0 * gripper.max_opening_m)
    out["long_axis"] = rs.GRASP_LONG_AXIS_BY_OBJECT.get(key, rs.GRASP_LONG_AXIS)

    limits = [j.limits_rad for j in spec.joints]
    rho = np.array([p.rho_gt for p in spec.parts])
    checker = rs.PoseChecker(spec, densities=rho, joint_limits_rad=limits,
                             min_distance_m=min_distance_m)
    indices = [j.position_start() for j in checker.arm_joints]

    axes = [np.linspace(lo, hi, steps) for lo, hi in limits]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(
        -1, len(spec.joints))
    reach, worst_gap, rows = [], float("inf"), []
    for theta in grid:
        solutions = checker.solutions_for(theta)
        good = sum(v is not None for v in solutions.values())
        gaps = [clearance_at(checker, q, theta)
                for q in solutions.values() if q is not None]
        gap = min((g for g, _ in gaps), default=float("nan"))
        pair = min(gaps, key=lambda t: t[0])[1] if gaps else ""
        rows.append(dict(theta_deg=[round(float(np.degrees(v)), 1) for v in theta],
                         reachable=good, clearance_mm=round(1000 * gap, 2)
                         if np.isfinite(gap) else None, pair=pair))
        if good == 3:
            reach.append((theta, solutions))
            worst_gap = min(worst_gap, gap)
    out["grid"] = len(grid)
    out["reachable"] = len(reach)
    out["worst_clearance_mm"] = (round(1000 * worst_gap, 2)
                                 if np.isfinite(worst_gap) else None)
    out["rows"] = rows

    if plan and reach:
        theta, solutions = reach[len(reach) // 2]
        start_q, _ = rs.find_starting_pose(checker, [t for t, _ in reach])
        if start_q is None:
            out["path"] = "시작 자세 없음"
        else:
            q = np.array(start_q).copy()
            for joint, value in zip(checker.object_joints, np.atleast_1d(theta)):
                q[joint.position_start()] = value
            planner = pp.ArmPathPlanner(checker.plant, checker.context,
                                        checker.arm_joints, min_distance_m, q)
            planner.set_fixed(q)
            home = np.array(start_q)[indices]
            legs = [np.array(a)[indices] for a in solutions.values()]
            chain = [home] + legs + [home]
            ok = all(planner.plan(chain[i], chain[i + 1]) is not None
                     for i in range(len(chain) - 1))
            out["path"] = ("사슬 연결됨 (start -> g1 -> g2 -> g3 -> start)"
                           if ok else "사슬 끊김")
            out["path_theta_deg"] = [round(float(np.degrees(v)), 1)
                                     for v in np.atleast_1d(theta)]
    elif plan:
        out["path"] = "도달 가능한 각도가 없어 계획 안 함"
    out["seconds"] = round(time.time() - started, 1)
    return out


def show(result):
    print(f"\n{'=' * 74}\n{result['object']}   ({result['seconds']:.0f}초)\n{'=' * 74}")
    print(f"  사슬       {result['chain']}")
    for part in result["parts"]:
        print(f"    {part['name']:12s} {part['volume_cm3']:8.1f} cm^3"
              f"   볼록조각 {part['pieces']:2d}")
    mark = "OK" if result["jaw_fits"] else "**개구 초과**"
    print(f"  파지 단면   {result['jaw_mm']:.1f} mm  (Robotiq 78 mm)  {mark}")
    print(f"  장축 규약   {result['long_axis']}")
    print(f"  도달·충돌   {result['reachable']}/{result['grid']} 각도가"
          f" 중력 3방향 모두 통과")
    if result["worst_clearance_mm"] is not None:
        print(f"  최소 간격   {result['worst_clearance_mm']:.2f} mm")
    print(f"  경로       {result.get('path', '-')}")
    bad = [r for r in result["rows"] if r["reachable"] < 3]
    for row in bad[:3]:
        print(f"    막힘 theta={row['theta_deg']}  도달 {row['reachable']}/3"
              f"  간격 {row['clearance_mm']} mm  {row['pair']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--object", action="append", default=None)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--min-distance-mm", type=float, default=10.0)
    ap.add_argument("--no-plan", action="store_true")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    keys = args.object or ["2link", "3link", "desklamp"]
    results = []
    for key in keys:
        try:
            result = check(key, args.steps, args.min_distance_mm / 1000.0,
                           not args.no_plan)
        except Exception as exc:                                  # noqa: BLE001
            result = dict(object=key, error=f"{type(exc).__name__}: {exc}")
            print(f"\n{key}: **실패** {result['error']}")
            results.append(result)
            continue
        show(result)
        results.append(result)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
        print(f"\n{args.json} 에 저장했습니다.")
    return 0 if all("error" not in r for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
