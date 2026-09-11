"""논문 IV-C: 도심(centroid) 섭동 ablation.

왜 이 실험인가
--------------
측정 모형은 y = A(theta, g_hat) rho 이고, 회귀행렬은 rho_i 에 오직 곱
V_i rho_i 와 V_i c_i (c_i = 부위 도심) 로만 의존한다 (재매개화). 그래서:

  - 부피 V_i 오차는 재매개화라서 부위 질량 추정에 전파되지 않는다
    (이미 실행으로 확인했다).
  - 도심 c_i 오차는 토크 행을 직접 바꾸므로 전파된다.

즉 재구성 오차가 추정에 끼치는 영향을 실제로 가르는 것은 도심이다. 논문
예측: 도심 1 mm 섭동 -> 부위 질량 오차 약 2.4%, 2 mm -> 약 6.1%. 이 스윕은
그 예측을 확인한다.

섭동을 어디에 넣었는가
-----------------------
참값 물체(density_id_drake.TRUTH_PLANT, alg.measure)는 절대 건드리지 않는다.
**추정기가 회귀행렬을 만들 때 쓰는 도심**만 몬키패치로 섭동시킨다.

density_id_drake.regressor(theta, g_dirs) 는 내부에서
part_centroids_in_S(theta) 를 불러 c_i(theta) 를 얻는다. 그 함수는
KIN_PLANT(단위밀도 쌍둥이 plant, bind_object 가 spec 의
bbox_center_in_link_mm 으로 몸체 프레임 원점을 잡아 만든다)의 몸체 프레임
원점을 순운동학으로 world/sensor 좌표로 옮긴 값을 그대로 c_i 로 쓴다
(density_id_drake.py:169-176, density_id_objects.py 의 build_plant:
"body frame = 각 part 의 외형 도심").

그래서 "재구성이 도심을 delta 만큼 틀리게 줬다"는 상황은, part_centroids_in_S
가 돌려주는 각 부위 위치에 그 부위의 **몸체 프레임에 고정된** 오프셋 벡터를
그 부위의 현재 회전으로 world 프레임에 돌려 더하는 것으로 정확히 흉내낼 수
있다 (재구성 오차는 물체에 붙어서 같이 움직이는 것이 물리적으로 맞다).
이 스크립트는 alg.part_centroids_in_S 를 그런 함수로 교체한다
(run-time monkeypatch, density_id_drake.py 파일 자체는 건드리지 않는다).
힌지는 섭동하지 않는다 — 힌지는 비전 재구성이 아니라 실측(저울) 하드웨어라
"재구성 도심 오차" 의 대상이 아니다.

물체
----
nlink.make_spec 으로 만든 직렬 사슬. study_scaling.py 와 같은 방식·같은
시드(spec = nlink.make_spec(n_part, seed=1000+s))를 써서 같은 물체를
가리킨다. nlink.py / density_id_objects.py 는 읽기만 하고 수정하지 않는다.

실행:
  ../robot_learning/scripts/run_drake_env.sh python -u study_centroid.py \
      --parts 3 4 5 --delta 0 0.5 1 2 5 --seeds 8
"""

import argparse
import json
import time

import numpy as np

import density_id_drake as alg
import density_id_objects as obj
import design_core as dc
import nlink

# 원본(비섭동) 도심 함수. 스크립트 전체에서 delta=0 기준선과, 종료 시
# 복원에 쓴다. bind_object 는 이 함수 자체를 건드리지 않으므로 임포트
# 시점에 한 번만 잡아두면 된다.
_ORIGINAL_CENTROIDS_FN = alg.part_centroids_in_S


def target_for(n_part, per_link=0.005):
    """목표 상대 반폭 tau(p) = per_link * p (study_scaling.py 규칙 복제)."""
    return per_link * n_part


def make_perturbed_centroids(n_part, delta_m, direction_seed):
    """part_centroids_in_S 를 대체할 함수를 만든다.

    부위(0..n_part-1) 마다 무작위 단위벡터 * delta_m [m] 을 몸체 프레임
    오프셋으로 고정하고, 매 호출(각 theta)마다 그 부위의 현재 회전으로
    world 프레임에 돌려 c_i(theta) 에 더한다. 힌지(n_part 이후 인덱스)는
    건드리지 않는다. alg.KIN_PLANT / alg.KIN_BODIES / alg.PARTS 는 매번
    현재 값을 다시 읽으므로(클로저에 캡처하지 않음), bind_object 가 나중에
    다른 spec 으로 다시 바인딩해도 안전하다 — 다만 이 함수 자체는 그
    spec(n_part) 전용으로만 다시 만들어 쓴다.
    """
    rng = np.random.default_rng(direction_seed)
    directions = rng.normal(size=(n_part, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    offsets = directions * delta_m  # (n_part, 3) m, body frame

    def perturbed(theta):
        alg.KIN_PLANT.SetPositions(alg.KIN_CTX, np.asarray(theta))
        pts = []
        for idx, (name, _dims, _rho) in enumerate(alg.PARTS):
            pose = alg.KIN_PLANT.EvalBodyPoseInWorld(alg.KIN_CTX, alg.KIN_BODIES[name])
            p = pose.translation()
            if idx < n_part:
                p = p + pose.rotation().matrix() @ offsets[idx]
            pts.append(p)
        return np.stack(pts)

    return perturbed


def run_cell(n_part, deltas_mm, seeds, target, max_rounds, rel, n_starts):
    """(n_part, delta) 격자를 한 번에 seeds 번 돈다.

    delta 마다 spec 을 새로 뽑지 않는다 — 같은 시드의 물체 하나에 대해
    delta 만 늘려가야 "재구성이 더 틀렸다면" 이라는 비교가 성립한다.
    도심 섭동 방향도 (n_part, seed) 당 하나로 고정하고 delta 로만 크기를
    늘린다 (방향이 델타마다 바뀌면 크기 효과와 방향 효과가 섞인다).
    """
    table = {d: dict(rounds=[], converged=[], errors=[]) for d in deltas_mm}
    for s in range(seeds):
        spec = nlink.make_spec(n_part, seed=1000 + s)
        obj.set_measurement_averaging()
        rho_gt = obj.bind_object(spec)
        obj.apply_weight_prior(spec, obj.assembled_mass_kg(spec))

        direction_seed = 900_000 + 977 * n_part + s
        for d in deltas_mm:
            if d == 0.0:
                alg.part_centroids_in_S = _ORIGINAL_CENTROIDS_FN
            else:
                alg.part_centroids_in_S = make_perturbed_centroids(
                    n_part, d * 1e-3, direction_seed)
            out = dc.closed_loop(spec, target=target, max_rounds=max_rounds,
                                 seed=100 * s, rel_error=rel, n_starts=n_starts)
            part = slice(0, n_part)  # 부위만 채점 (힌지는 아는 값)
            err = float(np.max(np.abs(out["rho_hat"][part] - rho_gt[part])
                               / rho_gt[part]))
            table[d]["rounds"].append(out["rounds"])
            table[d]["converged"].append(bool(out["converged"]))
            table[d]["errors"].append(err)
        alg.part_centroids_in_S = _ORIGINAL_CENTROIDS_FN
    return table


def summarize(cell, seeds):
    errs = cell["errors"]
    ok_rounds = [r for r, c in zip(cell["rounds"], cell["converged"]) if c]
    return dict(
        median_err=float(np.median(errs)),
        worst_err=float(np.max(errs)),
        median_rounds=float(np.median(ok_rounds)) if ok_rounds else None,
        n_converged=int(sum(cell["converged"])),
        n_seeds=seeds,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", type=int, nargs="+", default=[3, 4, 5])
    ap.add_argument("--delta", type=float, nargs="+",
                    default=[0.0, 0.5, 1.0, 2.0, 5.0],
                    help="도심 섭동 크기 [mm]")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--rel", type=float, default=0.05, help="관절각 상대 오차")
    ap.add_argument("--per-link-target", type=float, default=0.005)
    ap.add_argument("--max-rounds", type=int, default=30)
    ap.add_argument("--starts", type=int, default=6)
    ap.add_argument("--json", default="figures/centroid_v1.json")
    args = ap.parse_args()

    deltas = args.delta
    print(f"도심 섭동 ablation: parts={args.parts} delta(mm)={deltas} "
          f"seeds={args.seeds} rel={100*args.rel:.0f}% max_rounds={args.max_rounds}")
    print("섭동 지점: alg.part_centroids_in_S 몬키패치 (추정기 회귀행렬 도심만, "
          "참값 물체는 불변)\n")

    table = {}
    t_start = time.time()
    for n_part in args.parts:
        target = target_for(n_part, args.per_link_target)
        t0 = time.time()
        cell_by_delta = run_cell(n_part, deltas, args.seeds, target,
                                 args.max_rounds, args.rel, args.starts)
        dt = time.time() - t0
        print(f"P={n_part} (target={100*target:.2f}%)  {dt:.1f}s")
        for d in deltas:
            cell = cell_by_delta[d]
            summ = summarize(cell, args.seeds)
            table[f"{n_part}|{d}"] = dict(raw=cell, summary=summ)
            conv_s = f"{summ['n_converged']}/{summ['n_seeds']}"
            mr = f"{summ['median_rounds']:.0f}" if summ["median_rounds"] is not None else "—"
            print(f"    delta={d:>4.1f}mm  median_err={100*summ['median_err']:6.2f}%  "
                  f"worst_err={100*summ['worst_err']:6.2f}%  "
                  f"median_rounds={mr:>3}  converged={conv_s}")

    total = time.time() - t_start
    print(f"\n총 {total:.0f}초")

    payload = dict(parts=args.parts, delta_mm=deltas, seeds=args.seeds,
                   rel=args.rel, per_link_target=args.per_link_target,
                   max_rounds=args.max_rounds, starts=args.starts,
                   target={str(p): target_for(p, args.per_link_target)
                           for p in args.parts},
                   cells=table, total_seconds=total)
    with open(args.json, "w") as fh:
        json.dump(payload, fh, indent=1)
    print(f"수치 -> {args.json}")

    alg.part_centroids_in_S = _ORIGINAL_CENTROIDS_FN


if __name__ == "__main__":
    main()
