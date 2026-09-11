"""To-Do [8] — N-link 시뮬레이션 밀도 추정 (논문 "Density identification
experiment — Simulation" 절의 표 하나를 만든다).

무엇을 재는가
--------------
링크 수 N = 2..6 짜리 직렬 사슬에서 part-wise density identification 폐루프가
끝까지 도는지, 그리고 **부피 오차와 각도 오차 중 어느 쪽에 더 민감한지**를
같은 하이퍼파라미터 아래에서 잰다. 그래프·ablation·파지점 스윕은 만들지
않는다 (회의에서 제외).

조건 네 가지 (지시문 3절)

    C0  vol 기준(1 %)  ang 기준(1 %)   기본 성능
    C1  vol 상향(5 %)  ang 기준(1 %)   volume 민감도
    C2  vol 기준(1 %)  ang 상향(5 %)   angle 민감도
    C3  vol 상향(5 %)  ang 상향(5 %)   상호작용 (표 B 에는 미게재)

오차를 **어디에** 넣는가 — 이게 결과의 의미를 정한다
-----------------------------------------------------
둘 다 "참값 물체"가 아니라 "추정기가 믿는 모형"을 틀리게 만든다. 측정
(alg.measure) 은 언제나 참부피·참밀도로 세운 TRUTH_PLANT 에서 나온다.

* volume : bind_object 직후의 alg.VOLUMES (회귀행렬이 rho 의 계수로 쓰는
  부피) 중 **부위 몫만** 덮어쓴다. exp_v2_volume.py 와 같은 자리·같은 규약.
  힌지는 비전 재구성이 아니라 저울 실측이라 재구성 오차의 대상이 아니다.
  주입 방식은 파이프라인 구현(exp_v2_volume.py)을 그대로 따른다 —
  **링크마다 부호를 무작위로 하나 뽑아 고정하고, 크기는 정확히 x %**
  (상대오차). 같은 (N, seed) 면 C0~C3 가 같은 부호를 쓰므로 조건 사이
  비교는 "같은 물체에서 오차만 키운 것"이 된다.

* angle : design_core.closed_loop 의 rel_error. angle_aware.angle_covariance
  가 sigma_j = max(rel * |theta_j|, 0.5 deg) 인 **가우시안 상대오차**를
  만들고, closed_loop 이 그 sigma 로 두 번 흔든다 (지령각 -> 실제각,
  실제각 -> 측정각). 즉 추정기가 보는 각도는 참각도에서 sigma 만큼 틀어져
  있다. 절대 하한 0.5 deg 는 파이프라인 기본값이며 조건과 무관하게 유지한다.

부피 오차는 회귀행렬의 **열 스케일**이라 라운드를 늘려도 안 없어지는
치우침이고(exp_v2_volume.py 의 재매개화 논증: rho_hat_i ~ 1/(1+delta)),
각도 오차는 **잡음**이라 라운드로 줄어든다. 그래서 아래 표에서 C1 과 C2 가
비대칭으로 나오는 것은 예상된 결과다 — 이 실험은 그 비대칭의 크기를 잰다.

실물 실험과 맞춘 것 / 다른 것은 results/nlink_summary.md 에 적는다.

실행:
  cd my_work
  ../robot_learning/scripts/run_drake_env.sh python -u exp_nlink_density.py
"""

import argparse
import csv
import json
import os
import time

import numpy as np

import density_id_drake as alg
import density_id_objects as obj
import design_core as dc
import nlink

# 실물 실험(exp_v2_main.py)과 같은 값. 조건별로 바꾸지 않는다 (지시문 8절).
PER_LINK_TARGET = 0.005      # 목표 반폭 tau(p) = 0.5 % x 부위수
MAX_ROUNDS = 14
N_STARTS = 6
ESTIMATOR = "tls"
STOP_RULE = "residual"
MEAS_PER_CONFIG = 3          # 중력 3방향 = configuration 당 측정 3회

MAX_LINKS = 6                # CSV 열 개수 (L1..L6). 파이프라인 한계도 6링크.

# (vol_err_pct, ang_err_rel)
CONDITIONS = {
    "C0": (1.0, 0.01),
    "C1": (5.0, 0.01),
    "C2": (1.0, 0.05),
    "C3": (5.0, 0.05),
}

COLUMNS = (
    ["n_links", "cond_id", "vol_err_pct", "ang_err", "seed"]
    + [f"gt_density_L{i}" for i in range(1, MAX_LINKS + 1)]
    + [f"est_density_L{i}" for i in range(1, MAX_LINKS + 1)]
    + [f"err_pct_L{i}" for i in range(1, MAX_LINKS + 1)]
    + ["mean_err_pct", "n_configs", "n_measurements", "final_uncertainty",
       "converged", "grasp_point", "wall_time_sec"]
)

# 파지점: 파이프라인이 base(link1) 를 센서 프레임에 용접하는 자리 그대로다.
# nlink.make_spec 은 base_bbox_center_in_sensor_mm = (L1/2, 0, 0) 을 쓰므로
# 센서 원점(=파지점)은 link1 의 **몸쪽 끝면 중심**에 온다. 한 곳 고정.
GRASP_POINT = "link1_proximal_end_center (sensor origin; base welded at bbox center +L1/2 x)"


def target_for(n_part, per_link=PER_LINK_TARGET):
    """목표 상대 반폭은 부위 수에 비례한다: tau(p) = per_link * p.
    링크마다 미지수가 둘씩 늘어 달성 가능한 정밀도가 곱셈으로 나빠지므로,
    하나의 절대값으로 묶으면 긴 사슬에서만 도달 불가가 된다."""
    return per_link * n_part


def volume_sign_seed(n_part, trial_seed):
    """부피 오차 부호용 재현 가능한 시드.

    파이썬 hash() 는 프로세스마다 무작위화돼 있어 쓰면 안 된다
    (exp_v2_volume.direction_seed_for 와 같은 이유).
    """
    return 700_000 + 977 * n_part + trial_seed


def run_trial(n_part, cond_id, trial_seed, max_rounds, n_starts, per_link):
    """한 (N, 조건, seed) 조합을 끝까지 돌리고 기록 한 줄을 만든다.

    GT 는 채점에만 쓴다. 탐색·정지 판단은 rho_gt 를 한 번도 안 본다.
    """
    vol_err_pct, ang_rel = CONDITIONS[cond_id]

    # 물체는 seed 마다 새로 뽑는다 (링크 길이 70~150 mm 균일). 물체 하나에
    # 대한 결과가 아니라 '전형적인 사슬' 에 대한 결과가 되게 하려는 것이며,
    # study_scaling.run_cell 과 같은 규약(spec seed = 1000 + s)이다.
    spec = nlink.make_spec(n_part, seed=1000 + trial_seed)
    obj.set_measurement_averaging()
    rho_gt = obj.bind_object(spec)
    # 저울 사전분포는 **참부피**로 세운다 (exp_v2_volume.setup 과 같은 순서).
    obj.apply_weight_prior(spec, obj.assembled_mass_kg(spec))

    true_volumes_all = alg.VOLUMES.copy()          # 부위 + 힌지
    sign_rng = np.random.default_rng(volume_sign_seed(n_part, trial_seed))
    direction = sign_rng.choice([-1.0, 1.0], size=n_part)

    v_used_all = true_volumes_all.copy()
    v_used_all[:n_part] = (true_volumes_all[:n_part]
                           * (1.0 + direction * vol_err_pct / 100.0))
    alg.VOLUMES = v_used_all                       # 추정기가 '믿는' 부피만 교체
    try:
        t0 = time.time()
        out = dc.closed_loop(spec, target=target_for(n_part, per_link),
                             max_rounds=max_rounds, seed=100 * trial_seed,
                             rel_error=ang_rel, n_starts=n_starts,
                             estimator=ESTIMATOR, stop_rule=STOP_RULE)
        wall = time.time() - t0
    finally:
        alg.VOLUMES = true_volumes_all.copy()      # 다음 trial 로 새지 않게

    gt = np.asarray(rho_gt)[:n_part]
    est = np.asarray(out["rho_hat"])[:n_part]
    err = 100.0 * np.abs(est - gt) / gt

    row = {c: "" for c in COLUMNS}
    row.update(
        n_links=n_part, cond_id=cond_id, vol_err_pct=f"{vol_err_pct:g}",
        ang_err=f"{100 * ang_rel:g}%rel(min0.5deg)", seed=trial_seed,
        mean_err_pct=f"{err.mean():.4f}",
        n_configs=out["rounds"],
        n_measurements=out["rounds"] * MEAS_PER_CONFIG,
        final_uncertainty=f"{100 * out['worst']:.4f}",
        converged=bool(out["converged"]),
        grasp_point=GRASP_POINT,
        wall_time_sec=f"{wall:.2f}",
    )
    for i in range(n_part):
        row[f"gt_density_L{i + 1}"] = f"{gt[i]:.1f}"
        row[f"est_density_L{i + 1}"] = f"{est[i]:.1f}"
        row[f"err_pct_L{i + 1}"] = f"{err[i]:.4f}"

    # 링크 길이는 seed 마다 다르므로 재현에 필요하다. CSV 규격 밖이라
    # JSON 부산물에만 남긴다.
    extra = dict(lengths_mm=[p.bbox_mm[0] for p in spec.parts],
                 volume_sign=direction.tolist(),
                 target_half_width_pct=100 * target_for(n_part, per_link),
                 half_width_pct=[100 * h for h in out["half"][:n_part]])
    return row, extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--links", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    ap.add_argument("--conds", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--seeds", type=int, default=5, help="시드 0..seeds-1")
    ap.add_argument("--max-rounds", type=int, default=MAX_ROUNDS)
    ap.add_argument("--starts", type=int, default=N_STARTS)
    ap.add_argument("--per-link-target", type=float, default=PER_LINK_TARGET)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    csv_path = os.path.join(args.outdir, "nlink_density_results.csv")
    json_path = os.path.join(args.outdir, "nlink_density_extra.json")

    total = len(args.links) * len(args.conds) * args.seeds
    print(f"총 {total} 회 = {len(args.links)}(N) x {len(args.conds)}(조건) "
          f"x {args.seeds}(trial)")
    print(f"목표 반폭 tau(p) = {100 * args.per_link_target:g}% x p, "
          f"최대 {args.max_rounds}라운드, 시작점 {args.starts}개, "
          f"추정기 {ESTIMATOR}, 정지 {STOP_RULE}")
    print(f"파지점: {GRASP_POINT}\n")

    rows, extras, t_start, done = [], {}, time.time(), 0
    for n_part in args.links:
        for cond_id in args.conds:
            for s in range(args.seeds):
                row, extra = run_trial(n_part, cond_id, s, args.max_rounds,
                                       args.starts, args.per_link_target)
                rows.append(row)
                extras[f"{n_part}|{cond_id}|{s}"] = extra
                done += 1
                print(f"[{done:3d}/{total}] N={n_part} {cond_id} seed={s}  "
                      f"평균오차 {float(row['mean_err_pct']):7.3f}%  "
                      f"config {row['n_configs']:2d}  "
                      f"반폭 {float(row['final_uncertainty']):6.2f}%  "
                      f"{'수렴' if row['converged'] else '미수렴'}  "
                      f"{float(row['wall_time_sec']):5.1f}s", flush=True)

    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    with open(json_path, "w") as fh:
        json.dump(extras, fh, indent=1)

    print(f"\n총 {time.time() - t_start:.0f}초")
    print(f"원본 -> {csv_path}")
    print(f"부산물(링크 길이·부호·부위별 반폭) -> {json_path}")


if __name__ == "__main__":
    main()
