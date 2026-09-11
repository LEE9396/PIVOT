"""Fig. 4 — 부위별 VOLUME 재구성 오차가 추정에 어떻게 전파되는가.

논문 주장 (이 스크립트가 실증하는 것)
--------------------------------------
density_id_drake.regressor 는 rho_i 의 계수로 부위 부피 V_i 를 그대로 쓴다
(force_rows = G_ACC * outer(g_hat, VOLUMES), torque_rows 도 VOLUMES 를 곱한다).
그래서 이 회귀는 사실 rho_i 가 아니라 **질량 m_i = rho_i V_i** 를 재매개화해
푸는 것과 같다 — 열 하나를 상수 k_i 로 스케일하면(V_i -> k_i V_i) 최소자승
해는 정확히 1/k_i 배가 되어(정규방정식으로 바로 유도된다: A'=AD 이면
rho_hat' = D^-1 rho_hat), 곱 rho_hat_i * V_i 는 스케일과 무관하게 그대로
남는다. 즉:

  - 질량 m_hat_i = rho_hat_i * V_i(추정기가 믿는 부피) 는 V_i 를 얼마나
    틀리게 잡든 (사전분포 항의 미세한 영향을 빼면) 불변이다 — 부피오차가
    "평균화" 되는 게 아니라 애초에 재매개화라서 상쇄된다.
  - 밀도 rho_hat_i 는 그 몫이므로 V_i 오차를 1:1 로 그대로 받는다
    (V_i 가 (1+delta) 배면 rho_hat_i 는 대략 1/(1+delta) 배).
  - 반폭·라운드 수(정지 판단)는 회귀행렬의 조건수로 정해지는데, 열 하나를
    양수 상수로 스케일해도 조건수는 안 변한다(대각 재스케일은 랭크·상대
    조건 구조를 보존한다) — 그래서 이론적 하한(수렴에 필요한 라운드 수)도
    거의 그대로다.
  - 관성텐서는 다르다. 메시가 등방으로 s 배 잘못 스케일됐다면(부피는
    s^3 배 틀어진다) 단위관성(도심 기준, 질량당)은 길이^2 차원이라 s^2 배
    틀어진다. 질량을 정확히 맞혔어도(위 불변성) 관성텐서 I = m * J_unit 은
    J_unit 자체가 틀린 메시에서 나오므로 s^2 배 오차가 그대로 남는다 —
    이건 회귀로 고칠 수 있는 대상이 아니라(준정적 렌치는 질량과 위치만
    본다) URDF 로 내보낼 때만 나타나는 순수 기하 오차다. 이 스크립트는
    그 s^2 을 시뮬레이션 없이 delta 로부터 해석적으로 계산해 출력한다.

섭동을 어디에 넣었는가 (study_centroid.py 와 같은 자리, 같은 원리)
-------------------------------------------------------------------
참값 물체(density_id_drake.TRUTH_PLANT, alg.measure)는 절대 건드리지
않는다 — 측정은 항상 참부피*참밀도로 만든 실제 무게중심·총질량에서
나온다. **추정기가 회귀행렬을 만들 때 쓰는 부피(alg.VOLUMES)만** 몬키
패치 대신 직접 덮어써서 섭동한다. bind_object(spec) 은 alg.VOLUMES 를
참부피로 설정하지만, density_id_drake.regressor(theta) 는 호출될 때마다
모듈 전역 VOLUMES 를 다시 읽으므로(density_id_drake.py:185-186, 이름을
지역변수로 캡처하지 않는다), bind_object 직후 alg.VOLUMES 원소만 바꿔치기
하면 "추정기가 재구성에서 얻었다고 믿는 부피"만 바뀌고 회귀행렬의 도심
c_i(theta)·참값 측정은 그대로 남는다. 힌지는 섭동하지 않는다 — 힌지는
비전 재구성이 아니라 저울 실측(hinge_mass_kg, hinge_volume_cm3)이라
"재구성 부피 오차"의 대상이 아니다 (study_centroid.py 의 힌지 제외
이유와 같다).

채점 규약 (반드시 명시하라고 지시받은 부분)
--------------------------------------------
"밀도오차 = 질량오차" 인지 여부는 채점할 때 **어느 부피로 나누느냐/곱하느냐**
에 전적으로 달려 있다. exp_v2_main.py 의 부록 실험은 V 가 고정(참값)일 때
rho_err = |rho_hat-rho_gt|/rho_gt 와 mass_err = |rho_hat*V-rho_gt*V|/(rho_gt*V)
가 V 로 약분되어 항상 같은 수(1.7e-16 수준 차이)임을 이미 확인했다 — 그건
"V 가 안 틀렸을 때"의 항등식일 뿐이다.

여기서는 V 가 실제로 틀린다. 그래서 두 오차를 **서로 다른 부피**로 채점한다:

  mass_hat_i     = rho_hat_i * V_used_i     (V_used_i = 추정기가 실제로 회귀에
                                              쓴 부피, 즉 alg.VOLUMES — 틀렸다면
                                              틀린 그 값)
  mass_err_i%    = 100 * (mass_hat_i - m_true_i) / m_true_i

  density_err_i% = 100 * (rho_hat_i - rho_true_i) / rho_true_i
                   (rho_true_i = m_true_i / V_true_i, 참부피 기준 물리적 밀도)

mass_hat 을 rho_hat_i * V_true_i (참부피) 로 채점했다면 반대로 질량오차
쪽이 delta 를 그대로 받고 밀도오차가 불변으로 보였을 것이다 — "어느 부피로
스코어링하는지"가 결과의 부호를 정하는 선택이라는 뜻이며, 이 스크립트는
추정기 스스로가 내부적으로 쓴 부피(V_used = alg.VOLUMES)로 질량을 매기는
쪽을 "정직한" 채점으로 택했다: 이게 바로 회귀가 실제로 계산해낸 값이고,
불변성 주장("V 를 무엇으로 잡든 rho_hat*V 는 참질량에 수렴한다")이 말하는
바로 그 V 다.

물체
----
2link / 3link / desklamp (density_id_objects.get_spec, volume_source="gt" 기본값
— CAD/실측 참부피). nlink 합성물체가 아니라 논문 Table 3 과 같은 실물 세
물체를 쓴다.

실행:
  cd my_work
  ../robot_learning/scripts/run_drake_env.sh python -u exp_v2_volume.py
"""

import argparse
import json
import os
import time

import numpy as np

import density_id_drake as alg
import density_id_objects as obj
import design_core as dc

OBJECT_KEYS = ("2link", "3link", "desklamp")
PER_LINK_TARGET = 0.005   # tau(p) = 0.5% x p — exp_v2_main.py / study_baselines.py 와 같은 규칙


def target_for(n_part, per_link=PER_LINK_TARGET):
    """목표 상대 반폭은 부위 수에 비례한다: tau(p) = per_link * p."""
    return per_link * n_part


def setup(key):
    """물체 하나를 세우고 참값을 돌려준다. bind_object 가 alg.VOLUMES 를
    참부피로 (재)설정하므로, delta 를 바꾸기 전에는 매번 이 함수로 새로
    세워야 이전 delta 의 섭동이 새지 않는다."""
    spec = obj.get_spec(key)
    obj.set_measurement_averaging()
    rho_gt = obj.bind_object(spec)
    obj.apply_weight_prior(spec, obj.assembled_mass_kg(spec))
    return spec, rho_gt


def direction_seed_for(key, seed_index):
    """object 이름 + seed 로 결정되는 재현 가능한 시드.

    파이썬 str 의 내장 hash() 는 프로세스마다 무작위화돼 있어(PYTHONHASHSEED)
    실행할 때마다 값이 달라진다. 재현성을 위해 문자 코드 합으로 대신한다
    (study_centroid.py 의 `900_000 + 977 * n_part + s` 와 같은 발상).
    """
    return 800_000 + 131 * sum(ord(c) for c in key) + seed_index


def run_object(key, deltas_pct, seeds, target, max_rounds, rel, n_starts):
    """물체 하나에 대해 (delta, seed) 격자를 돈다.

    delta 마다 spec/참값을 새로 뽑지 않는다 — 같은 시드의 물체 하나에 대해
    delta 만 늘려가야 "재구성이 더 틀렸다면" 이라는 비교가 성립한다
    (study_centroid.run_cell 과 같은 이유). 부위별 섭동 방향(부호)도
    (object, seed) 당 하나로 고정하고 delta 크기만 키운다.
    """
    cells = {d: dict(seeds=[]) for d in deltas_pct}
    n_part_ref = None

    for s in range(seeds):
        spec, rho_gt = setup(key)
        table = obj.body_table(spec)
        n_part = len(spec.parts)
        n_part_ref = n_part

        # bind_object 직후의 alg.VOLUMES 는 참부피 그대로다 (density_id_objects
        # .bind_object: volumes = body_table(spec) 의 volume_m3). 이걸 기준선으로
        # 저장해 둔다 — delta 를 걸 때마다 이 배열에서부터 다시 섭동한다.
        true_volumes_all = alg.VOLUMES.copy()          # (P,) 부위+힌지
        true_volumes_parts = true_volumes_all[:n_part]
        rho_true_parts = rho_gt[:n_part]
        m_true_parts = rho_true_parts * true_volumes_parts

        rng = np.random.default_rng(direction_seed_for(key, s))
        direction = rng.choice([-1.0, 1.0], size=n_part)   # 부위마다 고정 부호

        for d in deltas_pct:
            delta_frac = d / 100.0
            v_used_all = true_volumes_all.copy()
            v_used_all[:n_part] = true_volumes_parts * (1.0 + direction * delta_frac)
            alg.VOLUMES = v_used_all      # <- 추정기가 "믿는" 부피만 갈아끼운다

            out = dc.closed_loop(spec, target=target, max_rounds=max_rounds,
                                 seed=100 * s, rel_error=rel, n_starts=n_starts)

            rho_hat_parts = np.asarray(out["rho_hat"])[:n_part]
            v_used_parts = v_used_all[:n_part]

            mass_hat_parts = rho_hat_parts * v_used_parts          # 추정기가 실제로 쓴 V 로 채점
            mass_err_pct = 100.0 * (mass_hat_parts - m_true_parts) / m_true_parts
            density_err_pct = 100.0 * (rho_hat_parts - rho_true_parts) / rho_true_parts

            cells[d]["seeds"].append(dict(
                seed=100 * s,
                direction=direction.tolist(),
                mass_err_pct=mass_err_pct.tolist(),
                density_err_pct=density_err_pct.tolist(),
                rounds=int(out["rounds"]),
                converged=bool(out["converged"]),
            ))

        # 다음 seed 의 setup() 이 alg.VOLUMES 를 다시 참값으로 세우긴 하지만,
        # 이 함수 밖(다른 물체·다른 호출)이 실수로 섭동된 값을 보지 않도록
        # 매 seed 끝에 원상복구해 둔다.
        alg.VOLUMES = true_volumes_all.copy()

    return cells, n_part_ref


def summarize_cell(cell):
    """(seed 개수, n_part) 배열들을 |오차| 부위평균 -> seed 평균/표준편차로 요약.

    부위마다 섭동 부호가 무작위라(direction), 부호 있는 오차를 그대로
    seed 에 대해 평균 내면 상쇄돼 항상 0에 가까워진다 — 밀도오차조차 "평평"
    해 보이는 착시가 생긴다. 그래서 절댓값을 취한 뒤 부위에 대해 평균낸
    스칼라 하나를 seed 당 하나씩 만들고, 그 스칼라들의 평균/표준편차를
    그림의 점/오차막대로 쓴다.
    """
    mass_per_seed = np.array([np.mean(np.abs(r["mass_err_pct"])) for r in cell["seeds"]])
    dens_per_seed = np.array([np.mean(np.abs(r["density_err_pct"])) for r in cell["seeds"]])
    rounds = np.array([r["rounds"] for r in cell["seeds"]])
    converged = np.array([r["converged"] for r in cell["seeds"]])
    ok_rounds = rounds[converged] if converged.any() else rounds
    return dict(
        mass_err_mean=float(mass_per_seed.mean()), mass_err_std=float(mass_per_seed.std()),
        density_err_mean=float(dens_per_seed.mean()), density_err_std=float(dens_per_seed.std()),
        rounds_median=float(np.median(ok_rounds)),
        n_converged=int(converged.sum()), n_seeds=len(cell["seeds"]),
    )


def inertia_factor_table(deltas_pct):
    """등방 재구성 스케일 오차 s 에서 관성텐서 오차 배율 s^2 을 해석적으로 낸다.

    가정: 부피오차 delta 가 부위 모양이 모든 축으로 똑같이 s 배 잘못
    스케일된 데서 온다고 보면 V_wrong/V_true = s^3 = 1+delta 이므로
    s = (1+delta)^(1/3). 단위관성(도심 기준, 질량당)은 길이^2 차원이라
    질량을 정확히 맞혀도(본문 불변성) J_unit 자체가 s^2 배 틀린 메시에서
    나온다 — 그래서 관성텐서 오차 배율은 s^2 = (1+delta)^(2/3) 이다.
    시뮬레이션이 아니라 순수 대수식이므로 seed 도 closed_loop 도 안 쓴다.
    """
    rows = []
    for d in deltas_pct:
        ratio = 1.0 + d / 100.0
        s = ratio ** (1.0 / 3.0)
        s2 = s ** 2
        rows.append(dict(delta_pct=d, volume_ratio=ratio, s=s, s2=s2,
                         inertia_err_pct=100.0 * (s2 - 1.0)))
    return rows


def print_object_table(key, n_part, deltas_pct, summaries):
    print(f"\n{'='*92}")
    print(f"[{key}]  부위 {n_part}개  (부위 부호 무작위, delta 는 |V_wrong/V_true - 1| 크기)")
    print('='*92)
    print(f"  {'delta%':>7}{'|mass err|% mean+-std':>26}{'|density err|% mean+-std':>28}"
          f"{'rounds(med)':>13}{'converged':>11}")
    print("  " + "-"*88)
    for d in deltas_pct:
        s = summaries[d]
        print(f"  {d:>6.1f}{s['mass_err_mean']:>14.3f} +-{s['mass_err_std']:<8.3f}"
              f"{s['density_err_mean']:>16.3f} +-{s['density_err_std']:<10.3f}"
              f"{s['rounds_median']:>13.0f}{s['n_converged']:>6d}/{s['n_seeds']:<4d}")


def print_inertia_table(deltas_pct):
    rows = inertia_factor_table(deltas_pct)
    print(f"\n{'='*70}")
    print("해석적 관성텐서 오차 배율 (등방 재구성 스케일 s 가정, 시뮬레이션 아님)")
    print('='*70)
    print(f"  {'delta%':>7}{'s (선형 배율)':>16}{'s^2 (관성 배율)':>18}{'관성 오차%':>14}")
    for r in rows:
        print(f"  {r['delta_pct']:>6.1f}{r['s']:>16.4f}{r['s2']:>18.4f}"
              f"{r['inertia_err_pct']:>14.2f}")
    return rows


def make_figure(all_results, deltas_pct, path):
    """오브젝트당 한 패널: x=부피오차%, mass err(평평) vs density err(기울기 1)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(all_results), figsize=(4.6 * len(all_results), 4.0),
                             squeeze=False)
    axes = axes[0]
    x = np.array(deltas_pct, dtype=float)

    for ax, (key, block) in zip(axes, all_results.items()):
        summ = block["summary"]
        mass_mean = np.array([summ[d]["mass_err_mean"] for d in deltas_pct])
        mass_std = np.array([summ[d]["mass_err_std"] for d in deltas_pct])
        dens_mean = np.array([summ[d]["density_err_mean"] for d in deltas_pct])
        dens_std = np.array([summ[d]["density_err_std"] for d in deltas_pct])

        ax.plot(x, x, ls=":", color="0.6", lw=1.4, label="ideal slope 1 (density)")
        ax.errorbar(x, mass_mean, yerr=mass_std, marker="o", capsize=3,
                   color="tab:blue", label="mass |err| % (expect flat)")
        ax.errorbar(x, dens_mean, yerr=dens_std, marker="s", capsize=3,
                   color="tab:red", label="density |err| % (expect ~slope 1)")
        ax.set_title(f"{key}  (n_part={block['n_part']})", fontsize=10)
        ax.set_xlabel("per-part volume error magnitude [%]")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("relative error [%]  (mean +- std across seeds)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("Fig. 4 — volume error propagation: mass is invariant, "
                 "density carries it 1:1 (per-part |err|, honest V_used scoring)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.10, 1, 0.93))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", nargs="+", default=list(OBJECT_KEYS))
    ap.add_argument("--delta", type=float, nargs="+",
                    default=[0.0, 2.0, 5.0, 10.0, 20.0, 30.0],
                    help="부위별 부피 섭동 크기 [%%]")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--rel", type=float, default=0.05, help="관절각 상대 오차")
    ap.add_argument("--max-rounds", type=int, default=14)
    ap.add_argument("--starts", type=int, default=6)
    ap.add_argument("--json", default="figures/v2/volume_propagation.json")
    ap.add_argument("--png", default="figures/v2/fig_volume.png")
    args = ap.parse_args()

    deltas = args.delta
    print("Fig. 4 실험 — 부위별 부피(volume) 재구성 오차의 전파")
    print(f"  물체={args.objects}  delta%={deltas}  seeds={args.seeds}  "
          f"rel={100*args.rel:.0f}%  max_rounds={args.max_rounds}  starts={args.starts}")
    print("  섭동 지점: bind_object 직후 alg.VOLUMES (추정기 회귀행렬 부피만). "
          "참값 측정(alg.TRUTH_PLANT/alg.measure)은 절대 안 건드린다.")
    print("  채점: mass_hat = rho_hat * V_used(추정기가 실제로 쓴 부피) vs m_true"
          " / density_err = rho_hat vs m_true/V_true\n")

    t0 = time.time()
    all_results = {}
    for key in args.objects:
        target = target_for(len(obj.get_spec(key).parts))
        print(f"[{key}] 실행 중 (목표 반폭 tau={100*target:.2f}%) ...", flush=True)
        t1 = time.time()
        cells, n_part = run_object(key, deltas, args.seeds, target,
                                   args.max_rounds, args.rel, args.starts)
        summary = {d: summarize_cell(cells[d]) for d in deltas}
        print(f"    {time.time()-t1:.1f}s")
        all_results[key] = dict(n_part=n_part, target=target,
                                raw=cells, summary=summary)
        print_object_table(key, n_part, deltas, summary)

    inertia_rows = print_inertia_table(deltas)

    make_figure(all_results, deltas, args.png)

    payload = dict(objects=args.objects, delta_pct=deltas, seeds=args.seeds,
                   rel_error=args.rel, max_rounds=args.max_rounds, n_starts=args.starts,
                   per_link_target=PER_LINK_TARGET,
                   scoring_convention=(
                       "mass_hat_i = rho_hat_i * V_used_i (V_used_i = alg.VOLUMES, "
                       "추정기가 회귀에 실제로 쓴 -- 어쩌면 틀린 -- 부피); "
                       "mass_err_pct = 100*(mass_hat_i-m_true_i)/m_true_i. "
                       "density_err_pct = 100*(rho_hat_i-rho_true_i)/rho_true_i, "
                       "rho_true_i = m_true_i/V_true_i (참부피 기준)."),
                   results={k: dict(n_part=v["n_part"], target=v["target"],
                                    summary={str(d): v["summary"][d] for d in deltas},
                                    raw={str(d): v["raw"][d] for d in deltas})
                           for k, v in all_results.items()},
                   inertia_factor_table=inertia_rows,
                   total_seconds=time.time() - t0)
    os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
    with open(args.json, "w") as fh:
        json.dump(payload, fh, indent=1)

    print(f"\n총 {time.time()-t0:.0f}초")
    print(f"수치 -> {args.json}")
    print(f"그림 -> {args.png}")


if __name__ == "__main__":
    main()
