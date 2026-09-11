"""Table 3 — 실물 세 물체(2link / 3link / desklamp)의 주 결과표.

study_baselines.py 와 무엇이 같고 무엇이 다른가
-----------------------------------------------
같은 것: 대조군 네 가지와 seed 사용 규약을 그대로 가져온다. 같은 seed 로
같은 물체를 돌려야 "추정기 차이" 가 "물체 차이" 에 섞이지 않기 때문이다.

  uniform  균질 가정 — 저울 총질량 / 총부피. GT 를 훔치지 않는다
           (저울값과 부피는 우리도 아는 값이다).
  single   우리 추정기인데 형상을 한 번만 쓴다 (R=1, target=0)
           -> 정리 1의 rank-4 천장을 직접 때린다. 미지수가 4개를 넘는
              3link(부위3+힌지2=5) 에서 반드시 무너져야 한다.
  wls      우리 폐루프인데 각도오차를 미지수로 안 푼다 (EIV 치우침 잔존).
  ours     TLS + D-최적 폐루프 (전체).

다른 것: study_baselines.py 는 합성 n-link 만 보고 **스칼라 하나**(부위 최악
오차)만 남긴다. 논문 Table 3 은 실물 물체를 **부위별로** 적어야 한다. 이유가
둘이다.
  (1) 힌지는 작고 가려져 있는데 밀도가 높은 부품이라, "따로 미지수로 잡아야
      한다" 는 주장을 부위별 숫자 없이 보일 수 없다. 최악값만 적으면 힌지가
      부위 뒤에 숨거나 반대로 부위를 가린다.
  (2) 물체가 셋뿐이라 중앙값만 적으면 퍼짐이 감춰진다. IQR 을 같이 낸다.

밀도오차 = 질량오차 인 이유 (표에 그대로 적어야 하는 성질)
----------------------------------------------------------
부피 V_i 는 추정 대상이 아니라 회귀행렬의 계수로 **고정**돼 들어간다
(density_id_drake.regressor 가 VOLUMES 를 곱한다). 그래서

  |rho_hat_i V_i - rho_gt_i V_i| / (rho_gt_i V_i) = |rho_hat_i - rho_gt_i| / rho_gt_i

로 V_i 가 약분된다. 두 열은 독립된 지표가 아니라 **같은 수**다. 논문은 이걸
두 개의 다른 성능인 양 나란히 싣지 말고 "부피 고정 하에서 동일" 이라고
적어야 한다. 이 스크립트는 두 열을 다 계산해 최대 차이를 출력함으로써 그
불변성을 매번 검증한다.

노트: 논문에 laptop 이 있지만 이 저장소에 자산이 없다. 넣지 않는다
(가짜 숫자를 만들지 않는다).

실행:
  cd my_work
  ../robot_learning/scripts/run_drake_env.sh python exp_v2_main.py
"""
import argparse
import json
import os
import time

import numpy as np

import density_id_objects as obj
import design_core as dc

OBJECT_KEYS = ("2link", "3link", "desklamp")
METHODS = ("uniform", "single", "wls", "ours")

PER_LINK_TARGET = 0.005   # tau(p) = 0.5 % x p — study_baselines.py 와 같은 규칙


def target_for(n_part, per_link=PER_LINK_TARGET):
    """목표 상대 반폭은 부위 수에 비례한다: tau(p) = per_link * p."""
    return per_link * n_part


def setup(key):
    """물체 하나를 세우고 GT 를 돌려준다. GT 는 채점에만 쓴다.

    bind_object 가 전역(alg.MU0 등)을 갈아끼우므로 seed 마다 다시 부른다 —
    앞선 물체의 사전분포가 남아 있으면 비교가 깨진다.
    """
    spec = obj.get_spec(key)
    obj.set_measurement_averaging()
    rho_gt = obj.bind_object(spec)
    obj.apply_weight_prior(spec, obj.assembled_mass_kg(spec))
    return spec, rho_gt


def uniform_estimate(spec, n_body):
    """균질 가정 — 저울에 읽힌 총질량을 총부피로 나눈다.

    저울값과 부피는 우리도 아는 것이므로 이 대조군은 GT 를 훔치지 않는다.
    (study_baselines.uniform_estimate 와 같은 코드다.)
    """
    table = obj.body_table(spec)
    vols = np.array([row["volume_m3"] for row in table])
    total_kg = obj.assembled_mass_kg(spec)
    return np.full(n_body, total_kg / float(vols.sum()))


def run_object(key, seeds, rel, max_rounds, n_starts, verbose=True):
    """한 물체에 대해 네 방법을 **같은 seed** 로 돌리고 per-seed 원자료를 남긴다."""
    spec, rho_gt = setup(key)
    table = obj.body_table(spec)
    n_part = len(spec.parts)
    n_body = len(table)
    target = target_for(n_part)
    volumes = np.array([row["volume_m3"] for row in table])

    per_seed = {m: [] for m in METHODS}
    for s in range(seeds):
        # seed 마다 다시 세운다. 사전분포·노이즈 모델을 매번 같은 상태에서
        # 출발시켜야 네 방법이 정말 같은 조건이 된다.
        spec, rho_gt = setup(key)

        rec = dict(seed=100 * s,
                   rho_hat=uniform_estimate(spec, n_body).tolist(),
                   half=None, rounds=0, converged=False, worst=None)
        per_seed["uniform"].append(rec)

        # single: 형상 하나 (R=1). target=0 이라 절대 멈추지 않고 1라운드로 끝난다.
        one = dc.closed_loop(spec, target=0.0, max_rounds=1, seed=100 * s,
                             rel_error=rel, n_starts=n_starts,
                             estimator="tls")
        per_seed["single"].append(dict(
            seed=100 * s, rho_hat=one["rho_hat"].tolist(),
            half=one["half"].tolist(), rounds=one["rounds"],
            converged=False,          # R=1 은 정지조건을 아예 안 본다
            worst=float(one["worst"])))

        for name, est in (("wls", "wls"), ("ours", "tls")):
            out = dc.closed_loop(spec, target=target, max_rounds=max_rounds,
                                 seed=100 * s, rel_error=rel,
                                 n_starts=n_starts, estimator=est)
            per_seed[name].append(dict(
                seed=100 * s, rho_hat=out["rho_hat"].tolist(),
                half=out["half"].tolist(), rounds=out["rounds"],
                converged=bool(out["converged"]),
                worst=float(out["worst"])))
        if verbose:
            print(f"    seed {100*s:<5} done", flush=True)

    return dict(key=key, label=spec.label, n_part=n_part, n_body=n_body,
                target=target,
                bodies=[dict(name=r["name"], kind=r["kind"],
                             volume_m3=float(r["volume_m3"]),
                             rho_gt=float(r["rho_gt"]),
                             mass_kg=float(r["volume_m3"] * r["rho_gt"]))
                        for r in table],
                rho_gt=[float(v) for v in rho_gt],
                volumes=[float(v) for v in volumes],
                total_mass_kg=float(obj.assembled_mass_kg(spec)),
                per_seed=per_seed)


# ---------------------------------------------------------------------------
# 집계
# ---------------------------------------------------------------------------
def errors(block, method):
    """(seeds, n_body) 밀도 상대오차와 질량 상대오차를 둘 다 만든다."""
    gt = np.asarray(block["rho_gt"])
    vol = np.asarray(block["volumes"])
    hat = np.array([r["rho_hat"] for r in block["per_seed"][method]])
    rho_err = np.abs(hat - gt) / gt
    mass_err = np.abs(hat * vol - gt * vol) / (gt * vol)
    return rho_err, mass_err


def med_iqr(x, axis=0):
    """중앙값과 사분위수. 물체가 적어 퍼짐을 숨기면 안 된다."""
    return (np.median(x, axis=axis),
            np.percentile(x, 25, axis=axis),
            np.percentile(x, 75, axis=axis))


def fmt(med, lo, hi, scale=100.0):
    return f"{scale*med:>7.2f} [{scale*lo:>6.2f},{scale*hi:>6.2f}]"


def report(block, seeds):
    """물체 하나의 부위별 표를 찍고, 표에 넣을 요약 dict 를 돌려준다."""
    key = block["key"]
    bodies = block["bodies"]
    n_part = block["n_part"]
    n_hinge = block["n_body"] - n_part
    print(f"\n{'='*104}")
    print(f"{key}  ({block['label']})")
    print(f"  부위 {n_part} + 힌지 {n_hinge} = 미지수 {block['n_body']}   "
          f"저울 총질량 {1000*block['total_mass_kg']:.1f} g   "
          f"목표 반폭 tau = {100*block['target']:.2f}%   seed {seeds}개")
    print('='*104)

    inv_max = 0.0
    summary = {}
    print(f"  {'method':<8}{'body':<14}{'kind':<7}"
          f"{'rho err % med [IQR]':>24}{'mass err % med [IQR]':>24}"
          f"{'half % med':>12}{'cover':>8}")
    print("  " + "-"*102)
    for method in METHODS:
        rho_err, mass_err = errors(block, method)
        inv_max = max(inv_max, float(np.max(np.abs(rho_err - mass_err))))
        recs = block["per_seed"][method]
        has_half = recs[0]["half"] is not None
        half = (np.array([r["half"] for r in recs]) if has_half else None)
        rm, rlo, rhi = med_iqr(rho_err)
        mm, mlo, mhi = med_iqr(mass_err)
        if has_half:
            hm, _, _ = med_iqr(half)
            cover = (rho_err <= half).sum(axis=0)
        for i, body in enumerate(bodies):
            tag = "HINGE" if body["kind"] == "hinge" else "part"
            hcol = f"{100*hm[i]:>11.2f}" if has_half else f"{'n/a':>11}"
            ccol = f"{cover[i]:>3d}/{seeds}" if has_half else f"{'n/a':>7}"
            mark = " <-" if body["kind"] == "hinge" else ""
            print(f"  {method:<8}{body['name']:<14}{tag:<7}"
                  f"{fmt(rm[i], rlo[i], rhi[i]):>24}"
                  f"{fmt(mm[i], mlo[i], mhi[i]):>24}"
                  f"{hcol}{ccol:>8}{mark}")
        # 방법 수준 요약 — 부위만 본다 (힌지는 저울로 아는 값, 탐색의 목적이 아님)
        part_worst = np.max(rho_err[:, :n_part], axis=1)
        pw, plo, phi = med_iqr(part_worst)
        rounds = np.array([r["rounds"] for r in recs])
        conv = int(sum(r["converged"] for r in recs))
        hinge_worst = (np.max(rho_err[:, n_part:], axis=1)
                       if n_hinge else None)
        print(f"  {'':<8}{'-> 부위 최악':<14}{'':<7}{fmt(pw, plo, phi):>24}"
              f"{'':>24}"
              f"{'라운드 ' + str(int(np.median(rounds))):>12}"
              f"{str(conv) + '/' + str(seeds):>8}")
        summary[method] = dict(
            part_worst_med=float(pw), part_worst_iqr=[float(plo), float(phi)],
            rounds_med=float(np.median(rounds)),
            rounds_iqr=[float(np.percentile(rounds, 25)),
                        float(np.percentile(rounds, 75))],
            converged=conv, n=seeds,
            hinge_worst_med=(float(np.median(hinge_worst))
                             if hinge_worst is not None else None),
            coverage_part=(int((rho_err[:, :n_part]
                                <= half[:, :n_part]).sum())
                           if has_half else None),
            coverage_part_n=(n_part * seeds if has_half else None))
        print("  " + "-"*102)
    print(f"  밀도오차 vs 질량오차 최대 차이 = {inv_max:.3e}  "
          f"-> 부피 V_i 가 고정이라 두 열은 같은 수다 (독립 지표가 아니다)")
    return summary, inv_max


def final_table(blocks, summaries, seeds):
    """논문 Table 3 후보 — 물체 x 방법, 부위 최악 밀도(=질량) 오차."""
    print(f"\n{'='*104}")
    print("Table 3 후보 — 부위 최악 상대 밀도(=질량) 오차 [%], 중앙값 [IQR]")
    print('='*104)
    print(f"  {'object':<10}{'unk':>4}{'tau%':>7}"
          + "".join(f"{m:>24}" for m in METHODS))
    for key in blocks:
        b, s = blocks[key], summaries[key]
        line = f"  {key:<10}{b['n_body']:>4}{100*b['target']:>7.2f}"
        for m in METHODS:
            c = s[m]
            line += (f"{100*c['part_worst_med']:>9.2f}"
                     f" [{100*c['part_worst_iqr'][0]:>5.2f},"
                     f"{100*c['part_worst_iqr'][1]:>6.2f}]")
        print(line)
    print(f"\n  {'object':<10}{'rounds(med) / converged':<28}"
          f"{'part coverage (|err|<=half)':<32}{'hinge worst % (ours)':<22}")
    for key in blocks:
        s = summaries[key]
        o = s["ours"]
        cov = (f"{o['coverage_part']}/{o['coverage_part_n']}"
               if o["coverage_part"] is not None else "n/a")
        hin = ("n/a (힌지 없음)" if o["hinge_worst_med"] is None
               else f"{100*o['hinge_worst_med']:.2f}")
        print(f"  {key:<10}{str(int(o['rounds_med'])) + ' / ' + str(o['converged']) + '/' + str(seeds):<28}"
              f"{cov:<32}{hin:<22}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", nargs="+", default=list(OBJECT_KEYS))
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--rel", type=float, default=0.05)
    ap.add_argument("--max-rounds", type=int, default=14)
    ap.add_argument("--starts", type=int, default=6)
    ap.add_argument("--json", default="figures/v2/main_real.json")
    a = ap.parse_args()

    os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
    t0 = time.time()
    print(f"실물 물체 주 결과표 (Table 3)")
    print(f"  각도오차 {100*a.rel:g}%   seed {a.seeds}개 (seed=100*s)   "
          f"예산 {a.max_rounds}라운드   시작점 {a.starts}개")
    print(f"  목표 반폭 tau(p) = {100*PER_LINK_TARGET:.1f}% x 부위수")
    print(f"  물체: {', '.join(a.objects)}   (laptop 은 저장소에 자산이 없어 뺀다)")

    blocks, summaries = {}, {}
    inv_all = 0.0
    for key in a.objects:
        print(f"\n[{key}] 실행 중 ...", flush=True)
        blocks[key] = run_object(key, a.seeds, a.rel, a.max_rounds, a.starts)
    for key in a.objects:
        summaries[key], inv = report(blocks[key], a.seeds)
        inv_all = max(inv_all, inv)

    final_table(blocks, summaries, a.seeds)

    json.dump(dict(objects=a.objects, seeds=a.seeds, rel_error=a.rel,
                   max_rounds=a.max_rounds, n_starts=a.starts,
                   per_link_target=PER_LINK_TARGET,
                   methods=list(METHODS),
                   density_mass_max_abs_diff=inv_all,
                   data=blocks, summary=summaries),
              open(a.json, "w"), indent=1)
    print(f"\n총 {time.time()-t0:.0f}초   per-seed 원자료 -> {a.json}")
    print("\n읽는 법")
    print("  uniform 이 크게 틀리는 것이 정리 1의 직접 증거다 — 형상을 안 바꾸면")
    print("  부위를 가를 방법이 없다. 특히 2link/3link 는 부위마다 밀도를 일부러")
    print("  다르게 만든 물체라 균질 가정이 구조적으로 깨진다.")
    print("  single 은 우리 추정기인데 형상을 한 번만 쓴 것이라, 차이가 '추정기'가")
    print("  아니라 '형상을 바꾼 것'에서 온다는 뜻이다. 미지수가 5개인 3link 에서")
    print("  rank-4 천장에 걸려 특히 나쁘다.")
    print("  힌지 행은 작고 가려진 고밀도 부품을 따로 미지수로 잡은 결과다.")


if __name__ == "__main__":
    main()
