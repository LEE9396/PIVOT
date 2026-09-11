#!/usr/bin/env python
"""N-link 시뮬레이션을 동적 여기 방법으로 (실험지시_N-link_밀도추정_시뮬레이션.md 의 동적판).

정적 실험과 같은 물체(nlink.make_spec, seed = 1000 + trial), 같은 조건 C0~C3
(부피 오차 1/5 %, 각도 오차 1/5 % 상대·하한 0.5°), 시드 5개. 다른 점:
  - 미지수가 밀도 P개가 아니라 부위당 10개 (질량·1차모멘트·관성)
  - 라운드 = 형상 선택 → 정적 3방향 → 여기 궤적 (8 s) → 갱신, 라운드 수 = P + 1
  - 부피는 추정에 안 쓰이고 사전분포와 '밀도 = 질량/부피' 환산에만 들어간다
  - 정적 실험처럼 힌지 유지토크·파지 용량 제약은 끈다(--hinge-torque inf), 대신
    필요했던 힌지 토크 최대값을 기록한다

    # 한 trial
    $R python -m dynex.exp_nlink --trial 3 C0 0 --outdir results/dynex_nlink
    # 집계
    $R python -m dynex.exp_nlink --aggregate --outdir results/dynex_nlink
"""
import argparse
import csv
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import density_id_objects as obj          # noqa: E402
import nlink                              # noqa: E402
import robot_scene as rs                  # noqa: E402

from .configs import closed_loop          # noqa: E402
from .design import Limits, Noise         # noqa: E402
from .identifiability import base_report  # noqa: E402
from .identify import apply_total_mass, prior_from_spec, part_errors  # noqa: E402
from .kinematics import SceneModel        # noqa: E402
from .simulate import PRESETS, perturb_tool, truth_phis  # noqa: E402

CONDITIONS = {"C0": (1.0, 0.01), "C1": (5.0, 0.01), "C2": (1.0, 0.05), "C3": (5.0, 0.05)}
MAX_LINKS = 6


def to_jsonable(v):
    """numpy 배열·스칼라를 JSON 으로 쓸 수 있게 재귀 변환."""
    if isinstance(v, dict):
        return {str(k): to_jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [to_jsonable(x) for x in v]
    if hasattr(v, "tolist"):
        return v.tolist()
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def believed_spec(spec, vol_err_pct, sign_seed):
    """추정기가 믿는 부피: 링크마다 부호를 고정으로 뽑고 크기는 정확히 x % (정적 실험과 같은 규약)."""
    rng = np.random.default_rng(sign_seed)
    direction = rng.choice([-1.0, 1.0], size=len(spec.parts))
    parts = [dataclasses.replace(p, volume_cm3=p.volume_cm3 * (1.0 + d * vol_err_pct / 100.0))
             for p, d in zip(spec.parts, direction)]
    return dataclasses.replace(spec, parts=parts), direction


def run_trial(n_part, cond_id, seed, sensor_name, outdir, hinge_torque=np.inf,
              rounds=None, tool_error=1, verbose=False):
    vol_err_pct, ang_rel = CONDITIONS[cond_id]
    t_start = time.time()
    spec_true = nlink.make_spec(n_part, seed=1000 + seed)
    spec_bel, direction = believed_spec(spec_true, vol_err_pct, 700_000 + 977 * n_part + seed)
    model = SceneModel(spec_true)
    checker = model.pose_checker(ik_restarts=6)
    rng = np.random.default_rng(100 * seed + n_part)

    phis_true = truth_phis(spec_true, variant="uniform")
    total_mass = sum(p[0] for p in phis_true)
    tool_true = model.tool_phi() * (0.925 / 0.414)
    tool_est = perturb_tool(tool_true, rng) if tool_error else tool_true.copy()

    est = prior_from_spec(spec_bel, total_mass_kg=total_mass, mass_rel_std=0.5,
                          com_frac_std=0.3, inertia_rel_std=0.5,
                          bias_sigma=(0.2, 0.2, 0.2, 0.01, 0.01, 0.01))
    est = apply_total_mass(est, total_mass * (1.0 + rng.normal(0.0, 0.002)), 0.002)

    sensor = PRESETS[sensor_name]
    noise = Noise(sensor.sigma_f, sensor.sigma_t, sensor.rate_hz)
    big = 1e6
    limits = Limits(workspace_lower=rs.WORKSPACE_LOWER_M - 0.05,
                    workspace_upper=rs.WORKSPACE_UPPER_M + 0.05,
                    min_distance_m=model.min_distance_m,
                    hinge_budget_nm=np.full(len(spec_true.joints),
                                            big if not np.isfinite(hinge_torque) else hinge_torque / 1.5),
                    grasp_force_cap_n=big, grasp_torque_cap_nm=big)
    n_rounds = rounds if rounds else min(n_part + 1, 8)
    opts = dict(period=8.0, harmonics=4, samples=32, starts=1, steps=4, top=1, poses=1,
                safety=1.5, hold_s=2.0, triad=True, max_candidates=20,
                angle_rel_error=ang_rel, angle_floor_deg=0.5)
    if not np.isfinite(hinge_torque):
        opts["hinge_hold_nm"] = np.full(len(spec_true.joints), big)

    est, hist = closed_loop(model, checker, spec_true, phis_true, tool_true, tool_est, est,
                            noise, sensor, limits, opts, rng, rounds=n_rounds, verbose=verbose)
    wall = time.time() - t_start

    # --- 채점 -------------------------------------------------------------
    hinge_on = {p.name: 0.0 for p in spec_true.parts}
    for j in spec_true.joints:
        hinge_on[j.child] += j.hinge_mass_kg
    gt_rho = np.array([p.rho_gt for p in spec_true.parts])
    est_mass = np.array([est.mean[10 * k] for k in range(n_part)])
    true_mass = np.array([phis_true[k][0] for k in range(n_part)])
    hinge = np.array([hinge_on[p.name] for p in spec_true.parts])
    vol_bel = np.array([p.volume_m3 for p in spec_bel.parts])
    est_rho = (est_mass - hinge) / vol_bel
    err_rho = 100.0 * np.abs(est_rho - gt_rho) / gt_rho
    err_mass = 100.0 * np.abs(est_mass - true_mass) / true_mass
    perr = [part_errors(est.mean[10 * k:10 * (k + 1)], phis_true[k]) for k in range(n_part)]
    base = base_report(est, phis_true, model, spec_true)
    stat, dyn = base["static"], base["dynamic_only"]
    converged = bool(np.median(stat["sd_rel"]) < 0.01)

    rec = dict(
        n_links=n_part, cond_id=cond_id, vol_err_pct=vol_err_pct,
        ang_err=f"{100 * ang_rel:g}%rel(min0.5deg)", seed=seed, sensor=sensor_name,
        gt_density=gt_rho.tolist(), est_density=est_rho.tolist(), err_pct=err_rho.tolist(),
        mean_err_pct=float(err_rho.mean()),
        mass_err_pct=err_mass.tolist(), mean_mass_err_pct=float(err_mass.mean()),
        com_err_mm=[float(e["com_mm"]) for e in perr],
        inertia_err_pct=[float(e["inertia_pct"]) for e in perr],
        base_static_n=int(stat["n"]), base_static_err_med=float(np.median(stat["err_rel"])),
        base_static_sd_med=float(np.median(stat["sd_rel"])),
        base_dyn_n=int(dyn["n"]), base_dyn_err_med=float(np.median(dyn["err_rel"])),
        base_dyn_sd_med=float(np.median(dyn["sd_rel"])),
        n_configs=len(hist), n_measurements=sum(4 for _ in hist),
        final_uncertainty=float(np.median(stat["sd_rel"]) * 100.0),
        converged=converged, grasp_point="link0 proximal end (sensor at AFT200 puck)",
        hinge_peak_nm=[float(max(h["hinge_peak_true_nm"])) if h["hinge_peak_true_nm"] else 0.0 for h in hist],
        qd_peak_deg=[float(h["qd_peak_deg"]) for h in hist],
        wall_time_sec=wall, lengths_mm=[p.bbox_mm[0] for p in spec_true.parts],
        volume_sign=direction.tolist(), rounds=[h["theta_deg"] for h in hist],
        hyper=dict(opts, hinge_torque=str(hinge_torque), tool_error=tool_error, rounds=n_rounds),
    )
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    rec = to_jsonable(rec)
    (out / f"trial_N{n_part}_{cond_id}_s{seed}_{sensor_name}.json").write_text(
        json.dumps(rec, indent=1))
    return rec


def aggregate(outdir, sensor_name):
    out = Path(outdir)
    recs = [json.loads(p.read_text()) for p in sorted(out.glob(f"trial_*_{sensor_name}.json"))]
    if not recs:
        print("결과 없음")
        return
    cols = (["n_links", "cond_id", "vol_err_pct", "ang_err", "seed"]
            + [f"gt_density_L{i}" for i in range(1, MAX_LINKS + 1)]
            + [f"est_density_L{i}" for i in range(1, MAX_LINKS + 1)]
            + [f"err_pct_L{i}" for i in range(1, MAX_LINKS + 1)]
            + ["mean_err_pct", "mean_mass_err_pct", "base_static_err_med", "base_static_sd_med",
               "base_dyn_err_med", "base_dyn_sd_med", "n_configs", "n_measurements",
               "final_uncertainty", "converged", "grasp_point", "wall_time_sec"])
    with open(out / f"nlink_dyn_results_{sensor_name}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in recs:
            row = {c: "" for c in cols}
            for c in cols:
                if c in r:
                    row[c] = r[c]
            for i in range(r["n_links"]):
                row[f"gt_density_L{i + 1}"] = f"{r['gt_density'][i]:.1f}"
                row[f"est_density_L{i + 1}"] = f"{r['est_density'][i]:.1f}"
                row[f"err_pct_L{i + 1}"] = f"{r['err_pct'][i]:.3f}"
            w.writerow(row)

    def cell(key, n, cond, fmt="{:.1f} ± {:.1f}"):
        vals = [r[key] for r in recs if r["n_links"] == n and r["cond_id"] == cond]
        return fmt.format(np.mean(vals), np.std(vals)) if vals else "—"

    links = sorted({r["n_links"] for r in recs})
    conds = [c for c in CONDITIONS if any(r["cond_id"] == c for r in recs)]
    lines = [f"# N-link 동적 여기 시뮬레이션 집계 ({sensor_name}, trial {len(recs)}개)", ""]
    for title, key, fmt in (("표 A — 링크 밀도 오차 [%] (밀도 = 추정질량 / 믿는 부피; 부피오차가 1:1로 들어감)", "mean_err_pct", "{:.1f} ± {:.1f}"),
                            ("표 A' — 링크 질량 오차 [%] (부피와 무관)", "mean_mass_err_pct", "{:.1f} ± {:.1f}"),
                            ("표 C — 식별 가능 조합 중 정적으로도 보이는 것: 오차 중앙값 [비율]", "base_static_err_med", "{:.3f} ± {:.3f}"),
                            ("표 D — 흔들어야 보이는 조합(관성 관련): 오차 중앙값 [비율]", "base_dyn_err_med", "{:.2f} ± {:.2f}"),
                            ("표 E — 흔들어야 보이는 조합: 사후 σ 중앙값 [비율] (사전 ≈ 0.6)", "base_dyn_sd_med", "{:.2f} ± {:.2f}")):
        lines += [f"## {title}", "", "| 조건 | " + " | ".join(f"{n} links" for n in links) + " |",
                  "|---|" + "---|" * len(links)]
        for c in conds:
            lines.append(f"| {c} | " + " | ".join(cell(key, n, c, fmt) for n in links) + " |")
        lines.append("")
    lines += ["## 부수 지표", "", "| 조건 | " + " | ".join(f"{n} links" for n in links) + " |", "|---|" + "---|" * len(links)]
    for c in conds:
        cells = []
        for n in links:
            rs_ = [r for r in recs if r["n_links"] == n and r["cond_id"] == c]
            if not rs_:
                cells.append("—"); continue
            conv = sum(r["converged"] for r in rs_)
            hp = np.mean([max(r["hinge_peak_nm"]) if r["hinge_peak_nm"] else 0 for r in rs_])
            cells.append(f"{conv}/{len(rs_)} 수렴, config {np.mean([r['n_configs'] for r in rs_]):.1f}, 힌지 {hp:.2f} N·m, {np.mean([r['wall_time_sec'] for r in rs_]) / 60:.0f} min")
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    six = [r for r in recs if r["n_links"] == 6 and r["cond_id"] == "C0"]
    if six:
        lines += ["", "## 6링크 C0 — 첫 링크 / 마지막 링크 밀도 오차 [%]", "",
                  "| seed | L1 | L6 |", "|---|---|---|"]
        for r in six:
            lines.append(f"| {r['seed']} | {r['err_pct'][0]:.1f} | {r['err_pct'][5]:.1f} |")
    (out / f"nlink_dyn_summary_{sensor_name}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trial", nargs=3, metavar=("N", "COND", "SEED"))
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--sensor", default="datasheet_50hz", choices=tuple(PRESETS))
    ap.add_argument("--outdir", default="results/dynex_nlink")
    ap.add_argument("--hinge-torque", type=float, default=np.inf)
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--tool-error", type=int, default=1)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    if args.aggregate:
        aggregate(args.outdir, args.sensor)
        return
    n, cond, seed = int(args.trial[0]), args.trial[1], int(args.trial[2])
    rec = run_trial(n, cond, seed, args.sensor, args.outdir, args.hinge_torque, args.rounds,
                    args.tool_error, args.verbose)
    print(f"N={n} {cond} seed={seed}: 밀도오차 {rec['mean_err_pct']:.2f}%  질량오차 {rec['mean_mass_err_pct']:.2f}%"
          f"  base(static) err {rec['base_static_err_med']:.3f} σ {rec['base_static_sd_med']:.3f}"
          f"  base(dyn) err {rec['base_dyn_err_med']:.2f} σ {rec['base_dyn_sd_med']:.2f}"
          f"  config {rec['n_configs']}  {rec['wall_time_sec'] / 60:.1f} min")


if __name__ == "__main__":
    main()
