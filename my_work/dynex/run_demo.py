#!/usr/bin/env python
"""동적 여기 식별 폐루프 시뮬레이션 + 하드웨어 보고서.

    cd my_work
    ../robot_learning/scripts/run_drake_env.sh python -m dynex.run_demo --object 3link --quick
    ../robot_learning/scripts/run_drake_env.sh python -m dynex.run_demo --object 3link \
        --rounds 4 --sensor datasheet_50hz --variant shell --hinge-torque 0.5 --hardware-report
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import density_id_objects as obj                       # noqa: E402
import robot_scene as rs                               # noqa: E402

from .configs import closed_loop                       # noqa: E402
from .design import Limits, Noise                      # noqa: E402
from .hardware import report                           # noqa: E402
from .identify import apply_total_mass, prior_from_spec  # noqa: E402
from .kinematics import SceneModel                     # noqa: E402
from .simulate import PRESETS, perturb_tool, truth_phis  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--object", default="3link", choices=tuple(obj.OBJECTS))
    ap.add_argument("--variant", default="uniform", choices=("uniform", "shell", "insert"))
    ap.add_argument("--insert-mass-frac", type=float, default=0.3)
    ap.add_argument("--densities", type=float, nargs="+", default=None,
                    help="부위별 밀도 [kg/m³]. 기본: 3link 는 제작품 실측 347 442 425, 그 외 spec 값")
    ap.add_argument("--hinge-torque", type=float, default=0.5, help="힌지 유지토크 [N·m]")
    ap.add_argument("--safety", type=float, default=1.5)
    ap.add_argument("--sensor", default="datasheet_50hz", choices=tuple(PRESETS))
    ap.add_argument("--jitter-ms", type=float, default=0.0)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--period", type=float, default=8.0)
    ap.add_argument("--harmonics", type=int, default=4)
    ap.add_argument("--samples", type=int, default=40)
    ap.add_argument("--starts", type=int, default=2)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--top", type=int, default=2)
    ap.add_argument("--tool-error", type=int, default=1, help="1 이면 툴 파라미터에 식별 오차")
    ap.add_argument("--tool-mass-scale", type=float, default=0.925 / 0.414)
    ap.add_argument("--qdd-max-deg", type=float, default=300.0)
    ap.add_argument("--qd-max-deg", type=float, default=180.0)
    ap.add_argument("--grip-force", type=float, default=205.0, help="experiment.conf 의 GRIPPER_FORCE")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bias-sigma", type=float, nargs=2, default=(0.2, 0.01), help="타어 뒤 남는 영점 불확실성 [N, N·m]")
    ap.add_argument("--lag-ms", type=float, default=0.0, help="서보 추종 지연 (참 궤적이 지령보다 늦음)")
    ap.add_argument("--poses", type=int, default=2, help="형상당 설계에 쓸 시작 자세 수")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--select", default="info", choices=("info", "random", "fixed"),
                    help="형상 선택: info=정보이득(제안), random=무작위(ablation), fixed=한 형상 고정(강체 baseline)")
    ap.add_argument("--fixed-theta-deg", type=float, nargs="+", default=None)
    ap.add_argument("--no-hinge-constraint", action="store_true",
                    help="ablation: 힌지 예산을 무시하고 설계 (미끄러짐이 slip 열에 나타난다)")
    ap.add_argument("--robust-k", type=float, default=2.0)
    ap.add_argument("--hold-s", type=float, default=2.0, help="정적 3방향 측정의 자세당 유지 시간")
    ap.add_argument("--no-triad", action="store_true", help="ablation: 정적 3방향 없이 동적만")
    ap.add_argument("--prior", type=float, nargs=3, default=(0.5, 0.3, 0.5),
                    help="사전분포 폭: 질량 상대, 무게중심(상자 반폭 비율), 관성 상대")
    ap.add_argument("--hardware-report", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.quick:
        args.rounds, args.starts, args.steps, args.top, args.samples = 1, 1, 3, 1, 24

    rng = np.random.default_rng(args.seed)
    spec = obj.OBJECTS[args.object]
    t0 = time.time()
    model = SceneModel(spec)
    checker = model.pose_checker()
    print(f"{spec.label}: 씬 준비 {time.time() - t0:.1f}s, 부위 {model.n_parts}, 관절 {len(spec.joints)}")

    densities = args.densities
    if densities is None and args.object == "3link":
        densities = [347.0, 442.0, 425.0]           # 논문 Table III 의 제작품 GT
    scale = None if densities is None else (np.asarray(densities, float)
                                            / np.array([p.rho_gt for p in spec.parts]))
    phis_true = truth_phis(spec, variant=args.variant, insert_mass_frac=args.insert_mass_frac,
                           density_scale=scale)
    total_mass = sum(p[0] for p in phis_true)
    tool_true = model.tool_phi() * args.tool_mass_scale
    tool_est = perturb_tool(tool_true, rng) if args.tool_error else tool_true.copy()
    print(f"물체 총질량 {1000 * total_mass:.1f} g (부위 {[round(1000 * p[0], 1) for p in phis_true]}),"
          f" 툴 {1000 * tool_true[0]:.0f} g")

    bf, bt = args.bias_sigma
    pm, pc, pi = args.prior
    est = prior_from_spec(spec, total_mass_kg=total_mass, mass_rel_std=pm, com_frac_std=pc,
                          inertia_rel_std=pi, bias_sigma=(bf, bf, bf, bt, bt, bt))
    est = apply_total_mass(est, total_mass * (1 + rng.normal(0, 0.002)), rel_error=0.002)

    sensor = PRESETS[args.sensor]
    sensor.jitter_s = args.jitter_ms * 1e-3
    noise = Noise(sigma_f=sensor.sigma_f, sigma_t=sensor.sigma_t, rate_hz=sensor.rate_hz)
    from .hardware import Robotiq2F85Spec
    f_cap, t_cap = Robotiq2F85Spec().capacity(args.grip_force)
    limits = Limits(qd_max=np.full(6, np.deg2rad(args.qd_max_deg)),
                    qdd_max=np.full(6, np.deg2rad(args.qdd_max_deg)),
                    workspace_lower=rs.WORKSPACE_LOWER_M - 0.05,
                    workspace_upper=rs.WORKSPACE_UPPER_M + 0.05,
                    min_distance_m=model.min_distance_m,
                    hinge_budget_nm=np.full(len(spec.joints),
                                            1e6 if args.no_hinge_constraint else args.hinge_torque / args.safety),
                    robust_k=args.robust_k,
                    grasp_force_cap_n=f_cap, grasp_torque_cap_nm=t_cap)
    opts = dict(period=args.period, harmonics=args.harmonics, samples=args.samples,
                starts=args.starts, steps=args.steps, top=args.top, safety=args.safety,
                lag=args.lag_ms * 1e-3, poses=args.poses, select=args.select,
                fixed_theta=(None if args.fixed_theta_deg is None else np.deg2rad(args.fixed_theta_deg)))
    opts["hold_s"] = args.hold_s
    opts["triad"] = not args.no_triad
    if args.no_hinge_constraint:
        opts["hinge_hold_nm"] = args.hinge_torque
    target = dict(mass_rel=0.01, com_mm=1.0, inertia_rel=0.10)

    est, history = closed_loop(model, checker, spec, phis_true, tool_true, tool_est, est,
                               noise, sensor, limits, opts, rng, rounds=args.rounds,
                               target=target, verbose=True)

    if history:
        print()
        print("=== 라운드 요약 ===")
        print(f"{'round':>5} {'theta [deg]':>16} {'IG':>6} {'hinge/budget':>12} {'slip':>5}"
              + "".join(f" | {p.name[:10]:>10} m% com_mm I%" for p in spec.parts))
        for h in history:
            row = (f"{h['round']:>5} {str(np.round(h['theta_deg'], 0)):>16} {h['ig']:>6.1f}"
                   f" {max(h['hinge_peak_true_nm']) / min(h['hinge_budget_nm']):>12.2f} {str(h['slip']):>5}")
            for e in h["errors"]:
                row += f" | {e['mass_pct']:>5.2f} {e['com_mm']:>6.2f} {e['inertia_pct']:>5.1f}"
            print(row)
        print(f"총 {sum(h['seconds'] for h in history):.0f}s")

    if args.hardware_report and history:
        last = history[-1]
        from .design import ExcitationDesigner
        d = ExcitationDesigner(model, np.deg2rad(last["theta_deg"]), np.array(last["q0"]),
                               est, noise, limits, period_s=args.period,
                               n_harmonics=args.harmonics, n_samples=args.samples)
        d.last = (np.array(last["x"]), None)
        print()
        print(report(d, phis_true, tool_true,
                     {k: v for k, v in PRESETS.items()}, limits.hinge_budget_nm,
                     grip_force_n=args.grip_force))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        clean = []
        for rec in history:
            r = {k: v for k, v in rec.items() if k not in ("hinge_true", "times")}
            r["errors"] = [{kk: (vv.tolist() if hasattr(vv, "tolist") else vv)
                            for kk, vv in e.items()} for e in rec["errors"]]
            clean.append(r)
        out.write_text(json.dumps(dict(args=vars(args), history=clean), indent=1, default=float))
        print(f"saved {out}")
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
            last = history[-1]
            for i in range(last["hinge_true"].shape[1]):
                axes[0].plot(last["times"], last["hinge_true"][:, i], label=f"joint{i + 1}")
                axes[0].axhline(last["hinge_budget_nm"][i], ls="--", c="k", lw=0.8)
                axes[0].axhline(-last["hinge_budget_nm"][i], ls="--", c="k", lw=0.8)
            axes[0].set_xlabel("t [s]"); axes[0].set_ylabel("hinge-axis torque [N·m]")
            axes[0].set_title(f"round {last['round']}: θ={np.round(last['theta_deg'], 0)} deg")
            axes[0].legend()
            rounds = [h["round"] for h in history]
            for k in range(model.n_parts):
                axes[1].plot(rounds, [h["errors"][k]["mass_pct"] for h in history], "o-",
                             label=f"{spec.parts[k].name} mass %")
                axes[1].plot(rounds, [h["errors"][k]["inertia_pct"] for h in history], "s--",
                             label=f"{spec.parts[k].name} inertia %")
            axes[1].set_yscale("log"); axes[1].set_xlabel("round"); axes[1].set_ylabel("error [%]")
            axes[1].legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(out.with_suffix(".png"), dpi=140)
            print(f"saved {out.with_suffix('.png')}")
        except Exception as exc:                                     # noqa: BLE001
            print("plot skipped:", exc)


if __name__ == "__main__":
    main()
