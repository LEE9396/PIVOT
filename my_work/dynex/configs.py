"""관절 형상 선택과 폐루프.

라운드마다: 후보 θ 를 훑어 (i) 시작 자세가 있고 (ii) 정지 힌지 토크가 예산
안인 것만 남기고, 값싼 대리 점수(공통 탐침 궤적의 정보이득)로 순위를 매긴
뒤, 상위 몇 개만 궤적을 제대로 최적화해서 정보이득이 가장 큰 (θ, 궤적)을
고른다. 측정 → 사후분포 갱신 → 물리 일관성 투영 → 목표 불확실성 검사.

한 형상에서 관측되는 것은 '그 형상의 조립체' 10개뿐이므로, 부위 P 개의
10P 개를 다 풀려면 형상이 최소 P 개 필요하다 (정적 하한 ⌈(P−1)/3⌉ 의
동적 대응). 형상 다양성이 여기서 관측성을 채운다.
"""
import itertools
import time

import numpy as np

from .design import ExcitationDesigner
from .identify import (apply_total_mass, part_errors, part_uncertainty,
                       project_physical, update)
from .simulate import execute_and_measure

G_DIRS = [np.array(v) for v in ([0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])]


def candidate_thetas(spec, steps=4, margin_deg=8.0):
    axes = []
    for joint in spec.joints:
        lo, hi = joint.limits_rad
        m = np.deg2rad(margin_deg)
        axes.append(np.linspace(lo + m, hi - m, steps))
    return [np.array(v) for v in itertools.product(*axes)]


def start_poses(model, checker, theta, g_dirs=G_DIRS, max_poses=3):
    poses = []
    checker._last_solution = None
    for g in g_dirs:
        full = checker.solve_robust(theta, g)
        if full is None:
            continue
        q_arm = np.array([full[j.position_start()] for j in checker.arm_joints])
        if model.min_distance(q_arm, theta) < model.min_distance_m:
            continue
        poses.append(q_arm)
        if len(poses) >= max_poses:
            break
    return poses


def make_designer(model, theta, q0, est, noise, limits, opts, seed=0):
    return ExcitationDesigner(model, theta, q0, est, noise, limits,
                              period_s=opts["period"], n_harmonics=opts["harmonics"],
                              n_samples=opts["samples"], seed=seed)


def proxy_score(model, theta, q0, est, noise, limits, opts, x_probe):
    d = make_designer(model, theta, q0, est, noise, limits, opts)
    ok, _ = d.static_feasible()
    if not ok:
        return -np.inf
    x = d.backoff(x_probe[:d.n_free], n_dense=60)
    ev = d.evaluate(x)
    return d.info_gain(ev["Yp"])


def rank_candidates(model, checker, candidates, est, noise, limits, opts, rng,
                    verbose=False):
    """후보 θ 를 대리 점수(공통 탐침 궤적의 정보이득)로 정렬한다."""
    x_probe = rng.normal(0.0, 0.08, 6 * 2 * (opts["harmonics"] - 1))
    ranked = []
    t0 = time.time()
    for theta in candidates:
        poses = start_poses(model, checker, theta)
        if not poses:
            continue
        score = proxy_score(model, theta, poses[0], est, noise, limits, opts, x_probe)
        if np.isfinite(score):
            ranked.append((score, theta, poses))
    ranked.sort(key=lambda r: -r[0])
    mode = opts.get("select", "info")
    if mode == "random" and ranked:                 # ablation: 형상을 무작위로
        rng.shuffle(ranked)
    elif mode == "fixed" and ranked:                # ablation: 강체처럼 한 형상만
        fixed = opts.get("fixed_theta")
        if fixed is not None:
            ranked.sort(key=lambda r: np.linalg.norm(r[1] - fixed))
        ranked = ranked[:1]
    if verbose:
        print(f"  후보 {len(candidates)}개 중 실현 가능 {len(ranked)}개"
              f" (대리 점수 {time.time() - t0:.1f}s)")
        for sc, th, _ in ranked[:opts.get("top", 1)]:
            print(f"    θ={np.round(np.degrees(th), 1)} deg  대리 IG {sc:.2f}")
    return ranked


def static_triad(model, theta, poses, est, noise, limits, opts, sensor,
                 phis_true, tool_true, tool_est, rng, verbose=False, theta_true=None):
    """논문의 정적 3방향 측정을 이 형상에서 먼저 한다.

    질량과 1차 모멘트가 여기서 좁아지고, 그래야 힌지 축 토크의 보수적 경계
    (k·σ) 가 풀려 그다음 여기 궤적이 빨라질 수 있다. 정적 → 동적 순서가
    '황금 비율' 의 실제 모습이다.
    """
    hold = opts.get("hold_s", 2.0)
    total_ig = 0.0
    for q0 in poses:
        d = ExcitationDesigner(model, theta, q0, est, noise, limits, period_s=hold,
                               n_harmonics=2, n_samples=4)
        x = np.zeros(d.n_free)
        ev = d.evaluate(x, times=d.traj.times(4))
        total_ig += d.info_gain(ev["Yp"])
        meas = execute_and_measure(model, d, x, phis_true, tool_true, tool_est, sensor, rng,
                                   theta_true=theta_true)
        est = update(est, meas["Yp"], meas["y"], meas["r_diag"])
    if verbose:
        print(f"  정적 3방향 ({len(poses)}자세 × {hold:.0f}s): 정보이득 {total_ig:.1f} nat")
    return est, total_ig


def design_at(model, theta, poses, est, noise, limits, opts, rng, verbose=False):
    best = None
    for q0 in poses[:opts.get("poses", 1)]:
        d = make_designer(model, theta, q0, est, noise, limits, opts,
                          seed=int(rng.integers(1 << 30)))
        ok, _ = d.static_feasible()
        if not ok:
            continue
        x, ver = d.optimize(n_starts=opts["starts"], verbose=verbose)
        if ver is None:
            continue
        if best is None or (ver["feasible"] and ver["ig"] > best["verify"]["ig"]):
            best = dict(theta=theta, q0=q0, designer=d, x=x, verify=ver)
    return best


def closed_loop(model, checker, spec, phis_true, tool_true, tool_est, est, noise,
                sensor, limits, opts, rng, rounds=3, target=None, verbose=True):
    """라운드 = 형상 선택 → (정적 3방향) → 여기 궤적 설계·실행 → 갱신 → 투영."""
    all_candidates = candidate_thetas(spec, steps=opts["steps"])
    history = []
    for r in range(1, rounds + 1):
        t0 = time.time()
        candidates = all_candidates
        cap = opts.get("max_candidates")
        if cap and len(all_candidates) > cap:
            idx = rng.choice(len(all_candidates), size=cap, replace=False)
            candidates = [all_candidates[i] for i in idx]
        ranked = rank_candidates(model, checker, candidates, est, noise, limits, opts,
                                 rng, verbose=verbose)
        if not ranked:
            print("  실현 가능한 형상이 없다 — 중단")
            break
        score, theta_cmd, poses = ranked[0]
        # 각도 오차: 작업자가 맞춘 실제 각도와 카메라가 읽은 각도는 지령과 다르다.
        rel, floor = opts.get("angle_rel_error", 0.0), np.deg2rad(opts.get("angle_floor_deg", 0.5))
        if rel > 0.0:
            sigma = np.maximum(rel * np.abs(theta_cmd), floor)
            theta_true = theta_cmd + rng.normal(0.0, sigma)
            theta = theta_true + rng.normal(0.0, sigma)          # 추정기가 믿는 각도
        else:
            theta_true, theta = theta_cmd, theta_cmd
        triad_ig = 0.0
        if opts.get("triad", True):
            est, triad_ig = static_triad(model, theta, poses, est, noise, limits, opts,
                                         sensor, phis_true, tool_true, tool_est, rng,
                                         verbose=verbose, theta_true=theta_true)
        pick = design_at(model, theta, poses, est, noise, limits, opts, rng, verbose=verbose)
        if pick is None:
            print("  여기 궤적을 못 만들었다 — 이 라운드는 정적 측정만")
            ver = dict(ig=0.0, hinge_peak_ratio=0.0, qd_peak_deg=0.0, qdd_peak_deg=0.0,
                       min_distance_m=0.0)
            meas = dict(hinge_peak_nm=np.zeros(len(spec.joints)),
                        hinge_true=np.zeros((1, len(spec.joints))), times=np.zeros(1))
            x, q0 = np.zeros(1), poses[0]
        else:
            d, x, ver, q0 = pick["designer"], pick["x"], pick["verify"], pick["q0"]
            meas = execute_and_measure(model, d, x, phis_true, tool_true, tool_est, sensor, rng,
                                       tracking_lag_s=opts.get("lag", 0.0), theta_true=theta_true)
            est = update(est, meas["Yp"], meas["y"], meas["r_diag"])
        est, projected = project_physical(est)
        hold = opts.get("hinge_hold_nm", None)
        hold = (np.asarray(limits.hinge_budget_nm) * opts.get("safety", 1.0)) if hold is None else hold
        slip = bool((np.asarray(meas["hinge_peak_nm"]) > hold).any())
        errors = [part_errors(est.mean[10 * k:10 * (k + 1)], phis_true[k])
                  for k in range(model.n_parts)]
        unc = [part_uncertainty(est, k) for k in range(model.n_parts)]
        rec = dict(round=r, theta_deg=np.degrees(theta).tolist(),
                   theta_true_deg=np.degrees(theta_true).tolist(), ig=ver["ig"], triad_ig=triad_ig,
                   proxy=score, hinge_peak_ratio=ver["hinge_peak_ratio"],
                   hinge_peak_true_nm=np.asarray(meas["hinge_peak_nm"]).tolist(),
                   hinge_budget_nm=np.asarray(limits.hinge_budget_nm).tolist(),
                   slip=slip, projected=projected,
                   qd_peak_deg=ver["qd_peak_deg"], qdd_peak_deg=ver["qdd_peak_deg"],
                   min_distance_mm=1000 * ver["min_distance_m"],
                   errors=errors, uncertainty=unc, seconds=time.time() - t0,
                   x=np.asarray(x).tolist(), q0=np.asarray(q0).tolist(),
                   hinge_true=meas["hinge_true"], times=meas["times"])
        history.append(rec)
        if verbose:
            print(f"[round {r}] θ={np.round(rec['theta_deg'], 1)} deg  정적 {triad_ig:.1f} + 동적 {ver['ig']:.1f} nat"
                  f"  힌지 최대 {max(rec['hinge_peak_true_nm']) if rec['hinge_peak_true_nm'] else 0:.3f} N·m"
                  f" (예산 {np.round(limits.hinge_budget_nm, 3)})"
                  f"  qd {ver['qd_peak_deg']:.0f} deg/s  qdd {ver['qdd_peak_deg']:.0f} deg/s²"
                  f"  간격 {rec['min_distance_mm']:.1f} mm  {rec['seconds']:.0f}s")
            for k, (e, u) in enumerate(zip(errors, unc)):
                print(f"    {spec.parts[k].name:<12} 질량 {e['mass_pct']:6.2f}%  "
                      f"무게중심 {e['com_mm']:6.2f} mm  관성 {e['inertia_pct']:6.1f}%"
                      f"   (σ: 질량 {100 * u['mass_rel']:.2f}%  무게중심 {u['com_mm']:.2f} mm"
                      f"  관성 {100 * u['inertia_rel']:.0f}%)")
        if target is not None and all(
                u["mass_rel"] <= target["mass_rel"] and u["com_mm"] <= target["com_mm"]
                and u["inertia_rel"] <= target["inertia_rel"] for u in unc):
            if verbose:
                print("  목표 불확실성 도달")
            break
    return est, history
