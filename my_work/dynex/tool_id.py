#!/usr/bin/env python
"""툴(그리퍼 + AFT200 퍽 위쪽) 10개 파라미터 식별 — 물체 없이, 같은 벌림량으로.

센서 아래에 매달린 것은 전부 측정에 섞인다. 물체를 재기 전에 툴만 먼저 재서
빼야 하고, 툴/물체 질량비가 2.7 이라 툴 질량 1 % 오차가 물체 2.7 % 로 샌다.

절차
  1. 지금 자세에서 무작위 푸리에 여기 궤적 K 개를 만든다 (관절·속도·가속·충돌 한계 안).
  2. 궤적마다 실행하면서 (t, q) 와 렌치를 100 Hz 로 모은다.
  3. [Y(t) | I6] · [φ_tool; b] = 렌치  를 최소제곱으로 푼다 (영점 b 는 세션에 하나).
  4. calibration/tool_dynamic.json 으로 저장. 세션이 --tool-file 로 읽는다.

    # 모의 (회수율 확인)
    $R python -m dynex.tool_id --hardware sim --out /tmp/tool_sim.json
    # 실물 — 물체를 잡지 않은 상태, 벌림량은 실험과 같게. 작은 진폭(--amp 0.15)부터.
    $R python -m dynex.tool_id --hardware real --robot-host 192.168.50.51 --aft-host 192.168.50.51 \
        --opening-m 0.045 --amp 0.15 --out calibration/tool_dynamic.json
    # 아는 강체 검사: 툴 파일을 기준으로 빼고, 잡은 덩어리의 질량·무게중심을 보고한다
    $R python -m dynex.tool_id --hardware real --baseline calibration/tool_dynamic.json \
        --expect-mass-kg 0.500 --expect-com-mm 0 0 120 --out /tmp/payload_check.json

실물 스트리밍(rb5_stream / Aft200Stream) 은 **미검증** 이다. 비상정지를 손에 들고 시작한다.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import density_id_objects as obj                 # noqa: E402
import robot_scene as rs                         # noqa: E402

from .configs import start_poses                 # noqa: E402
from .design import ExcitationDesigner, Limits, Noise  # noqa: E402
from .identify import prior_from_spec            # noqa: E402
from .inertial import com, central_inertia       # noqa: E402
from .kinematics import SceneModel               # noqa: E402
from .regressor import rigid_body_regressor      # noqa: E402
from .simulate import PRESETS                    # noqa: E402


def make_trajectories(model, designer, n, amp, rng):
    """관절·속도·가속·충돌 한계 안의 무작위 여기 궤적 n 개 (진폭은 이분법으로 줄인다)."""
    out = []
    for _ in range(n):
        x = rng.normal(0.0, amp, designer.n_free)
        x = designer.backoff(x, n_dense=200)
        ver = designer.verify(x)
        out.append((x, ver))
    return out


def collect_sim(model, designer, x, theta, sensor, phi_true, rng):
    ts = np.arange(0.0, designer.traj.T, 1.0 / sensor.rate_hz)
    rows, ys, log = [], [], []
    for t in ts:
        q, qd, qdd = (v[0] for v in designer.traj.evaluate(x, [t]))
        kin = model.evaluate(q, qd, qdd, theta)
        Y = rigid_body_regressor(kin["a_o"], kin["omega"], kin["alpha"], kin["g"])
        w = Y @ phi_true + sensor.bias
        noise = np.concatenate([rng.normal(0, sensor.sigma_f, 3), rng.normal(0, sensor.sigma_t, 3)])
        y = sensor.quantize(w + noise)
        rows.append(Y)
        ys.append(y)
        log.append([t, *q, *y])
    return np.vstack(rows), np.concatenate(ys), log


def collect_real(model, designer, x, theta, arm, wrench, servo_dt):
    import hardware_real as hr
    from scipy.signal import savgol_filter
    joints = arm.stream(lambda t: tuple(v[0] for v in designer.traj.evaluate(x, [t])),
                        designer.traj.T, servo_dt)
    samples = wrench.drain()
    tj = np.array([t for t, _ in joints]); qj = np.vstack([q for _, q in joints])
    tw = np.array([t for t, _ in samples]); w = np.vstack([r for _, r in samples])
    keep = (tw >= tj[0]) & (tw <= tj[-1])
    tw, w = tw[keep], w[keep]
    q = np.column_stack([np.interp(tw, tj, qj[:, i]) for i in range(6)])
    dt = np.median(np.diff(tw)) if tw.size > 1 else 0.01
    win = max(5, int(round(0.2 / dt)) | 1)
    qd = savgol_filter(q, win, 3, deriv=1, delta=dt, axis=0)
    qdd = savgol_filter(q, win, 3, deriv=2, delta=dt, axis=0)
    rows, log = [], []
    for k in range(tw.size):
        kin = model.evaluate(q[k], qd[k], qdd[k], theta)
        rows.append(rigid_body_regressor(kin["a_o"], kin["omega"], kin["alpha"], kin["g"]))
        log.append([tw[k], *q[k], *w[k]])
    return np.vstack(rows), w.ravel(), log


def solve(Y_all, y_all, sigma_f, sigma_t):
    n = Y_all.shape[0] // 6
    B = np.zeros((6 * n, 6))
    B[np.arange(6 * n), np.arange(6 * n) % 6] = 1.0
    A = np.hstack([Y_all, B])
    w = np.tile(np.array([1 / sigma_f] * 3 + [1 / sigma_t] * 3), n)
    sol, *_ = np.linalg.lstsq(A * w[:, None], y_all * w, rcond=None)
    resid = (y_all - A @ sol).reshape(-1, 6)
    cov = np.linalg.pinv((A * w[:, None]).T @ (A * w[:, None]))
    return sol[:10], sol[10:], resid, np.sqrt(np.diag(cov))[:10]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hardware", choices=("sim", "real"), default="sim")
    ap.add_argument("--object", default="3link", choices=tuple(obj.OBJECTS), help="운동학·충돌 검사용 씬 (물체는 없다고 가정, 보수적)")
    ap.add_argument("--sensor", default="datasheet_100hz", choices=tuple(PRESETS))
    ap.add_argument("--n-traj", type=int, default=3)
    ap.add_argument("--period", type=float, default=8.0)
    ap.add_argument("--harmonics", type=int, default=3)
    ap.add_argument("--amp", type=float, default=0.25, help="푸리에 계수 표준편차 [rad]. 실물 첫 시험은 0.15")
    ap.add_argument("--qd-max-deg", type=float, default=90.0)
    ap.add_argument("--qdd-max-deg", type=float, default=150.0)
    ap.add_argument("--opening-m", type=float, default=None, help="기록용: 이때 쓴 그리퍼 벌림량")
    ap.add_argument("--baseline", default=None, help="이전 툴 파일: 빼서 추가 페이로드를 보고 (아는 강체 검사)")
    ap.add_argument("--expect-mass-kg", type=float, default=None)
    ap.add_argument("--expect-com-mm", type=float, nargs=3, default=None)
    ap.add_argument("--out", default="calibration/tool_dynamic.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--robot-host", default="192.168.50.51")
    ap.add_argument("--aft-host", default="192.168.50.51")
    ap.add_argument("--aft-hz", type=float, default=100.0)
    ap.add_argument("--servo-dt", type=float, default=0.005)
    ap.add_argument("--move-duration", type=float, default=8.0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    spec = obj.OBJECTS[args.object]
    model = SceneModel(spec)
    checker = model.pose_checker(ik_restarts=6)
    theta = np.array([np.pi / 2] * len(spec.joints))
    sensor = PRESETS[args.sensor]
    noise = Noise(sensor.sigma_f, sensor.sigma_t, sensor.rate_hz)
    big = 1e6
    limits = Limits(qd_max=np.full(6, np.deg2rad(args.qd_max_deg)),
                    qdd_max=np.full(6, np.deg2rad(args.qdd_max_deg)),
                    workspace_lower=rs.WORKSPACE_LOWER_M - 0.15, workspace_upper=rs.WORKSPACE_UPPER_M + 0.15,
                    min_distance_m=model.min_distance_m,
                    hinge_budget_nm=np.full(len(spec.joints), big),
                    grasp_force_cap_n=big, grasp_torque_cap_nm=big)
    est = prior_from_spec(spec, total_mass_kg=0.3)

    if args.hardware == "sim":
        q0 = start_poses(model, checker, theta)[0]
        phi_true = model.tool_phi() * (0.925 / 0.414)
        arm = wrench = None
    else:
        import hardware_real as hr
        arm = hr.Rb5Driver(hr.RbpodoBackend(host=args.robot_host))
        wrench = hr.Aft200Stream(args.aft_host, args.aft_hz)
        q0 = arm.joint_positions()
        phi_true = None
        print(f"  시작 자세(현재) {np.round(np.degrees(q0), 1)} deg — 물체를 잡고 있지 않은지 확인하세요")

    designer = ExcitationDesigner(model, theta, q0, est, noise, limits, period_s=args.period,
                                  n_harmonics=args.harmonics, n_samples=24, seed=args.seed)
    ok, _ = designer.static_feasible()
    if not ok:
        raise SystemExit("시작 자세가 충돌·작업공간 조건을 어깁니다. 팔을 열린 공간으로 옮기세요.")
    trajs = make_trajectories(model, designer, args.n_traj, args.amp, rng)
    Y_all, y_all, logs = [], [], []
    for k, (x, ver) in enumerate(trajs):
        print(f"  궤적 {k + 1}/{len(trajs)}: qd {ver['qd_peak_deg']:.0f} deg/s, qdd {ver['qdd_peak_deg']:.0f} deg/s², 간격 {1000 * ver['min_distance_m']:.1f} mm")
        if args.hardware == "sim":
            Y, y, log = collect_sim(model, designer, x, theta, sensor, phi_true, rng)
        else:
            input("  >>> 궤적을 실행합니다. 비상정지를 손에 들고 Enter")
            arm.move_to(q0, args.move_duration) if hasattr(arm, "move_to") else arm.follow([q0], args.move_duration)
            time.sleep(0.5)
            wrench.drain()
            Y, y, log = collect_real(model, designer, x, theta, arm, wrench, args.servo_dt)
        Y_all.append(Y); y_all.append(y); logs += log
    Y_all, y_all = np.vstack(Y_all), np.concatenate(y_all)
    phi, bias, resid, sd = solve(Y_all, y_all, sensor.sigma_f, sensor.sigma_t)
    rms = np.sqrt((resid ** 2).mean(axis=0))
    print(f"툴: 질량 {1000 * phi[0]:.1f} g ± {1000 * sd[0]:.1f}, 무게중심 {np.round(1000 * com(phi), 1)} mm,"
          f" 영점 {np.round(bias, 3)}")
    print(f"잔차 RMS 힘 {rms[:3].round(3)} N, 토크 {rms[3:].round(4)} N·m  (센서 잡음 {sensor.sigma_f} N / {sensor.sigma_t} N·m 수준이어야 한다;"
          f" 훨씬 크면 케이블 힘·시각 지연·추종 지연을 의심)")
    if phi_true is not None:
        print(f"  [모의] 참값 질량 {1000 * phi_true[0]:.1f} g, 무게중심 {np.round(1000 * com(phi_true), 1)} mm →"
              f" 오차 질량 {100 * abs(phi[0] - phi_true[0]) / phi_true[0]:.2f} %,"
              f" 무게중심 {1000 * np.linalg.norm(com(phi) - com(phi_true)):.2f} mm,"
              f" 관성 {100 * np.linalg.norm(central_inertia(phi) - central_inertia(phi_true)) / np.linalg.norm(central_inertia(phi_true)):.0f} %")
    out = dict(phi_tool=phi.tolist(), bias=bias.tolist(), sigma=sd.tolist(), residual_rms=rms.tolist(),
               n_samples=int(Y_all.shape[0] // 6), opening_m=args.opening_m, sensor=args.sensor,
               hardware=args.hardware, q0_rad=np.asarray(q0).tolist(),
               trajectories=[np.asarray(x).tolist() for x, _ in trajs], period_s=args.period,
               harmonics=args.harmonics)
    if args.baseline:
        base = np.asarray(json.loads(Path(args.baseline).read_text())["phi_tool"], float)
        payload = phi - base
        m, c = payload[0], com(payload) if payload[0] > 1e-6 else np.full(3, np.nan)
        print(f"페이로드(툴 기준선을 뺀 것): 질량 {1000 * m:.1f} g, 무게중심 {np.round(1000 * c, 1)} mm")
        if args.expect_mass_kg is not None:
            print(f"  기대 질량 {1000 * args.expect_mass_kg:.1f} g → 오차 {100 * abs(m - args.expect_mass_kg) / args.expect_mass_kg:.2f} %")
        if args.expect_com_mm is not None:
            print(f"  기대 무게중심 {args.expect_com_mm} mm → 오차 {1000 * np.linalg.norm(c - np.asarray(args.expect_com_mm) / 1000):.2f} mm")
        out["payload"] = dict(mass_kg=float(m), com_m=np.asarray(c).tolist())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    Path(args.out).with_suffix(".samples.csv").write_text(
        "t,q0,q1,q2,q3,q4,q5,fx,fy,fz,tx,ty,tz\n" + "\n".join(",".join(f"{v:.6f}" for v in row) for row in logs) + "\n")
    print(f"saved {args.out} (+ .samples.csv)")


if __name__ == "__main__":
    main()
