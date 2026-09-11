#!/usr/bin/env python
"""dynex 세션 — 통합 UI 의 탐색 단계를 동적 여기 방법으로 돈다.

pivot_ui.py 는 experiment.conf 에 METHOD=dynamic 이면 dual_view.py 대신 이것을
부른다. 세션 폴더 규약은 tools/pivot_session.py 와 같고, 라운드마다 남기는 것:

    angle_round_N.json      이번 라운드에 쓴 θ (추천·측정), 다음 라운드 추천 θ
    path_round_N.json       정적 3자세 팔 관절각 + 여기 궤적 경유점 + 예측 최대값
    wrench_round_N.csv      t, q(6), 원시 렌치(6), 툴을 뺀 렌치(6)  ← 오프라인 재추정용
    posterior_round_N.json  부위별 추정·σ, 식별 가능 좌표의 σ, 수렴 여부
    dynex_state.npz         사후분포 (라운드 사이에 이어 쓴다)
    export/asset_dynamic.json  최종 파라미터와 '약속으로 채운 방향' 표시

--hardware sim 이면 가짜 팔 + AFT200 사양 잡음(분해능·잡음·영점·기록 속도)을 넣은
모의 센서 + 각도 오차를 섞는 모의 추적기로 실물 경로를 그대로 밟는다.
--hardware real 은 RB5(rbpodo)·AFT200(Modbus 스트림)·FoundationPose 파일을 쓴다.
여기 궤적의 실물 실행(move_servo_j 스트리밍)은 **미검증** 이다 (hardware_real.stream).
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent / "tools"))

import density_id_objects as obj                   # noqa: E402
import robot_scene as rs                           # noqa: E402
from pivot_session import Session                  # noqa: E402

from .configs import (G_DIRS, candidate_thetas, design_at, rank_candidates,  # noqa: E402
                      start_poses)
from .design import ExcitationDesigner, Limits, Noise  # noqa: E402
from .hardware import Robotiq2F85Spec              # noqa: E402
from .identifiability import base_basis, base_report  # noqa: E402
from .identify import (GaussianEstimate, apply_total_mass, part_uncertainty,  # noqa: E402
                       prior_from_spec, project_physical, update)
from .inertial import central_inertia, com, N_PARAM  # noqa: E402
from .kinematics import SceneModel                 # noqa: E402
from .regressor import rigid_body_regressor        # noqa: E402
from .simulate import PRESETS, perturb_tool, truth_phis  # noqa: E402


# ---------------------------------------------------------------------------
# 장비 추상화 — 세션이 쓰는 최소 인터페이스
# ---------------------------------------------------------------------------
class ArmSim:
    """가짜 팔. 정지 자세는 즉시 도달, 궤적은 시각 t 에 q(t) 를 그대로 실행한다."""

    def __init__(self, q0, publish=None):
        self.q = np.asarray(q0, float).copy()
        self.publish = publish or (lambda q: None)

    def joint_positions(self):
        return self.q.copy()

    def move_to(self, q, duration_s):
        self.q = np.asarray(q, float).copy()
        self.publish(self.q)

    def stream(self, q_fn, duration_s, dt):
        """궤적을 실행하며 (t, q_measured) 를 돌려준다. 가짜 팔은 지령을 정확히 따른다."""
        ts = np.arange(0.0, duration_s, dt)
        out = []
        for t in ts:
            q, _, _ = q_fn(t)
            self.q = q
            out.append((t, q.copy()))
        self.publish(self.q)
        return out


class WrenchSim:
    """AFT200 사양의 모의 센서. 지금 팔 상태와 참 물체로 렌치를 만든다."""

    def __init__(self, model, sensor, phis_true, tool_true, theta_true_getter, rng):
        self.model, self.sensor, self.rng = model, sensor, rng
        self.Phi = np.concatenate(phis_true)
        self.tool = tool_true
        self.theta_true_getter = theta_true_getter
        self._Mstack = None
        self._theta_cached = None

    def _mstack(self):
        theta = np.atleast_1d(self.theta_true_getter())
        if self._theta_cached is None or not np.allclose(theta, self._theta_cached):
            self._Mstack = np.hstack(self.model.part_transports(theta))
            self._theta_cached = theta.copy()
        return self._Mstack

    def read_at(self, q, qd, qdd):
        kin = self.model.evaluate(q, qd, qdd, np.atleast_1d(self.theta_true_getter()))
        Y = rigid_body_regressor(kin["a_o"], kin["omega"], kin["alpha"], kin["g"])
        w = Y @ (self._mstack() @ self.Phi + self.tool) + self.sensor.bias
        noise = np.concatenate([self.rng.normal(0, self.sensor.sigma_f, 3),
                                self.rng.normal(0, self.sensor.sigma_t, 3)])
        return self.sensor.quantize(w + noise)


class PoseSim:
    """FoundationPose 흉내: 실제 각도에 상대오차(하한 0.5°)를 섞는다."""

    def __init__(self, truth_getter, rel_error=0.05, floor_deg=0.5, rng=None):
        self.truth_getter, self.rel, self.floor = truth_getter, rel_error, floor_deg
        self.rng = rng or np.random.default_rng(0)

    def object_joint_deg(self):
        truth = np.degrees(np.atleast_1d(self.truth_getter()))
        sigma = np.maximum(self.rel * np.abs(truth), self.floor)
        return truth + self.rng.normal(0.0, sigma), sigma


# ---------------------------------------------------------------------------
class DynexSession:
    def __init__(self, args):
        self.args = args
        self.spec = obj.OBJECTS[args.object]
        self.model = SceneModel(self.spec)
        self.checker = self.model.pose_checker(ik_restarts=6)
        self.session = Session(args.session) if args.session else Session.new(str(HERE.parent / "sessions"))
        self.rng = np.random.default_rng(args.seed)
        self.sensor = PRESETS[args.sensor]
        self.noise = Noise(self.sensor.sigma_f, self.sensor.sigma_t, self.sensor.rate_hz)
        f_cap, t_cap = Robotiq2F85Spec().capacity(args.grip_force)
        self.limits = Limits(workspace_lower=rs.WORKSPACE_LOWER_M - 0.05,
                             workspace_upper=rs.WORKSPACE_UPPER_M + 0.05,
                             min_distance_m=self.model.min_distance_m,
                             hinge_budget_nm=np.full(len(self.spec.joints), 1e-3),   # 라운드마다 _budget() 이 정한다
                             robust_k=args.robust_k, grasp_force_cap_n=f_cap, grasp_torque_cap_nm=t_cap)
        self.opts = dict(period=args.period, harmonics=args.harmonics, samples=args.samples,
                         starts=args.starts, steps=args.steps, top=args.top, poses=1,
                         max_candidates=args.max_candidates)
        self.candidates = candidate_thetas(self.spec, steps=args.steps)
        self.V_all, self.V_static, self.V_dyn = base_basis(self.model, self.spec)
        self.console = None
        self.meshcat = None
        self._setup_view()
        self._setup_hardware()
        self._load_state()

    # -- 화면 --------------------------------------------------------------
    def _setup_view(self):
        if self.args.no_meshcat:
            return
        try:
            from pydrake.geometry import MeshcatVisualizer, StartMeshcat
            from pydrake.systems.framework import DiagramBuilder
            from operator_ui import Console
            builder = DiagramBuilder()
            scene = rs.build_scene(self.spec, builder=builder, include_visuals=True)
            self.meshcat = StartMeshcat()
            MeshcatVisualizer.AddToBuilder(builder, scene["scene_graph"], self.meshcat)
            self.view_diagram = builder.Build()
            self.view_context = self.view_diagram.CreateDefaultContext()
            self.view_plant = scene["plant"]
            self.view_plant_context = self.view_plant.GetMyContextFromRoot(self.view_context)
            self.view_arm_idx = [self.view_plant.GetJointByName(n, scene["arm"]).position_start()
                                 for n in rs.ARM_JOINT_NAMES]
            self.view_obj_idx = [self.view_plant.GetJointByName(j.name, scene["payload"]).position_start()
                                 for j in self.spec.joints]
            self.console = Console(self.meshcat, auto=self.args.auto)
            print(f"  [화면] 계획·로봇 화면 {self.meshcat.web_url()}")
        except Exception as exc:                                  # noqa: BLE001
            print(f"  [주의] Meshcat 을 못 띄웁니다 ({exc}) — 터미널로 갑니다")
            self.meshcat = None

    def show(self, q_arm, theta):
        if self.meshcat is None:
            return
        q = self.view_plant.GetPositions(self.view_plant_context).copy()
        q[self.view_arm_idx] = q_arm
        q[self.view_obj_idx] = np.atleast_1d(theta)
        self.view_plant.SetPositions(self.view_plant_context, q)
        self.view_diagram.ForcedPublish(self.view_context)

    def ask(self, label):
        if self.args.auto:
            print(f"  [자동] {label}")
            return
        if self.console is None:
            input(f"  >>> {label} — Enter")
            return
        name = self.console.button(label)
        self.console.wait_for(name)
        self.console.clear()

    def lamp(self, kind, text):
        if self.console is None:
            print(f"  [{kind}] {text}")
        elif kind == "moving":
            self.console.moving(text)
        elif kind == "stopped":
            self.console.stopped(text)
        else:
            self.console.lamp(__import__("pydrake.geometry", fromlist=["Rgba"]).Rgba(0.85, 0.55, 0.05, 1.0), text)

    # -- 장비 --------------------------------------------------------------
    def _setup_hardware(self):
        a = self.args
        self.theta_true = np.zeros(len(self.spec.joints))
        if a.hardware == "sim":
            densities = a.densities or ([347.0, 442.0, 425.0] if a.object == "3link" else None)
            scale = None if densities is None else (np.asarray(densities, float)
                                                    / np.array([p.rho_gt for p in self.spec.parts]))
            self.phis_true = truth_phis(self.spec, variant=a.variant, density_scale=scale)
            tool_true = self.model.tool_phi() * (0.925 / 0.414)
            self.tool_est = perturb_tool(tool_true, self.rng) if a.tool_error else tool_true.copy()
            self.arm = ArmSim(np.zeros(6), publish=lambda q: self.show(q, self.theta_true))
            self.wrench = WrenchSim(self.model, self.sensor, self.phis_true, tool_true,
                                    lambda: self.theta_true, self.rng)
            self.pose = PoseSim(lambda: self.theta_true, a.angle_rel_error, a.angle_floor_deg, self.rng)
            self.total_mass = sum(p[0] for p in self.phis_true)
        else:
            import hardware_real as hr
            self.phis_true = None
            if not a.tool_file:
                raise SystemExit("--tool-file 이 필요합니다 (dynex.session --identify-tool 로 먼저 만드세요)")
            self.tool_est = np.asarray(json.loads(Path(a.tool_file).read_text())["phi_tool"], float)
            backend = hr.RbpodoBackend(host=a.robot_host)
            self.arm = hr.Rb5Driver(backend)
            self.wrench = hr.Aft200Stream(a.aft_host, a.aft_hz)          # 스트리밍 + 호스트 시각
            self.pose = hr.FoundationPoseSensor(*hr.pose_file_reader(a.pose_file, a.pose_keys),
                                                n_joint=len(self.spec.joints))
            self.total_mass = a.total_mass_kg
            if self.total_mass is None:
                raise SystemExit("--total-mass-kg (저울) 이 필요합니다")

    # -- 상태 --------------------------------------------------------------
    def _load_state(self):
        path = self.session.path("dynex_state.npz")
        if path.is_file():
            d = np.load(path)
            self.est = GaussianEstimate(d["mean"], d["cov"], int(d["n_bias"]))
            self.round_done = int(d["round"])
            self.certified = d["certified"] if "certified" in d else np.zeros(len(self.spec.joints))
            self.cap = d["cap"] if "cap" in d else np.full(len(self.spec.joints), np.inf)
            print(f"  사후분포 이어받음: 라운드 {self.round_done} 까지")
        else:
            bf, bt = self.args.bias_sigma
            self.est = prior_from_spec(self.spec, total_mass_kg=self.total_mass,
                                       mass_rel_std=0.5, com_frac_std=0.3, inertia_rel_std=0.5,
                                       bias_sigma=(bf, bf, bf, bt, bt, bt))
            self.est = apply_total_mass(self.est, self.total_mass, 0.002)
            self.round_done = 0
            self.certified = np.zeros(len(self.spec.joints))      # 버틴 것으로 확인된 최대 힌지 토크
            self.cap = np.full(len(self.spec.joints), np.inf)     # 미끄러진 토크 (상한)
        self._budget()

    def _save_state(self, r):
        np.savez(self.session.path("dynex_state.npz"), mean=self.est.mean, cov=self.est.cov,
                 n_bias=self.est.n_bias, round=r, certified=self.certified, cap=self.cap)

    def _budget(self):
        """이번 라운드에 허용할 힌지 축 토크 예산 (관절별).

        자동 모드: 이미 버틴 최대값 × growth, 최소 floor. 미끄러진 적이 있으면 그 값 × 0.8 아래.
        토크 게이지 값이 있으면 그 값/안전계수를 넘지 않는다.
        """
        a = self.args
        if a.hinge_torque is not None:
            # 토크 게이지 값이 있으면 그 값/안전계수를 바로 믿는다 (미끄러지면 상한만 내린다)
            b = np.full(len(self.spec.joints), a.hinge_torque / a.safety)
        else:
            b = np.maximum(self.certified, a.hinge_floor_nm) * a.hinge_growth   # 버틴 값(최소 floor)의 growth 배
        b = np.minimum(b, self.cap * 0.8)
        self.limits.hinge_budget_nm = b
        return b

    def _slipped(self, theta_ref):
        """궤적·자세 뒤 카메라 각도가 허용치보다 움직였으면 관절별 True."""
        theta_after, sigma = self.read_theta()
        drift = np.degrees(np.abs(theta_after - theta_ref))
        # 두 판독 모두 잡음이 섞이므로 허용치는 카메라 잡음의 3배 이상이어야 한다
        tol = np.maximum(self.args.slip_tol_deg, 3.0 * np.degrees(sigma) * np.sqrt(2.0))
        return drift > tol, theta_after

    def _sim_slip(self, peak_true):
        """모의: 참 유지토크를 넘긴 관절을 몇 도 밀어 미끄러짐을 흉내낸다."""
        for i, tau in enumerate(np.atleast_1d(peak_true)):
            if tau > self.args.sim_hinge_torque:
                self.theta_true[i] += np.deg2rad(self.args.slip_drift_deg) * self.rng.choice([-1.0, 1.0])

    # -- 한 라운드 ------------------------------------------------------------
    def recommend(self):
        self._budget()                      # 형상 선택도 지금 예산 안에서
        ranked = rank_candidates(self.model, self.checker, self._subset(), self.est, self.noise,
                                 self.limits, self.opts, self.rng, verbose=True)
        if not ranked:
            return None
        score, theta, poses = ranked[0]
        return theta, poses, score

    def _subset(self):
        cap = self.opts.get("max_candidates")
        if cap and len(self.candidates) > cap:
            idx = self.rng.choice(len(self.candidates), size=cap, replace=False)
            return [self.candidates[i] for i in idx]
        return self.candidates

    def read_theta(self):
        deg, sigma = self.pose.object_joint_deg()
        return np.deg2rad(np.atleast_1d(deg)), np.deg2rad(np.atleast_1d(sigma))

    def hold_and_sample(self, q, theta, hold_s, writer, t_offset):
        """정지 자세에서 hold_s 동안 샘플을 모아 (Yp, y) 를 만든다."""
        n = int(round(self.sensor.rate_hz * hold_s))
        d = ExcitationDesigner(self.model, theta, q, self.est, self.noise, self.limits,
                               period_s=hold_s, n_harmonics=2, n_samples=4)
        Yp, y = [], []
        zero = np.zeros(6)
        for k in range(n):
            if self.args.hardware == "sim":
                raw = self.wrench.read_at(q, zero, zero)
            else:
                raw = self.wrench.read_one()
            kin = self.model.evaluate(q, zero, zero, theta)
            Y = rigid_body_regressor(kin["a_o"], kin["omega"], kin["alpha"], kin["g"])
            Yp.append(Y @ d.Mstack)
            y.append(raw - Y @ self.tool_est)
            writer.writerow([t_offset + k / self.sensor.rate_hz, *q, *raw, *(raw - Y @ self.tool_est)])
        return np.vstack(Yp), np.concatenate(y), n

    def run_round(self, r):
        a = self.args
        t0 = time.time()
        self.session.set_phase("angle", r)
        # 1) 각도: 추천(전 라운드가 남긴 것)이 있으면 그것, 없으면 지금 자세
        rec = self.session.read(f"angle_round_{r}.json") or {}
        if "recommended_deg" in rec:
            theta_cmd = np.deg2rad(rec["recommended_deg"])
        else:
            pick = self.recommend()
            if pick is None:
                print(f"  실현 가능한 형상이 없습니다 (힌지 예산 {np.round(self._budget(), 3)} N·m) —"
                      f" 힌지가 최소 예산도 못 버티거나 도달 가능한 자세가 없습니다. 라운드 중단")
                return False
            theta_cmd = pick[0]
            rec["recommended_deg"] = np.degrees(theta_cmd).tolist()
        self.lamp("stopped", f"관절을 {np.round(np.degrees(theta_cmd), 1)} deg 로 맞추세요 (round {r})")
        if a.hardware == "sim":
            sigma = np.maximum(a.angle_rel_error * np.abs(theta_cmd), np.deg2rad(a.angle_floor_deg))
            self.theta_true = theta_cmd + self.rng.normal(0.0, sigma)     # 작업자가 맞춘 실제 각도
        self.ask("각도 조정 완료")
        theta, theta_sigma = self.read_theta()
        rec.update(measured_deg=np.degrees(theta).tolist(), measured_sigma_deg=np.degrees(theta_sigma).tolist())
        self.session.write(f"angle_round_{r}.json", rec)
        print(f"[round {r}] 측정 각도 {np.round(np.degrees(theta), 2)} deg")

        # 2) 경로: 정적 3자세 IK + 여기 궤적 설계
        self.session.set_phase("path", r)
        poses = start_poses(self.model, self.checker, theta)
        if not poses:
            print("  이 각도에서 도달 가능한 자세가 없습니다 — 라운드 중단")
            return False
        wrench_path = self.session.path(f"wrench_round_{r}.csv")
        fh = open(wrench_path, "w", newline="")
        writer = csv.writer(fh)
        writer.writerow(["t"] + [f"q{i}" for i in range(6)] + [f"raw{i}" for i in range(6)] + [f"obj{i}" for i in range(6)])
        t_clock = 0.0
        # 정적 3방향 먼저 (사후분포를 좁혀야 힌지 예산의 보수적 경계가 풀린다).
        # 예산 안의 자세만, 예측 토크가 작은 것부터. 자세마다 카메라로 버텼는지 확인한다.
        self.session.set_phase("explore", r)
        budget = self._budget()
        print(f"  힌지 예산 {np.round(budget, 3)} N·m (인증 {np.round(self.certified, 3)},"
              f" 상한 {np.round(self.cap, 3)})")
        predicted = []
        for q in poses:
            d0 = ExcitationDesigner(self.model, theta, q, self.est, self.noise, self.limits,
                                    period_s=a.hold_s, n_harmonics=2, n_samples=4)
            ev0 = d0.evaluate(np.zeros(d0.n_free), times=np.array([0.0]))
            predicted.append((ev0["hinge_bound"][0], np.abs(ev0["hinge_mean"][0]), q))
        predicted.sort(key=lambda t: t[0].max())
        used_poses = []
        for k, (bound, mean_t, q) in enumerate(predicted):
            if (bound > budget).any():
                print(f"  정적 자세 {k + 1}: 예측 힌지 토크 {np.round(bound, 3)} 가 예산 밖 — 건너뜀")
                continue
            self.lamp("moving", f"정적 자세 {k + 1}/{len(predicted)} 로 이동")
            self.arm.move_to(q, a.move_duration)
            self.show(q, theta if a.hardware == "real" else self.theta_true)
            time.sleep(a.settle_s if a.hardware == "real" else 0.0)
            if a.hardware == "sim":
                Mt = np.hstack(self.model.part_transports(self.theta_true))
                kin0 = self.model.evaluate(q, np.zeros(6), np.zeros(6), self.theta_true)
                Y0 = rigid_body_regressor(kin0["a_o"], kin0["omega"], kin0["alpha"], kin0["g"])
                from .regressor import hinge_axis_row
                peak_true = np.array([abs((hinge_axis_row(Y0, rh, ah) @ Mt * mask) @ np.concatenate(self.phis_true))
                                      for (rh, ah), mask in zip(self.model.hinge_geometry(self.theta_true), d0.down_mask)])
                self._sim_slip(peak_true)
            Yp, y, n = self.hold_and_sample(q, theta, a.hold_s, writer, t_clock)
            t_clock += n / self.sensor.rate_hz
            slipped, theta_after = self._slipped(theta)
            if slipped.any():
                # 미끄러진 토크는 '시도한 예산' 이하라는 것만 안다 (예측 평균이 0 에 가까워도)
                self.cap = np.where(slipped, np.minimum(self.cap, np.maximum(mean_t, budget)), self.cap)
                print(f"  [주의] 정적 자세 {k + 1} 에서 관절 {np.where(slipped)[0] + 1} 이 움직였습니다"
                      f" (예측 토크 {np.round(mean_t, 3)}) — 이 자세 데이터는 버리고 상한을 내립니다")
                theta = theta_after                      # 각도가 바뀌었으니 이후는 새 각도로
                budget = self._budget()
                continue
            self.certified = np.maximum(self.certified, mean_t)
            self.est = update(self.est, Yp, y, np.tile(self.noise.r_diag6, n))
            used_poses.append(q)
        poses = used_poses or poses[:1]
        budget = self._budget()
        # 여기 궤적 설계 (좁아진 사후분포로)
        pick = design_at(self.model, theta, poses, self.est, self.noise, self.limits, self.opts,
                         self.rng, verbose=True)
        if pick is None:
            print("  여기 궤적을 못 만들었습니다 — 정적 측정만으로 갱신")
            d, x, ver = None, None, dict(ig=0.0, hinge_peak_ratio=0.0, qd_peak_deg=0.0,
                                         qdd_peak_deg=0.0, min_distance_m=0.0,
                                         grasp_torque_peak_nm=0.0)
        else:
            d, x, ver = pick["designer"], pick["x"], pick["verify"]
            ts = np.arange(0.0, d.traj.T, 1.0 / self.sensor.rate_hz)
            q_path, _, _ = d.traj.evaluate(x, ts)
            self.session.write(f"path_round_{r}.json", dict(
                static_poses_rad=[np.asarray(p).tolist() for p in poses],
                excitation=dict(period_s=d.traj.T, rate_hz=self.sensor.rate_hz, q0=np.asarray(pick["q0"]).tolist(),
                                coefficients=np.asarray(x).tolist(), waypoints_rad=q_path.tolist()),
                predicted=dict(ig_nat=ver["ig"], hinge_peak_ratio=ver["hinge_peak_ratio"],
                               qd_peak_deg=ver["qd_peak_deg"], qdd_peak_deg=ver["qdd_peak_deg"],
                               min_distance_mm=1000 * ver["min_distance_m"],
                               grasp_torque_peak_nm=ver["grasp_torque_peak_nm"])))
            print(f"  여기 궤적: 정보이득 {ver['ig']:.1f} nat, 힌지 {ver['hinge_peak_ratio']:.2f}×예산,"
                  f" qd {ver['qd_peak_deg']:.0f} deg/s, 간격 {1000 * ver['min_distance_m']:.1f} mm")
            # 미리보기
            if self.meshcat is not None:
                for q in q_path[::10]:
                    self.show(q, theta)
                    time.sleep(0.02)
            self.ask("여기 궤적 승인 — 로봇이 8초간 움직입니다")
            # 3) 실행 + 샘플
            self.lamp("moving", "여기 궤적 실행 중")
            self.arm.move_to(pick["q0"], a.move_duration)
            dt = 1.0 / self.sensor.rate_hz
            if a.hardware == "sim":
                samples = self.arm.stream(lambda t: tuple(v[0] for v in d.traj.evaluate(x, [t])), d.traj.T, dt)
                Yp, y = [], []
                from .regressor import hinge_axis_row
                Mt = np.hstack(self.model.part_transports(self.theta_true))
                hinges_t = self.model.hinge_geometry(self.theta_true)
                Phi_t = np.concatenate(self.phis_true)
                peak_true = np.zeros(len(hinges_t))
                for t, q in samples:
                    q_, qd_, qdd_ = (v[0] for v in d.traj.evaluate(x, [t]))
                    raw = self.wrench.read_at(q_, qd_, qdd_)
                    kin = self.model.evaluate(q_, qd_, qdd_, theta)
                    Y = rigid_body_regressor(kin["a_o"], kin["omega"], kin["alpha"], kin["g"])
                    Yp.append(Y @ d.Mstack)
                    y.append(raw - Y @ self.tool_est)
                    writer.writerow([t_clock + t, *q_, *raw, *(raw - Y @ self.tool_est)])
                    for i, ((rh, ah), mask) in enumerate(zip(hinges_t, d.down_mask)):
                        peak_true[i] = max(peak_true[i], abs((hinge_axis_row(Y, rh, ah) @ Mt * mask) @ Phi_t))
                Yp, y = np.vstack(Yp), np.concatenate(y)
                self._sim_slip(peak_true)
            else:
                Yp, y = self._execute_real(d, x, theta, writer, t_clock)
            t_clock += d.traj.T
            self.lamp("stopped", "여기 궤적 종료")
            slipped, theta_after = self._slipped(theta)
            peak_mean = np.abs(ver["ev"]["hinge_mean"]).max(axis=0)
            if slipped.any():
                self.cap = np.where(slipped, np.minimum(self.cap, np.maximum(peak_mean, budget)), self.cap)
                print(f"  [주의] 궤적 뒤 관절 {np.where(slipped)[0] + 1} 이 움직였습니다 — 데이터를 버리고"
                      f" 힌지 상한을 {np.round(self.cap, 3)} N·m 로 내립니다")
                ver = dict(ver, ig=0.0, slipped=True)
            else:
                self.certified = np.maximum(self.certified, peak_mean)
                self.est = update(self.est, Yp, y, np.tile(self.noise.r_diag6, Yp.shape[0] // 6))
        fh.close()
        self.est, projected = project_physical(self.est)

        # 4) 기록
        unc = [part_uncertainty(self.est, k) for k in range(self.model.n_parts)]
        base = self._base_sigma()
        converged = bool(base["static_sd_med"] < a.target_static and base["dyn_sd_med"] < a.target_dyn)
        parts = []
        for k, p in enumerate(self.spec.parts):
            mu = self.est.mean[10 * k:10 * (k + 1)]
            parts.append(dict(name=p.name, mass_kg=float(mu[0]), com_m=com(mu).tolist(),
                              inertia_central=central_inertia(mu).tolist(),
                              sigma=dict(mass_rel=unc[k]["mass_rel"], com_mm=unc[k]["com_mm"],
                                         inertia_rel=unc[k]["inertia_rel"])))
            if self.phis_true is not None:
                from .identify import part_errors
                parts[-1]["error_vs_truth"] = {kk: (vv.tolist() if hasattr(vv, "tolist") else float(vv))
                                               for kk, vv in part_errors(mu, self.phis_true[k]).items()}
        rec_post = dict(round=r, theta_deg=np.degrees(theta).tolist(), parts=parts, base=base,
                        converged=converged, projected=projected, ig=ver["ig"],
                        hinge_peak_ratio=ver["hinge_peak_ratio"], slipped=bool(ver.get("slipped", False)),
                        hinge_budget_nm=np.asarray(budget).tolist(), hinge_certified_nm=self.certified.tolist(),
                        hinge_cap_nm=[None if not np.isfinite(c) else float(c) for c in self.cap],
                        seconds=time.time() - t0)
        self.session.write(f"posterior_round_{r}.json", rec_post)
        self._save_state(r)
        # 다음 라운드 추천 각도
        if not converged:
            pick = self.recommend()
            if pick is not None:
                self.session.write(f"angle_round_{r + 1}.json", dict(recommended_deg=np.degrees(pick[0]).tolist()))
        print(f"[round {r}] σ(정적조합) {100 * base['static_sd_med']:.2f}%  σ(관성조합) {100 * base['dyn_sd_med']:.0f}%"
              f"  힌지 인증 {np.round(self.certified, 3)} N·m  {'수렴' if converged else '계속'}  {time.time() - t0:.0f}s")
        for p in parts:
            e = p.get("error_vs_truth")
            extra = f"  오차: 질량 {e['mass_pct']:.1f}% 무게중심 {e['com_mm']:.1f} mm 관성 {e['inertia_pct']:.0f}%" if e else ""
            print(f"    {p['name']:<12} m {1000 * p['mass_kg']:6.1f} g  σ {100 * p['sigma']['mass_rel']:.1f}%{extra}")
        return converged

    def _base_sigma(self):
        n = self.est.n_phi
        cov = self.est.cov[:n, :n]
        out = {}
        for name, V in (("static", self.V_static), ("dyn", self.V_dyn)):
            sd = np.sqrt(np.einsum("ij,jk,ik->i", V.T, cov, V.T))
            ref = np.abs(V.T @ self.est.mean[:n])
            out[f"{name}_sd_med"] = float(np.median(sd / np.maximum(ref, 1e-9)))
        return out

    def _execute_real(self, d, x, theta, writer, t_clock):
        """실물: 궤적 스트리밍 + 렌치 스트리밍, 시각 정렬 후 회귀행렬 (미검증)."""
        import hardware_real as hr
        joints = self.arm.stream(lambda t: tuple(v[0] for v in d.traj.evaluate(x, [t])), d.traj.T,
                                 self.args.servo_dt)
        wrenches = self.wrench.drain()                                  # [(t_host, raw6)]
        return hr.align_and_regress(self, d, joints, wrenches, theta, writer, t_clock)

    def export(self):
        out = self.session.path("export")
        out.mkdir(exist_ok=True)
        n = self.est.n_phi
        null = np.linalg.svd(self.V_all.T, full_matrices=True)[2][self.V_all.shape[1]:]
        asset = dict(object=self.spec.key, sensor=self.args.sensor, rounds=self.round_done,
                     parts=[], invisible_combinations=null.tolist(),
                     note="invisible_combinations 방향의 값은 측정이 아니라 사전분포(균일 밀도, 힌지는 자식 링크 핀 축)로 정해졌다. 이 방향은 시뮬레이션 결과를 바꾸지 않는다.")
        for k, p in enumerate(self.spec.parts):
            mu = self.est.mean[10 * k:10 * (k + 1)]
            asset["parts"].append(dict(name=p.name, mass_kg=float(mu[0]), com_m=com(mu).tolist(),
                                       inertia_central=central_inertia(mu).tolist()))
        (out / "asset_dynamic.json").write_text(json.dumps(asset, indent=1))
        print(f"  자산 → {out / 'asset_dynamic.json'}")

    def run(self):
        r = self.round_done + 1 if self.args.round is None else self.args.round
        last = r + (self.args.rounds - 1)
        while r <= last:
            done = self.run_round(r)
            self.round_done = r
            if done:
                break
            r += 1
        self.export()
        return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--object", default="3link", choices=tuple(obj.OBJECTS))
    ap.add_argument("--hardware", choices=("sim", "real"), default="sim")
    ap.add_argument("--sensor", default="datasheet_100hz", choices=tuple(PRESETS))
    ap.add_argument("--session", default=os.environ.get("PIVOT_SESSION"))
    ap.add_argument("--round", type=int, default=None, help="이 라운드 하나만 (pivot_ui 가 라운드마다 부른다)")
    ap.add_argument("--rounds", type=int, default=1, help="이어서 돌 라운드 수")
    ap.add_argument("--auto", action="store_true")
    ap.add_argument("--no-meshcat", action="store_true")
    ap.add_argument("--hinge-torque", type=float, default=None,
                    help="힌지 유지토크 [N·m] (토크 게이지). 없으면 '버틴 만큼만' 자동 모드")
    ap.add_argument("--hinge-growth", type=float, default=1.5,
                    help="자동 모드: 이미 버틴 최대 토크의 몇 배까지 다음 라운드에 시도하나")
    ap.add_argument("--hinge-floor-nm", type=float, default=0.05,
                    help="자동 모드: 아무것도 버틴 적 없을 때 허용하는 최소 예산")
    ap.add_argument("--sim-hinge-torque", type=float, default=0.5,
                    help="모의 장비의 참 유지토크. 넘으면 관절이 미끄러진 것으로 흉내낸다")
    ap.add_argument("--slip-drift-deg", type=float, default=10.0)
    ap.add_argument("--safety", type=float, default=1.5)
    ap.add_argument("--robust-k", type=float, default=2.0)
    ap.add_argument("--grip-force", type=float, default=205.0)
    ap.add_argument("--period", type=float, default=8.0)
    ap.add_argument("--harmonics", type=int, default=4)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--starts", type=int, default=1)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--top", type=int, default=1)
    ap.add_argument("--max-candidates", type=int, default=20)
    ap.add_argument("--hold-s", type=float, default=2.0)
    ap.add_argument("--move-duration", type=float, default=8.0)
    ap.add_argument("--settle-s", type=float, default=0.5)
    ap.add_argument("--slip-tol-deg", type=float, default=2.0)
    ap.add_argument("--target-static", type=float, default=0.01, help="정적 조합 σ 목표 (비율)")
    ap.add_argument("--target-dyn", type=float, default=0.30, help="관성 조합 σ 목표 (비율)")
    ap.add_argument("--bias-sigma", type=float, nargs=2, default=(0.2, 0.01))
    ap.add_argument("--seed", type=int, default=0)
    # 모의 장비
    ap.add_argument("--variant", default="shell", choices=("uniform", "shell", "insert"))
    ap.add_argument("--densities", type=float, nargs="+", default=None)
    ap.add_argument("--tool-error", type=int, default=1)
    ap.add_argument("--angle-rel-error", type=float, default=0.01, help="모의 추적기 상대오차 (논문 C0 조건)")
    ap.add_argument("--angle-floor-deg", type=float, default=2.0, help="모의 추적기 절대오차 하한 (체크리스트 실측 1~3°)")
    # 실물
    ap.add_argument("--tool-file", default=None)
    ap.add_argument("--total-mass-kg", type=float, default=None)
    ap.add_argument("--pose-file", default=None)
    ap.add_argument("--pose-keys", nargs="+", default=None)
    ap.add_argument("--robot-host", default="192.168.50.51")
    ap.add_argument("--aft-host", default="192.168.50.51")
    ap.add_argument("--aft-hz", type=float, default=100.0)
    ap.add_argument("--servo-dt", type=float, default=0.005)
    args = ap.parse_args()
    return DynexSession(args).run()


if __name__ == "__main__":
    sys.exit(main())
