"""실사양 센서로 측정을 흉내낸다.

참값 물체는 부위마다 균일 직육면체 + (선택) 셸 도심 이동 + (선택) 삽입 추 +
힌지 점질량으로 만든다. 센서는 AFT200 데이터시트 값(잡음 0.4 N / 0.025 N·m,
분해능 0.15 N / 0.015 N·m)으로 두고, 기록 속도는 지금 팀 드라이버의 50 Hz 와
센서 자체의 1 kHz 를 모두 시험한다. 툴(그리퍼)은 참값과 조금 다른 추정값으로
빼서, 툴 식별 오차가 물체 추정에 어떻게 새는지 본다.
"""
from dataclasses import dataclass, field

import numpy as np

from .inertial import point_mass_phi, solid_box_phi
from .regressor import hinge_axis_row, rigid_body_regressor


@dataclass
class SensorModel:
    sigma_f: float = 0.4
    sigma_t: float = 0.025
    res_f: float = 0.15
    res_t: float = 0.015
    rate_hz: float = 50.0
    jitter_s: float = 0.0             # F/T 와 관절 상태 사이 시각 어긋남 (균등 ±)
    bias: np.ndarray = field(default_factory=lambda: np.array([1.5, -0.8, 2.0, 0.03, -0.02, 0.01]))

    def quantize(self, w):
        out = w.copy()
        if self.res_f > 0:
            out[:3] = np.round(out[:3] / self.res_f) * self.res_f
        if self.res_t > 0:
            out[3:] = np.round(out[3:] / self.res_t) * self.res_t
        return out


PRESETS = {
    "datasheet_50hz": SensorModel(),
    "datasheet_1khz": SensorModel(rate_hz=1000.0),
    "datasheet_100hz": SensorModel(rate_hz=100.0),      # 팀이 Modbus 로 확인한 폴링 속도
    "datasheet_200hz": SensorModel(rate_hz=200.0),
    "paper": SensorModel(sigma_f=0.10, sigma_t=0.003, res_f=0.0, res_t=0.0, rate_hz=1000.0),
}


def truth_phis(spec, variant="uniform", insert_frac=0.3, insert_mass_frac=0.0,
               density_scale=None):
    """부위별 참값 phi (각 부위 몸체 프레임 = 외형 중심)."""
    parts = spec.parts
    hinge_add = {p.name: [] for p in parts}
    for joint in spec.joints:
        if joint.hinge_mass_kg > 0.0:
            child = next(p for p in parts if p.name == joint.child)
            origin = np.array(joint.origin_in_child_link_mm
                              if joint.origin_in_child_link_mm is not None
                              else (0.0, 0.0, 0.0))
            pos = (origin - np.array(child.bbox_center_in_link_mm)) * 1e-3
            hinge_add[joint.child].append((joint.hinge_mass_kg, pos))
    out = []
    for k, part in enumerate(parts):
        rho = part.rho_gt if density_scale is None else part.rho_gt * density_scale[k]
        mass = rho * part.volume_m3
        dims = np.array(part.bbox_mm) * 1e-3
        if variant == "shell":
            offset = (np.array(part.shell_centroid_in_link_mm)
                      - np.array(part.bbox_center_in_link_mm)) * 1e-3
            phi = solid_box_phi(mass, dims, offset)
        elif variant == "insert":
            m_ins = insert_mass_frac * mass
            phi = solid_box_phi(mass - m_ins, dims)
            phi = phi + point_mass_phi(m_ins, np.array([insert_frac * dims[0] / 2, 0.0, 0.0]))
        else:
            phi = solid_box_phi(mass, dims)
        for m_h, pos in hinge_add[part.name]:
            phi = phi + point_mass_phi(m_h, pos)
        out.append(phi)
    return out


def perturb_tool(phi_tool, rng, mass_rel=0.02, com_mm=1.0, inertia_rel=0.05):
    """툴 식별 오차: 질량 2 %, 무게중심 1 mm, 관성 5 % 수준."""
    m, h, I = phi_tool[0], phi_tool[1:4].copy(), phi_tool[4:10].copy()
    m2 = m * (1.0 + rng.normal(0.0, mass_rel))
    c = h / m + rng.normal(0.0, com_mm * 1e-3, 3)
    I2 = I * (1.0 + rng.normal(0.0, inertia_rel, 6))
    return np.concatenate([[m2], m2 * c, I2])


def execute_and_measure(model, designer, x, phis_true, tool_true, tool_est,
                        sensor, rng, tracking_lag_s=0.0, theta_true=None):
    """설계 궤적을 '실행' 하고 센서 샘플을 만든다.

    돌려주는 것: 추정용 (Yp_stack, y_stack, r_diag) 와 진단(힌지 축 토크 참값
    시계열, 힌지 최대비, 렌치 최대).
    """
    traj = designer.traj
    T = traj.T
    n = int(round(sensor.rate_hz * T))
    times = (np.arange(n) + 0.5) / sensor.rate_hz
    Phi_true = np.concatenate(phis_true)
    # 추정기는 designer.theta(측정된 각도)를 믿고, 참 렌치는 실제 각도로 만든다.
    if theta_true is None:
        Mstack_t, hinges_t = designer.Mstack, designer.hinges
    else:
        Mstack_t = np.hstack(model.part_transports(np.atleast_1d(theta_true)))
        hinges_t = model.hinge_geometry(np.atleast_1d(theta_true))
    Yp_stack = np.zeros((6 * n, designer.P))
    y_stack = np.zeros(6 * n)
    hinge_true = np.zeros((n, len(designer.hinges)))
    wrench_true = np.zeros((n, 6))
    for k, t in enumerate(times):
        t_true = t + (rng.uniform(-sensor.jitter_s, sensor.jitter_s) if sensor.jitter_s else 0.0)
        q, qd, qdd = traj.evaluate(x, [t_true])
        if tracking_lag_s > 0.0:
            # 1차 지연 서보: 참 궤적은 지령보다 조금 늦고 부드럽다 (근사)
            q2, qd2, qdd2 = traj.evaluate(x, [t_true - tracking_lag_s])
            q, qd, qdd = q2, qd2, qdd2
        kin = model.evaluate(q[0], qd[0], qdd[0], designer.theta)
        Y = rigid_body_regressor(kin["a_o"], kin["omega"], kin["alpha"], kin["g"])
        w = Y @ (Mstack_t @ Phi_true + tool_true) + sensor.bias
        wrench_true[k] = w
        for i, ((r_h, a_hat), mask) in enumerate(zip(hinges_t, designer.down_mask)):
            hinge_true[k, i] = (hinge_axis_row(Y, r_h, a_hat) @ Mstack_t * mask) @ Phi_true
        noise = np.concatenate([rng.normal(0, sensor.sigma_f, 3), rng.normal(0, sensor.sigma_t, 3)])
        y = sensor.quantize(w + noise)
        # 추정기는 지령 시각의 운동학을 믿는다
        q, qd, qdd = traj.evaluate(x, [t])
        kin_r = model.evaluate(q[0], qd[0], qdd[0], designer.theta)
        Yr = rigid_body_regressor(kin_r["a_o"], kin_r["omega"], kin_r["alpha"], kin_r["g"])
        Yp_stack[6 * k:6 * k + 6] = Yr @ designer.Mstack
        y_stack[6 * k:6 * k + 6] = y - Yr @ tool_est
    r_diag = np.tile(np.array([sensor.sigma_f ** 2] * 3 + [sensor.sigma_t ** 2] * 3), n)
    return dict(Yp=Yp_stack, y=y_stack, r_diag=r_diag, times=times,
                hinge_true=hinge_true, wrench_true=wrench_true,
                hinge_peak_nm=np.abs(hinge_true).max(axis=0) if hinge_true.size else np.zeros(0))
