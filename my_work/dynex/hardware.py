"""RB5-850E / AFT200-D80 / Robotiq 2F-85 의 한계와 실현 가능성 보고서.

숫자의 출처를 함께 적는다. '가정' 이라고 표시한 값은 실측으로 바꿔야 한다.
"""
from dataclasses import dataclass

import numpy as np

from .regressor import hinge_axis_row, rigid_body_regressor


@dataclass
class RB5Spec:
    joints: int = 6
    payload_kg: float = 5.0                       # 제조사
    qd_max_deg: float = 180.0                     # 저장소 URDF (velocity=3.14 rad/s). 제조사 표 미확인
    qdd_max_deg: float = 300.0                    # 가정 — 서보 스트리밍으로 실측할 것
    servo_period_s: float = 0.005                 # rbpodo 예제 move_servo_j 5 ms 주기
    state_fields: tuple = ("time", "jnt_ang", "jnt_ref", "jnt_cur")  # rbpodo SystemState
    joint_torque_sensor: bool = False             # 전류만 있다 (관절 토크 센서 없음)


@dataclass
class AFT200Spec:
    range_f_n: float = 200.0
    range_t_nm: float = 15.0
    res_f_n: float = 0.15                         # 데이터시트 분해능
    res_t_nm: float = 0.015
    noise_f_n: float = 0.4                        # 데이터시트 noise-free resolution (STD)
    noise_t_nm: float = 0.025
    internal_rate_hz: float = 1000.0
    current_interface: str = "Modbus TCP 폴링 50 Hz, 타임스탬프 없음 (aft_tare.py)"
    imu: str = "EtherCAT 판에 가속도계·자이로 내장 (Modbus 노출 여부 미확인)"


@dataclass
class Robotiq2F85Spec:
    mass_kg: float = 0.925                        # 제조사 (URDF 는 0.414 — 저평가)
    grip_force_n: tuple = (20.0, 235.0)
    stroke_m: float = 0.085
    payload_kg: float = 5.0
    mu: float = 0.5                               # 가정 — 패드/PLA 마찰계수
    pad_radius_m: float = 0.010                   # 가정 — 패드 접촉 반경

    def capacity(self, grip_force_n):
        f_cap = 2.0 * self.mu * grip_force_n
        t_cap = 2.0 * self.mu * grip_force_n * self.pad_radius_m
        return f_cap, t_cap


def hinge_sensitivity(designer, phis_true):
    """힌지 축 토크가 단위 병진가속(3축)·단위 각가속(3축)에 얼마나 민감한가.

    중력·각속도 0 에서 계산한다. 0 에 가까운 방향이 '힌지에 하중을 안 주는'
    자유 운동이다.
    """
    Phi = np.concatenate(phis_true)
    rows = []
    for (r_h, a_hat), mask in zip(designer.hinges, designer.down_mask):
        sens = np.zeros(6)
        for k in range(3):
            e = np.zeros(3)
            e[k] = 1.0
            Y = rigid_body_regressor(e, np.zeros(3), np.zeros(3), np.zeros(3))
            sens[k] = (hinge_axis_row(Y, r_h, a_hat) @ designer.Mstack * mask) @ Phi
            Y = rigid_body_regressor(np.zeros(3), np.zeros(3), e, np.zeros(3))
            sens[3 + k] = (hinge_axis_row(Y, r_h, a_hat) @ designer.Mstack * mask) @ Phi
        rows.append(sens)
    return np.array(rows)


def report(designer, phis_true, tool_true, sensor_presets, budget_nm, grip_force_n=100.0):
    rb5, aft, grip = RB5Spec(), AFT200Spec(), Robotiq2F85Spec()
    lines = []
    P = lines.append
    P("=== 하드웨어 조건 보고서 ===")
    P(f"RB5-850E : 6축, 관절속도 한계 {rb5.qd_max_deg:.0f} deg/s (URDF), 가속 {rb5.qdd_max_deg:.0f} deg/s² (가정),"
      f" 서보 스트리밍 move_servo_j {1000 * rb5.servo_period_s:.0f} ms 주기 (rbpodo 예제), 관절 토크 센서 없음")
    P(f"AFT200   : {aft.range_f_n:.0f} N / {aft.range_t_nm:.0f} N·m, 분해능 {aft.res_f_n} N / {aft.res_t_nm} N·m,"
      f" 잡음 STD {aft.noise_f_n} N / {aft.noise_t_nm} N·m, 내부 {aft.internal_rate_hz:.0f} Hz")
    P(f"           현재 인터페이스: {aft.current_interface}")
    P(f"           IMU: {aft.imu}")
    f_cap, t_cap = grip.capacity(grip_force_n)
    P(f"2F-85    : {grip.mass_kg} kg, 파지력 {grip.grip_force_n[0]:.0f}~{grip.grip_force_n[1]:.0f} N,"
      f" 파지력 {grip_force_n:.0f} N 일 때 마찰 용량 ≈ 힘 {f_cap:.0f} N, 토크 {t_cap:.2f} N·m (μ={grip.mu} 가정)")
    m_obj = sum(p[0] for p in phis_true)
    P(f"물체 {1000 * m_obj:.0f} g vs 툴(센서 아래) {1000 * tool_true[0]:.0f} g → 툴/물체 질량비 {tool_true[0] / m_obj:.2f}."
      f" 툴 질량 오차 1 % 가 물체 질량에 {100 * tool_true[0] / m_obj * 0.01:.2f} % 로 샌다")
    sens = hinge_sensitivity(designer, phis_true)
    P("힌지 축 토크 민감도 [N·m per 1 m/s² 병진 (x y z) | per 1 rad/s² 각가속 (x y z)], S 프레임:")
    for i, s in enumerate(sens):
        with np.errstate(divide="ignore"):
            a_allow = budget_nm[i] / np.maximum(np.abs(s[:3]), 1e-12)
            al_allow = budget_nm[i] / np.maximum(np.abs(s[3:]), 1e-12)
        P(f"  관절 {i + 1}: 병진 {np.round(s[:3], 4)}  각가속 {np.round(s[3:], 5)}")
        P(f"           예산 {budget_nm[i]:.3f} N·m → 허용 병진가속 {np.round(np.minimum(a_allow, 99), 2)} m/s²,"
          f" 허용 각가속 {np.round(np.minimum(al_allow, 999), 1)} rad/s²  (∞에 가까운 축 = 자유 운동)")
    # 정적 중력 토크 (시작 자세)
    ok, ev0 = designer.static_feasible()
    P(f"시작 자세 정지 힌지 토크 {np.round(np.abs(ev0['hinge_mean'][0]), 4)} N·m, 예산 {np.round(budget_nm, 3)}"
      f" → 동적 여유 {np.round(budget_nm - np.abs(ev0['hinge_mean'][0]), 4)} N·m")
    # 신호 크기 대 잡음
    x = designer.last[0] if designer.last is not None else np.zeros(designer.n_free)
    ev = designer.evaluate(x, times=designer.traj.times(200))
    Phi = np.concatenate(phis_true)
    w = (ev["Yp"] @ Phi).reshape(-1, 6)
    # 관성 텐서 열만 남긴 몫
    Yp_I = ev["Yp"].copy()
    for k in range(designer.n_parts):
        Yp_I[:, 10 * k:10 * k + 4] = 0.0
    w_I = (Yp_I @ Phi).reshape(-1, 6)
    P(f"설계 궤적의 물체 렌치 최대: 힘 {np.abs(w[:, :3]).max():.2f} N, 토크 {np.abs(w[:, 3:]).max():.3f} N·m;"
      f" 그중 관성 텐서 항만: 힘 {np.abs(w_I[:, :3]).max():.3f} N, 토크 {np.abs(w_I[:, 3:]).max():.4f} N·m"
      f"  (센서 토크 분해능 {aft.res_t_nm} N·m, 잡음 STD {aft.noise_t_nm} N·m)")
    P(f"궤적 최고 주파수 {designer.traj.K / designer.traj.T:.2f} Hz — 50 Hz 기록이면 나이퀴스트 여유 {25 / (designer.traj.K / designer.traj.T):.0f}배")
    for name, preset in sensor_presets.items():
        from .design import Noise
        from .identify import information_gain
        n_t = ev["Yp"].shape[0] // 6
        factor = preset.rate_hz * designer.traj.T / n_t
        r = np.tile(np.array([preset.sigma_f ** 2] * 3 + [preset.sigma_t ** 2] * 3), n_t) / factor
        ig = information_gain(designer.est, ev["Yp"], r)
        P(f"  잡음 모형 {name:<15}: 이 궤적 한 번의 정보이득 {ig:.1f} nat")
    return "\n".join(lines)
