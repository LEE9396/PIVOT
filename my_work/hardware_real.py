"""실물 장비 구현 — 벤더 API 는 백엔드 4개 함수로 격리한다.

왜 이렇게 나누는가
------------------
RB5 나 AFT200 의 API 를 코드 곳곳에 흩뿌리면 두 가지가 나빠진다.

  1. 장비가 바뀌면 여러 파일을 고쳐야 한다.
  2. **장비 없이는 아무것도 검증할 수 없다.** 그런데 실물에서 처음 돌리는
     날에 디버깅할 것이 많으면 안 된다. 사람이 옆에 서 있는 상황이다.

그래서 벤더가 주는 것은 아래 네 함수로만 받는다.

    backend.joint_positions()        -> (6,) rad
    backend.move_to(q, duration_s)   -> 도착까지 블로킹
    backend.halt()                   -> 즉시 정지
    backend.set_servo(on: bool)      -> 서보 on/off

이 네 개 바깥의 모든 것 — 속도 상한, 준정적 조건, 도착 검증, 서보 상태기계,
타어링 유효기간, 센서 프레임 부호, 단위 변환 — 은 여기서 구현하고
**지금 장비 없이 검증한다** (`python hardware_real.py --check`).

실물 붙이는 날 팀이 할 일은 `RbpodoBackend` 의 네 함수를 채우는 것뿐이다.

실행:
  ../robot_learning/scripts/run_drake_env.sh python hardware_real.py --check
"""

import argparse
import time

import numpy as np

import hardware as hw

# ---------------------------------------------------------------------------
# 안전 한계 — 소프트웨어 쪽 방어선. 컨트롤러에도 같은 값을 걸어야 한다.
# ---------------------------------------------------------------------------
MAX_JOINT_SPEED_RAD_S = 0.35      # 사람이 옆에 있는 준정적 작업 기준
ARRIVAL_TOL_RAD = np.deg2rad(0.5)  # 도착 판정 허용오차
TARE_MAX_AGE_S = 30 * 60           # AFT200 온도 드리프트. 30분마다 다시.
N_ARM_JOINT = 6


class SafetyViolation(RuntimeError):
    """안전 불변식이 깨졌다. 절대 삼키지 말 것."""


def quasi_static_duration(q_from, q_to, ratio=0.01, reach_m=0.85):
    """준정적으로 움직이려면 최소 몇 초가 필요한가.

    싸이클로이드 프로파일의 최대 가속도는 a_max = 2*pi*L / T^2 이다 (L 은 이동량).
    말단에서 그 가속도가 만드는 관성력이 중력의 `ratio` 미만이어야 하므로

        (2*pi*L_max*reach) / T^2  <  ratio * g      ->  T > sqrt(...)

    L_max 는 관절 이동량의 최댓값, reach 는 말단까지의 팔 길이다. 정확한
    야코비안 대신 reach 를 쓰는 것은 **보수적인 쪽으로 틀리기** 위해서다.
    """
    L = float(np.max(np.abs(np.asarray(q_to) - np.asarray(q_from))))
    if L <= 0.0:
        return 0.0
    return float(np.sqrt(2.0 * np.pi * L * reach_m / (ratio * 9.81)))


# ---------------------------------------------------------------------------
# 백엔드 — 벤더 API 가 들어오는 유일한 자리
# ---------------------------------------------------------------------------
class ArmBackend:
    """네 함수만 구현하면 된다. 단위는 전부 rad / 초."""

    def joint_positions(self):
        raise NotImplementedError

    def move_to(self, q, duration_s):
        """**도착할 때까지 반환하지 않는다.** 이게 안전의 핵심이다."""
        raise NotImplementedError

    def halt(self):
        raise NotImplementedError

    def set_servo(self, on):
        raise NotImplementedError


class RbpodoBackend(ArmBackend):
    """Rainbow Robotics RB5-850E (rbpodo SDK).

    ★ 팀이 채울 곳은 아래 네 군데뿐입니다. 나머지는 전부 구현돼 있습니다.

    확인해야 할 것 두 가지
      1) **관절 순서**가 URDF(`third_party/HTD/assets/rbpodo_description`)와
         같은지. 다르면 팔이 엉뚱한 곳으로 갑니다.
      2) **단위**가 rad 인지 deg 인지. rbpodo 는 deg 를 쓰는 API 가 있습니다.
         이 클래스는 **바깥에 항상 rad 로 내보내야** 합니다.

    처음 붙이는 날 순서
      a) set_servo/halt 만 연결하고 `--check --backend rbpodo` 로 배선 확인
      b) joint_positions 연결 후 팔을 손으로 옮겨가며 값이 맞는지 눈으로 확인
      c) move_to 는 **아주 짧은 이동(1~2도)** 부터. 비상정지 손에 들고.
    """

    def __init__(self, host, port=5000, deg_api=True):
        self.host, self.port, self.deg_api = host, port, deg_api
        self._cobot = None
        raise NotImplementedError(
            "rbpodo 연결을 채우세요. 예: self._cobot = rbpodo.Cobot(host)\n"
            "  joint_positions / move_to / halt / set_servo 네 개만 채우면 됩니다.")

    def _to_rad(self, q):
        return np.deg2rad(q) if self.deg_api else np.asarray(q, dtype=float)

    def _from_rad(self, q):
        return np.rad2deg(q) if self.deg_api else np.asarray(q, dtype=float)

    def joint_positions(self):       # TODO(팀): 현재 관절각 6개
        ...

    def move_to(self, q, duration_s):  # TODO(팀): 블로킹 이동
        ...

    def halt(self):                  # TODO(팀): 즉시 정지
        ...

    def set_servo(self, on):         # TODO(팀): 서보 on/off
        ...


class FakeArm(ArmBackend):
    """장비 없이 전 경로를 돌리기 위한 가짜 팔. `--check` 가 이걸 쓴다.

    일부러 **불완전하게** 만들었다. 도착 오차를 조금 남기고(0.05도), 서보가
    꺼진 동안 중력으로 처지게 한다. 그래야 바깥 코드의 검증 로직이 실제로
    일하는지 확인된다.
    """

    def __init__(self, q0=None, sag_rad=np.deg2rad(0.8), rng=None):
        self.q = np.zeros(N_ARM_JOINT) if q0 is None else np.array(q0, float)
        self.servo = True
        self.sag_rad = sag_rad
        self.rng = rng or np.random.default_rng(0)
        self.log = []

    def joint_positions(self):
        return self.q.copy()

    def move_to(self, q, duration_s):
        if not self.servo:
            raise SafetyViolation("서보가 꺼진 채로 이동 명령이 들어왔다")
        self.log.append(("move", float(duration_s)))
        time.sleep(min(duration_s, 0.01))          # 검사에서는 실제로 안 기다린다
        self.q = np.asarray(q, float) + self.rng.normal(0, np.deg2rad(0.05),
                                                        N_ARM_JOINT)

    def halt(self):
        self.log.append(("halt", 0.0))

    def set_servo(self, on):
        if self.servo and not on:                  # 끄면 중력으로 처진다
            self.q = self.q + self.rng.normal(0, self.sag_rad, N_ARM_JOINT)
        self.servo = bool(on)
        self.log.append(("servo", 1.0 if on else 0.0))


# ---------------------------------------------------------------------------
# 팔 드라이버 — 안전 로직은 전부 여기 있다
# ---------------------------------------------------------------------------
class Rb5Driver(hw.RobotDriver):
    """백엔드를 감싸 안전 불변식을 강제한다.

    강제하는 것
      1. 이동 시간이 속도 상한과 준정적 조건을 **둘 다** 만족하는지.
         모자라면 늘린다 (줄이지 않는다).
      2. 도착을 **확인**한다. 백엔드가 블로킹이라고 주장해도 믿지 않는다.
      3. 서보가 꺼진 상태에서는 이동 명령을 거부한다.
      4. 서보를 켠 직후에는 관절각을 **다시 읽는다** (꺼진 동안 처졌을 수 있다).
    """

    def __init__(self, backend, max_speed=MAX_JOINT_SPEED_RAD_S,
                 arrival_tol=ARRIVAL_TOL_RAD, reach_m=0.85, log=print):
        self.backend = backend
        self.max_speed = float(max_speed)
        self.arrival_tol = float(arrival_tol)
        self.reach_m = float(reach_m)
        self.log = log
        self._servo_on = True
        self.stretched_s = 0.0        # 안전 때문에 늘린 총 시간 (보고용)

    # -- 상태 ---------------------------------------------------------------
    def joint_positions(self):
        q = np.asarray(self.backend.joint_positions(), dtype=float)
        if q.shape != (N_ARM_JOINT,):
            raise SafetyViolation(
                f"관절각이 {q.shape} 로 왔다. ({N_ARM_JOINT},) 이어야 한다 — "
                "백엔드의 관절 순서/개수를 확인하세요.")
        if not np.all(np.isfinite(q)):
            raise SafetyViolation("관절각에 NaN/Inf 가 있다")
        return q

    # -- 이동 ---------------------------------------------------------------
    def safe_duration(self, q_from, q_to, requested_s):
        """요청 시간이 안전 조건을 만족하는지 보고, 모자라면 늘린다."""
        move = float(np.max(np.abs(np.asarray(q_to) - np.asarray(q_from))))
        by_speed = move / self.max_speed if self.max_speed > 0 else 0.0
        by_quasi = quasi_static_duration(q_from, q_to, reach_m=self.reach_m)
        need = max(by_speed, by_quasi)
        if requested_s < need - 1e-9:
            self.stretched_s += need - requested_s
            self.log(f"[안전] 이동 시간 {requested_s:.1f}s -> {need:.1f}s 로 늘림 "
                     f"(속도 {by_speed:.1f}s / 준정적 {by_quasi:.1f}s)")
            return need
        return float(requested_s)

    def follow(self, waypoints, duration_s):
        if not self._servo_on:
            raise SafetyViolation(
                "서보가 꺼진 상태에서 이동을 시도했다. servo_on() 을 먼저 부르세요.")
        q = self.joint_positions()
        for target in waypoints:
            target = np.asarray(target, dtype=float)
            if target.shape != (N_ARM_JOINT,):
                raise SafetyViolation(f"경유점 차원이 {target.shape} 다")
            dt = self.safe_duration(q, target, duration_s)
            self.backend.move_to(target, dt)
            q = self.joint_positions()                 # 도착을 **확인**한다
            gap = float(np.max(np.abs(q - target)))
            if gap > self.arrival_tol:
                self.backend.halt()
                raise SafetyViolation(
                    f"도착 실패: 목표와 {np.rad2deg(gap):.2f}도 차이 "
                    f"(허용 {np.rad2deg(self.arrival_tol):.2f}도). "
                    "백엔드의 move_to 가 블로킹인지 확인하세요.")
        return q

    def stop(self):
        self.backend.halt()

    # -- 서보 ---------------------------------------------------------------
    def servo_off(self):
        self.backend.halt()
        self.backend.set_servo(False)
        self._servo_on = False
        self.log("[안전] 서보 OFF — 작업자 접근 가능")

    def servo_on(self):
        self.backend.set_servo(True)
        self._servo_on = True
        q = self.joint_positions()      # 처졌을 수 있으므로 반드시 다시 읽는다
        self.log(f"[안전] 서보 ON — 현재 관절각 재확인 "
                 f"{np.round(np.rad2deg(q), 2)} deg")
        return q


# ---------------------------------------------------------------------------
# 렌치 센서
# ---------------------------------------------------------------------------
class Aft200Sensor(hw.WrenchSensor):
    """AFT200-D80. 샘플 소스만 백엔드로 받고 나머지는 여기서 한다.

    sample_fn() -> (6,) [Fx Fy Fz Tx Ty Tz], 단위 N / N·m

    여기서 하는 일
      - n_samples 평균 (잡음 모형이 이 평균을 전제한다)
      - **튀는 샘플 제거**: 중앙값에서 6 MAD 넘는 것은 버린다.
        케이블 충격이나 통신 결손이 평균을 통째로 오염시키는 것을 막는다.
      - 부호 규약 검사: 정지 상태에서 중력 방향과 힘의 방향이 반대인지.
        **이게 틀리면 밀도가 통째로 틀어지는데 결과만 봐서는 모른다.**
    """

    def __init__(self, sample_fn, rate_hz=1000.0, mad_k=6.0, log=print):
        self.sample_fn = sample_fn
        self.rate_hz = float(rate_hz)
        self.mad_k = float(mad_k)
        self.log = log
        self.n_rejected = 0

    def read_raw(self, n_samples):
        n = int(max(1, n_samples))
        buf = np.empty((n, 6))
        for k in range(n):
            s = np.asarray(self.sample_fn(), dtype=float)
            if s.shape != (6,) or not np.all(np.isfinite(s)):
                raise SafetyViolation(f"렌치 샘플이 이상하다: {s}")
            buf[k] = s
        if n < 4:
            return buf.mean(axis=0)
        med = np.median(buf, axis=0)
        mad = np.median(np.abs(buf - med), axis=0) * 1.4826
        mad = np.where(mad > 0, mad, np.inf)          # 완전 무잡음이면 안 버린다
        keep = np.all(np.abs(buf - med) <= self.mad_k * mad, axis=1)
        self.n_rejected += int(n - keep.sum())
        if keep.sum() < max(4, 0.5 * n):
            raise SafetyViolation(
                f"샘플 {n}개 중 {keep.sum()}개만 살아남았다. 케이블/통신 확인.")
        return buf[keep].mean(axis=0)

    def check_sign_convention(self, g_hat, n_samples=200, tol_deg=25.0):
        """정지 상태에서 힘의 방향이 중력과 반대인지 확인한다.

        물체를 잡고 가만히 있으면 센서는 물체를 **떠받치는** 힘을 읽으므로
        `f` 는 `-g_hat` 쪽을 향해야 한다. 벗어나면 축 순서나 부호가 틀린 것이다.
        """
        f = self.read_raw(n_samples)[:3]
        norm = np.linalg.norm(f)
        if norm < 1e-6:
            self.log("[센서] 힘이 거의 0 — 물체를 잡은 상태에서 다시 확인하세요.")
            return None
        cos = float(np.dot(f / norm, -np.asarray(g_hat, dtype=float)))
        ang = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
        ok = ang <= tol_deg
        self.log(f"[센서] 힘 방향과 -g 사이 {ang:.1f}도  "
                 + ("OK" if ok else "*** 축 순서/부호 확인 필요 ***"))
        if not ok:
            raise SafetyViolation(
                f"힘 방향이 중력과 {ang:.1f}도 어긋난다. 센서 프레임이 모형의 "
                "센서 프레임과 다르게 놓였을 수 있습니다. 이 상태로 측정하면 "
                "밀도가 통째로 틀어집니다.")
        return ang


# ---------------------------------------------------------------------------
# 물체 관절각 센서
# ---------------------------------------------------------------------------
class FoundationPoseSensor(hw.PoseSensor):
    """D456 + FoundationPose.

    pose_fn() -> (angles_deg, sigma_deg) 또는 angles_deg 하나.
    sigma 를 안 주면 `default_sigma_deg` 를 쓴다.

    여기서 하는 일
      - 차원·유한성 검사
      - 오래된 값 거부 (stale). 트래커가 멈췄는데 마지막 값을 계속 주면
        **알고리즘은 각도가 맞다고 믿고 측정을 진행한다.** 가장 위험한 고장이다.
      - sigma 하한 적용 (angle_aware 의 floor 와 같은 취지)
    """

    def __init__(self, pose_fn, n_joint, default_sigma_deg=None,
                 floor_deg=0.5, max_age_s=2.0, stamp_fn=None, log=print):
        self.pose_fn = pose_fn
        self.n_joint = int(n_joint)
        self.default_sigma_deg = default_sigma_deg
        self.floor_deg = float(floor_deg)
        self.max_age_s = float(max_age_s)
        self.stamp_fn = stamp_fn
        self.log = log

    def object_joint_deg(self):
        out = self.pose_fn()
        ang, sig = (out if isinstance(out, (tuple, list)) and len(out) == 2
                    else (out, None))
        ang = np.atleast_1d(np.asarray(ang, dtype=float))
        if ang.size != self.n_joint or not np.all(np.isfinite(ang)):
            raise SafetyViolation(
                f"관절각이 {ang} 로 왔다. 관절 {self.n_joint}개가 와야 한다.")
        if self.stamp_fn is not None:
            age = float(time.time() - self.stamp_fn())
            if age > self.max_age_s:
                raise SafetyViolation(
                    f"물체 자세가 {age:.1f}초 전 값이다 (허용 {self.max_age_s}초). "
                    "트래커가 멈췄을 수 있습니다 — 이 값으로 측정하면 안 됩니다.")
        if sig is None:
            if self.default_sigma_deg is None:
                raise SafetyViolation(
                    "트래커가 불확실성을 안 준다. default_sigma_deg 를 주거나 "
                    "고정 장면 반복측정으로 값을 재서 넣으세요.")
            sig = np.full(self.n_joint, float(self.default_sigma_deg))
        sig = np.maximum(np.atleast_1d(np.asarray(sig, dtype=float)),
                         self.floor_deg)
        return ang, sig


# ---------------------------------------------------------------------------
# 타어링 유효기간
# ---------------------------------------------------------------------------
class TimedTare(hw.TareTable):
    """타어링에 시각을 붙여 오래된 값을 거부한다.

    AFT200 은 온도 드리프트가 있다. 아침에 잰 값을 오후에 쓰면 그리퍼 무게를
    잘못 빼게 되고, 그건 라운드를 늘려도 안 없어지는 치우침이 된다.
    """

    def __init__(self, max_age_s=TARE_MAX_AGE_S, clock=time.time):
        super().__init__()
        self.max_age_s = float(max_age_s)
        self.clock = clock
        self.stamped = {}

    def record(self, g_hat, wrench):
        super().record(g_hat, wrench)
        self.stamped[self.key(g_hat)] = self.clock()

    def apply(self, g_hat, wrench):
        t = self.stamped.get(self.key(g_hat))
        if t is None:
            raise KeyError(f"중력 방향 {np.round(g_hat, 3)} 타어링 값이 없다.")
        age = self.clock() - t
        if age > self.max_age_s:
            raise SafetyViolation(
                f"타어링이 {age/60:.0f}분 전 값이다 (허용 {self.max_age_s/60:.0f}분). "
                "온도 드리프트로 그리퍼 몫이 달라졌을 수 있으니 다시 재세요.")
        return super().apply(g_hat, wrench)


# ---------------------------------------------------------------------------
# 검증 — 장비 없이 위 전부를 시험한다
# ---------------------------------------------------------------------------
def self_check(verbose=True):
    log = (lambda *a: None) if not verbose else print
    ok = []

    def case(name, fn):
        try:
            fn()
            ok.append((name, True, ""))
        except Exception as exc:                       # noqa: BLE001
            ok.append((name, False, f"{type(exc).__name__}: {exc}"))

    # 1. 이동 시간이 안전 조건에 맞춰 늘어나는가
    def t_stretch():
        arm = FakeArm()
        d = Rb5Driver(arm, log=lambda *a: None)
        q0 = np.zeros(6); q1 = np.zeros(6); q1[0] = np.deg2rad(90)
        got = d.safe_duration(q0, q1, requested_s=0.1)
        need_speed = np.deg2rad(90) / MAX_JOINT_SPEED_RAD_S
        assert got >= need_speed - 1e-9, f"속도 상한 미적용 {got} < {need_speed}"
        assert got >= quasi_static_duration(q0, q1) - 1e-9, "준정적 조건 미적용"
        assert d.safe_duration(q0, q1, 999.0) == 999.0, "충분한 시간을 줄이면 안 된다"

    # 2. 도착 실패를 잡아내는가 (블로킹이 아닌 백엔드 흉내)
    def t_arrival():
        class Lazy(FakeArm):
            def move_to(self, q, duration_s):          # 절반만 가고 반환
                self.q = self.q + 0.5 * (np.asarray(q, float) - self.q)
        d = Rb5Driver(Lazy(), log=lambda *a: None)
        tgt = np.full(6, np.deg2rad(10))
        try:
            d.follow([tgt], 30.0)
        except SafetyViolation:
            return
        raise AssertionError("도착하지 않았는데 통과시켰다")

    # 3. 서보 OFF 중 이동 거부 + ON 후 각도 재확인
    def t_servo():
        arm = FakeArm(); d = Rb5Driver(arm, log=lambda *a: None)
        d.servo_off()
        try:
            d.follow([np.zeros(6)], 10.0)
            raise AssertionError("서보 OFF 인데 이동을 허용했다")
        except SafetyViolation:
            pass
        before = arm.q.copy()
        q = d.servo_on()
        assert np.allclose(q, before), "servo_on 이 실제 각도를 안 읽었다"

    # 4. 렌치 평균과 튀는 샘플 제거
    def t_wrench():
        rng = np.random.default_rng(1)
        base = np.array([0.0, 0.0, -5.0, 0.0, 0.0, 0.0])
        state = {"k": 0}

        def sample():
            state["k"] += 1
            s = base + rng.normal(0, 0.01, 6)
            if state["k"] % 50 == 0:                   # 20 개 중 1 개가 튄다
                s = s + 100.0
            return s
        w = Aft200Sensor(sample, log=lambda *a: None)
        got = w.read_raw(1000)
        assert w.n_rejected > 0, "튀는 샘플을 하나도 안 걸렀다"
        assert abs(got[2] - (-5.0)) < 0.02, f"평균이 오염됐다 {got[2]}"

    # 5. 센서 부호 규약 검사가 뒤집힌 축을 잡는가
    def t_sign():
        g = np.array([0.0, 0.0, -1.0])
        good = Aft200Sensor(lambda: np.array([0, 0, 5.0, 0, 0, 0]),
                            log=lambda *a: None)
        good.check_sign_convention(g, n_samples=10)     # 예외 없어야 함
        bad = Aft200Sensor(lambda: np.array([0, 0, -5.0, 0, 0, 0]),
                           log=lambda *a: None)
        try:
            bad.check_sign_convention(g, n_samples=10)
        except SafetyViolation:
            return
        raise AssertionError("부호가 뒤집혔는데 통과시켰다")

    # 6. 물체 자세가 오래된 값이면 거부하는가
    def t_stale():
        now = [1000.0]
        p = FoundationPoseSensor(lambda: (np.array([30.0]), np.array([1.0])),
                                 n_joint=1, max_age_s=2.0,
                                 stamp_fn=lambda: now[0] - 10.0,
                                 log=lambda *a: None)
        old_time = time.time
        try:
            time.time = lambda: now[0]
            try:
                p.object_joint_deg()
                raise AssertionError("10초 지난 자세를 통과시켰다")
            except SafetyViolation:
                pass
        finally:
            time.time = old_time

    # 7. sigma 하한이 걸리는가
    def t_sigma_floor():
        p = FoundationPoseSensor(lambda: (np.array([30.0]), np.array([0.01])),
                                 n_joint=1, floor_deg=0.5, log=lambda *a: None)
        _, sig = p.object_joint_deg()
        assert sig[0] >= 0.5, f"sigma 하한 미적용 {sig}"

    # 8. 타어링 유효기간
    def t_tare_age():
        clock = [0.0]
        t = TimedTare(max_age_s=60.0, clock=lambda: clock[0])
        g = np.array([0.0, 0.0, -1.0])
        t.record(g, np.ones(6))
        t.apply(g, np.ones(6) * 2)                     # 바로 쓰면 OK
        clock[0] = 3600.0
        try:
            t.apply(g, np.ones(6) * 2)
            raise AssertionError("1시간 지난 타어링을 통과시켰다")
        except SafetyViolation:
            pass

    # 9. 타어링이 없으면 측정이 진행되지 않는가
    def t_tare_missing():
        t = TimedTare()
        try:
            t.apply(np.array([1.0, 0, 0]), np.ones(6))
            raise AssertionError("타어링 없이 측정을 허용했다")
        except KeyError:
            pass

    # 10. run_tare 가 모든 중력 방향을 채우는가
    def t_run_tare():
        arm = FakeArm(); d = Rb5Driver(arm, log=lambda *a: None)
        w = Aft200Sensor(lambda: np.arange(6, dtype=float), log=lambda *a: None)
        dirs = [tuple(v) for v in np.eye(3)]
        poses = {g: np.zeros(6) for g in dirs}
        table = hw.run_tare(d, w, poses, n_samples=8, settle_s=0.0,
                            duration_s=0.01, log=lambda *a: None)
        assert table.is_complete([np.array(g) for g in dirs]), "방향이 빠졌다"

    for name, fn in [
            ("이동 시간이 속도·준정적 조건에 맞춰 늘어난다", t_stretch),
            ("도착하지 않으면 예외를 던진다 (블로킹 검증)", t_arrival),
            ("서보 OFF 중 이동 거부 / ON 후 각도 재확인", t_servo),
            ("렌치 평균 + 튀는 샘플 제거", t_wrench),
            ("센서 힘 방향 부호 규약 검사", t_sign),
            ("오래된 물체 자세를 거부한다", t_stale),
            ("자세 불확실성 하한 적용", t_sigma_floor),
            ("타어링 유효기간(30분) 적용", t_tare_age),
            ("타어링 없으면 측정 진행 불가", t_tare_missing),
            ("run_tare 가 중력 3방향을 모두 채운다", t_run_tare)]:
        case(name, fn)

    width = max(len(n) for n, _, _ in ok)
    for name, good, msg in ok:
        log(f"  [{'OK ' if good else 'FAIL'}] {name.ljust(width)}  {msg}")
    n_bad = sum(1 for _, g, _ in ok if not g)
    log(f"\n  {len(ok) - n_bad}/{len(ok)} 통과")
    return n_bad == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="장비 없이 안전 로직 전체를 시험한다")
    args = ap.parse_args()
    if args.check:
        print("실물 코드 자가검증 (장비 없이)\n")
        raise SystemExit(0 if self_check() else 1)
    ap.print_help()


if __name__ == "__main__":
    main()
