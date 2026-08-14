"""실물 하드웨어 추상화 — 시뮬레이션과 실제 로봇을 같은 코드로 몬다.

왜 이 파일이 필요한가
---------------------
dual_view 의 오른쪽 화면은 이미 '로봇에게 시키고 렌치를 받는' 일만 한다.
그 일을 하는 주체를 갈아끼울 수 있게 인터페이스로 뽑아 둔 것이 이 모듈이다.

    RobotDriver   팔을 궤적대로 움직이고 현재 관절각을 알려준다
    WrenchSensor  손목 렌치를 읽는다 (평균 + 타어링)
    PoseSensor    물체 관절각을 알려준다 (FoundationPose)

각각 Sim* 구현이 기본으로 들어 있어 하드웨어 없이도 전 과정을 돌릴 수 있고,
Rb5Driver / Aft200Sensor / FoundationPoseSensor 는 실물 붙일 자리다.

실물에서 반드시 해야 하는 것 — 타어링
--------------------------------------
모형의 y 는 **물체만** 만드는 렌치다. 그런데 센서는 이것까지 다 읽는다.

    센서가 읽는 값 = 물체 + 그리퍼 + 센서 마운트 + 손가락

그리퍼(PGC-140)만 해도 물체보다 무겁다. 반드시 빼야 한다. 그리고 자세마다
그리퍼 무게가 다른 방향으로 실리므로 **중력 방향별로 따로** 재야 한다.

    y_물체(g) = y_측정(g) - y_공회전(g)

이 절차가 TareTable 이다. 물체를 잡기 전에 한 번 돌려 두고, 이후 매 측정에서
같은 방향의 값을 뺀다.
"""

from abc import ABC, abstractmethod

import numpy as np

import density_id_drake as alg


# ---------------------------------------------------------------------------
# 인터페이스
# ---------------------------------------------------------------------------
class RobotDriver(ABC):
    """팔. 경로는 이미 RRT 로 만들어져 있으므로 '따라가기'만 하면 된다."""

    @abstractmethod
    def joint_positions(self):
        """현재 팔 관절각 [rad]."""

    @abstractmethod
    def follow(self, waypoints, duration_s):
        """경유점 목록을 duration_s 에 걸쳐 따라간다. 도착하면 반환."""

    @abstractmethod
    def stop(self):
        """즉시 정지. 작업자가 접근하기 전에 반드시 호출."""

    def servo_off(self):
        """서보를 끈다. 사람이 작업 영역에 들어갈 때. 기본은 stop 과 같다."""
        self.stop()

    def servo_on(self):
        """서보를 다시 켠다. 사람이 작업 영역에서 나온 뒤.

        servo_off 와 반드시 짝으로 쓴다. 끄기만 하는 인터페이스는 덫이다 —
        작업자가 물체를 물린 뒤 로봇이 다시 움직여야 하는데 켤 방법이 없다.

        켠 직후에는 **joint_positions 를 다시 읽어야 한다.** 서보가 꺼진
        동안 중력으로 처져서 실제 각도가 껐을 때와 다를 수 있고, 그 차이를
        모르면 첫 경로가 엉뚱한 곳에서 출발한다.
        """


class WrenchSensor(ABC):
    """손목 F/T. 정지 상태에서 여러 샘플을 평균낸다."""

    @abstractmethod
    def read_raw(self, n_samples):
        """평균 렌치 [Fx Fy Fz Tx Ty Tz]. 타어링 전 원값."""


class PoseSensor(ABC):
    """물체 관절각 관측기 (FoundationPose)."""

    @abstractmethod
    def object_joint_deg(self):
        """물체 관절각 [deg] 과 그 불확실성 [deg] 을 함께 돌려준다."""


# ---------------------------------------------------------------------------
# 타어링 — 실물에서 가장 놓치기 쉬운 단계
# ---------------------------------------------------------------------------
class TareTable:
    """중력 방향별 공회전 렌치.

    물체를 잡지 않은 상태로 탐색에 쓸 자세들을 그대로 밟으며 렌치를 재둔다.
    그 값이 그리퍼 + 마운트 + 손가락이 만드는 몫이다.

    주의: 그리퍼를 열고 닫으면 손가락 위치가 바뀌어 무게중심이 움직인다.
    타어링과 측정에서 **같은 개구량** 을 써야 한다.
    """

    def __init__(self):
        self.table = {}

    def key(self, g_hat):
        return tuple(np.round(np.asarray(g_hat, dtype=float), 6))

    def record(self, g_hat, wrench):
        self.table[self.key(g_hat)] = np.asarray(wrench, dtype=float)

    def apply(self, g_hat, wrench):
        offset = self.table.get(self.key(g_hat))
        if offset is None:
            raise KeyError(
                f"중력 방향 {np.round(g_hat, 3)} 의 타어링 값이 없다. "
                "물체를 잡기 전에 run_tare() 를 먼저 돌려야 한다.")
        return np.asarray(wrench, dtype=float) - offset

    def is_complete(self, g_dirs):
        return all(self.key(g) in self.table for g in g_dirs)


def run_tare(robot, sensor, arm_poses, n_samples, settle_s=1.0,
             duration_s=8.0, log=print):
    """물체를 잡지 않은 상태로 탐색 자세를 밟으며 공회전 렌치를 잰다.

    arm_poses : {중력방향: 팔 관절각} — 탐색 때 쓸 자세와 같아야 한다.
    """
    import time

    table = TareTable()
    log("[타어] 물체를 잡지 않은 상태인지 확인하세요. 그리퍼 개구량은"
        " 측정 때와 같아야 합니다.")
    for g_hat, arm_q in arm_poses.items():
        robot.follow([arm_q], duration_s)
        robot.stop()
        time.sleep(settle_s)
        raw = sensor.read_raw(n_samples)
        table.record(g_hat, raw)
        log(f"[타어] g={np.round(g_hat, 2)}  "
            f"F={np.round(raw[:3], 4)} N  T={np.round(raw[3:], 5)} N·m")
    return table


# ---------------------------------------------------------------------------
# 시뮬레이션 구현 — 하드웨어 없이 전 과정을 돌린다
# ---------------------------------------------------------------------------
class SimRobot(RobotDriver):
    """Drake plant 위에서 도는 팔. dual_view 의 기존 동작과 같다."""

    def __init__(self, plant, context, arm_joints, publish=None):
        self.plant = plant
        self.context = context
        self.arm_joints = arm_joints
        self.publish = publish or (lambda: None)
        self._indices = [j.position_start() for j in arm_joints]

    def joint_positions(self):
        return self.plant.GetPositions(self.context)[self._indices].copy()

    def follow(self, waypoints, duration_s):
        import time
        from robot_scene import cycloidal
        q = self.plant.GetPositions(self.context).copy()
        for target in waypoints:
            start = q[self._indices].copy()
            steps = max(2, int(duration_s / 0.03))
            for k in range(1, steps + 1):
                s = cycloidal(k / steps)
                q[self._indices] = start + s * (np.asarray(target) - start)
                self.plant.SetPositions(self.context, q)
                self.publish()
                time.sleep(duration_s / steps)

    def stop(self):
        self.publish()


class SimWrench(WrenchSensor):
    """진리 plant 에서 계산한 렌치 + 잡음. 타어링 값은 0 이다."""

    def __init__(self, theta_getter, g_hat_getter, rng=None):
        self.theta_getter = theta_getter
        self.g_hat_getter = g_hat_getter
        self.rng = rng or np.random.default_rng(0)

    def read_raw(self, n_samples):
        theta = self.theta_getter()
        g_hat = self.g_hat_getter()
        return alg.measure(theta, g_dirs=[np.asarray(g_hat)], rng=self.rng)


class SimPose(PoseSensor):
    """FoundationPose 흉내. 실제 각도에 상대오차를 섞어 돌려준다."""

    def __init__(self, truth_getter, rel_error=0.05, floor_deg=0.5, rng=None):
        self.truth_getter = truth_getter
        self.rel_error = rel_error
        self.floor_deg = floor_deg
        self.rng = rng or np.random.default_rng(0)

    def object_joint_deg(self):
        truth = np.atleast_1d(self.truth_getter())
        sigma = np.maximum(self.rel_error * np.abs(truth), self.floor_deg)
        return truth + self.rng.normal(0.0, sigma), sigma


# ---------------------------------------------------------------------------
# 실물 구현 자리 — 여기만 채우면 같은 파이프라인이 실물을 몬다
# ---------------------------------------------------------------------------
class Rb5Driver(RobotDriver):
    """Rainbow Robotics RB5-850E.

    채울 것
      joint_positions : 드라이버의 조인트 상태 토픽/API 에서 6개 각도
      follow          : 경유점을 궤적으로 만들어 전송하고 도착까지 대기
                        (RRT 가 준 경유점을 그대로 쓰면 된다)
      stop / servo_off/ servo_on : 즉시 정지, 서보 오프/온

    joint_positions 는 세션 맨 앞에서도 불린다
      파이프라인은 팔이 어디 있는지 **가정하지 않는다**. 작업자가 물체를
      물리는 동안 로봇은 전원을 켠 자리에 그대로 서 있고, 거기서 시작
      자세까지 경로를 계획해 이동한다. 그러니 이 함수는 세션 시작 시점에
      실제 각도를 돌려줘야 한다 (dual_view.RobotScreen.sync_from_robot).

    안전
      - follow 는 반드시 **블로킹** 이어야 한다. 도착 전에 반환하면 작업자가
        움직이는 로봇에 접근할 수 있다.
      - 속도 상한을 소프트웨어와 컨트롤러 양쪽에 건다. 준정적 조건은
        소프트웨어 계산일 뿐이라 하드웨어 리밋이 따로 있어야 한다.
    """

    def __init__(self, *_, **__):
        raise NotImplementedError(
            "RB5 드라이버를 연결하세요. ROS1 이면 joint_trajectory 액션, "
            "직접 제어면 제조사 API 를 이 클래스 안에서만 씁니다.")

    def joint_positions(self): ...
    def follow(self, waypoints, duration_s): ...
    def stop(self): ...
    def servo_off(self): ...
    def servo_on(self): ...


class Aft200Sensor(WrenchSensor):
    """AFT200-D80 (1000 Hz).

    채울 것
      read_raw : n_samples 개를 모아 축별 평균

    주의
      - 온도 드리프트가 있다. 세션 시작 때, 그리고 30분마다 타어링을 다시 한다.
      - 코드의 잡음 모형은 1000 샘플 평균 + 평균으로 안 줄어드는 5 % 바이어스
        를 가정한다 (obj.set_measurement_averaging).
      - 센서 좌표계가 모형의 센서 프레임과 같은 방향인지 확인해야 한다.
        틀리면 부호만 바뀌어도 밀도가 통째로 틀어진다.
    """

    def __init__(self, *_, **__):
        raise NotImplementedError(
            "AFT200 노드를 연결하세요. 1000 Hz 스트림에서 n_samples 평균.")

    def read_raw(self, n_samples): ...


class FoundationPoseSensor(PoseSensor):
    """D456 + FoundationPose.

    채울 것
      object_joint_deg : 물체 관절각과 그 불확실성

    주의
      - 손-눈 보정이 되어 있어야 카메라 좌표의 자세를 로봇 좌표로 옮긴다.
      - 코드는 상대오차 +-5 % 를 가정한다. 실제 값을 고정 장면에서 반복
        측정해 확인하고, 다르면 --angle-error 로 바꿔 넣는다.
      - 그 반복 측정에서 '매번 같은 방향으로 치우친 몫' 도 같이 나온다.
        그것이 --systematic 에 넣을 값이다.
    """

    def __init__(self, *_, **__):
        raise NotImplementedError(
            "FoundationPose 노드를 연결하세요. 관절각과 공분산을 함께 받습니다.")

    def object_joint_deg(self): ...
