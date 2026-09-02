"""Robotiq 2F-85 실물 그리퍼 드라이버 (FTDI / Modbus-RTU).

왜 이 파일이 필요한가
---------------------
이 파이프라인은 파지를 **용접**으로 모형화한다 (grippers.py 머리말). 손가락이
시뮬레이션 안에서 움직일 필요가 없다는 뜻이지, **실물에서 안 움직여도 된다는
뜻이 아니다.** 실물 세션은 매 회차 이렇게 시작한다.

    그리퍼를 연다 -> 물체의 파지 부위를 죠 사이에 넣는다 -> 계획된 개구량으로
    문다 -> 손을 뗀다 -> 로봇이 움직인다

지금까지 이 네 줄은 화면에 **글자로만** 떠 있었고 (dual_view.await_grasp),
사람이 별도 스크립트로 그리퍼를 여닫아야 했다. 그러면 두 가지가 어긋난다.

  1. **개구량이 계획과 달라진다.** robot_scene.jaw_opening_for() 가 정한
     개구량은 물체가 죠 안에서 어디에 앉는지를 정하고, 그게 곧 파지점이며,
     파지점은 regressor 의 모든 모멘트팔이다. 눈대중으로 5 mm 다르게 물면
     모멘트팔이 5 mm 어긋난 채로 밀도를 푼다 (study_grasp.py: 2 mm 에 113 %).
  2. **물었는지 아무도 확인하지 않는다.** 사람이 버튼을 누르면 로봇은
     움직인다. 2F-85 는 gOBJ 비트로 "접촉해서 멈췄다" 를 알려주는데,
     그걸 안 읽으면 빈 손으로 8 초짜리 준정적 이동을 하게 된다.

그래서 개구량을 **명령하고 되읽는** 최소한의 드라이버를 둔다.

무엇을 쓰나
-----------
pyserial 을 안 쓴다. 부트스트랩에 꾸러미를 더하지 않으려는 것이고, Modbus-RTU
프레임 몇 개면 되는 일이라 termios 로 충분하다 (원본: robotiq_keyboard_toggle.py).

카운트와 개구량
---------------
2F-85 의 위치 명령은 0..255 카운트다. 0 이 활짝, 255 가 꽉 닫힘이고 그 사이는
벤더 문서 기준 **거의 선형**이다. 정확한 값이 필요하면 CALIBRATION 을 실측으로
갈아끼우면 된다. grippers.ROBOTIQ_GAP_M (최대 78.5 mm) 와 다른데, 그건 URDF
충돌 메시의 볼록 껍질 기준이라 실물 패드보다 뚱뚱하기 때문이다. 여기서는
실물 사양값 85 mm 를 쓴다.

단독 실행
---------
    ../robot_learning/scripts/run_drake_env.sh python gripper_hw.py --check
    ../robot_learning/scripts/run_drake_env.sh python gripper_hw.py \
        --port /dev/ttyUSB0 --demo
"""

import argparse
import os
import select
import sys
import termios
import time
import tty
from dataclasses import dataclass

# 실물 2F-85 사양. 카운트 0 = 활짝, 255 = 꽉 닫힘.
MAX_OPENING_M = 0.085
COUNTS_CLOSED = 255
COUNTS_OPEN = 0
DEFAULT_SLAVE = 9
DEFAULT_PORT = "/dev/ttyUSB0"
# 속도 (0..255). 사람이 옆에 있는 단계라 낮게 둔다.
DEFAULT_SPEED = 64

# 파지력. rFR 0..255 가 벤더 문서 기준 대략 20..235 N 에 선형 대응한다.
FORCE_MIN_N = 20.0
FORCE_MAX_N = 235.0
# 원본 스크립트는 32 (약 47 N) 였다. 그건 "집어 올리기" 에는 충분해도
# **이 실험에는 모자란다.**
#
# 미끄러지지 않는 것만 따지면 몇 N 이면 된다 (램프 0.56 kg 을 마찰계수
# 0.5 로 잡으면 5.5 N). 문제는 미끄럼이 아니라 **자세가 바뀌는 것**이다.
# 이 실험은 물체를 문 채로 중력 3방향으로 손목을 돌린다. 그때마다 무게중심이
# 파지점에서 떨어져 있는 만큼 모멘트가 걸려 물체를 죠 안에서 비튼다. 몇 mm
# 만 돌아가도 그게 곧 파지점 어긋남이고, 파지점은 회귀행렬의 모든
# 모멘트팔이다 (study_grasp.py: 2 mm 에 밀도 오차 113 %).
#
# 그래서 넉넉히 조인다. 220 은 약 205 N 으로 최대(235 N)의 87 % 이고,
# 위쪽을 조금 남겨 둔 것은 작업자가 --gripper-force 로 더 올릴 수 있게
# 하기 위해서다. **얇은 플라스틱처럼 눌리는 물체는 반대로 낮추어야 한다** —
# 형상이 눌리면 스캔 메시와 실물이 달라져 부피가 틀린다.
DEFAULT_FORCE = 220

# 물체를 **물 때** 목표 개구보다 이만큼 더 좁게 명령한다 [m].
#
# 위치 명령만으로는 물리지 않는다. 물체 단면과 똑같은 개구를 명령하면 죠가
# 그 자리에 도달해 버리고(gOBJ=3 '요청 위치 도달') 힘이 하나도 안 걸린다.
# 물체보다 좁게 명령해야 물체에 **걸려서** 멈추고(gOBJ=2), 그때부터 rFR
# 만큼의 힘으로 조인다. 실제 도달 개구는 물체 단면 그대로가 된다.
CLAMP_SQUEEZE_M = 0.004

# 키보드 조작 한 틱의 크기. 키를 누르고 있으면 터미널 자동 반복으로
# 초당 수십 번 들어오므로, 한 번에 조금씩 움직여야 손맛이 난다.
KEY_OPEN_STEP_M = 0.0015       # 1.5 mm
KEY_FORCE_STEP = 4             # rFR 카운트 (약 3.4 N)

# gOBJ (상태 바이트 6~7 비트)
OBJ_MOVING = 0
OBJ_STOPPED_OPENING = 1        # 열다가 뭔가에 걸려 멈춤
OBJ_STOPPED_CLOSING = 2        # 닫다가 물체를 물어 멈춤  <- 정상 파지
OBJ_AT_REQUEST = 3             # 요청 위치 도달 (= 아무것도 안 물었다)

# gSTA (상태 바이트 4~5 비트)
STA_RESET = 0
STA_ACTIVATING = 1
STA_ACTIVE = 3


class GripperError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Modbus-RTU 프레임
# ---------------------------------------------------------------------------
def crc16(data):
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return bytes((crc & 0xFF, crc >> 8))


def _write_frame(slave, action, position, speed, force):
    """레지스터 0x03E8 에 3워드(6바이트)를 쓴다."""
    body = bytes((slave, 0x10, 0x03, 0xE8, 0, 3, 6,
                  action, 0, 0, position, speed, force))
    return body + crc16(body)


def command_frame(position, slave=DEFAULT_SLAVE,
                  speed=DEFAULT_SPEED, force=DEFAULT_FORCE):
    """gACT=1, gGTO=1 + 위치. 첫 명령이 활성화도 겸한다."""
    return _write_frame(slave, 0x09, int(position), int(speed), int(force))


def activate_frame(slave=DEFAULT_SLAVE):
    return _write_frame(slave, 0x01, 0, 0, 0)


def reset_frame(slave=DEFAULT_SLAVE):
    return _write_frame(slave, 0x00, 0, 0, 0)


def status_frame(slave=DEFAULT_SLAVE):
    """0x07D0 에서 3워드(6바이트)를 읽는다."""
    body = bytes((slave, 0x03, 0x07, 0xD0, 0, 3))
    return body + crc16(body)


# 응답 길이. 쓰기 확인은 8바이트, 읽기 응답은 11바이트로 고정이다.
WRITE_REPLY_LEN = 8
READ_REPLY_LEN = 11


# ---------------------------------------------------------------------------
# 카운트 <-> 개구량
# ---------------------------------------------------------------------------
def counts_for_opening(opening_m, max_opening_m=MAX_OPENING_M):
    """그 개구량을 만드는 위치 카운트 0..255."""
    fraction = float(opening_m) / max_opening_m
    fraction = min(max(fraction, 0.0), 1.0)
    return int(round(COUNTS_CLOSED * (1.0 - fraction)))


def opening_for_counts(counts, max_opening_m=MAX_OPENING_M):
    """그 카운트에서의 개구량 [m]."""
    counts = min(max(int(counts), COUNTS_OPEN), COUNTS_CLOSED)
    return max_opening_m * (1.0 - counts / float(COUNTS_CLOSED))


# ---------------------------------------------------------------------------
# 파지력
# ---------------------------------------------------------------------------
def force_newton(counts):
    """rFR 카운트가 실제로 만드는 파지력 [N] (벤더 문서의 선형 근사)."""
    counts = min(max(int(counts), 0), 255)
    return FORCE_MIN_N + (FORCE_MAX_N - FORCE_MIN_N) * counts / 255.0


def counts_for_force(newton):
    """원하는 파지력 [N] 을 만드는 rFR 카운트."""
    span = FORCE_MAX_N - FORCE_MIN_N
    fraction = (float(newton) - FORCE_MIN_N) / span
    return int(round(255 * min(max(fraction, 0.0), 1.0)))


def slip_margin(counts, mass_kg, mu=0.5):
    """패드 마찰이 물체 무게의 몇 배를 버티는가.

    패드 두 장이 각각 normal force 만큼 누르므로 마찰은 2*mu*F 다. 이 값이
    1 보다 크면 '미끄러지지는 않는다' 는 뜻일 뿐, **죠 안에서 안 돌아간다는
    보장은 아니다.** 돌아가는 것은 무게중심이 파지점에서 떨어진 만큼 걸리는
    모멘트가 정하고, 그건 패드 접촉면 크기까지 알아야 풀린다. 그래서 이
    값은 하한을 보는 용도로만 쓴다.
    """
    weight = max(float(mass_kg), 1e-9) * 9.80665
    return 2.0 * mu * force_newton(counts) / weight


@dataclass(frozen=True)
class GripperStatus:
    """상태 바이트 6개를 사람이 읽는 형태로 푼 것."""
    activated: bool
    moving: bool
    object_detected: bool       # gOBJ 1 또는 2 — 뭔가에 걸려 멈췄다
    holding: bool               # gOBJ 2 — 닫다가 물어서 멈췄다 (정상 파지)
    fault: int
    counts: int                 # 지금 위치 (되읽은 값)
    requested_counts: int       # 마지막으로 요청한 위치
    opening_m: float

    def describe(self):
        if not self.activated:
            return "활성화 안 됨"
        if self.moving:
            return f"이동 중 (개구 {1000 * self.opening_m:.1f} mm)"
        if self.holding:
            return f"물었습니다 (개구 {1000 * self.opening_m:.1f} mm)"
        if self.object_detected:
            return f"열다가 걸림 (개구 {1000 * self.opening_m:.1f} mm)"
        return f"정지 · 물체 없음 (개구 {1000 * self.opening_m:.1f} mm)"


def parse_status(data, max_opening_m=MAX_OPENING_M):
    """읽기 응답 11바이트를 GripperStatus 로 푼다."""
    if len(data) < READ_REPLY_LEN:
        raise GripperError(f"상태 응답이 짧습니다 ({len(data)} 바이트): "
                           f"{data.hex(' ')}")
    if crc16(data[:READ_REPLY_LEN - 2]) != data[READ_REPLY_LEN - 2:READ_REPLY_LEN]:
        raise GripperError(f"상태 응답 CRC 불일치: {data.hex(' ')}")
    payload = data[3:9]
    gripper_status, _, fault, requested, position, _current = payload
    sta = (gripper_status >> 4) & 0x03
    obj = (gripper_status >> 6) & 0x03
    return GripperStatus(
        activated=(sta == STA_ACTIVE),
        moving=(obj == OBJ_MOVING),
        object_detected=obj in (OBJ_STOPPED_OPENING, OBJ_STOPPED_CLOSING),
        holding=(obj == OBJ_STOPPED_CLOSING),
        fault=fault & 0x0F,
        counts=position,
        requested_counts=requested,
        opening_m=opening_for_counts(position, max_opening_m))


# ---------------------------------------------------------------------------
# 키보드 조작
# ---------------------------------------------------------------------------
KEY_ACTIONS = {
    "a": "open", "left": "open",
    "d": "close", "right": "close",
    "w": "force_up", "up": "force_up",
    "s": "force_down", "down": "force_down",
    "o": "full_open",
    "c": "full_close",
    "g": "grip",
    "q": "quit", "\x03": "quit",          # Ctrl-C
    "\r": "accept", "\n": "accept",
}

KEY_HELP = (
    "  a / <-  열기 (누르고 있으면 계속)      w / ^  파지력 +\n"
    "  d / ->  닫기 (누르고 있으면 계속)      s / v  파지력 -\n"
    "  o  활짝 열기      c  꽉 닫기      g  계획 개구로 물기\n"
    "  Enter  확정      q  나가기")


class KeyReader:
    """터미널을 cbreak 로 두고 눌린 키를 **논블로킹**으로 읽는다.

    터미널에는 '키를 뗐다' 는 신호가 없다. 대신 키를 누르고 있으면 OS 의
    자동 반복이 같은 글자를 초당 수십 번 보낸다. 그래서 "누르고 있으면
    계속 움직인다" 는 한 번 들어올 때마다 조금씩 움직이는 것으로 만든다
    (KEY_OPEN_STEP_M / KEY_FORCE_STEP).

    한 번 poll 에 여러 글자가 몰려 들어오므로, 부르는 쪽은 그것을 **합쳐서
    한 번만 명령해야 한다.** 글자마다 명령하면 도착을 기다리는 사이에
    입력이 더 쌓여 조작이 끈적해진다.

    stdin 이 터미널이 아니면(로그로 넘겨 돌릴 때) 조용히 꺼진다.
    """

    ARROWS = {b"A": "up", b"B": "down", b"C": "right", b"D": "left"}

    def __init__(self, stream=None):
        self.stream = stream or sys.stdin
        self._saved = None

    @property
    def usable(self):
        try:
            return self.stream.isatty()
        except Exception:                              # noqa: BLE001
            return False

    def __enter__(self):
        if self.usable:
            self._saved = termios.tcgetattr(self.stream)
            tty.setcbreak(self.stream.fileno())
        return self

    def __exit__(self, *_):
        if self._saved is not None:
            termios.tcsetattr(self.stream, termios.TCSADRAIN, self._saved)
            self._saved = None

    @classmethod
    def decode(cls, data):
        keys, index = [], 0
        while index < len(data):
            byte = data[index:index + 1]
            if byte == b"\x1b" and data[index + 1:index + 2] == b"[":
                keys.append(cls.ARROWS.get(data[index + 2:index + 3], "esc"))
                index += 3
            else:
                keys.append(byte.decode("utf-8", "replace").lower())
                index += 1
        return keys

    def poll(self):
        """지금 버퍼에 와 있는 키들. 없으면 빈 목록 (기다리지 않는다)."""
        if not self.usable:
            return []
        keys = []
        while select.select([self.stream], [], [], 0)[0]:
            data = os.read(self.stream.fileno(), 64)
            if not data:
                break
            keys.extend(self.decode(data))
        return keys


def fold_keys(keys, opening_m, force_counts, max_opening_m=MAX_OPENING_M,
              plan_opening_m=None):
    """눌린 키들을 **하나의 목표**로 접는다.

    돌려주는 것: (목표 개구 [m], 목표 힘 [카운트], 물기인가, 특수동작)
    특수동작은 "accept" / "quit" / None.
    """
    opening = float(opening_m)
    force = int(force_counts)
    clamp = False
    special = None
    for key in keys:
        action = KEY_ACTIONS.get(key)
        if action == "open":
            opening += KEY_OPEN_STEP_M
        elif action == "close":
            opening -= KEY_OPEN_STEP_M
            clamp = False
        elif action == "force_up":
            force += KEY_FORCE_STEP
        elif action == "force_down":
            force -= KEY_FORCE_STEP
        elif action == "full_open":
            opening = max_opening_m
        elif action == "full_close":
            opening = 0.0
        elif action == "grip" and plan_opening_m is not None:
            opening, clamp = float(plan_opening_m), True
        elif action in ("accept", "quit"):
            special = action
    opening = min(max(opening, 0.0), max_opening_m)
    force = min(max(force, 0), 255)
    return opening, force, clamp, special


# ---------------------------------------------------------------------------
# 드라이버
# ---------------------------------------------------------------------------
class Robotiq2F85:
    """USB(FTDI) 로 붙은 2F-85 하나.

    쓰는 쪽에서 알아야 할 것은 네 개뿐이다.

        activate()          전원 켠 뒤 한 번. 손가락이 한 번 왕복한다.
        grip(m)             그 단면의 물체를 **문다** (좁게 명령해 조인다)
        set_opening(m)      그 개구량으로 이동. 도착까지 기다린다.
        open() / close()    끝까지 열기 / 닫기
        status()            지금 개구량과 '물었는지'

    물릴 때는 set_opening 이 아니라 **grip 을 써야 한다.** 이유는
    CLAMP_SQUEEZE_M 주석에 있다.
    """

    def __init__(self, port=DEFAULT_PORT, slave=DEFAULT_SLAVE,
                 max_opening_m=MAX_OPENING_M, speed=DEFAULT_SPEED,
                 force=DEFAULT_FORCE, log=print):
        self.port = port
        self.slave = slave
        self.max_opening_m = float(max_opening_m)
        self.speed = int(speed)
        self.force = int(force)
        self.log = log
        self._fd = None
        # 마지막으로 명령한 개구량. 화면 안내에 쓴다.
        self.commanded_opening_m = None
        self.open_port()

    # -- 포트 ----------------------------------------------------------
    def open_port(self):
        if self._fd is not None:
            return
        try:
            fd = os.open(self.port, os.O_RDWR | os.O_NOCTTY | os.O_SYNC)
        except OSError as exc:
            raise GripperError(
                f"그리퍼 포트를 열지 못했습니다: {self.port} ({exc})\n"
                f"  - 케이블과 전원을 확인하세요\n"
                f"  - 권한이면: sudo usermod -aG dialout $USER (재로그인)\n"
                f"  - 그리퍼 없이 돌리려면 --no-gripper") from exc
        # 115200 8N1, 흐름제어 없음.
        attrs = termios.tcgetattr(fd)
        attrs[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
        attrs[4] = termios.B115200
        attrs[5] = termios.B115200
        attrs[0] = attrs[1] = attrs[3] = 0
        attrs[6][termios.VMIN] = 0
        attrs[6][termios.VTIME] = 5          # 0.5 s 읽기 타임아웃
        termios.tcsetattr(fd, termios.TCSANOW, attrs)
        self._fd = fd

    def close_port(self):
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close_port()

    # -- 저수준 --------------------------------------------------------
    def _transact(self, frame, expect_len, timeout_s=0.6):
        """프레임 하나를 보내고 정해진 길이만큼 읽는다.

        RTU 는 길이 필드가 없어 '얼마나 읽을지' 를 쓰는 쪽이 알아야 한다.
        여기서 오가는 프레임은 두 종류뿐이라 길이가 상수다.
        """
        if self._fd is None:
            raise GripperError("포트가 닫혀 있습니다")
        os.write(self._fd, frame)
        data = b""
        deadline = time.monotonic() + timeout_s
        while len(data) < expect_len and time.monotonic() < deadline:
            chunk = os.read(self._fd, expect_len - len(data))
            if chunk:
                data += chunk
            else:
                time.sleep(0.005)
        if len(data) < expect_len:
            raise GripperError(
                f"그리퍼가 응답하지 않습니다 ({len(data)}/{expect_len} 바이트)."
                f" 전원·슬레이브 주소({self.slave})·포트({self.port}) 확인")
        return data

    def status(self):
        return parse_status(self._transact(status_frame(self.slave),
                                           READ_REPLY_LEN),
                            self.max_opening_m)

    # -- 고수준 --------------------------------------------------------
    def activate(self, timeout_s=6.0):
        """리셋 후 활성화. 손가락이 한 번 완전히 왕복한다.

        **물체를 물린 채로 부르면 안 된다.** 활성화는 스트로크 전체를 훑는
        보정 동작이라 물려 있던 것을 떨어뜨린다. 그래서 세션에서는 파지
        단계 맨 앞에서만 부른다.
        """
        self._transact(reset_frame(self.slave), WRITE_REPLY_LEN)
        time.sleep(0.3)
        self._transact(activate_frame(self.slave), WRITE_REPLY_LEN)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            time.sleep(0.2)
            state = self.status()
            if state.fault:
                raise GripperError(f"그리퍼 결함 코드 {state.fault}")
            if state.activated and not state.moving:
                self.log(f"[그리퍼] 활성화 완료 — {state.describe()}")
                return state
        raise GripperError(f"활성화가 {timeout_s:.0f} 초 안에 끝나지 않았습니다")

    def move_to_counts(self, counts, wait=True, timeout_s=4.0):
        counts = int(min(max(int(counts), COUNTS_OPEN), COUNTS_CLOSED))
        self._transact(command_frame(counts, self.slave, self.speed,
                                     self.force), WRITE_REPLY_LEN)
        self.commanded_opening_m = opening_for_counts(counts,
                                                      self.max_opening_m)
        if not wait:
            return None
        # 명령 직후 한 번은 이전 상태가 읽힌다. gOBJ 가 '이동 중' 으로
        # 바뀔 틈을 준 뒤부터 도착을 본다.
        time.sleep(0.1)
        deadline = time.monotonic() + timeout_s
        state = self.status()
        while time.monotonic() < deadline:
            state = self.status()
            if state.fault:
                raise GripperError(f"그리퍼 결함 코드 {state.fault}")
            if not state.moving:
                return state
            time.sleep(0.05)
        return state

    def set_opening(self, opening_m, wait=True, timeout_s=4.0):
        """개구량을 [m] 로 명령한다. 돌려주는 것은 도착 후 상태.

        **물체를 물 때는 이걸 쓰면 안 된다** — grip() 을 쓴다. 물체 단면과
        같은 개구를 명령하면 죠가 닿기만 하고 힘이 안 걸린다.
        """
        return self.move_to_counts(
            counts_for_opening(opening_m, self.max_opening_m),
            wait=wait, timeout_s=timeout_s)

    def grip(self, opening_m, squeeze_m=CLAMP_SQUEEZE_M, wait=True,
             timeout_s=4.0):
        """단면이 opening_m 인 물체를 문다. 그보다 좁게 명령해 조인다."""
        return self.set_opening(max(0.0, float(opening_m) - float(squeeze_m)),
                                wait=wait, timeout_s=timeout_s)

    def open(self, wait=True):
        return self.move_to_counts(COUNTS_OPEN, wait=wait)

    def close(self, wait=True):
        return self.move_to_counts(COUNTS_CLOSED, wait=wait)


class SimGripper:
    """장비 없이 절차를 리허설할 때 쓰는 가짜 그리퍼.

    Robotiq2F85 와 같은 자리에 꽂힌다. 물체가 죠를 **막는 것**까지 흉내낸다:
    단면(hold_below_m) 보다 좁게 명령하면 죠는 단면에서 멈추고 gOBJ 는
    '물었다' 가 된다. 이게 있어야 grip() 의 여유분(CLAMP_SQUEEZE_M) 을 준
    뒤에도 되읽은 개구량이 계획값과 같게 나온다 — 실물과 같은 모양이다.
    """

    def __init__(self, max_opening_m=MAX_OPENING_M, hold_below_m=None,
                 force=DEFAULT_FORCE, log=print):
        self.max_opening_m = float(max_opening_m)
        self.hold_below_m = hold_below_m
        self.force = int(force)
        self.speed = DEFAULT_SPEED
        self.log = log
        self.port = "(모의)"
        self.commanded_opening_m = None
        self._counts = COUNTS_OPEN
        self._activated = False

    def open_port(self):
        pass

    def close_port(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def activate(self, timeout_s=6.0):
        self._activated = True
        self._counts = COUNTS_OPEN
        self.log("[그리퍼] (모의) 활성화 완료")
        return self.status()

    def status(self):
        requested = opening_for_counts(self._counts, self.max_opening_m)
        # 물체가 죠를 막는다. 단면보다 좁게 명령했으면 거기서 멈춘 것으로 본다.
        blocked = (self.hold_below_m is not None
                   and requested < self.hold_below_m - 1e-9)
        opening = self.hold_below_m if blocked else requested
        holding = bool(self._activated and blocked)
        return GripperStatus(
            activated=self._activated, moving=False,
            object_detected=holding, holding=holding, fault=0,
            counts=counts_for_opening(opening, self.max_opening_m),
            requested_counts=self._counts, opening_m=opening)

    def move_to_counts(self, counts, wait=True, timeout_s=4.0):
        self._counts = int(min(max(int(counts), COUNTS_OPEN), COUNTS_CLOSED))
        self.commanded_opening_m = opening_for_counts(self._counts,
                                                      self.max_opening_m)
        return self.status()

    def set_opening(self, opening_m, wait=True, timeout_s=4.0):
        return self.move_to_counts(
            counts_for_opening(opening_m, self.max_opening_m), wait=wait)

    def grip(self, opening_m, squeeze_m=CLAMP_SQUEEZE_M, wait=True,
             timeout_s=4.0):
        return self.set_opening(max(0.0, float(opening_m) - float(squeeze_m)),
                                wait=wait)

    def open(self, wait=True):
        return self.move_to_counts(COUNTS_OPEN, wait=wait)

    def close(self, wait=True):
        return self.move_to_counts(COUNTS_CLOSED, wait=wait)


# ---------------------------------------------------------------------------
# 키보드로 직접 몰기
# ---------------------------------------------------------------------------
def keyboard_drive(gripper, plan_opening_m=None, poll_s=0.04, log=print):
    """키보드로 개구량과 파지력을 실시간으로 조작한다.

    원본 robotiq_keyboard_toggle.py 는 스페이스바 하나로 열기/닫기만 했다.
    그걸로는 **이 물체에 얼마가 맞는지** 를 못 찾는다. 여기서는 개구와 힘을
    각각 따로, 키를 누르고 있는 동안 연속으로 바꾼다.

    돌려주는 것: 마지막 상태 (GripperStatus) 또는 None.
    """
    reader = KeyReader()
    if not reader.usable:
        raise GripperError("키보드 조작은 터미널에서만 됩니다 (stdin 이 tty 가 아님)")
    state = gripper.status()
    opening = state.opening_m
    log("\n" + KEY_HELP + "\n")
    with reader:
        last_draw = 0.0
        while True:
            keys = reader.poll()
            if keys:
                # 몰려 들어온 키를 **한 번의 명령**으로 접는다.
                opening, force, clamp, special = fold_keys(
                    keys, opening, gripper.force, gripper.max_opening_m,
                    plan_opening_m)
                if special == "quit":
                    break
                gripper.force = force
                try:
                    # wait=False. 키를 누르고 있는 동안 도착을 기다리면
                    # 입력이 쌓여 조작이 끈적해진다.
                    if clamp:
                        gripper.grip(opening, wait=False)
                    else:
                        gripper.set_opening(opening, wait=False)
                except GripperError as exc:
                    log(f"\n[그리퍼] 명령 실패: {exc}")
                if special == "accept":
                    state = gripper.status()
                    break
            if time.time() - last_draw > 0.1:
                try:
                    state = gripper.status()
                except GripperError:
                    pass
                else:
                    # 한 줄을 제자리에서 갱신한다 (스크롤을 안 흘리려고).
                    sys.stdout.write(
                        f"\r  개구 {1000*state.opening_m:5.1f} mm"
                        f" (목표 {1000*opening:5.1f})"
                        f"   힘 {force_newton(gripper.force):5.0f} N"
                        f"   {state.describe():<28}")
                    sys.stdout.flush()
                last_draw = time.time()
            time.sleep(poll_s)
    sys.stdout.write("\n")
    sys.stdout.flush()
    return state


# ---------------------------------------------------------------------------
# 자가 진단 — 장비 없이 프레임과 환산을 검증한다
# ---------------------------------------------------------------------------
def self_check(verbose=True):
    failures = []

    def case(name, fn):
        try:
            fn()
            if verbose:
                print(f"  [통과] {name}")
        except Exception as exc:                       # noqa: BLE001
            failures.append((name, exc))
            print(f"  [실패] {name}: {exc}")

    def t_frames():
        # 원본 robotiq_keyboard_toggle.py 가 내보내던 바이트와 같아야 한다.
        assert command_frame(0).hex(" ").startswith("09 10 03 e8 00 03 06 09")
        assert command_frame(255)[10] == 255
        assert len(command_frame(0)) == 15      # 본문 13 + CRC 2
        assert len(status_frame()) == 8
        assert crc16(command_frame(0)[:-2]) == command_frame(0)[-2:]

    def t_round_trip():
        for mm in (0.0, 12.5, 40.0, 85.0):
            counts = counts_for_opening(mm * 1e-3)
            back = 1000.0 * opening_for_counts(counts)
            assert abs(back - mm) < 0.4, (mm, counts, back)

    def t_bounds():
        assert counts_for_opening(-1.0) == COUNTS_CLOSED
        assert counts_for_opening(10.0) == COUNTS_OPEN

    def t_status_parse():
        # gACT=1, gGTO=1, gSTA=3, gOBJ=2 -> 0b10_11_1_001 = 0xB9
        payload = bytes((0xB9, 0x00, 0x00, 200, 198, 12))
        body = bytes((DEFAULT_SLAVE, 0x03, 0x06)) + payload
        state = parse_status(body + crc16(body))
        assert state.activated and state.holding and not state.moving
        assert state.counts == 198 and state.fault == 0
        assert abs(1000 * state.opening_m - 19.0) < 0.6, state.opening_m

    def t_status_crc():
        body = bytes((DEFAULT_SLAVE, 0x03, 0x06)) + bytes(6)
        bad = body + b"\x00\x00"
        try:
            parse_status(bad)
        except GripperError:
            return
        raise AssertionError("CRC 가 틀렸는데 통과했습니다")

    def t_sim():
        sim = SimGripper(hold_below_m=0.030, log=lambda *_: None)
        sim.activate()
        sim.open()
        assert not sim.status().holding
        sim.set_opening(0.028)
        assert sim.status().holding
        assert abs(sim.commanded_opening_m - 0.028) < 0.001

    def t_force():
        assert abs(force_newton(0) - FORCE_MIN_N) < 1e-6
        assert abs(force_newton(255) - FORCE_MAX_N) < 1e-6
        for newton in (20.0, 50.0, 120.0, 205.0, 235.0):
            back = force_newton(counts_for_force(newton))
            assert abs(back - newton) < 1.0, (newton, back)
        assert counts_for_force(1e6) == 255 and counts_for_force(-1) == 0
        # 기본값이 원본 스크립트(32, 약 47 N)보다 확실히 세야 한다.
        assert force_newton(DEFAULT_FORCE) > 150.0, force_newton(DEFAULT_FORCE)
        # 램프 0.56 kg 기준 미끄럼 여유가 넉넉해야 한다.
        assert slip_margin(DEFAULT_FORCE, 0.56) > 20.0

    def t_grip_squeezes():
        # 물릴 때는 단면보다 좁게 명령해야 힘이 걸린다.
        sim = SimGripper(hold_below_m=0.044, log=lambda *_: None)
        sim.activate()
        sim.set_opening(0.044)                 # 단면과 같은 개구 -> 안 물린다
        assert not sim.status().holding
        state = sim.grip(0.044)                # grip 은 더 좁게 명령한다
        assert state.holding, state
        # 그런데 되읽은 개구는 물체 단면 그대로여야 한다 (죠가 물체에 걸림).
        assert abs(state.opening_m - 0.044) < 1e-9, state.opening_m
        assert sim.commanded_opening_m < 0.044

    def t_keys():
        assert KeyReader.decode(b"ad") == ["a", "d"]
        assert KeyReader.decode(b"\x1b[A\x1b[D") == ["up", "left"]
        assert KeyReader.decode(b"w\x1b[Bq") == ["w", "down", "q"]
        # 몰려 들어온 키가 하나의 목표로 접혀야 한다.
        opening, force, clamp, special = fold_keys(
            ["d"] * 4, 0.060, 100)
        assert abs(opening - (0.060 - 4 * KEY_OPEN_STEP_M)) < 1e-9, opening
        assert force == 100 and not clamp and special is None
        opening, force, _, _ = fold_keys(["w"] * 3 + ["s"], 0.05, 100)
        assert force == 100 + 2 * KEY_FORCE_STEP, force
        # 범위를 넘어가지 않는다.
        assert fold_keys(["a"] * 500, 0.05, 100)[0] == MAX_OPENING_M
        assert fold_keys(["d"] * 500, 0.05, 100)[0] == 0.0
        assert fold_keys(["w"] * 500, 0.05, 100)[1] == 255
        # 특수 키
        assert fold_keys(["q"], 0.05, 100)[3] == "quit"
        assert fold_keys(["\r"], 0.05, 100)[3] == "accept"
        assert fold_keys(["g"], 0.05, 100, plan_opening_m=0.044)[2] is True
        assert abs(fold_keys(["g"], 0.05, 100, plan_opening_m=0.044)[0]
                   - 0.044) < 1e-9

    case("명령 프레임", t_frames)
    case("키보드 입력 해석", t_keys)
    case("파지력 <-> N 환산", t_force)
    case("물릴 때 더 좁게 명령한다", t_grip_squeezes)
    case("개구량 <-> 카운트 왕복", t_round_trip)
    case("범위 넘어간 개구량", t_bounds)
    case("상태 바이트 해석", t_status_parse)
    case("상태 CRC 검사", t_status_crc)
    case("모의 그리퍼", t_sim)
    return failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=DEFAULT_PORT)
    ap.add_argument("--slave", type=int, default=DEFAULT_SLAVE)
    ap.add_argument("--check", action="store_true",
                    help="장비 없이 프레임·환산을 검증한다")
    ap.add_argument("--demo", action="store_true",
                    help="실제로 붙여 활성화 후 열고 닫는다")
    ap.add_argument("--opening-mm", type=float, default=None,
                    help="이 개구량으로 이동하고 끝낸다")
    ap.add_argument("--force-n", type=float, default=None,
                    help=f"파지력 [N] {FORCE_MIN_N:.0f}~{FORCE_MAX_N:.0f}"
                         f" (기본 {force_newton(DEFAULT_FORCE):.0f})")
    ap.add_argument("--grip-mm", type=float, default=None,
                    help="이 단면의 물체를 문다 (그보다 좁게 명령해 조인다)."
                         " --force-n 을 바꿔 가며 흔들어 보는 용도")
    ap.add_argument("--keyboard", action="store_true",
                    help="키보드로 개구량과 파지력을 직접 조작한다")
    ap.add_argument("--plan-mm", type=float, default=None,
                    help="--keyboard 에서 g 키가 물 개구량 [mm]")
    args = ap.parse_args()

    if args.check or not (args.demo or args.keyboard
                          or args.opening_mm is not None
                          or args.grip_mm is not None):
        print("gripper_hw 자가 진단")
        failures = self_check()
        print(f"\n{'모두 통과' if not failures else f'{len(failures)} 개 실패'}")
        return 1 if failures else 0

    force = (DEFAULT_FORCE if args.force_n is None
             else counts_for_force(args.force_n))
    print(f"파지력 {force_newton(force):.0f} N (rFR {force}/255)")
    with Robotiq2F85(port=args.port, slave=args.slave, force=force) as grip:
        grip.activate()
        if args.keyboard:
            plan = (None if args.plan_mm is None else args.plan_mm * 1e-3)
            state = keyboard_drive(grip, plan_opening_m=plan)
            if state is not None:
                print(f"[그리퍼] 마지막 상태 {state.describe()}"
                      f"  힘 {force_newton(grip.force):.0f} N"
                      f"  (--gripper-force {force_newton(grip.force):.0f} 로"
                      f" 세션에 넘기면 됩니다)")
            return 0
        if args.grip_mm is not None:
            state = grip.grip(args.grip_mm * 1e-3)
            print(f"[그리퍼] {state.describe()}"
                  f"  물림 {'예' if state.holding else '아니오'}")
            return 0
        if args.opening_mm is not None:
            state = grip.set_opening(args.opening_mm * 1e-3)
            print(f"[그리퍼] {state.describe()}")
            return 0
        for label, opening_mm in (("활짝", 85.0), ("절반", 40.0), ("닫힘", 0.0)):
            state = grip.set_opening(opening_mm * 1e-3)
            print(f"[그리퍼] {label} {opening_mm:.0f} mm -> {state.describe()}")
            time.sleep(0.5)
    return 0


if __name__ == "__main__":
    sys.exit(main())
