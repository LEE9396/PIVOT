"""계획 쪽과 로봇 쪽을 잇는 통신 경로.

dual_view 는 두 화면이 **버스로만** 대화하도록 만들어져 있다. 메시지는 전부
JSON 직렬화 가능한 dict 라, 버스 구현만 갈아끼우면 로봇 쪽이 다른 프로세스나
다른 PC, 나아가 실물 로봇이 된다. 알고리즘 코드는 한 줄도 안 바뀐다.

    planner --- exploration_target ---> robot     (관절각 목표)
    planner <--- measurement ---------- robot     (렌치 18개)

메시지 규약
-----------
exploration_target
    {"round": int,
     "object_joint_deg": [float, ...],      # 작업자가 맞출 목표 각도
     "reason": str}

measurement
    {"round": int,
     "object_joint_deg": [float, ...],          # 실제로 맞춘 각도(시뮬만)
     "object_joint_deg_measured": [float, ...], # FoundationPose 가 읽은 값
     "wrench": [float x 6*방향수],              # **타어링 끝난** 물체만의 렌치
     "aborted": bool}

wrench 는 반드시 타어링을 끝낸 값이어야 한다. 그리퍼·마운트·손가락 무게가
섞여 들어오면 밀도가 통째로 틀어진다 (hardware.TareTable 참고).

실행 (연결 확인용 반향 서버):
    ../robot_learning/scripts/run_drake_env.sh python pose_bus.py --echo
"""

import json
import queue
import socket
import threading
from abc import ABC, abstractmethod


class PoseBus(ABC):
    @abstractmethod
    def send_target(self, message): ...

    @abstractmethod
    def recv_target(self): ...

    @abstractmethod
    def send_measurement(self, message): ...

    @abstractmethod
    def recv_measurement(self): ...

    def close(self):
        pass


class LocalBus(PoseBus):
    """같은 프로세스 안에서 큐로 주고받는다. 기본 구현."""

    def __init__(self):
        self._targets = queue.Queue()
        self._measurements = queue.Queue()

    def send_target(self, message):
        self._targets.put(message)

    def recv_target(self):
        return self._targets.get()

    def send_measurement(self, message):
        self._measurements.put(message)

    def recv_measurement(self):
        return self._measurements.get()


# ---------------------------------------------------------------------------
# TCP/IP — 로봇 쪽을 다른 프로세스나 다른 PC 에서 돌릴 때
# ---------------------------------------------------------------------------
class _JsonLineChannel:
    """한 줄에 JSON 하나. 사람이 읽을 수 있어 디버깅이 쉽다."""

    def __init__(self, sock):
        self.sock = sock
        self.reader = sock.makefile("r", encoding="utf-8")

    def send(self, message):
        line = json.dumps(message, ensure_ascii=False) + "\n"
        self.sock.sendall(line.encode("utf-8"))

    def recv(self):
        line = self.reader.readline()
        if not line:
            raise ConnectionError("상대가 연결을 끊었습니다")
        return json.loads(line)

    def close(self):
        try:
            self.reader.close()
        finally:
            self.sock.close()


class TcpBus(PoseBus):
    """소켓 하나로 양방향. 계획 쪽이 서버, 로봇 쪽이 클라이언트다.

    계획 쪽 (왼쪽 화면):
        bus = TcpBus.serve(port=5555)        # 로봇이 붙을 때까지 기다림
    로봇 쪽 (오른쪽 화면 / 실물):
        bus = TcpBus.connect("192.168.0.10", 5555)

    양쪽 다 같은 send_*/recv_* 를 쓴다. 어느 쪽이 어느 메시지를 보내는지는
    dual_view 의 루프가 정한다.
    """

    def __init__(self, channel, role):
        self.channel = channel
        self.role = role            # "planner" | "robot"

    # -- 연결 --------------------------------------------------------
    @classmethod
    def serve(cls, port=5555, host="0.0.0.0", timeout_s=None, log=print):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((host, port))
        listener.listen(1)
        listener.settimeout(timeout_s)
        log(f"[버스] {host}:{port} 에서 로봇 쪽 연결을 기다립니다...")
        sock, peer = listener.accept()
        listener.close()
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        log(f"[버스] {peer[0]}:{peer[1]} 연결됨")
        return cls(_JsonLineChannel(sock), "planner")

    @classmethod
    def connect(cls, host, port=5555, timeout_s=10.0, log=print):
        sock = socket.create_connection((host, port), timeout=timeout_s)
        sock.settimeout(None)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        log(f"[버스] {host}:{port} 에 연결됨")
        return cls(_JsonLineChannel(sock), "robot")

    # -- 메시지 ------------------------------------------------------
    #    한 소켓을 쓰므로 종류를 태그로 구분한다.
    def _send(self, kind, message):
        self.channel.send({"kind": kind, "body": message})

    def _recv(self, kind):
        while True:
            packet = self.channel.recv()
            if packet.get("kind") == kind:
                return packet["body"]
            # 다른 종류가 먼저 오는 일은 이 프로토콜에선 없다. 오면 알린다.
            raise ValueError(f"{kind} 를 기다렸는데 {packet.get('kind')} 가 왔다")

    def send_target(self, message):
        self._send("target", message)

    def recv_target(self):
        return self._recv("target")

    def send_measurement(self, message):
        self._send("measurement", message)

    def recv_measurement(self):
        return self._recv("measurement")

    def close(self):
        self.channel.close()


# ---------------------------------------------------------------------------
# ROS1 — 실물 RB5 드라이버·AFT200 노드가 ROS1 에 있을 때
# ---------------------------------------------------------------------------
class Ros1Bus(PoseBus):
    """ROS1 토픽 위의 버스.

    이 파일이 도는 파이썬에는 Drake 가 들어 있고 ROS1 은 보통 없다. 그래서
    두 가지 방법 중 하나를 쓴다.

      (A) 권장 — 로봇 쪽만 ROS1 파이썬으로 따로 띄우고, 계획 쪽과는 TcpBus 로
          잇는다. 그러면 이 클래스가 필요 없다. 환경이 섞이지 않아 안전하다.

      (B) 같은 프로세스에서 rospy 를 쓸 수 있다면 아래를 채운다.
              /exploration_target   (std_msgs/String, JSON 한 줄)
              /object_measurement   (std_msgs/String, JSON 한 줄)
          커스텀 msg 를 만들지 않고 JSON 문자열을 실어 보내면 규약이 그대로
          유지되고, 나중에 TcpBus 로 되돌리기도 쉽다.
    """

    def __init__(self, target_topic="/exploration_target",
                 measurement_topic="/object_measurement", queue_size=1):
        try:
            import rospy
            from std_msgs.msg import String
        except ImportError as exc:                       # pragma: no cover
            raise ImportError(
                "rospy 를 찾지 못했습니다. 권장 방식은 로봇 쪽을 ROS1 파이썬으로"
                " 따로 띄우고 TcpBus 로 잇는 것입니다 (환경이 섞이지 않습니다)."
            ) from exc
        self._rospy = rospy
        self._String = String
        self._targets = queue.Queue()
        self._measurements = queue.Queue()
        self._pub_target = rospy.Publisher(target_topic, String,
                                           queue_size=queue_size)
        self._pub_meas = rospy.Publisher(measurement_topic, String,
                                         queue_size=queue_size)
        rospy.Subscriber(target_topic, String,
                         lambda m: self._targets.put(json.loads(m.data)))
        rospy.Subscriber(measurement_topic, String,
                         lambda m: self._measurements.put(json.loads(m.data)))

    def send_target(self, message):
        self._pub_target.publish(self._String(json.dumps(message)))

    def recv_target(self):
        return self._targets.get()

    def send_measurement(self, message):
        self._pub_meas.publish(self._String(json.dumps(message)))

    def recv_measurement(self):
        return self._measurements.get()


# ---------------------------------------------------------------------------
def _echo_server(port):
    """연결 확인용. 목표를 받아 그대로 되돌려 준다."""
    bus = TcpBus.serve(port=port)
    try:
        while True:
            target = bus.recv_target()
            print(f"[반향] 받음 {target}")
            bus.send_measurement(dict(round=target.get("round", 0),
                                      object_joint_deg=target.get(
                                          "object_joint_deg", []),
                                      wrench=[0.0] * 18, aborted=False))
    except (ConnectionError, KeyboardInterrupt):
        print("[반향] 종료")
    finally:
        bus.close()


def _loopback_test(port=5599):
    """서버와 클라이언트를 한 프로세스에서 띄워 왕복을 확인한다."""
    result = {}

    def robot_side():
        bus = TcpBus.connect("127.0.0.1", port, log=lambda *_: None)
        target = bus.recv_target()
        bus.send_measurement(dict(round=target["round"],
                                  object_joint_deg=target["object_joint_deg"],
                                  object_joint_deg_measured=[45.3, 44.8],
                                  wrench=[0.1] * 18, aborted=False))
        bus.close()

    # serve() 는 상대가 붙을 때까지 막히므로 다른 스레드에서 띄운다.
    def planner_side():
        server = TcpBus.serve(port=port, log=lambda *_: None)
        server.send_target(dict(round=1, object_joint_deg=[45.0, 45.0],
                                reason="D-optimal=39.2"))
        result["reply"] = server.recv_measurement()
        server.close()

    thread = threading.Thread(target=planner_side, daemon=True)
    thread.start()
    import time
    for _ in range(50):                 # 서버가 listen 할 때까지 잠깐 기다린다
        try:
            robot_side()
            break
        except (ConnectionRefusedError, OSError):
            time.sleep(0.1)
    thread.join(timeout=5.0)
    print(f"왕복 성공: {result['reply']}")
    assert result["reply"]["round"] == 1
    assert len(result["reply"]["wrench"]) == 18


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--echo", action="store_true", help="반향 서버를 띄운다")
    ap.add_argument("--test", action="store_true", help="왕복 자체 확인")
    ap.add_argument("--port", type=int, default=5555)
    args = ap.parse_args()
    if args.echo:
        _echo_server(args.port)
    else:
        _loopback_test()
