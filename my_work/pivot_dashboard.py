#!/usr/bin/env python3
"""HTML 통합 지휘 화면. 기존 프로세스의 JSON과 Meshcat을 한곳에 모은다."""

import json
import re
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class Dashboard:
    def __init__(self, session, conf, meshcat_url, host="127.0.0.1", port=8080):
        self.session = session
        self.conf = conf
        self.meshcat_url = meshcat_url
        self.html = Path(__file__).with_name("pivot_dashboard.html")
        self.host, self.port = host, int(port)
        self.status = "시작 중"
        self.current_prompt = None
        self.clicks = []
        self.jobs = {}
        self.lock = threading.Lock()
        self.server = self.thread = None

    @staticmethod
    def _read_json(path, default=None):
        try:
            return json.loads(Path(path).read_text())
        except (OSError, ValueError, TypeError):
            return default

    def set_status(self, text):
        with self.lock:
            self.status = str(text)

    def prompt(self, label):
        with self.lock:
            self.current_prompt = str(label)

    def clear_prompt(self):
        with self.lock:
            self.current_prompt = None

    def consume(self, label):
        with self.lock:
            if label not in self.clicks:
                return False
            self.clicks.remove(label)
            return True

    def _state(self):
        with self.lock:
            status, prompt, jobs = self.status, self.current_prompt, dict(self.jobs)
        output = Path(self.conf.get("FP_OUTPUT", "/tmp/lamp_foundationpose_live"))
        fp = self._read_json(output / "latest.json", {}) or {}
        hardware = self._read_json(output / "hardware.json", {}) or {}
        preflight = self.session.read("preflight.json", {}) or {}
        posterior_files = self.session.rounds()
        posterior = self._read_json(posterior_files[-1], {}) if posterior_files else {}
        operator = self._read_json(self.session.path("operator_ui.json"), {}) or {}
        image = output / "latest.jpg"
        if not image.is_file():
            image = output / "latest.png"
        if not image.is_file():
            image = Path(self.conf.get("FP_INIT_RGB", ""))
        return {
            "phase": self.session.phase(),
            "status": status,
            "prompt": prompt,
            "preflight": preflight,
            "foundationpose": fp,
            "hardware": hardware,
            "camera_ready": image.is_file(),
            "posterior": posterior or {},
            "meshcat_url": (operator.get("planner_url")
                            if self.session.phase().get("index") == 4
                            and operator.get("planner_url") else self.meshcat_url),
            "scene_url": (operator.get("scene_url")
                          if self.session.phase().get("index") == 4 else None),
            "operator": operator,
            "hardware_window": "AFT200·Robotiq 장치 백엔드 실행 중",
            "gripper_port": self.conf.get("GRIPPER_PORT", "/dev/ttyUSB0"),
            "jobs": {name: ("실행 중" if process.poll() is None else
                            "완료" if process.returncode == 0 else
                            f"실패 ({process.returncode})")
                     for name, process in jobs.items()},
        }

    def hardware_command(self, action, port=None, speed=None, force=None):
        if action not in {"refresh_ports", "reconnect", "activate", "open", "close",
                          "toggle_recording"}:
            return False
        if port is not None and not re.fullmatch(r"/dev/tty(?:USB|ACM)\d+", str(port)):
            return False
        try:
            speed, force = int(speed), int(force)
        except (TypeError, ValueError):
            return False
        if not 1 <= speed <= 255 or not 1 <= force <= 255:
            return False
        output = Path(self.conf.get("FP_OUTPUT", "/tmp/lamp_foundationpose_live"))
        output.mkdir(parents=True, exist_ok=True)
        target = output / "hardware_command.json"
        payload = {"id": time.time_ns(), "action": action, "port": port,
                   "speed": speed, "force": force}
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload) + "\n")
        temporary.replace(target)
        return True

    def start_job(self, name, command, log_path):
        with self.lock:
            running = self.jobs.get(name)
            if running is not None and running.poll() is None:
                return False
            log = open(log_path, "a", buffering=1)
            self.jobs[name] = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
            log.close()
        return True

    def _handler(self):
        dashboard = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                return

            def send_bytes(self, body, content_type, status=200):
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/" or self.path.startswith("/?"):
                    return self.send_bytes(dashboard.html.read_bytes(),
                                           "text/html; charset=utf-8")
                if self.path.startswith("/api/state"):
                    body = json.dumps(dashboard._state(), ensure_ascii=False).encode()
                    return self.send_bytes(body, "application/json; charset=utf-8")
                if self.path.startswith("/camera.jpg"):
                    output = Path(dashboard.conf.get(
                        "FP_OUTPUT", "/tmp/lamp_foundationpose_live"))
                    image = output / "latest.jpg"
                    if not image.is_file():
                        image = output / "latest.png"
                    if not image.is_file():
                        image = Path(dashboard.conf.get("FP_INIT_RGB", ""))
                    if image.is_file():
                        kind = "image/png" if image.suffix.lower() == ".png" else "image/jpeg"
                        return self.send_bytes(image.read_bytes(), kind)
                    return self.send_bytes(b"camera image pending", "text/plain", 404)
                return self.send_bytes(b"not found", "text/plain", 404)

            def do_POST(self):
                try:
                    size = min(int(self.headers.get("Content-Length", "0")), 4096)
                    request = json.loads(self.rfile.read(size))
                except (ValueError, TypeError):
                    request = {}
                if self.path == "/api/hardware":
                    accepted = dashboard.hardware_command(
                        request.get("action"), request.get("port"),
                        request.get("speed"), request.get("force"))
                    return self.send_bytes(
                        json.dumps({"accepted": accepted}).encode(),
                        "application/json", 200 if accepted else 409)
                root = Path(dashboard.conf.get("PIVOT_ROOT", Path(__file__).parents[1]))
                if self.path == "/api/remask":
                    accepted = dashboard.start_job(
                        "FoundationPose 재마스킹",
                        [str(root / "setup/launch_experiment.sh"), "--remask"],
                        "/tmp/pivot_remask.log")
                    return self.send_bytes(
                        json.dumps({"accepted": accepted}).encode(),
                        "application/json", 200 if accepted else 409)
                if self.path == "/api/zero-adjustment":
                    accepted = dashboard.start_job(
                        "영점 다시 측정",
                        [str(root / "setup/wrist_tare.sh"), "--run"],
                        "/tmp/pivot_zero_adjustment.log")
                    return self.send_bytes(
                        json.dumps({"accepted": accepted}).encode(),
                        "application/json", 200 if accepted else 409)
                if self.path == "/api/operator":
                    state = dashboard._read_json(
                        dashboard.session.path("operator_ui.json"), {}) or {}
                    action = request.get("action")
                    accepted = action in state.get("buttons", [])
                    if accepted:
                        target = dashboard.session.path("operator_action.json")
                        temporary = target.with_suffix(".tmp")
                        temporary.write_text(json.dumps(
                            {"id": time.time_ns(), "action": action},
                            ensure_ascii=False) + "\n")
                        temporary.replace(target)
                    return self.send_bytes(
                        json.dumps({"accepted": accepted}).encode(),
                        "application/json", 200 if accepted else 409)
                if self.path != "/api/action":
                    return self.send_bytes(b"not found", "text/plain", 404)
                action = request.get("action")
                with dashboard.lock:
                    allowed = action and action == dashboard.current_prompt
                    if allowed:
                        dashboard.clicks.append(action)
                body = json.dumps({"accepted": bool(allowed)}).encode()
                self.send_bytes(body, "application/json", 200 if allowed else 409)

        return Handler

    def start(self):
        try:
            self.server = ThreadingHTTPServer((self.host, self.port), self._handler())
        except OSError:
            self.server = ThreadingHTTPServer((self.host, 0), self._handler())
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return self

    def stop(self):
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()

    def web_url(self):
        return f"http://{self.host}:{self.port}"
