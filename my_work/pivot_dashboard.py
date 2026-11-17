#!/usr/bin/env python3
"""HTML 통합 지휘 화면. 기존 프로세스의 JSON과 Meshcat을 한곳에 모은다."""

import json
import math
import os
import re
import signal
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
        self.experiment_process = None
        self.experiment_stop_requested = False
        self.lock = threading.Lock()
        self.server = self.thread = None
        self.ft_stream = None
        self.ft_error = None
        if self.conf.get('FT_GRIPPER_MODEL'):
            try:
                import sys
                sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
                from ft_compensation_stream import CompensationStream
                self.ft_stream = CompensationStream(self.conf['ROBOT_HOST'], self.conf['FT_GRIPPER_MODEL'],
                    Path(self.conf['FP_OUTPUT'])/'hardware.json', self.session.path('ft_display'), continuous_display=True)
            except Exception as exc:
                self.ft_error = str(exc)

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
        now = time.time()
        stamp = fp.get("timestamp_s")
        fp_age = (now - stamp if isinstance(stamp, (int, float))
                  and math.isfinite(stamp) else None)
        image_age = now - image.stat().st_mtime if image.is_file() else None
        stamp = hardware.get("timestamp_s")
        hardware_age = (now - stamp if isinstance(stamp, (int, float))
                        and math.isfinite(stamp) else None)
        compensation = None
        if self.conf.get('FT_GRIPPER_MODEL'):
            compensation = self.ft_stream.snapshot() if self.ft_stream else dict(error=self.ft_error)
            if self.ft_stream:
                self.ft_stream.discard()
                compensation = dict(compensation, selection_path=self.conf['FT_GRIPPER_MODEL'],
                    model_path=self.ft_stream.model['_source_path'],
                    model_sha256=self.ft_stream.model['_source_sha256'],
                    frozen_bias=self.ft_stream.model['frozen_bias'],
                    zero_adjustment=self.ft_stream.model.get('_zero_adjustment'),
                    average_window_s=.3, sensor_delay_s=.025)
        return {
            "ft_compensation": compensation,
            "phase": self.session.phase(),
            "status": status,
            "prompt": prompt,
            "preflight": preflight,
            "foundationpose": fp,
            "foundationpose_fresh": fp_age is not None and 0 <= fp_age <= 2.0,
            "foundationpose_age_s": fp_age,
            "hardware": hardware,
            "hardware_fresh": hardware_age is not None and 0 <= hardware_age <= 2.0,
            "camera_ready": image_age is not None and 0 <= image_age <= 2.0,
            "camera_age_s": image_age,
            "ft_validation": self.session.read("ft_validation.json", {}) or {},
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
            "exploration_active": (self.experiment_process is not None
                                   and self.experiment_process.poll() is None),
            "exploration_stop_requested": self.experiment_stop_requested,
        }

    def set_experiment_process(self, process):
        with self.lock:
            self.experiment_process = process
            self.experiment_stop_requested = False

    def stop_exploration(self):
        with self.lock:
            process = self.experiment_process
            if (self.experiment_stop_requested or process is None
                    or process.poll() is not None):
                return False
            self.experiment_stop_requested = True
        os.kill(process.pid, signal.SIGUSR1)
        return True

    def hardware_command(self, action, port=None, speed=None, force=None):
        if action not in {'refresh_ports', 'reconnect', 'activate', 'open', 'close'}:
            return False
        if port is not None and not re.fullmatch(r'/dev/tty(?:USB|ACM)\d+', str(port)):
            return False
        try:
            speed, force = int(speed), int(force)
        except (TypeError, ValueError):
            return False
        if not 1 <= speed <= 255 or not 1 <= force <= 255:
            return False
        output = Path(self.conf.get('FP_OUTPUT', '/tmp/lamp_foundationpose_live'))
        output.mkdir(parents=True, exist_ok=True)
        target = output/'hardware_command.json'
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(id=time.time_ns(), action=action, port=port,
                                            speed=speed, force=force))+'\n')
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

    def reload_ft(self):
        from ft_compensation_stream import CompensationStream
        with self.lock:
            if self.experiment_process is not None and self.experiment_process.poll() is None:
                return False, '실험을 마친 뒤 보정 파일을 다시 읽으세요'
            try:
                stream = CompensationStream(self.conf['ROBOT_HOST'], self.conf['FT_GRIPPER_MODEL'],
                    Path(self.conf['FP_OUTPUT'])/'hardware.json', self.session.path('ft_display'), continuous_display=True)
            except Exception as exc:
                return False, str(exc)
            previous, self.ft_stream = self.ft_stream, stream
            self.ft_error = None
        if previous:
            previous.close()
        return True, '보정 파일을 읽었습니다. 새 표본을 기다립니다'

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
                if self.path == '/api/ft-reload':
                    accepted, message = dashboard.reload_ft()
                    return self.send_bytes(json.dumps(dict(accepted=accepted, message=message), ensure_ascii=False).encode(),
                                           'application/json', 200 if accepted else 409)
                if self.path == '/api/hardware':
                    accepted = dashboard.hardware_command(request.get('action'), request.get('port'),
                                                          request.get('speed'), request.get('force'))
                    return self.send_bytes(json.dumps(dict(accepted=accepted)).encode(),
                                           'application/json', 200 if accepted else 409)
                if self.path == '/api/zero-adjustment':
                    return self.send_bytes(json.dumps({'accepted': False, 'error': '이 화면은 F/T 표시 전용입니다. 장치 조작은 FT 실험 GUI를 사용하세요'}, ensure_ascii=False).encode(), 'application/json', 409)
                if self.path == "/api/stop-exploration":
                    accepted = dashboard.stop_exploration()
                    return self.send_bytes(
                        json.dumps({"accepted": accepted}).encode(),
                        "application/json", 200 if accepted else 409)
                root = Path(dashboard.conf.get("PIVOT_ROOT", Path(__file__).parents[1]))
                if self.path == "/api/remask":
                    # 파지 후 재마스킹은 FoundationPose 좌표계를 바꾸어
                    # grasp.json·가상환경·밀도 회귀가 서로 어괋난다.
                    if dashboard.session.phase().get("index", 0) >= 1:
                        return self.send_bytes(
                            json.dumps({"accepted": False,
                                        "error": "파지 후에는 재마스킹할 수 없습니다. 실험을 재시작하세요."},
                                       ensure_ascii=False).encode(),
                            "application/json; charset=utf-8", 409)
                    accepted = dashboard.start_job(
                        "FoundationPose 재마스킹",
                        [str(root / "setup/launch_experiment.sh"), "--remask"],
                        "/tmp/pivot_remask.log")
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
        if self.ft_stream:
            self.ft_stream.close()
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()

    def web_url(self):
        return f"http://{self.host}:{self.port}"
