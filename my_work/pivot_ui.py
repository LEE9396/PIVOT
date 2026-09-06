#!/usr/bin/env python
"""PIVOT 통합 UI — 창 1(지휘자). 단계 0~5 를 하나로 묶는다.

설계는 pivot_ui_design.html 그대로다. 핵심 두 가지.

**로봇은 나중에 들어온다.** 0~2 단계는 물체만 올린다. 파지점을 고르고 각도를
추천하는 동안 로봇·그리퍼는 화면에도 판정에도 없다. 예전에는 `prepare()` 가
시작하자마자 로봇 씬을 세워서, 파지점 짐작이 어긋나면 "힌지·도달·충돌을 모두
통과하는 자세가 없다" 로 **시작조차 못 했다**. 2026-09-02 사고가 그것이었다.

**용접 변환은 짐작이 아니라 측정이다.** 1단계 「파지 완료」 시점에 창 2 의
물체 자세 + 핸드아이 + 로봇 q 로 X_G_O 를 계산해 세션에 남긴다. 3단계에서
로봇 씬을 세울 때 그 값을 WeldFrames 에 그대로 쓴다. GRASP_LONG_AXIS 규약,
grasp_rotation, 볼록 조각 정점평균 같은 짐작이 전부 필요 없어진다.

창 2·3 은 세션 폴더의 phase.json 을 폴링해 자기 표시를 바꾼다. 창 1 만
쓰고 나머지는 읽는다 — 파이썬 환경이 서로 달라도(conda/venv) 파일 하나면
된다.

실행
    $R python pivot_ui.py --conf ../setup/experiment.conf
    $R python pivot_ui.py --dry-run          # 장비·Drake 없이 단계 기계만
"""

import argparse
import fcntl
import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "tools"))

from pivot_session import PHASES, Session, describe          # noqa: E402

TOOLS = HERE.parent / "tools"
INSTANCE_LOCK = Path("/tmp/pivot_ui.lock")


def acquire_instance_lock():
    """통합 UI가 두 세션을 동시에 조종하지 못하게 막는다."""
    lock = INSTANCE_LOCK.open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        return None
    lock.seek(0)
    lock.truncate()
    lock.write(f"{os.getpid()}\n")
    lock.flush()
    return lock


# ---------------------------------------------------------------------------
class Conductor:
    """단계 기계. 창 1 이 이걸 돌린다."""

    def __init__(self, conf_path, session, console=None, auto=False,
                 dashboard=None):
        self.conf_path = Path(conf_path)
        self.conf = self._read_conf()
        self.session = session
        self.console = console
        self.dashboard = dashboard
        self.auto = auto
        self.round = 0

    # -- 설정 -------------------------------------------------------------
    def _read_conf(self):
        sys.path.insert(0, str(TOOLS))
        from preflight import read_conf
        return read_conf(self.conf_path)

    def python(self):
        """PIVOT 쪽 파이썬은 반드시 run_drake_env.sh 를 거친다."""
        root = Path(self.conf.get("PIVOT_ROOT", HERE.parent)).expanduser()
        return [str(root / "robot_learning" / "scripts" / "run_drake_env.sh"),
                "python"]

    # -- 화면 -------------------------------------------------------------
    def bar(self):
        line = describe(self.session)
        print(f"\n{'=' * 78}\n{line}\n{'=' * 78}")
        if self.dashboard is not None:
            self.dashboard.set_status(line)
        if self.console is not None:
            from pydrake.geometry import Rgba
            self.console.lamp(Rgba(0.10, 0.37, 0.72, 1.0), line)

    def ask(self, label):
        """사람이 누를 때까지 기다린다. 콘솔이 없으면 터미널로."""
        if self.auto:
            print(f"  [자동] {label}")
            return True
        if self.dashboard is not None:
            self.dashboard.prompt(label)
        if self.console is None:
            if self.dashboard is None:
                input(f"  >>> {label} — Enter")
            else:
                while not self.dashboard.consume(label):
                    time.sleep(0.05)
                self.dashboard.clear_prompt()
            return True
        self.console.clear()
        name = self.console.button(label)
        start = self.console.meshcat.GetButtonClicks(name)
        while self.console.meshcat.GetButtonClicks(name) == start:
            if self.dashboard is not None and self.dashboard.consume(label):
                break
            time.sleep(0.05)
        if self.dashboard is not None:
            self.dashboard.clear_prompt()
        return True

    # -- 0단계 ------------------------------------------------------------
    def wait_for_tracker(self, timeout_s=180):
        output = self.conf.get("FP_OUTPUT")
        if not output:
            return True
        latest = Path(output).expanduser() / "latest.json"
        print(f"  FoundationPose 첫 각도를 기다립니다: {latest}")
        if self.dashboard is not None:
            self.dashboard.set_status("[0 준비] 카메라 마스크와 FoundationPose 대기")
        if self.console is not None:
            from pydrake.geometry import Rgba
            self.console.lamp(
                Rgba(0.10, 0.37, 0.72, 1.0),
                "[0 준비] 카메라 마스크와 FoundationPose 대기")
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if latest.is_file() and time.time() - latest.stat().st_mtime <= 2.0:
                if self.dashboard is not None:
                    self.dashboard.set_status("[0 준비] FoundationPose 연결됨")
                return True
            time.sleep(0.5)
        print("  [실패] FoundationPose가 180초 안에 새 각도를 내지 않았습니다.")
        return False

    def phase_preflight(self):
        self.session.set_phase("preflight")
        self.bar()
        self.show_object_only()  # 재마스킹 중에도 창 1에서 부위 이름을 대조한다.
        while True:
            tracker_ready = self.wait_for_tracker()
            out = self.session.path("preflight.json")
            result = subprocess.run(
                self.python() + [str(TOOLS / "preflight.py"),
                                 "--conf", str(self.conf_path),
                                 "--json", str(out)])
            data = self.session.read("preflight.json", {})
            verdict = data.get("verdict", "FAIL")
            if tracker_ready and verdict != "FAIL" and result.returncode == 0:
                if verdict == "WARN":
                    print("\n  주의 항목이 있습니다. 결과가 조용히 나빠질 수 있습니다.")
                    self.ask("주의를 알고도 계속한다")
                return True
            print("\n  준비 점검에 **실패**가 있습니다. 고치기 전에는 못 갑니다.")
            if self.dashboard is not None:
                self.dashboard.set_status("[0 준비] 실패 항목을 고친 뒤 다시 점검하세요")
            if self.auto:
                return False
            self.ask("준비 점검 다시 실행")

    # -- 1단계 ------------------------------------------------------------
    def phase_grasp(self):
        self.session.set_phase("grasp")
        self.bar()
        print("  창 1 에 물체만 띄웁니다. 파지 후보를 보고 창 3 으로 무세요.")
        print("  창 2 에도 같은 점이 겹쳐 보입니다.")
        self.show_object_only()
        self.ask("파지 완료 — 물체를 물렸고 손을 뗐습니다")

        print("\n  파지 변환을 **잽니다** (짐작하지 않습니다)")
        pose_file = self.conf.get("FP_OUTPUT", "")
        command = self.python() + [
            str(TOOLS / "grasp_measure.py"),
            "--pose-file", str(Path(pose_file) / "latest.json"),
            "--session", str(self.session.root),
            "--part", self.conf.get("FP_GRASP_PART", "support")]
        host = self.conf.get("ROBOT_HOST")
        if host:
            command += ["--robot-host", host]
        if subprocess.run(command).returncode != 0:
            print("  [실패] 파지 변환을 못 쟀습니다.")
            print("  창 2 가 각도를 내고 있는지, 핸드아이가 있는지 보세요.")
            print("  실측 파지 없이는 자세·충돌 계산이 실제 물체 배치와 다르므로"
                  " 진행하지 않습니다.")
            return False
        return True

    def show_object_only(self):
        """물체만 올린 가벼운 Meshcat 뷰. 로봇은 여기 없다."""
        try:
            import numpy as np
            from pydrake.geometry import Mesh, Rgba, Sphere, StartMeshcat
            from pydrake.math import RigidTransform, RotationMatrix
            from pydrake.perception import BaseField, Fields, PointCloud
            import desk_lamp
            import density_id_objects as density_obj
            from make_part_legend import read_ply_cloud
        except Exception as exc:                                # noqa: BLE001
            print(f"  [주의] 물체 뷰를 못 띄웁니다: {exc}")
            return None
        spec = desk_lamp.build_spec()
        self.object_spec = spec
        meshcat = (self.console.meshcat if self.console is not None
                   else StartMeshcat())
        self.object_meshcat = meshcat
        label_colors = {
            "base": (20 / 255, 100 / 255, 1.0, 1.0),
            "support": (40 / 255, 210 / 255, 40 / 255, 1.0),
            "head": (240 / 255, 40 / 255, 30 / 255, 1.0),
        }
        plant, bodies = density_obj.build_plant(
            spec, np.ones(len(density_obj.body_table(spec))))
        context = plant.CreateDefaultContext()
        latest = Path(self.conf.get("FP_OUTPUT", "")) / "latest.json"
        pose = self.session.read(latest, {}) if latest.is_file() else {}
        from dual_view import observed_to_model_deg
        theta = np.deg2rad(observed_to_model_deg(
            "desklamp", [pose.get("base_support_deg", 0.0),
                         pose.get("support_head_deg", 0.0)]))
        if theta.size == plant.num_positions():
            plant.SetPositions(context, theta)
        upright = RigidTransform(RotationMatrix(desk_lamp.DISPLAY_ROTATION))
        poses = {part.name: upright @ plant.EvalBodyPoseInWorld(context, bodies[part.name])
                 for part in spec.parts}
        origins = np.array([body_pose.translation() for body_pose in poses.values()])
        extents = np.array([np.asarray(part.bbox_mm) * 1e-3 for part in spec.parts])
        center = 0.5 * (origins.min(axis=0) + origins.max(axis=0))
        size = float(np.max(origins.max(axis=0) - origins.min(axis=0)
                            + extents.max(axis=0)))
        separation = 1.35 * size
        mesh_shift = RigidTransform([-0.5 * separation, 0.0, 0.0])
        gaussian_shift = RigidTransform([0.5 * separation, 0.0, 0.0])
        meshcat.Delete("object")
        meshcat.Delete("grasp")
        for part in spec.parts:
            path = f"object/urdf/{part.name}"
            offset = RigidTransform(np.array(part.mesh_offset_m))
            if part.visual_mesh:
                meshcat.SetObject(path, Mesh(str(part.visual_mesh), 1.0),
                                  Rgba(*label_colors[desk_lamp.FINAL_PART[part.name]]))
                meshcat.SetTransform(path, mesh_shift @ poses[part.name] @ offset)

        gaussian_dir = Path(self.conf.get("GAUSSIAN_DIR", "")).expanduser()
        gaussian_files = dict(item.split("=", 1) for item in
                              self.conf.get("GAUSSIAN_FILES", "").split(",")
                              if "=" in item)
        for part in spec.parts:
            semantic = desk_lamp.FINAL_PART[part.name]
            source = gaussian_dir / gaussian_files.get(semantic, "")
            if not source.is_file():
                continue
            points, colors = read_ply_cloud(source, limit=20000)
            cloud = PointCloud(len(points), Fields(BaseField.kXYZs |
                                                    BaseField.kRGBs))
            cloud.mutable_xyzs()[:] = points.T
            cloud.mutable_rgbs()[:] = colors.T
            path = f"object/3dgs/{part.name}"
            meshcat.SetObject(path, cloud, point_size=0.0025)
            meshcat.SetTransform(
                path, gaussian_shift @ poses[part.name]
                @ RigidTransform(-points.mean(axis=0)))
        meshcat.SetCameraPose(center + np.array([0.2, -3.8, 1.0]) * size,
                              center)
        # 파지 후보 — 설정에서 고른 부위의 볼록 조각 중심들.
        grasp_part = self.conf.get("GRASP_PART", spec.parts[0].name)
        root = next((part for part in spec.parts if part.name == grasp_part),
                    spec.parts[0])
        for index, piece in enumerate(root.collision_meshes):
            points = np.array([[float(t) for t in line.split()[1:4]]
                               for line in open(piece)
                               if line.startswith("v ")])
            centre = points.mean(axis=0) + np.array(root.mesh_offset_m)
            meshcat.SetObject(f"grasp/cand_{index}", Sphere(0.006),
                              Rgba(0.95, 0.35, 0.05, 0.9))
            meshcat.SetTransform(f"grasp/cand_{index}",
                                 mesh_shift @ poses[root.name]
                                 @ RigidTransform(centre))
        print(f"  물체 뷰: {meshcat.web_url()}")
        print("  왼쪽=URDF/충돌 메시, 오른쪽=3DGS Gaussian 중심점")
        print(f"  파지 후보 {len(root.collision_meshes)}개를 주황 점으로 표시했습니다"
              f" (잡는 부위 = {root.name})")
        return meshcat

    def show_density_meshes(self, posterior=None):
        """창 1의 실제 메시를 탐색 전/후 밀도 색으로 나란히 표시한다."""
        try:
            import numpy as np
            from density_view import DensityPanel
            spec = getattr(self, "object_spec", None)
            meshcat = getattr(self, "object_meshcat", None)
            if spec is None or meshcat is None:
                return
            latest = Path(self.conf.get("FP_OUTPUT", "")) / "latest.json"
            pose = self.session.read(latest) if latest.is_file() else {}
            from dual_view import observed_to_model_deg
            theta = (observed_to_model_deg(
                "desklamp", [pose.get("base_support_deg"),
                             pose.get("support_head_deg")])
                     if pose and all(pose.get(key) is not None for key in
                                     ("base_support_deg", "support_head_deg"))
                     else None)
            meshcat.Delete("object")
            meshcat.Delete("grasp")
            panel = DensityPanel(spec, meshcat, theta_deg=theta)
            prior = np.asarray((posterior or {}).get(
                "prior_densities_kg_m3", np.full(len(spec.parts), 1000.0)))
            panel.begin(prior, target_rel=float(self.conf.get("TARGET", 0.05)),
                        density_range=(300.0, 1700.0))
            if posterior and posterior.get("densities_kg_m3"):
                panel.update(posterior["densities_kg_m3"],
                             posterior.get("relative_half_width",
                                           np.zeros(len(spec.parts))),
                             posterior.get("measurement_round", 0),
                             bool(posterior.get("converged")))
            self.density_panel = panel
        except Exception as exc:                                # noqa: BLE001
            print(f"  [주의] 창 1 밀도 메시 갱신 실패: {exc}")

    # -- 2단계 ------------------------------------------------------------
    def phase_angle(self):
        self.session.set_phase("angle", self.round)
        self.bar()
        print("  창 1 이 정보이득이 가장 큰 각도를 추천합니다.")
        print("  창 2 를 보며 물체를 그 각도로 접으세요.")
        recommended = self.session.read(f"angle_round_{self.round}.json")
        if recommended is None:
            print("  (추천 각도는 4단계 탐색기가 계산합니다 — 첫 라운드는"
                  " 지금 자세 그대로 갑니다)")
            return True
        self.ask("각도 조정 완료")
        return True

    # -- 3단계 ------------------------------------------------------------
    def phase_path(self):
        self.session.set_phase("path", self.round)
        self.bar()
        grasp = self.session.read("grasp.json")
        if grasp and grasp.get("source") == "measured":
            print("  로봇 씬을 세웁니다 — 파지 변환은 **잰 값**을 씁니다.")
        else:
            print("  [주의] 잰 파지 변환이 없어 **짐작**으로 씬을 세웁니다.")
        print("  도달·충돌·경로를 검사합니다. 실패하면 어느 쌍이 몇 mm"
              " 겹치는지 알려 줍니다.")
        return True

    # -- 4~5단계 ----------------------------------------------------------
    def phase_explore(self):
        """탐색은 기존 dual_view 에 맡긴다. 세션을 환경변수로 물려준다."""
        self.session.set_phase("explore", self.round)
        self.bar()
        self.show_density_meshes()
        env = dict(os.environ, PIVOT_SESSION=str(self.session.root),
                   PIVOT_OUTER_ROUND=str(self.round))
        if self.conf.get("FP_INTRINSICS"):
            env["PIVOT_CAMERA_INTRINSICS"] = self.conf["FP_INTRINSICS"]
        grasp = self.session.path("grasp.json")
        if grasp.is_file():
            env["PIVOT_GRASP_FILE"] = str(grasp)
        for name in ("operator_ui.json", "operator_action.json"):
            try:
                self.session.path(name).unlink()
            except FileNotFoundError:
                pass
        conf = self.conf
        urdf_out = self.session.path("export/estimated_desklamp.urdf")
        urdf_out.parent.mkdir(parents=True, exist_ok=True)
        command = self.python() + [
            str(HERE / "dual_view.py"),
            "--mode", "deploy", "--bus", "local",
            "--hardware", "real" if conf.get("ROBOT_HOST") else "sim",
            "--object", conf.get("OBJECT", "desklamp"),
            "--grasp", "pinch", "--grasp-part", conf.get("GRASP_PART", "link_3"),
            "--prior", "water",
            "--target", conf.get("TARGET", "0.05"),
            "--max-rounds", conf.get("MAX_ROUNDS", "8"),
            "--angle-floor-deg", conf.get("ANGLE_FLOOR_DEG", "2.0"),
            "--move-duration", conf.get("MOVE_DURATION", "8"),
            "--dashboard-session", str(self.session.root),
            "--skip-grasp", "--no-gripper",
            "--no-density-view", "--urdf-out", str(urdf_out)]
        for flag, key in (("--gripper-port", "GRIPPER_PORT"),
                          ("--gripper-force", "GRIPPER_FORCE"),
                          ("--tare-file", "TARE_FILE"),
                          ("--tare-max-age-s", "TARE_MAX_AGE_S"),
                          ("--meshpca-root", "MESHPCA_ROOT"),
                          ("--aft-host", "AFT_HOST"),
                          # 밀도 계산의 모멘트팔을 무엇으로 세울지.
                          #   legacy    자산에 적어둔 짐작 파지 (지금까지)
                          #   measured  사람이 실제로 잡힌 자리를 카메라가
                          #             읽어 온 값. 이 연구의 설계다.
                          # 로봇 장면(충돌·도달)은 이미 실측값을 쓰는데
                          # 밀도 계산만 짐작값을 쓰고 있었다. 설정으로 켤
                          # 방법조차 없어서 코드를 고쳐야 했다.
                          ("--grasp-frame", "GRASP_FRAME"),
                          ("--robot-host", "ROBOT_HOST")):
            if conf.get(key):
                command += [flag, conf[key]]
        if conf.get("FP_OUTPUT"):
            command += ["--pose-file", str(Path(conf["FP_OUTPUT"]) / "latest.json")]
        if conf.get("START_ARM_DEG"):
            command += ["--start-arm-deg", *conf["START_ARM_DEG"].split()]
        print("  탐색을 시작합니다 (dual_view). 창 3·4 가 갱신됩니다.")
        ok = subprocess.run(command, env=env,
                            stdin=subprocess.DEVNULL).returncode == 0
        posterior = self.session.read(f"posterior_round_{self.round}.json")
        if posterior:
            self.show_density_meshes(posterior)
        return ok

    def phase_export(self):
        self.session.set_phase("export", self.round)
        self.bar()
        target = self.session.path("export/estimated_desklamp.urdf")
        if target.is_file():
            print(f"  {target} 에 sim-ready 자산을 냈습니다.")
        else:
            print("  [주의] 실험 결과 URDF가 없습니다. 밀도 JSON은 세션에 남았습니다.")
        return True

    # -- 전체 -------------------------------------------------------------
    def run(self):
        print(f"세션 {self.session.root}")
        print("창 2·3 은 이 값을 읽습니다:")
        print(f"  export PIVOT_SESSION={self.session.root}\n")
        if not self.phase_preflight():
            return 1
        if not self.phase_grasp():
            return 1
        while True:
            self.phase_angle()
            self.phase_path()
            if not self.phase_explore():
                return 1
            posterior = self.session.read(f"posterior_round_{self.round}.json")
            done = bool(posterior and posterior.get("converged"))
            if done or self.auto:
                break
            print("\n  불확실성이 목표 밖입니다. 각도를 다시 조정합니다.")
            self.round += 1
            if self.round >= int(self.conf.get("MAX_ROUNDS", 8)):
                print("  최대 라운드에 도달했습니다.")
                break
        self.phase_export()
        self.session.set_phase("export", self.round, note="완료")
        self.bar()
        return 0


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conf", type=Path,
                    default=HERE.parent / "setup" / "experiment.conf")
    ap.add_argument("--sessions", type=Path, default=HERE / "sessions")
    ap.add_argument("--session", type=Path, default=None,
                    help="이어서 할 세션 폴더 (없으면 새로 만든다)")
    ap.add_argument("--auto", action="store_true",
                    help="사람 확인을 건너뛴다 (리허설용)")
    ap.add_argument("--dry-run", action="store_true",
                    help="단계 기계만 돌려 본다 (Drake·장비 불필요)")
    args = ap.parse_args()

    instance_lock = None if args.dry_run else acquire_instance_lock()
    if not args.dry_run and instance_lock is None:
        print("[중단] PIVOT 통합 UI가 이미 실행 중입니다. 기존 창을 사용하세요.")
        return 2

    session = (Session(args.session) if args.session
               else Session.new(args.sessions))
    os.environ["PIVOT_SESSION"] = str(session.root)

    if args.dry_run:
        print(f"세션 {session.root}\n")
        for key, label in PHASES:
            session.set_phase(key)
            print(describe(session))
        print("\n단계 기계 OK. 창 2·3 은 phase.json 을 폴링합니다:")
        print(session.read("phase.json"))
        return 0

    console = dashboard = None
    try:
        from pydrake.geometry import StartMeshcat
        from operator_ui import Console
        console = Console(StartMeshcat(), auto=args.auto)
    except Exception as exc:                                    # noqa: BLE001
        print(f"[주의] Meshcat 콘솔을 못 띄웁니다 ({exc}) — 터미널로 갑니다")
    conductor = Conductor(args.conf, session, console, args.auto)
    if console is not None:
        from pivot_dashboard import Dashboard
        dashboard = Dashboard(session, conductor.conf,
                              console.meshcat.web_url()).start()
        conductor.dashboard = dashboard
        url = dashboard.web_url()
        print(f"통합 지휘 UI: {url}")
        webbrowser.open(url)
    try:
        return conductor.run()
    finally:
        if dashboard is not None:
            dashboard.stop()


if __name__ == "__main__":
    sys.exit(main())
