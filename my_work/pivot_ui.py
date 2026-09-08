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
                 dashboard=None, rehearse=False):
        self.conf_path = Path(conf_path)
        self.conf = self._read_conf()
        self.session = session
        self.console = console
        self.dashboard = dashboard
        self.auto = auto
        # 리허설: 장비·추적기 없이 **같은 단계 기계**를 끝까지 밟는다.
        #   - 0단계: preflight 를 돌리되 FAIL 에 멈추지 않는다 (목록만 보여 준다)
        #   - 1단계: FoundationPose 로 파지를 재는 대신 자산의 짐작 파지로 간다
        #   - 4단계: dual_view 는 conf 의 ROBOT_HOST 가 비어 있으면 --hardware sim
        # 실물 세션과 다른 길은 위 둘뿐이고, 화면(창 1·대시보드)은 같다.
        self.rehearse = bool(rehearse)
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
            if self.rehearse:
                failed = [r["name"] for r in data.get("rows", []) if r.get("level") == "FAIL"]
                print("\n  [리허설] 준비 점검 결과를 보기만 하고 넘어갑니다."
                      + (f" 실물이라면 막혔을 항목: {', '.join(failed)}" if failed else ""))
                if self.dashboard is not None:
                    self.dashboard.set_status("[0 준비] 리허설 — 장비 없이 진행")
                return True
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

        if self.rehearse:
            print("\n  [리허설] 카메라가 없어 파지 변환을 재지 않습니다 —"
                  " 자산에 적힌 짐작 파지로 갑니다 (3단계에 [주의]로 표시됩니다).")
            return True
        print("\n  파지 변환을 **잽니다** (짐작하지 않습니다)")
        pose_file = self.conf.get("FP_OUTPUT", "")
        command = self.python() + [
            str(TOOLS / "grasp_measure.py"),
            "--pose-file", str(Path(pose_file) / "latest.json"),
            "--session", str(self.session.root),
            "--part", self.conf.get("FP_GRASP_PART", "support"),
            "--grasp-target", str(HERE / "outputs" /
                                  f"grasp_target_{self.conf.get('OBJECT', 'desklamp')}.json")]
        host = self.conf.get("ROBOT_HOST")
        if host:
            command += ["--robot-host", host]
        if subprocess.run(command).returncode != 0:
            print("  [실패] 파지 변환을 못 쟀습니다.")
            print("  창 2 가 각도를 내고 있는지, 핸드아이가 있는지 보세요.")
            print("  실측 파지 없이는 자세·충돌 계산이 실제 물체 배치와 다르므로"
                  " 진행하지 않습니다.")
            return False
        return self.grasp_frame_check()

    GRASP_ROTATION_MAX_DEG = 30.0

    def grasp_frame_check(self):
        """잰 파지로 두 가상환경의 좌표계 규약이 같은지 바로 검사한다.

        회전 차이가 수십 도면 사람의 파지 오차가 아니라 카메라 메시와 밀도
        모델 메시의 좌표계가 다른 것이다. 그 상태로 measured 프레임을 켜면
        173.9 mm 가 다른 숫자로 돌아온다. 예전에는 사람이 파지 뒤에
        tools/check_grasp_frames.py 를 손으로 돌려야 했다.
        """
        grasp = self.session.path("grasp.json")
        if not grasp.is_file():
            return True
        print("\n  파지 좌표계 검사 (tools/check_grasp_frames.py)")
        result = subprocess.run(
            self.python() + [str(TOOLS / "check_grasp_frames.py"),
                             "--object", self.conf.get("OBJECT", "desklamp"),
                             "--grasp", str(grasp)],
            capture_output=True, text=True)
        text = (result.stdout or "") + (result.stderr or "")
        for line in text.splitlines():
            if "회전 차이" in line or "두 점 사이 거리" in line or "위치 차이" in line:
                print("  " + line.strip())
        import re
        m = re.search(r"회전 차이\s+([0-9.]+)\s*deg", text)
        if m is None:
            print("  [주의] 회전 차이를 못 읽었습니다 — 검사 출력을 확인하세요")
            return self.ask("좌표계 검사를 못 했지만 계속한다")
        rot = float(m.group(1))
        if rot > self.GRASP_ROTATION_MAX_DEG:
            print(f"  [중단] 회전 차이 {rot:.1f}° > {self.GRASP_ROTATION_MAX_DEG:.0f}° —"
                  " 카메라 메시와 밀도 모델 메시의 좌표계가 다릅니다."
                  " conf 의 LAMP_ASSET_DIR / FP_MESH_DIR 가 같은 트리인지 보세요.")
            if self.dashboard is not None:
                self.dashboard.set_status(f"[1 파지점] 좌표계 불일치 {rot:.0f}° — 메시 설정을 확인하세요")
            return False
        print(f"  OK  회전 차이 {rot:.2f}° (허용 {self.GRASP_ROTATION_MAX_DEG:.0f}°)")
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
            # 저울 총무게를 실제로 쓰는 사전분포. water 는 등방이라
            # 총질량을 전혀 안 묶고, 그래서 부위가 하한(50)에 붙어도
            # 막지 못한다 (session_20260904_1736: 416.9 g vs 저울 571.0 g).
            "--prior", "weight",
            "--target", conf.get("TARGET", "0.05"),
            "--max-rounds", conf.get("MAX_ROUNDS", "8"),
            "--angle-floor-deg", conf.get("ANGLE_FLOOR_DEG", "3.0"),
            "--angle-error", conf.get("ANGLE_REL_ERROR", "0"),
            "--grasp-sigma-mm", conf.get("GRASP_SIGMA_MM", "10"),
            "--move-duration", conf.get("MOVE_DURATION", "8"),
            "--dashboard-session", str(self.session.root),
            "--skip-grasp", "--no-gripper",
            "--no-density-view", "--urdf-out", str(urdf_out)]
        for flag, key in (("--gripper-port", "GRIPPER_PORT"),
                          ("--gripper-force", "GRIPPER_FORCE"),
                          ("--tare-file", "TARE_FILE"),
                          ("--tare-mode", "TARE_MODE"),
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
                          # 저울로 잰 총질량 [kg]. 토크만 쓰는 설정에서는
                          # 규모를 정하는 유일한 값이라 반드시 있어야 한다.
                          ("--total-mass-kg", "TOTAL_MASS_KG"),
                          # 탐색 속도. 리허설에서 STEPS=3 SELECT=grid 로 두면
                          # 램프도 몇 분 안에 끝난다 (연속 탐색은 수십 분).
                          ("--steps", "STEPS"),
                          ("--select", "SELECT"),
                          ("--plan-iters", "PLAN_ITERS"),
                          ("--robot-host", "ROBOT_HOST")):
            if conf.get(key):
                command += [flag, conf[key]]
        # FoundationPose 가 잰 파지점 어긋남 명목값 [mm] (예: "-49.2 82.0 -145.2")
        if conf.get("GRASP_MU_MM"):
            command += ["--grasp-mu-mm", *conf["GRASP_MU_MM"].split()]
        # 힘 3축을 추정에 다시 넣고 싶을 때만 켠다. 기본은 토크 전용이다.
        if str(conf.get("USE_FORCE", "")).strip().lower() in ("1", "true", "yes"):
            command += ["--use-force"]
        if conf.get("FP_OUTPUT"):
            command += ["--pose-file", str(Path(conf["FP_OUTPUT"]) / "latest.json")]
            command += ["--gripper-status-file", str(Path(conf["FP_OUTPUT"]) / "hardware.json")]
        if conf.get("START_ARM_DEG"):
            command += ["--start-arm-deg", *conf["START_ARM_DEG"].split()]
        print("  탐색을 시작합니다 (dual_view). 창 3·4 가 갱신됩니다.")
        process = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL)
        if self.dashboard is not None:
            self.dashboard.set_experiment_process(process)
        ok = process.wait() == 0
        posterior = self.session.read(f"posterior_round_{self.round}.json")
        if posterior:
            self.show_density_meshes(posterior)
        self.round_check()
        return ok

    def round_check(self):
        """탐색 뒤 숫자 세 개로 '센서가 모형이 말하는 것을 재고 있나'를 판정한다.

        session_20260904_1736 은 이 셋(힘 크기 58.7 N 오프셋, 잔차팽창 5 527,
        파지 오프셋 상자에 붙음)이 전부 틀렸는데 사람이 파일을 열어야 알 수
        있었다. 여기서 바로 찍고 대시보드에도 올린다. 판정은 tools/round_check.py.
        """
        if not self.session.path("exploration_round_1.json").is_file():
            return
        print("\n  --- 라운드 검산 (tools/round_check.py) ---")
        result = subprocess.run(
            self.python() + [str(TOOLS / "round_check.py"), str(self.session.root),
                             "--conf", str(self.conf_path)],
            capture_output=True, text=True)
        text = (result.stdout or "") + (result.stderr or "")
        print("\n".join("  " + line for line in text.rstrip().splitlines()))
        if self.dashboard is not None:
            verdict = "통과" if result.returncode == 0 else "실패 — 터미널의 검산을 보세요"
            self.dashboard.set_status(f"[4 탐색] 라운드 검산 {verdict}")

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
    ap.add_argument("--rehearse", action="store_true",
                    help="장비·추적기 없이 통합 UI 를 끝까지 돌린다."
                         " preflight FAIL 에 멈추지 않고, 파지는 짐작값을 쓴다."
                         " conf 의 ROBOT_HOST 가 비어 있어야 한다 (--hardware sim)")
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
    if args.rehearse:
        probe = Conductor(args.conf, session, None, args.auto)
        if probe.conf.get("ROBOT_HOST"):
            print("[중단] --rehearse 인데 conf 에 ROBOT_HOST 가 있습니다. 실물로 갈 뻔했습니다."
                  " setup/experiment.sim.conf 처럼 ROBOT_HOST 를 비운 conf 를 쓰세요.")
            return 2
    conductor = Conductor(args.conf, session, console, args.auto,
                          rehearse=args.rehearse)
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
