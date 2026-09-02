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
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "tools"))

from pivot_session import PHASES, Session, describe          # noqa: E402

TOOLS = HERE.parent / "tools"


# ---------------------------------------------------------------------------
class Conductor:
    """단계 기계. 창 1 이 이걸 돌린다."""

    def __init__(self, conf_path, session, console=None, auto=False):
        self.conf_path = Path(conf_path)
        self.conf = self._read_conf()
        self.session = session
        self.console = console
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
        if self.console is not None:
            self.console.lamp("blue", line)

    def ask(self, label):
        """사람이 누를 때까지 기다린다. 콘솔이 없으면 터미널로."""
        if self.auto:
            print(f"  [자동] {label}")
            return True
        if self.console is None:
            input(f"  >>> {label} — Enter")
            return True
        name = self.console.button(label)
        self.console.wait_for(name)
        return True

    # -- 0단계 ------------------------------------------------------------
    def phase_preflight(self):
        self.session.set_phase("preflight")
        self.bar()
        out = self.session.path("preflight.json")
        result = subprocess.run(
            self.python() + [str(TOOLS / "preflight.py"),
                             "--conf", str(self.conf_path), "--json", str(out)])
        data = self.session.read("preflight.json", {})
        verdict = data.get("verdict", "FAIL")
        if verdict == "FAIL" or result.returncode != 0:
            print("\n  준비 점검에 **실패**가 있습니다. 고치기 전에는 못 갑니다.")
            print("  (고친 뒤 이 프로그램을 다시 실행하세요)")
            return False
        if verdict == "WARN":
            print("\n  주의 항목이 있습니다. 결과가 조용히 나빠질 수 있습니다.")
            self.ask("주의를 알고도 계속한다")
        return True

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
            print("  그냥 진행하면 예전처럼 **짐작한 파지**로 돕니다.")
            return self.ask("짐작한 파지로 계속한다 (권장하지 않음)")
        return True

    def show_object_only(self):
        """물체만 올린 가벼운 Meshcat 뷰. 로봇은 여기 없다."""
        try:
            import numpy as np
            from pydrake.geometry import Mesh, Rgba, Sphere, StartMeshcat
            from pydrake.math import RigidTransform
            import desk_lamp
        except Exception as exc:                                # noqa: BLE001
            print(f"  [주의] 물체 뷰를 못 띄웁니다: {exc}")
            return None
        spec = desk_lamp.build_spec()
        meshcat = self.object_meshcat = StartMeshcat()
        for index, part in enumerate(spec.parts):
            path = f"object/{part.name}"
            offset = RigidTransform(np.array(part.mesh_offset_m))
            if part.visual_mesh:
                meshcat.SetObject(path, Mesh(str(part.visual_mesh), 1.0),
                                  Rgba(*part.color))
                meshcat.SetTransform(path, offset)
        # 파지 후보 — 볼록 조각 중심들. 사람이 보고 고른다.
        root = spec.parts[0]
        for index, piece in enumerate(root.collision_meshes):
            points = np.array([[float(t) for t in line.split()[1:4]]
                               for line in open(piece)
                               if line.startswith("v ")])
            centre = points.mean(axis=0) + np.array(root.mesh_offset_m)
            meshcat.SetObject(f"grasp/cand_{index}", Sphere(0.006),
                              Rgba(0.95, 0.35, 0.05, 0.9))
            meshcat.SetTransform(f"grasp/cand_{index}",
                                 RigidTransform(centre))
        print(f"  물체 뷰: {meshcat.web_url()}")
        print(f"  파지 후보 {len(root.collision_meshes)}개를 주황 점으로 표시했습니다"
              f" (잡는 부위 = {root.name})")
        return meshcat

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
        env = dict(os.environ, PIVOT_SESSION=str(self.session.root))
        grasp = self.session.path("grasp.json")
        if grasp.is_file():
            env["PIVOT_GRASP_FILE"] = str(grasp)
        conf = self.conf
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
            "--move-duration", conf.get("MOVE_DURATION", "8")]
        for flag, key in (("--gripper-port", "GRIPPER_PORT"),
                          ("--gripper-force", "GRIPPER_FORCE"),
                          ("--tare-file", "TARE_FILE"),
                          ("--aft-host", "AFT_HOST"),
                          ("--robot-host", "ROBOT_HOST")):
            if conf.get(key):
                command += [flag, conf[key]]
        if conf.get("FP_OUTPUT"):
            command += ["--pose-file", str(Path(conf["FP_OUTPUT"]) / "latest.json")]
        print("  탐색을 시작합니다 (dual_view). 창 3·4 가 갱신됩니다.")
        return subprocess.run(command, env=env).returncode == 0

    def phase_export(self):
        self.session.set_phase("export", self.round)
        self.bar()
        target = self.session.path("export")
        command = self.python() + [str(HERE / "export_urdf.py"),
                                   "--output", str(target)]
        result = subprocess.run(command)
        if result.returncode == 0:
            print(f"  {target} 에 sim-ready 자산을 냈습니다.")
        else:
            print("  [주의] 내보내기가 실패했습니다 — export_urdf.py 인자를"
                  " 확인하세요. 추정 결과는 세션 폴더에 그대로 있습니다.")
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
            if not self.ask("승인 — 로봇을 움직입니다"):
                return 1
            self.phase_explore()
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

    console = None
    try:
        from pydrake.geometry import StartMeshcat
        from operator_ui import Console
        console = Console(StartMeshcat(), auto=args.auto)
    except Exception as exc:                                    # noqa: BLE001
        print(f"[주의] Meshcat 콘솔을 못 띄웁니다 ({exc}) — 터미널로 갑니다")
    return Conductor(args.conf, session, console, args.auto).run()


if __name__ == "__main__":
    sys.exit(main())
