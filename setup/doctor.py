"""새 PC 에서 이 저장소가 실제로 돌아가는지 하나씩 확인한다.

setup/bootstrap.sh 가 마지막에 부른다. 따로 부를 수도 있다.

    cd my_work
    ../robot_learning/scripts/run_drake_env.sh python ../setup/doctor.py

각 항목은 **실패해도 계속 간다**. 마지막에 무엇이 왜 막혔는지 한꺼번에
보여 주고, 고치는 방법까지 적는다. 하나 막혔다고 나머지를 못 보면
새 PC 에서 왕복만 늘어난다.
"""

import socket
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "my_work"
sys.path.insert(0, str(WORK))

OK, BAD = "  [통과]", "  [실패]"
results = []


def check(title, fix=""):
    """검사 하나를 등록하는 장식자. 함수가 문자열을 돌려주면 그걸 덧붙인다."""
    def wrap(function):
        print(f"{title} ...", end="", flush=True)
        try:
            detail = function() or ""
            print(f"\r{OK} {title}  {detail}")
            results.append((True, title, ""))
        except Exception as exc:                      # noqa: BLE001
            print(f"\r{BAD} {title}")
            print(f"         {type(exc).__name__}: {exc}")
            results.append((False, title, fix))
            if "-v" in sys.argv:
                traceback.print_exc()
        return function
    return wrap


print(f"저장소: {ROOT}\n")

# ---------------------------------------------------------------- 파이썬·꾸러미
@check("파이썬 3.12", "bootstrap.sh 가 만든 환경 밖에서 돌리고 있습니다. "
                      "run_drake_env.sh 를 앞에 붙이세요.")
def _python():
    version = sys.version_info
    assert (version.major, version.minor) == (3, 12), f"{sys.version.split()[0]}"
    return sys.version.split()[0]


@check("Drake 1.54", "pip install -r robot_learning/requirements/drake.txt")
def _drake():
    import importlib.metadata as metadata

    import pydrake.multibody.inverse_kinematics  # noqa: F401  실제로 쓰는 모듈
    version = metadata.version("drake")
    assert version.startswith("1.54"), f"버전 {version} (1.54 가 필요)"
    return version


@check("numpy / scipy / matplotlib", "pip install -r ... drake.txt")
def _stack():
    import matplotlib
    import numpy
    import scipy
    return f"numpy {numpy.__version__}, scipy {scipy.__version__}"


# ---------------------------------------------------------------- 자산
@check("RB5 / PGC / AFT200 자산 (third_party/HTD)",
       "third_party/HTD 가 비었습니다. clone 이 덜 됐거나 LFS 가 안 받아졌습니다.")
def _htd():
    sys.path.insert(0, str(ROOT / "robot_learning" / "scripts"))
    import visualize_drake_rb5_hammer_payload as rb5
    rb5.validate_htd_source(rb5.DEFAULT_HTD_ROOT)
    return "해시까지 일치"


@check("Robotiq 2F-85 (third_party/robotiq_arg85_description)",
       "third_party/robotiq_arg85_description 이 없습니다.")
def _robotiq():
    import grippers as gr
    urdf = gr.robotiq_urdf_string(0.04)
    assert "robotiq_85_base_link" in urdf
    return f"개구 최대 {1000 * gr.GRIPPERS['robotiq2f85'].max_opening_m:.0f} mm"


@check("데스크 램프 스캔 (assets/desk_lamp_minimal_sim)",
       "assets/desk_lamp_minimal_sim 이 없습니다. "
       "다른 곳에 뒀다면 export DESK_LAMP_DELIVERY=/그/경로")
def _lamp():
    import desk_lamp as dl
    meshes = dl.ensure_obj()
    n_color = sum(len(v["pieces"]) for v in meshes.values())
    n_collision = sum(len(v["collision"]) for v in meshes.values())
    return (f"{dl.LAYOUT} 구조, 색 조각 {n_color}개, 충돌 조각 {n_collision}개")


# ---------------------------------------------------------------- 실제로 세워 보기
@check("카메라 자세 출처", "calibration/README.md 를 보세요.")
def _camera():
    import numpy as np
    import robot_scene as rs
    position = np.round(rs.camera_pose(rs.CAMERA).translation(), 3)
    return f"{rs.CAMERA['source']}, 위치 {position} m"


@check("3-link 물체로 Drake 씬 세우기", "위 항목들을 먼저 고치세요.")
def _scene_3link():
    import density_id_objects as obj
    import robot_scene as rs
    spec = obj.OBJECTS["3link"]
    obj.set_measurement_averaging()
    rho = obj.bind_object(spec)
    scene = rs.build_scene(spec, rho, rs.parse_joint_range(spec, None))
    return f"몸체 {scene['plant'].num_bodies()}개"


@check("데스크 램프로 Drake 씬 세우기", "위 항목들을 먼저 고치세요.")
def _scene_lamp():
    import density_id_objects as obj
    import desk_lamp as dl
    import robot_scene as rs
    spec = dl.build_spec()
    obj.set_measurement_averaging()
    rho = obj.bind_object(spec)
    scene = rs.build_scene(spec, rho, rs.parse_joint_range(spec, None))
    return f"몸체 {scene['plant'].num_bodies()}개, 파지 {dl.DEFAULT_GRASP_PART}"


@check("추정기 한 바퀴 (측정 없이 수식만)", "design_core.py 를 확인하세요.")
def _estimator():
    import density_id_objects as obj
    import design_core as dc
    spec = obj.OBJECTS["2link"]
    obj.set_measurement_averaging()
    obj.bind_object(spec)
    obj.apply_weight_prior(spec, obj.assembled_mass_kg(spec))
    result = dc.closed_loop(spec, target=0.02, max_rounds=3, seed=0)
    return f"{result['rounds']}라운드, 주장 반폭 {100 * result['worst']:.2f}%"


# ---------------------------------------------------------------- 화면 자리
@check("화면 포트 7000 / 7001",
       "이미 떠 있는 화면이 있으면 정상입니다. 없는데도 막혔다면 "
       "lsof -i:7000 으로 누가 쓰는지 보세요.")
def _ports():
    busy = []
    for port in (7000, 7001):
        with socket.socket() as probe:
            probe.settimeout(0.3)
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                busy.append(port)
    assert not busy, f"이미 쓰는 중: {busy} (다른 화면이 떠 있을 수 있음)"
    return "비어 있음"


@check("TCP 버스 왕복 (작업 PC <-> 로봇 PC 규약)",
       "방화벽이 루프백까지 막고 있는지 보세요.")
def _bus():
    import contextlib
    import io

    import pose_bus
    with contextlib.redirect_stdout(io.StringIO()):     # 왕복 내용은 안 보여도 된다
        pose_bus._loopback_test()
    return "JSON 한 줄 왕복 성공"


# ---------------------------------------------------------------- 정리
failed = [(title, fix) for ok, title, fix in results if not ok]
print()
if not failed:
    print(f"모두 통과했습니다 ({len(results)}/{len(results)}). "
          "다음으로 무엇을 할지는 SETUP.md 4장을 보세요.")
    sys.exit(0)

print(f"{len(results) - len(failed)}/{len(results)} 통과, "
      f"{len(failed)}개가 막혔습니다.\n")
for title, fix in failed:
    print(f"  * {title}")
    if fix:
        print(f"    -> {fix}")
print("\n자세한 오류를 보려면 -v 를 붙여 다시 실행하세요.")
sys.exit(1)
