"""Robotiq 2F-85 로 세 물체의 후보 자세가 얼마나 나오는지."""

import sys as _sys, pathlib as _pathlib
# 이 폴더는 my_work 밖이라 형제 모듈이 안 보인다. run_drake_env.sh 가
# PYTHONPATH 를 지우므로 (ROS 오염 제거) 환경변수로는 못 넣는다.
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import io, contextlib, time
import numpy as np
import density_id_objects as obj, robot_scene as rs, dual_view as dv
import desk_lamp as lamp, grippers as gr

cases = [("3link", obj.OBJECTS["3link"], [0.0, 180.0]),
         ("2link", obj.OBJECTS["2link"], [0.0, 180.0]),
         ("desklamp", lamp.build_spec(), None)]
print(f"{'물체':<10}{'그리퍼':<13}{'파지단면':>10}{'개구':>8}"
      f"{'최종후보':>10}{'준비[s]':>9}")
for key, spec, rng in cases:
    for gname in ("pgc140", "robotiq2f85"):
        limits = rs.parse_joint_range(spec, rng)
        obj.set_measurement_averaging()
        t0 = time.perf_counter()
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                setup = dv.prepare(spec, None, limits, obj.DEFAULT_SAFETY, 5,
                                   0.006, 1.0, prior="weight", gripper=gname)
            n = f"{len(setup['feasible'])}/{setup['n_grid']}"
        except RuntimeError as exc:
            n = f"실패({str(exc)[:14]})"
        dt = time.perf_counter() - t0
        print(f"{key:<10}{gname:<13}{1000*rs.jaw_dimension_m(spec):>9.1f}mm"
              f"{1000*gr.GRIPPERS[gname].max_opening_m:>7.0f}mm{n:>10}{dt:>9.1f}")
