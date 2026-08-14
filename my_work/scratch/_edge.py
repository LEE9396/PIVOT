"""가설 검증: 정보이득이 고르는 '극단 자세' 가 정말 IK 가 약한 구역인가."""

import sys as _sys, pathlib as _pathlib
# 이 폴더는 my_work 밖이라 형제 모듈이 안 보인다. run_drake_env.sh 가
# PYTHONPATH 를 지우므로 (ROS 오염 제거) 환경변수로는 못 넣는다.
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import angle_aware as aa, density_id_drake as alg, density_id_objects as obj
import design_core as dc, desk_lamp as lamp, robot_scene as rs

spec = lamp.build_spec()
limits = rs.parse_joint_range(spec, None)
obj.set_measurement_averaging()
rho = obj.bind_object(spec)
gt_mass = np.array([lamp.GROUND_TRUTH[p.name]["mass_kg"] for p in spec.parts])
obj.apply_weight_prior(spec, float(gt_mass.sum()))
ck = rs.PoseChecker(spec, densities=rho, joint_limits_rad=limits,
                    min_distance_m=0.006)
lows = np.array([lo for lo, _ in limits]); highs = np.array([hi for _, hi in limits])

def ik_ok(th):
    return all(v is not None for v in ck.solutions_for(np.asarray(th)).values())

def margin_ok(th):
    th = np.asarray(th)
    sig = np.sqrt(np.diag(aa.angle_covariance(th, 0.05)))
    for code in range(4):
        s = np.array([1.0 if (code >> k) & 1 else -1.0 for k in range(2)])
        if not ik_ok(np.clip(th + 1.96*sig*s, lows, highs)):
            return False
    return ik_ok(th)

# 1) |joint1| 크기별 IK 통과율
n = 13
ax = np.linspace(lows[0], highs[0], n)
ay = np.linspace(lows[1], highs[1], n)
print("가로=joint2, 세로=joint1   O=IK 통과, .=실패")
print("        " + "".join(f"{np.degrees(b):6.0f}" for b in ay))
grid_ok = np.zeros((n, n), bool)
for i, a in enumerate(ax):
    row = ""
    for j, b in enumerate(ay):
        grid_ok[i, j] = ik_ok([a, b]); row += "     O" if grid_ok[i,j] else "     ."
    print(f"{np.degrees(a):8.0f}" + row)

band = np.abs(np.degrees(ax))
for lo_deg, hi_deg in ((0, 30), (30, 60), (60, 80), (80, 90)):
    sel = (band >= lo_deg) & (band <= hi_deg)
    if sel.any():
        print(f"  |joint1| {lo_deg:>2}~{hi_deg:<3} deg : IK 통과 "
              f"{100*grid_ok[sel].mean():5.1f}%")

# 2) 정보이득 최적화가 실제로 고르는 자세 (라운드별)
print("\n정보이득이 고르는 자세와 그 자리의 실현 가능성")
S, rho_hat = alg.SIGMA0.copy(), alg.MU0.copy()
g = dc.CANONICAL_TRIAD
for k in range(1, 7):
    th, _ = dc.continuous_best(limits,
        lambda t: dc.utility(t, rho_hat, S, g, "D", 0.05), n_starts=8, seed=k)
    th = np.atleast_1d(th)
    print(f"  round {k}: q={np.round(np.degrees(th),1)} deg"
          f"   |joint1|={abs(np.degrees(th[0])):5.1f}"
          f"   IK {'O' if ik_ok(th) else '.'}"
          f"   여유 {'O' if margin_ok(th) else '.'}")
    A = dc.regressor(th, g); R = dc.effective_cov(th, rho_hat, g, 0.05)
    S = dc.posterior(S, A, R)
