"""램프가 왜 자주 중단되는지 — 원인을 층별로 분리해 잰다."""
import numpy as np
import angle_aware as aa, density_id_drake as alg, density_id_objects as obj
import design_core as dc, desk_lamp as lamp, robot_scene as rs, path_planning as pp

def audit(spec, limits, tag, rel=0.05):
    obj.set_measurement_averaging()
    rho = obj.bind_object(spec)
    ck = rs.PoseChecker(spec, densities=rho, joint_limits_rad=limits,
                        min_distance_m=0.006)
    lows = np.array([lo for lo, _ in limits]); highs = np.array([hi for _, hi in limits])
    n = 7
    axes = [np.linspace(lo, hi, n) for lo, hi in limits]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 2)

    def ik_ok(th):
        return all(v is not None for v in ck.solutions_for(th).values())

    def margin_ok(th):
        sig = np.sqrt(np.diag(aa.angle_covariance(th, rel)))
        probes = [th]
        for code in range(4):
            s = np.array([1.0 if (code >> k) & 1 else -1.0 for k in range(2)])
            probes.append(np.clip(th + 1.96*sig*s, lows, highs))
        return all(all(v is not None for v in ck.solutions_for(p).values())
                   for p in probes)

    ik = np.array([ik_ok(t) for t in grid])
    mg = np.array([margin_ok(t) for t in grid if ik_ok(t)])
    sig_mid = np.degrees(np.sqrt(np.diag(aa.angle_covariance(
        np.array([0.5*(l+h) for l, h in limits]), rel))))
    sig_end = np.degrees(np.sqrt(np.diag(aa.angle_covariance(highs, rel))))
    print(f"\n=== {tag}   구동범위 {np.round(np.degrees(lows),0)} ~ {np.round(np.degrees(highs),0)} deg")
    print(f"  IK 통과            {ik.sum():>3}/{len(grid)} = {100*ik.mean():>3.0f}%")
    print(f"  + 여유검사까지     {mg.sum():>3}/{ik.sum()} = {100*mg.mean() if len(mg) else 0:>3.0f}%"
          f"  (IK 통과분 중)")
    print(f"  각도오차 1sigma    중앙 {np.round(sig_mid,2)} deg   끝 {np.round(sig_end,2)} deg")
    print(f"  95% 구간 폭        끝에서 +-{np.round(1.96*sig_end,1)} deg")
    return ck, grid, ik

lamp_spec = lamp.build_spec()
lamp_lim = rs.parse_joint_range(lamp_spec, None)
audit(lamp_spec, lamp_lim, "desk lamp")
audit(obj.OBJECTS["3link"], rs.parse_joint_range(obj.OBJECTS["3link"], [0.,180.]), "3link")
audit(lamp_spec, [(np.deg2rad(-60), np.deg2rad(60))]*2, "desk lamp (범위 -60~60)")
