"""저장된 힘을 실제 회귀행렬/추정기에 넣는 오프라인 검산. 로봇 연결 없음.

my_work에서 ../robot_learning/scripts/run_drake_env.sh python ../tools/replay_ft_matrix.py
물병의 도심을 모르므로 실측 검산은 힘 행만 쓴다. 램프의 전체 TLS는 합성 렌치로 검사한다.
"""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "my_work"))
import density_id_drake as alg
import density_id_objects as obj
import design_core as dc
import desk_lamp
import hardware as hw
import robot_scene as rs


def main():
    out = ROOT / "my_work/outputs"
    spec = desk_lamp.build_spec(grasp_at="pinch", grasp_part="link_3")
    grasp = rs.load_measured_grasp()
    scene = rs.build_scene(spec, include_visuals=False, grasp_transform=grasp)
    plant = scene["plant"]
    ctx = plant.CreateDefaultContext()
    aft = plant.GetBodyByName("ft_mount").body_frame()
    X_S_O = (aft.CalcPoseInWorld(ctx).inverse()
             @ scene["parts"][spec.parts[0].name].body_frame().CalcPoseInWorld(ctx))
    obj.bind_object(spec, X_sensor_object=X_S_O.GetAsMatrix4())
    theta = np.radians([24.43, 71.39])

    def gravity(q):
        for name, value in zip(rs.ARM_JOINT_NAMES, q):
            plant.GetJointByName(name, scene["arm"]).set_angle(ctx, np.radians(value))
        return aft.CalcPoseInWorld(ctx).rotation().matrix().T @ np.array([0., 0., -1.])

    def load(stamp):
        path = out / f"local_ft_20260907_{stamp}.json"
        data = json.loads(path.read_text())
        blocks = data["blocks"]
        return path, data, np.mean([b["wrench_mean"] for b in blocks], axis=0), gravity(
            blocks[0]["joint_before_deg"])

    def force_fit(force, g):
        # 물병 형상을 가정하지 않고 실제 밀도 행렬의 각 열을 질량 1 kg으로 정규화한다.
        A = alg.regressor(theta, [g])[:3]
        mass_columns = A / alg.VOLUMES[None, :]
        np.testing.assert_allclose(mass_columns, np.tile((alg.G_ACC * g)[:, None],
                                                       (1, alg.P)), atol=1e-12)
        a = mass_columns[:, :1]
        mass = float(np.linalg.lstsq(a, force, rcond=None)[0][0])
        # 실제 MAP 구현을 쓴다. 알려진 물병 질량은 사전분포에 넣지 않는다.
        mass_map = float(dc.wls_map([(a, force, np.eye(3) * alg.SIGMA_F**2)],
                                   np.array([1.]), np.array([[9.]]), (0., 20.))[0])
        rho_map = dc.wls_map([(A, force, np.eye(3) * alg.SIGMA_F**2)],
                            alg.MU0, alg.SIGMA0, alg.RHO_BOUNDS)
        return dict(force_n=force.tolist(), g_hat=g.tolist(), A_force=A.tolist(),
                    A_mass=a.tolist(), mass_ls_kg=mass, mass_map_kg=mass_map,
                    density_map_total_kg=float(alg.VOLUMES @ rho_map),
                    force_only_rank=int(np.linalg.matrix_rank(A)),
                    norm_mass_kg=float(np.linalg.norm(force) / alg.G_ACC),
                    perpendicular_residual_n=float(np.linalg.norm(force - a[:, 0] * mass)))

    cases = []
    for name, ls, bs, known in [
        ("aligned_0.541kg", "133048_584877", "133219_646212", .541),
        ("aligned_1.332kg_exploratory", "135128_611319", "133752_927964.stable", 1.332),
        ("tilted_0.541kg", "123436_990980", "123748_709132", .541),
        ("tilted_1.332kg", "124022_039338", "124257_622448", 1.332),
    ]:
        lp, ld, lw, lg = load(ls)
        bp, bd, bw, bg = load(bs)
        tare = hw.TareTable()
        tare.record(lg, bw)
        delta = tare.apply(lg, lw)
        np.testing.assert_allclose(delta, lw - bw)
        result = force_fit(delta[:3], lg)
        result.update(name=name, loaded_file=str(lp), empty_file=str(bp),
                      known_mass_kg=known, delta_wrench=delta.tolist(),
                      empty_g_hat=bg.tolist(), gravity_change_deg=float(np.degrees(
                          np.arccos(np.clip(lg @ bg, -1., 1.)))),
                      loaded_stability=ld.get("stability"),
                      calibration_valid=False,
                      mass_error_kg=result["mass_ls_kg"] - known)
        cases.append(result)

    ideal = [force_fit(np.array([0., 0., f]), np.array([0., 0., 1.]))
             for f in (13., 1.332 * alg.G_ACC, .541 * alg.G_ACC)]
    for result, expected in zip(ideal, (13. / alg.G_ACC, 1.332, .541)):
        assert abs(result["mass_ls_kg"] - expected) < 1e-12
        assert abs(result["mass_map_kg"] - expected) < 1e-5
        assert abs(result["density_map_total_kg"] - expected) < .0002

    # 합성 입력은 measure()/A@rho 대신 부위별 질량과 모멘트에서 구성한다.
    rho_test = np.linspace(700., 1500., alg.P)
    rounds, blocks = [], []
    for angles in ([27.48, 51.27], [10.15, 48.08], [24.43, 71.39]):
        th = np.radians(angles)
        c = alg.part_centroids_in_S(th)
        masses = alg.VOLUMES * rho_test
        y = []
        for g in alg.G_DIRS:
            f = masses[:, None] * (alg.G_ACC * g)
            y.extend(np.r_[f.sum(axis=0), np.cross(c, f).sum(axis=0)])
        y = np.array(y)
        # 모든 방향의 힘/토크에 공구 영점이 있는 합성 원시값을 추정기 입구로 전달한다.
        offset = np.tile([3., 5., 46., .12, -.08, .03], len(alg.G_DIRS))
        A, corrected = dc.measurement_equation(th, dict(
            wrench=y, wrench_raw=y + offset, tare_applied=offset,
            tare_required=True), alg.G_DIRS)
        np.testing.assert_allclose(corrected, y, atol=1e-12)
        y = corrected
        np.testing.assert_allclose(A @ rho_test, y, atol=1e-12)
        rounds.append((th, y))
        blocks.append((A, y, dc.sensor_cov(alg.G_DIRS)))
    wls = dc.wls_map(blocks, alg.MU0, alg.SIGMA0, alg.RHO_BOUNDS)
    tls, info = dc.tls_map(rounds, alg.MU0, alg.SIGMA0, alg.RHO_BOUNDS,
                          alg.G_DIRS, rho_init=wls)
    assert np.max(np.abs(tls - rho_test) / rho_test) < .005

    tare_path = ROOT / "calibration/aft_tare_current.json"
    tare_data = json.loads(tare_path.read_text())
    report = dict(hardware_connected=False, calibration_valid=False,
                  ideal_inputs=ideal, recorded_inputs=cases,
                  lamp_synthetic=dict(input_density_kg_m3=rho_test.tolist(),
                      measured_grasp_available=grasp is not None,
                      tare_applied=offset.tolist(),
                      wls_density_kg_m3=wls.tolist(), tls_density_kg_m3=tls.tolist(),
                      expected_mass_kg=float(alg.VOLUMES @ rho_test),
                      tls_mass_kg=float(alg.VOLUMES @ tls), tls_cost=info["cost"],
                      A=blocks[0][0].tolist(), input_wrench=rounds[0][1].tolist(),
                      X_sensor_object=X_S_O.GetAsMatrix4().tolist()),
                  current_tare=dict(file=str(tare_path),
                      wrench_frame=tare_data.get("wrench_frame"),
                      measured_passed=tare_data.get("measured", {}).get("passed")),
                  source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in (ROOT / "my_work/density_id_drake.py", ROOT / "my_work/design_core.py",
                                ROOT / "my_work/angle_aware.py", ROOT / "my_work/robot_scene.py")})
    dest = out / "ft_matrix_replay_20260907.json"
    dest.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(output=str(dest), ideal=ideal,
        cases=[{k: r[k] for k in ("name", "mass_ls_kg", "mass_map_kg", "norm_mass_kg",
                                 "density_map_total_kg", "perpendicular_residual_n",
                                 "gravity_change_deg")} for r in cases],
        synthetic=report["lamp_synthetic"]), indent=2))


if __name__ == "__main__":
    main()
