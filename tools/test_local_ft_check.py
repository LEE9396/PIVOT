"""실물 연결 없이 프레임·차분·작은 이동의 안전 조건을 검증한다."""

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from copy import deepcopy

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "my_work"))
import design_core as dc
import hardware_real as hr
import hardware as hw
import robot_scene as rs
import path_planning as pp
import preflight
from local_ft_check import compare_load, local_plan, record, select_stable_tail


# 실제 연결 후 관측한 빈 그리퍼 자세. 과거 자세는 link3/link5 관통으로 거부해야 한다.
START = np.deg2rad([84.254257, -33.486546, -85.785820,
                   -91.585732, 89.660400, 83.299644])


class LocalFTChecks(unittest.TestCase):
    def test_continuous_empty_gravity_tare_preserves_loaded_wrench(self):
        import dual_view
        directions = np.vstack([np.eye(3), -np.eye(3)])
        mass, com = 1.05, np.array([.01, -.02, .06])
        bias = np.array([3., 5., 46., .1, -.2, .03])

        def empty(g):
            f = mass * 9.81 * g
            return bias + np.r_[f, np.cross(com, f)]

        payload = dict(wrench_frame='ft_mount', measured=dict(passed=True),
                       created_at_s=100., entries=[dict(g_hat=g.tolist(),
                           achieved_g_hat=g.tolist(), wrench=empty(g).tolist())
                           for g in directions])
        tare = hw.GravityTare(payload, max_age_s=20., clock=lambda: 110.)
        np.testing.assert_allclose(tare.bias, bias, atol=1e-12)
        np.testing.assert_allclose(tare.com_m, com, atol=1e-12)
        self.assertAlmostEqual(tare.mass_kg, mass)
        for g in (np.array([.3, -.4, .5]), np.array([-.7, .1, .2])):
            g /= np.linalg.norm(g)
            force = 1.332 * 9.81 * g
            load = np.r_[force, np.cross([.02, .03, .15], force)]
            np.testing.assert_allclose(tare.apply(g, empty(g) + load), load, atol=1e-12)
            np.testing.assert_allclose(tare.apply(g, empty(g)), 0., atol=1e-12)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'tare.json'
            path.write_text(json.dumps(payload))
            loaded = dual_view._load_tare(SimpleNamespace(tare_file=str(path),
                tare_mode='gravity', tare_max_age_s=0.))
            self.assertIsInstance(loaded, hw.GravityTare)
        for field, value in (('wrench_frame', 'obj_sensor'),
                              ('measured', dict(passed=False)),
                              ('entries', [payload['entries'][0]] * 6)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                hw.GravityTare({**payload, field: value})
        bad = deepcopy(payload)
        bad['entries'][0]['wrench'][0] = float('nan')
        with self.assertRaises(ValueError):
            hw.GravityTare(bad)
        with self.assertRaises(RuntimeError):
            hw.GravityTare(payload, max_age_s=5., clock=lambda: 110.).apply(
                directions[0], empty(directions[0]))
        # 센서 취득 경로도 명령 방향 대신 현재 FK로 영점을 차감해야 한다.
        _, planner, scene, _, _ = local_plan(START)
        plant, ctx = scene['plant'], planner.context
        plant.SetPositions(ctx, planner.full_q(START))
        g = plant.GetBodyByName('ft_mount').body_frame().CalcPoseInWorld(
            ctx).rotation().matrix().T @ np.array([0., 0., -1.])
        load = np.r_[1.332 * 9.81 * g, np.zeros(3)]
        robot = dual_view.RobotScreen.__new__(dual_view.RobotScreen)
        robot.__dict__.update(tare=tare, plant=plant, plant_context=ctx,
            driver=SimpleNamespace(joint_positions=lambda: START.copy()),
            arm_joints=[plant.GetJointByName(n, scene['arm']) for n in rs.ARM_JOINT_NAMES],
            q=planner.full_q(START), samples_per_hold=1,
            wrench_sensor=SimpleNamespace(read_raw=lambda _: empty(g) + load),
            last_raw=[], last_tare=[], last_measurement_poses=[])
        from pydrake.math import RotationMatrix
        requested = RotationMatrix.MakeXRotation(.01).matrix() @ g
        np.testing.assert_allclose(robot.read_one(requested, []), load, atol=1e-12)
        np.testing.assert_allclose(robot.last_measurement_poses[0]['achieved_g_hat'], g)

    def test_planner_uses_actual_gravity_shared_grasp_and_three_degree_sigma(self):
        import angle_aware as aa
        import density_id_drake as alg
        import density_id_objects as obj
        import desk_lamp
        import dual_view
        from pydrake.math import RotationMatrix

        spec = desk_lamp.build_spec(grasp_at='pinch', grasp_part='link_3')
        obj.bind_object(spec)
        truth = np.linspace(700., 1500., alg.P)
        masses = alg.VOLUMES * truth
        grasp = np.array([.004, -.003, .002])
        planner = dual_view.PlannerScreen.__new__(dual_view.PlannerScreen)
        planner.__dict__.update(spec=spec, blocks=[], rounds=[], g_history=[],
            rho_hat=alg.MU0.copy(), Sigma=alg.SIGMA0.copy(), rho_gt=truth,
            grasp_sigma_m=.010, grasp_hat=np.zeros(3), angle_rel_error=0.,
            angle_floor_deg=3., total_mass_kg=float(alg.VOLUMES @ alg.MU0),
            estimator='tls', stop_rule='residual', bias_cov=None, target=.05,
            show=lambda *args: None)
        with patch.object(dual_view.time, 'sleep'), patch('builtins.print'), \
                patch.object(dc, 'tls_map', wraps=dc.tls_map) as solver:
            for i, angles in enumerate(([27.48, 51.27], [10.15, 48.08], [24.43, 71.39])):
                th = np.radians(angles)
                rotation = RotationMatrix.MakeXRotation(.007 * (i + 1)).matrix()
                directions = np.array(alg.G_DIRS) @ rotation.T
                # 회귀행렬 없이 부위 질량과 모멘트로 입력을 만든다.
                centers = alg.part_centroids_in_S(th) - grasp
                y = np.concatenate([np.r_[(masses[:, None] * 9.81 * g).sum(axis=0),
                    np.cross(centers, masses[:, None] * 9.81 * g).sum(axis=0)]
                    for g in directions])
                offset = np.tile([3., 5., 46., .12, -.08, .03], len(directions))
                planner.update(dict(round=i + 1, object_joint_deg_measured=angles,
                    wrench=y, wrench_raw=y + offset, tare_applied=offset,
                    tare_required=True, gravity_tare=True,
                    measurement_poses=[dict(achieved_g_hat=g) for g in directions]))
                np.testing.assert_allclose(planner.blocks[-1][0], dc.regressor(th, directions))
                np.testing.assert_allclose(planner.g_history[-1], directions)
                self.assertEqual(solver.call_args.kwargs['floor_deg'], 3.)
                self.assertEqual(solver.call_args.kwargs['rel_error'], 0.)
                self.assertEqual(solver.call_args.kwargs['grasp_sigma_m'], .010)
                self.assertNotIn('total_mass_kg', solver.call_args.kwargs)
        self.assertLess(abs(planner.total_mass_kg - masses.sum()), .001)
        # 공통 파지 오차와 부위 밀도는 서로 섞일 수 있다. 정답을 강제하지 않고
        # 불확실성이 합성 입력의 밀도 오차를 포함하는지 확인한다.
        self.assertTrue(np.all(np.abs(planner.rho_hat - truth) < planner.absolute_half_width()),
                        (planner.rho_hat, truth, planner.absolute_half_width()))
        self.assertEqual(len(planner.grasp_hat), 3)
        self.assertEqual(len(solver.call_args.args[4]), 3)
        self.assertTrue(np.all(np.linalg.eigvalsh(planner.Sigma) > 0))
        for theta in (np.zeros(2), np.radians([80., 130.])):
            np.testing.assert_allclose(aa.angle_covariance(theta, 0., 3.),
                                       np.eye(2) * np.radians(3.) ** 2)
        np.testing.assert_allclose(planner.absolute_half_width(),
                                   planner.half_width(per_part=True) * planner.rho_hat)

    def test_preflight_does_not_accept_failed_tare_or_invent_angle_accuracy(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'calibration').mkdir()
            tare = root / 'tare.json'
            tare.write_text(json.dumps(dict(entries=[], measured=dict(passed=False))))
            (root / 'calibration/angle_signs.json').write_text(json.dumps(
                {'joint': dict(sign=-1, offset_deg=116, method='urdf_scan_pose_geometry')}))
            report = preflight.Report()
            preflight.check_calibration(report, {'TARE_FILE': str(tare)}, root)
            rows = {r['name']: r for r in report.rows}
            self.assertEqual(rows['3자세 영점 조정']['level'], preflight.FAIL)
            self.assertEqual(rows['영점 물리 검산']['level'], preflight.FAIL)
            self.assertEqual(rows['각도 부호·영점']['level'], preflight.WARN)
            self.assertNotIn('0.00', rows['각도 부호·영점']['detail'])

    def test_measured_grasp_mode_requires_measurement(self):
        import dual_view
        with patch.object(rs, 'load_measured_grasp', return_value=None):
            with self.assertRaisesRegex(RuntimeError, 'grasp.json'):
                dual_view.prepare(None, None, [], 1., 1, .02, 1., grasp_frame='measured')

    def test_calibrated_table_preserves_heading_and_top_plane(self):
        for normal in (np.array([0., 0., 1.]), np.array([.002, .006, 1.])):
            normal /= np.linalg.norm(normal)
            offset = -.67
            size, pose = rs.table_box_pose(dict(normal=normal, offset=offset))
            self.assertGreater(pose.rotation().matrix()[0, 0], .999)
            for x in (-size[0]/2, size[0]/2):
                for y in (-size[1]/2, size[1]/2):
                    corner = pose @ np.array([x, y, size[2]/2])
                    self.assertAlmostEqual(float(normal @ corner + offset), 0., places=12)
            if normal[0] == 0 and normal[1] == 0:
                np.testing.assert_allclose(pose.rotation().matrix(), np.eye(3))

    def test_estimator_applies_force_and_torque_tare_once(self):
        directions = np.eye(3)
        A = np.vstack([np.r_[9.81 * g, np.cross([.02, .03, .1], 9.81 * g)]
                       for g in directions]).reshape(18, 1)
        offset = np.arange(18, dtype=float) / 10 + 2
        corrected = A[:, 0] * 1.332
        raw = corrected + offset
        reply = dict(wrench=corrected, wrench_raw=raw, tare_applied=offset,
                     tare_required=True)
        with patch.object(dc, "regressor", return_value=A):
            matrix, y = dc.measurement_equation([], reply, directions)
        np.testing.assert_allclose(y, corrected, atol=1e-12)
        # 알려진 영점을 고정한 확장 행렬과 우변 차감은 같은 방정식이다.
        np.testing.assert_allclose(np.column_stack((A, np.eye(18)))
                                   @ np.r_[1.332, offset], raw)
        mass = dc.wls_map([(matrix, y, np.eye(18) * .01)],
                          np.array([1.]), np.array([[9.]]), (0., 20.))[0]
        self.assertAlmostEqual(mass, 1.332, places=5)

    def test_estimator_rejects_missing_or_double_tare(self):
        offset = np.array([1., 2., 46., .1, .2, .3])
        y = np.array([0., 0., 13., 0., 0., 0.])
        good = dict(wrench=y, wrench_raw=y + offset, tare_applied=offset,
                    tare_required=True)
        with patch.object(dc, "regressor", return_value=np.ones((6, 1))):
            for changes in (dict(wrench=y + offset), dict(wrench=y - offset),
                            dict(wrench_raw=None), dict(tare_applied=None),
                            dict(wrench_raw=None, tare_applied=None),
                            dict(tare_applied=[0.]), dict(wrench_raw=[np.nan]*6)):
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    dc.measurement_equation([], {**good, **changes}, [[0, 0, 1]])
            _, sim = dc.measurement_equation([], dict(wrench=y), [[0, 0, 1]])
            np.testing.assert_array_equal(sim, y)

    def test_stability_uses_final_blocks_and_both_channels(self):
        def make(values):
            return {'blocks': [dict(wrench_mean=[0, 0, z, 0, 0, 0],
                controller_wrench_mean=[0, 0, z, 0, 0, 0]) for z in values]}
        good = make([2, 1, .00, .01, .02, .03, .04])
        select_stable_tail(good)
        self.assertTrue(good['stability']['passed'])
        self.assertEqual(len(good['all_recorded_blocks']), 7)
        self.assertEqual(len(good['blocks']), 5)
        bad = make([0, 0, 0, 0, 0, .2, .4])
        with self.assertRaises(ValueError):
            select_stable_tail(bad)
        self.assertEqual(len(bad['blocks']), 7)
        other_channel = make([0]*5)
        other_channel['blocks'][-1]['controller_wrench_mean'][2] = .2
        with self.assertRaises(ValueError):
            select_stable_tail(other_channel)

    def test_driver_rejects_whole_route_before_moving(self):
        arm = hr.FakeArm()
        driver = hr.Rb5Driver(arm)
        driver.limit_to_start(np.zeros(6), np.deg2rad(2))
        for bad in (np.ones(6) * np.deg2rad(3), np.full(6, np.nan)):
            with self.assertRaises(hr.SafetyViolation):
                driver.follow([np.ones(6) * np.deg2rad(1), bad], 4.)
            self.assertEqual(arm.log, [])

    def test_live_feedback_outside_window_is_rejected(self):
        backend = hr.RbpodoBackend.__new__(hr.RbpodoBackend)
        backend.deg_api = True
        driver = hr.Rb5Driver(backend)
        driver.limit_to_start(np.zeros(6), np.deg2rad(2))
        backend._read_status = lambda: SimpleNamespace(jnt_ang=[3.] * 6)
        with self.assertRaises(hr.SafetyViolation):
            backend._wait_for_arrival([3.] * 6, 1.)

    def test_old_tare_is_not_used_after_frame_change(self):
        from dual_view import _load_tare
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tare.json"
            args = SimpleNamespace(tare_file=path, tare_max_age_s=0)
            for value in ({"entries": []}, {"wrench_frame": "ft_mount", "measured": {"passed": False}}):
                path.write_text(json.dumps(value))
                with self.assertRaisesRegex(RuntimeError, "프레임|검산"):
                    _load_tare(args)

    def test_local_collision_and_no_detour(self):
        report, planner, scene, diagram, context = local_plan(START, 6, 2.)
        route = np.deg2rad(report["route_joint_deg"])
        np.testing.assert_allclose(route[:, :5], np.tile(START[:5], (3, 1)))
        self.assertTrue(planner.edge_valid(route[0], route[1]))
        self.assertTrue(planner.edge_valid(route[1], route[2]))
        self.assertIsNone(planner.plan(START, START + np.deg2rad(3)))
        # 끝점은 통과하지만 중간이 막히는 가상 장애물을 넣는다.
        original = planner.valid
        planner._paths.clear()
        planner.valid = lambda q: original(q) and not (
            np.deg2rad(0.6) < q[5] - START[5] < np.deg2rad(1.4))
        self.assertIsNone(planner.plan(route[0], route[1]))
        self.assertFalse(planner.edge_valid(np.full(6, np.nan), START))

    def test_self_cable_and_fixture_collision_coverage(self):
        _, planner, scene, diagram, context = local_plan(START)
        plant = scene["plant"]
        query = planner.query_port.Eval(planner.context)
        inspector = query.inspector()
        lab = plant.GetModelInstanceByName("lab")
        def ids(name, model):
            return list(plant.GetCollisionGeometriesForBody(plant.GetBodyByName(name, model)))
        cable = list(plant.GetCollisionGeometriesForBody(scene["cable"]))
        shape = inspector.GetShape(cable[0])
        self.assertAlmostEqual(shape.radius(), rs.AFT_CABLE_RADIUS_M)
        self.assertAlmostEqual(shape.length(), rs.AFT_CABLE_LENGTH_M)
        pairs = [(ids("link3", scene["arm"]), ids("link5", scene["arm"]))]
        pairs += [(cable, ids("link5", scene["arm"]))]
        for body in plant.GetBodyIndices(lab):
            name = plant.get_body(body).name()
            pairs += [(cable, ids(name, lab)), (ids("link3", scene["arm"]), ids(name, lab))]
        for left, right in pairs:
            active = [(a, b) for a in left for b in right
                      if not inspector.CollisionFiltered(a, b)]
            self.assertTrue(active, "필수 충돌쌍이 제외되거나 형상이 없음")
            a, b = active[0]
            injected = SimpleNamespace(inspector=lambda: inspector,
                ComputeSignedDistancePairwiseClosestPoints=lambda _: [
                    SimpleNamespace(id_A=a, id_B=b, distance=-0.0001)])
            self.assertFalse(pp.collision_free(plant, injected, .02, scene["arm"]))
        # 예전 5 mm 관통 허용을 다시 넣으면 이 자세가 통과하므로 회귀를 잡는다.
        with self.assertRaises(RuntimeError):
            local_plan(np.deg2rad([90.706818, -56.703594, -77.287407,
                                  155.141312, -119.928932, -107.496109]))

    def test_driver_requires_collision_checks_before_and_during_motion(self):
        backend = hr.RbpodoBackend.__new__(hr.RbpodoBackend)
        driver = hr.Rb5Driver(backend)
        with self.assertRaisesRegex(hr.SafetyViolation, "충돌 검사기"):
            driver.follow([np.zeros(6)], 4.)
        backend.deg_api = True
        driver.set_collision_planner(SimpleNamespace(valid=lambda q: q[0] < .01))
        backend._read_status = lambda: SimpleNamespace(jnt_ang=[1.] * 6)
        with patch("workspace_obstacles.require_current_scene"), \
                self.assertRaisesRegex(hr.SafetyViolation, "충돌·관절"):
            backend._wait_for_arrival([1.] * 6, 1.)
        arm = hr.FakeArm()
        driver = hr.Rb5Driver(arm)
        checker = SimpleNamespace(valid=lambda q: True,
                                  edge_valid=lambda a, b: b[0] < .02)
        driver.set_collision_planner(checker)
        with self.assertRaisesRegex(hr.SafetyViolation, "전체 이동 경로"):
            driver.follow([np.full(6, .01), np.full(6, .03)], 4.)
        self.assertEqual(arm.log, [])
        # 계획 뒤 환경이 바뀌면 명령 직전 재검사에서 거부한다.
        with patch.object(checker, "edge_valid", side_effect=[True, False]):
            with self.assertRaisesRegex(hr.SafetyViolation, "명령 직전"):
                driver.follow([np.full(6, .01)], 4.)
        self.assertEqual(arm.log, [])

    def test_pending_or_changed_obstacles_block_real_commands(self):
        import workspace_obstacles as wo
        from local_ft_check import empty_scene
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "obstacles.json"
            data = dict(status="pending", frame="robot_base", units="m",
                        unresolved=["support"], boxes=[])
            with patch.object(wo, "PATH", path):
                backend = hr.RbpodoBackend.__new__(hr.RbpodoBackend)
                driver = hr.Rb5Driver(backend)
                driver.set_collision_planner(SimpleNamespace(plant=SimpleNamespace()))
                for exists in (False, True):
                    if exists:
                        path.write_text(json.dumps(data))
                    with self.assertRaisesRegex(hr.SafetyViolation, "충돌 장면"):
                        driver.follow([np.zeros(6)], 1.)
                data.update(status="validated", unresolved=[], boxes=[dict(name="support",
                    center_m=[.4, .2, .1], size_m=[.2, .3, .1], rpy_deg=[0, 0, 0], padding_m=.01)])
                path.write_text(json.dumps(data))
                scene = empty_scene()
                wo.require_current_scene(scene["plant"])
                body = scene["plant"].GetBodyByName("obstacle_support")
                self.assertGreater(len(scene["plant"].GetCollisionGeometriesForBody(body)), 0)
                pose = body.EvalPoseInWorld(scene["plant"].CreateDefaultContext())
                expected = rs._robot_base_pose().multiply(np.array([.4, .2, .1]))
                np.testing.assert_allclose(pose.translation(), expected)
                path.write_text(path.read_text() + "\n")
                with self.assertRaisesRegex(ValueError, "바뀌었습니다"):
                    wo.require_current_scene(scene["plant"])
                data["boxes"][0]["size_m"][0] = -1
                path.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    wo.load()

    def test_physical_ft_frame_does_not_follow_grasp_rotation(self):
        report, planner, first, diagram, context = local_plan(START)
        matrix = np.eye(4)
        from pydrake.math import RotationMatrix
        matrix[:3, :3] = RotationMatrix.MakeXRotation(0.7).matrix()
        second = rs.build_scene(first["spec"], joint_limits_rad=[],
                                gripper="robotiq2f85", include_visuals=False,
                                grasp_transform=matrix)
        second_diagram = second["builder"].Build()
        second_root = second_diagram.CreateDefaultContext()
        second_context = second["plant"].GetMyContextFromRoot(second_root)
        for name, value in zip(rs.ARM_JOINT_NAMES, START):
            second["plant"].GetJointByName(name, second["arm"]).set_angle(second_context, value)
        first["plant"].SetPositions(planner.context, planner.full_q(START))
        def rotation(scene, ctx, key):
            return scene[key].CalcPoseInWorld(ctx).rotation().matrix()
        np.testing.assert_allclose(rotation(first, planner.context, "wrench_frame"),
                                   rotation(second, second_context, "wrench_frame"), atol=1e-12)
        self.assertGreater(np.linalg.norm(rotation(first, planner.context, "sensor_frame")
                                         - rotation(second, second_context, "sensor_frame")), 0.5)
        # 실제 PoseChecker와 영점 함수가 이 물리 프레임을 쓰는지 확인한다.
        with patch.object(rs, "build_scene", return_value=rs.build_scene(
                first["spec"], joint_limits_rad=[], gripper="robotiq2f85",
                include_visuals=False, grasp_transform=matrix)):
            checker = rs.PoseChecker(first["spec"], joint_limits_rad=[], gripper="robotiq2f85")
        self.assertEqual(checker.wrench_frame.name(), "ft_mount")
        checker.seed_q = START.copy()
        for joint, value in zip(checker.arm_joints, START):
            joint.set_angle(checker.context, value)
        desired = checker.wrench_frame.CalcPoseInWorld(checker.context).rotation().matrix().T @ [0., 0., -1.]
        position = checker.sensor_frame.CalcPoseInWorld(checker.context).translation()
        seen = []
        def inspect_gravity_constraint(program):
            # 충돌/FOV와 독립적으로 실제 IK 중력식이 알려진 시작 자세를 만족하는지 본다.
            guess = program.GetInitialGuess(program.decision_variables())
            for binding in program.GetAllConstraints():
                evaluator = binding.evaluator()
                if type(evaluator).__name__ == "AngleBetweenVectorsConstraint":
                    value = program.EvalBinding(binding, guess)
                    self.assertTrue(np.all(value >= evaluator.lower_bound() - 1e-10))
                    self.assertTrue(np.all(value <= evaluator.upper_bound() + 1e-10))
                    seen.append(value)
            return SimpleNamespace(is_success=lambda: False)
        with patch.object(rs, "Solve", side_effect=inspect_gravity_constraint):
            checker.solve(np.array([]), desired, warm_start=False,
                          workspace=(position - .05, position + .05))
        self.assertEqual(len(seen), 1)

    def test_bottle_difference_cancels_tool_bias(self):
        bias = [1., 2., 3., .1, .2, .3]
        def observation(values):
            return dict(status="recorded", sensor_sha256="same", blocks=[dict(
                joint_before_deg=[0.] * 6, joint_after_deg=[0.] * 6, wrench_mean=values)])
        base = observation(bias)
        loaded = observation([1., 2., 3. - .541 * 9.81, .1, .2, .3])
        result = compare_load(loaded, base, .541)
        self.assertAlmostEqual(result["observed_force_norm_n"], 5.30721)
        self.assertAlmostEqual(result["force_norm_relative_error"], 0.)
        loaded["blocks"][0]["joint_after_deg"][0] = 1.
        with self.assertRaises(ValueError):
            compare_load(loaded, base, .541)

    def test_absolute_relative_intervals_agree(self):
        covariance = np.diag([4., 9.])
        bias = np.diag([1., 4.])
        rho = np.array([100., 200.])
        half = dc.absolute_half_width(covariance, bias, inflate=3.)
        np.testing.assert_allclose(half, 1.96 * np.sqrt([37., 85.]))
        np.testing.assert_allclose(half / rho, dc.half_width(covariance, rho, bias, inflate=3.))

    def test_paired_record_keeps_channels_separate_and_rejects_motion(self):
        status = SimpleNamespace(robot_state=1, task_state=1,
            op_stat_collision_occur=0, op_stat_sos_flag=0,
            op_stat_soft_estop_occur=0, op_stat_ems_flag=0, jnt_ang=[0.] * 6,
            eft_fx=1.01, eft_fy=2.01, eft_fz=3.01,
            eft_mx=.101, eft_my=.201, eft_mz=.301)
        data = SimpleNamespace(request_data=lambda _: SimpleNamespace(sdata=status))
        class Sensor:
            def stream(self):
                for i in range(2):
                    self.last_registers = [50, 100, 150, 5, 10, 15]
                    self.last_sample_time_s = float(i)
                    yield np.array(self.last_registers) * .02
        report = dict(status="recorded", sensor_sha256="same")
        record(Sensor(), data, 2, 1, report)
        block = report["blocks"][0]
        np.testing.assert_allclose(block["wrench_mean"], [1, 2, 3, .1, .2, .3])
        np.testing.assert_allclose(block["controller_wrench_mean"],
                                   [1.01, 2.01, 3.01, .101, .201, .301])
        self.assertEqual(len(block["controller_request_receive_time_s"]), 2)
        loaded = json.loads(json.dumps(report))
        loaded["blocks"][0]["controller_wrench_mean"][2] += .541 * 9.81
        self.assertAlmostEqual(compare_load(loaded, report, .541,
            "controller_wrench_mean")["force_norm_relative_error"], 0.)
        self.assertAlmostEqual(compare_load(loaded, report, .541)
                               ["observed_force_norm_n"], 0.)
        loaded["blocks"][0]["controller_joint_deg"][0][0] = 1.
        with self.assertRaises(ValueError):
            compare_load(loaded, report, .541)
        with patch("local_ft_check.read_state", side_effect=[
                (np.zeros(6), status), (np.zeros(6), status),
                (np.deg2rad([1., 0, 0, 0, 0, 0]), status)]):
            with self.assertRaises(hr.SafetyViolation):
                record(Sensor(), data, 2, 1, {})


if __name__ == "__main__":
    unittest.main()
