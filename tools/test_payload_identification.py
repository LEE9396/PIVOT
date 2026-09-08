"""물체 파라미터 복원·오차 은폐 방지·기록 검사를 하드웨어 없이 확인한다."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "my_work"))
import density_id_drake as alg
import design_core as dc
from payload_identification import fit_static, separate_payload
from identify_ft_payload import identify, load_record
from local_ft_check import gripper_snapshot, record
from analyze_ft_drift import analyze
import hardware_real as hr


class PayloadChecks(unittest.TestCase):
    def setUp(self):
        self.g = np.eye(3)
        self.bias = np.array([3., -2., 45., .1, -.2, .3])
        self.tool_mass = 1.05
        self.tool_com = np.array([.01, -.02, .06])
        self.mass = 1.332
        self.com = np.array([.03, .04, .15])

    def wrench(self, mass, com):
        force = alg.FORCE_SIGN * 9.81 * mass * self.g
        return np.column_stack([force, np.cross(com, force)])

    def data(self):
        empty = self.bias + self.wrench(self.tool_mass, self.tool_com)
        return empty, empty + self.wrench(self.mass, self.com)

    def test_parameter_difference_preserves_measured_wrench_and_density_input(self):
        empty, loaded = self.data()
        loaded[0, 1] += .03
        result = separate_payload(self.g, empty, self.g, loaded)
        self.assertTrue(result["passed"])
        self.assertAlmostEqual(result["object_mass_kg"], self.mass)
        np.testing.assert_allclose(result["object_com_m"], self.com, atol=1e-12)
        np.testing.assert_allclose(result["corrected_wrench"], loaded - empty, atol=1e-12)
        # 실제 공통 추정기 입구가 동일한 원시값/영점/차분을 받는지 검사한다.
        with patch.object(dc, "regressor", return_value=np.zeros((18, 2))):
            _, y = dc.measurement_equation([0.], result["estimator_reply"], self.g)
        np.testing.assert_allclose(y, (loaded - empty).ravel(), atol=1e-12)
        self.assertFalse(result["uncertainty_calibrated"])

    def test_torque_change_does_not_change_mass(self):
        empty, _ = self.data()
        for com in ([0., 0., .095], [0., 0., .155]):
            result = separate_payload(self.g, empty, self.g,
                                      empty + self.wrench(self.mass, com))
            self.assertTrue(result["passed"])
            self.assertAlmostEqual(result["object_mass_kg"], self.mass)
            np.testing.assert_allclose(result["object_com_m"], com, atol=1e-12)

    def test_original_simulator_and_density_matrix_after_tool_subtraction(self):
        import density_id_objects as obj
        part = obj.Part("payload", (100., 100., 100.), 1000., 1332.,
                        (30., 40., 150.), (30., 40., 150.), (.3, .3, .3, 1.))
        obj.bind_object(obj.ObjectSpec("payload_check", "payload", [part], [], (0., 0., 0.)))
        empty, _ = self.data()
        with patch.object(alg, "R_EPS_DIAG", np.zeros(6)):
            ideal = alg.measure([], self.g).reshape(-1, 6)
        result = separate_payload(self.g, empty, self.g, empty + ideal)
        self.assertTrue(result["passed"])
        matrix, y = dc.measurement_equation([], result["estimator_reply"], self.g)
        np.testing.assert_allclose(y, ideal.ravel(), atol=1e-12)
        np.testing.assert_allclose(matrix @ np.array([1332.]), y, atol=1e-12)
        rho = dc.wls_map([(matrix, y, np.eye(18)*1e-6)],
                         np.array([1000.]), np.array([[1e6]]), (0., 10000.))
        self.assertAlmostEqual(float(alg.VOLUMES @ rho), 1.332, places=7)

    def test_large_fz_is_rejected_not_absorbed_as_new_bias(self):
        empty, loaded = self.data()
        loaded[1, 2] += 18.84
        result = separate_payload(self.g, empty, self.g, loaded)
        self.assertFalse(result["passed"])
        self.assertIsNone(result["estimator_reply"])
        self.assertGreater(result["loaded"]["max_force_residual_n"], 18.)
        np.testing.assert_allclose(result["loaded"]["bias"], self.bias, atol=1e-12)

    def test_repeated_pose_and_invalid_data_do_not_identify_tool(self):
        empty, _ = self.data()
        result = fit_static(np.tile(self.g[1], (5, 1)), np.tile(empty[1], (5, 1)))
        self.assertFalse(result["identifiable"])
        self.assertNotIn("mass_kg", result)
        with self.assertRaises(ValueError):
            fit_static(self.g, np.full((3, 6), np.nan))

    def test_opening_selection_and_configuration_guards(self):
        empty, loaded = self.data()
        def records(values, count, group):
            return [dict(g=g, wrench=w, opening_count=count, configuration_id=group,
                         sensor_sha256="sensor", kinematics_sha256="fk", ft_session_id="epoch")
                    for g, w in zip(self.g, values)]
        e = records(empty, 150, "empty150") + records(empty + 10, 200, "empty200")
        l = records(loaded, 151, "bottle_A")
        self.assertTrue(identify(e, l)["passed"])
        self.assertEqual(identify(e, l)["selected_empty_group"], "empty150")
        for field, value in [("configuration_id", "bottle_B"),
                             ("ft_session_id", "reset"), ("opening_count", 180)]:
            bad = deepcopy(l)
            bad[0][field] = value
            with self.assertRaises(ValueError):
                identify(e, bad)

    def test_record_parser_and_stale_gripper_status(self):
        empty, _ = self.data()
        block = dict(wrench_mean=empty[0].tolist(), achieved_g_hat=self.g[0].tolist(),
                     joint_before_deg=[0.] * 6, joint_after_deg=[0.] * 6,
                     gripper_before=dict(position=150), gripper_after=dict(position=150),
                     wrench_samples=[empty[0].tolist()] * 3)
        data = dict(status="recorded", load_state="empty", wrench_frame="ft_mount",
                    sensor_sha256="sensor", kinematics_sha256="fk", ft_session_id="epoch",
                    configuration_id="empty150", blocks=[deepcopy(block) for _ in range(3)])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "record.json"
            path.write_text(json.dumps(data))
            self.assertEqual(load_record(path, "empty")["opening_count"], 150.)
            data["blocks"][2]["joint_after_deg"][0] = .2
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                load_record(path, "empty")
            status = dict(gripper_busy=False, gripper_sample=dict(
                timestamp_s=100., activated=True, fault=0, position=150, status=3, object=2))
            path.write_text(json.dumps(status))
            with patch("local_ft_check.time.time", return_value=101.):
                self.assertEqual(gripper_snapshot(path)["position"], 150)
            with patch("local_ft_check.time.time", return_value=103.):
                with self.assertRaises(ValueError):
                    gripper_snapshot(path)
            status["gripper_sample"]["object"] = 0
            path.write_text(json.dumps(status))
            with patch("local_ft_check.time.time", return_value=101.):
                with self.assertRaises(ValueError):
                    gripper_snapshot(path)

    def test_live_sensor_excludes_gripper_motion_before_averaging(self):
        stable = dict(position=150, requested_position=160)
        changed = dict(position=152, requested_position=160)
        calls = []
        sensor = hr.Aft200Sensor(lambda: calls.append(1) or np.ones(6),
                                gripper_status_file="status.json")
        with patch.object(hr, "gripper_snapshot", side_effect=[stable, stable, changed]), \
                patch.object(hr.time, "sleep") as sleep:
            with self.assertRaises(hr.SafetyViolation):
                sensor.read_raw(2)
            sleep.assert_called_once_with(1.)
        self.assertEqual(len(calls), 1)

    def test_record_adds_actual_gravity_and_gripper_feedback(self):
        class Sensor:
            last_registers = [0] * 6
            last_sample_time_s = 100.

            def stream(self):
                while True:
                    yield np.arange(6, dtype=float)

        status = SimpleNamespace(**{"eft_" + k: float(i) for i, k in
                                    enumerate(("fx", "fy", "fz", "mx", "my", "mz"))})
        sample = dict(position=150, timestamp_s=100.)
        report = {}
        with patch("local_ft_check.read_q", return_value=np.zeros(6)), \
                patch("local_ft_check.read_state", return_value=(np.zeros(6), status)), \
                patch("local_ft_check.gripper_snapshot", return_value=sample):
            record(Sensor(), None, 2, 1, report, "unused.json")
        block = report["blocks"][0]
        self.assertAlmostEqual(np.linalg.norm(block["achieved_g_hat"]), 1.)
        self.assertEqual(block["gripper_after"]["position"], 150)
        np.testing.assert_allclose(block["wrench_mean"], np.arange(6))

    def test_drift_analysis_keeps_trend_and_duration_record_uses_clock(self):
        blocks = []
        for start in range(0, 620, 20):
            t = np.arange(start, start+20, dtype=float)
            y = np.zeros((20, 6))
            y[:, 2] = t * .001
            blocks.append(dict(sample_time_s=(t+1000).tolist(), wrench_samples=y.tolist(),
                               wrench_mean=y.mean(axis=0).tolist(),
                               controller_joint_deg=np.zeros((20, 6)).tolist()))
        result = analyze(dict(status="recorded", load_state="empty", blocks=blocks))
        self.assertTrue(result["requested_10min_completed"])
        self.assertAlmostEqual(result["fitted_slope_per_minute"][2], .06)
        self.assertGreater(result["force_endpoint_change_n"], .55)
        self.assertFalse(result["thermal_cause_confirmed"])

        class Sensor:
            last_registers = [0] * 6
            last_sample_time_s = 100.
            def stream(self):
                while True:
                    yield np.zeros(6)
        status = SimpleNamespace(**{"eft_"+k: 0. for k in ("fx", "fy", "fz", "mx", "my", "mz")})
        report, checkpoints = {}, []
        with patch("local_ft_check.read_q", return_value=np.zeros(6)), \
                patch("local_ft_check.read_state", return_value=(np.zeros(6), status)), \
                patch("local_ft_check.time.monotonic", side_effect=[0., 0., 2., 2., 4., 4., 4.]):
            record(Sensor(), None, 2, 99, report, duration_s=3.,
                   checkpoint=lambda r: checkpoints.append(len(r["blocks"])))
        self.assertEqual(checkpoints, [1, 2])
        self.assertEqual(report["record_elapsed_s"], 4.)


if __name__ == "__main__":
    unittest.main()
