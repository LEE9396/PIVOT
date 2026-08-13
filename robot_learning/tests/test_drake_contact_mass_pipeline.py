#!/usr/bin/env python3

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "scripts"),
)

import run_drake_contact_mass_pipeline as pipeline  # noqa: E402


class DrakeContactMassPipelineTest(unittest.TestCase):
    def test_geometry_action_plan_uses_only_safe_current_object_regressors(self):
        candidates = [
            {
                "wrist_pitch_deg": pitch,
                "wrist_roll_deg": roll,
                "gravity_sensor_m_s2": [0.0, 0.0, -9.81],
                "part_com_sensor_m": [
                    [0.0, 0.0, 0.0],
                    [0.1, 0.0, 0.0],
                    [0.0, 0.1, 0.0],
                ],
                "grasp_safe": True,
                "camera_angle_span_deg": (
                    [3.0] if index == 1 else [0.2]
                ),
            }
            for index, (pitch, roll) in enumerate(
                ((-20, -20), (0, 20), (20, -20), (20, 20))
            )
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "geometry_actions.json"
            path.write_text(json.dumps({"candidates": candidates}))
            result = pipeline.plan_actions_from_geometry(path, 3)
        self.assertEqual(result["ground_truth_usage"], "none")
        self.assertEqual(result["selected"]["indices"], [0, 2, 3])
        degenerate = np.ones((6, 3))
        with self.assertRaisesRegex(RuntimeError, "no identifiable"):
            pipeline.select_geometry_action_set(
                (-20.0, 0.0, 20.0),
                (-20.0, 20.0, -20.0),
                [[degenerate, degenerate, degenerate]],
                [{0, 1, 2}],
                3,
            )

    def test_next_action_maximizes_information_after_current_holds(self):
        current = [np.array([[1.0, 0.0, 0.0]] * 6)]
        candidates = [
            np.array([[0.0, 1.0, 0.0]] * 6),
            np.eye(6, 3),
        ]
        selected = pipeline.select_next_geometry_action(
            current,
            candidates,
            {0, 1},
        )
        self.assertEqual(selected["selected"]["index"], 1)
        with self.assertRaisesRegex(ValueError, "movement cost"):
            pipeline.select_next_geometry_action(
                current,
                candidates,
                {0, 1},
                [0.0],
            )

    def test_next_action_prefers_easier_pose_when_information_is_comparable(self):
        current = [np.eye(6, 3)]
        candidates = [np.eye(6, 3) * 1.01, np.eye(6, 3)]
        selected = pipeline.select_next_geometry_action(
            current,
            candidates,
            {0, 1},
            [10.0, 1.0],
        )
        self.assertEqual(selected["selected"]["index"], 1)

    def test_joint_response_counts_only_informative_unresponsive_actions(self):
        counts = np.zeros(2, dtype=int)
        counts, informative, responsive = pipeline._joint_response_counts(
            np.array((0.0, 0.0)),
            np.array((0.2, 2.0)),
            np.array((0.1, 0.01)),
            counts,
        )
        np.testing.assert_array_equal(informative, (True, False))
        np.testing.assert_array_equal(responsive, (False, True))
        np.testing.assert_array_equal(counts, (1, 0))
        counts, _, _ = pipeline._joint_response_counts(
            np.array((0.2, 2.0)),
            np.array((0.3, 4.0)),
            np.array((0.1, 0.01)),
            counts,
        )
        np.testing.assert_array_equal(counts, (2, 0))

    def test_bootstrap_uncertainty_summary_reports_mass_and_density(self):
        summary = pipeline._bootstrap_uncertainty_summary(
            [np.array((0.8, 0.4)), np.array((1.2, 0.6))],
            [np.array((80.0, 40.0)), np.array((120.0, 60.0))],
            np.array((1.0, 0.5)),
            np.array((0.01, 0.01)),
        )
        self.assertAlmostEqual(
            summary["maximum_relative_95_half_width"],
            0.19,
        )

    def test_mass_fit_enforces_each_link_minimum(self):
        design = np.array(
            (
                (1.0, 1.0, 1.0),
                (0.0, 1.0, 2.0),
                (1.0, 0.0, -1.0),
            )
        )
        estimate = pipeline._bounded_mass_fit(
            design,
            design @ np.array((0.05, 0.35, 0.60)),
        )
        self.assertGreaterEqual(estimate.min(), 0.1)

    def test_systematic_com_error_is_rotated_from_each_link_body_frame(self):
        hold = SimpleNamespace(
            gravity_sensor_m_s2=(0.0, 0.0, -9.81),
            part_com_sensor_m=((0.0, 0.0, 0.0),),
            part_rotations_sensor=(
                ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
            ),
            joint_origins_sensor_m=(),
            joint_axes_sensor=(),
            actual_opening_angles_deg=(),
        )
        regressor = pipeline._systematic_error_regressor(
            hold,
            (),
            np.array(((0.01, 0.0, 0.0),)),
            np.array(()),
        )
        self.assertAlmostEqual(regressor[3, 0], 0.0981)

    def test_three_link_default_action_stays_within_camera_range(self):
        self.assertEqual(
            len(pipeline.WRIST_PITCH_FREE_3LINK_DEG),
            len(pipeline.WRIST_ROLL_FREE_3LINK_DEG),
        )
        self.assertEqual(len(pipeline.WRIST_PITCH_FREE_3LINK_DEG), 8)
        self.assertLessEqual(
            max(map(abs, pipeline.WRIST_PITCH_FREE_3LINK_DEG)),
            20.0,
        )
        self.assertLessEqual(
            max(map(abs, pipeline.WRIST_ROLL_FREE_3LINK_DEG)),
            20.0,
        )
        with self.assertRaisesRegex(ValueError, "minimum static holds"):
            pipeline.run(
                opening_angle_deg=180.0,
                initial_opening_angle_deg=180.0,
                steps=3,
                seed=1,
                part_count=3,
                minimum_static_holds=2,
            )

    def test_foundationpose_pose_filter_rejects_flips(self):
        result = pipeline.foundationpose_joint_angle(
            120.0,
            np.random.default_rng(7),
            joint_min_deg=0.0,
            joint_max_deg=360.0,
        )
        self.assertAlmostEqual(result["fused_opening_deg"], 120.0, delta=0.5)
        self.assertEqual(len(result["per_camera"]), 3)

    def test_foundationpose_pose_filter_preserves_signed_joint_axis(self):
        result = pipeline.foundationpose_joint_angle(
            -35.0,
            np.random.default_rng(7),
            joint_min_deg=-133.0,
            joint_max_deg=133.0,
            joint_axis=(0.0, 0.0, 1.0),
        )
        self.assertAlmostEqual(result["fused_opening_deg"], -35.0, delta=0.5)

    def test_camera_regressor_does_not_use_simulated_com_truth(self):
        pose = np.eye(4)
        tracked = {
            "mass_order_links": ["parent"],
            "frames": [{"index": 0, "fused_world_poses": {"parent": pose.tolist()}}],
        }
        hold = SimpleNamespace(
            gravity_sensor_m_s2=(0.0, 0.0, -9.81),
            sensor_world_pose=np.eye(4),
            part_com_sensor_m=((999.0, 999.0, 999.0),),
        )
        regressor = pipeline._foundationpose_static_mass_regressor(
            hold,
            tracked,
            {"frame_indices": [0]},
            ((0.1, 0.0, 0.0),),
        )
        expected = np.array([[0.0], [0.0], [9.81], [0.0], [-0.981], [0.0]])
        np.testing.assert_allclose(regressor, expected)

    def test_contact_camera_ft_pipeline_closes(self):
        result = pipeline.run(
            opening_angle_deg=180.0,
            initial_opening_angle_deg=180.0,
            steps=8,
            seed=20260728,
            part_count=3,
            vary_internal_angles=True,
            part_masses_kg=(0.8, 0.4, 0.25),
        )
        self.assertTrue(all(row["accepted"] for row in result["trace"]))
        self.assertTrue(
            all(
                len(row["three_camera_foundationpose_filter"]["per_camera"])
                == 3
                for row in result["trace"]
            )
        )
        self.assertLess(
            max(result["evaluator_only_ground_truth"]["mass_relative_error"]),
            0.12,
        )
        self.assertEqual(
            result["termination"]["physical_trajectory"],
            "single_grasp_without_reset",
        )

    def test_three_link_unexcited_upper_joint_is_rejected(self):
        result = pipeline.run(
            opening_angle_deg=180.0,
            initial_opening_angle_deg=180.0,
            steps=8,
            seed=20260728,
            part_count=3,
            vary_internal_angles=True,
            part_masses_kg=(0.55, 0.7, 0.35),
        )
        observed = [
            row["three_camera_foundationpose_filter"][
                "fused_opening_angles_deg"
            ][0]
            for row in result["trace"]
        ]
        self.assertGreater(max(observed) - min(observed), 0.5)
        self.assertTrue(all(row["accepted"] for row in result["trace"]))
        self.assertLess(
            result["termination"]["joint_excitation_range_deg"][1],
            pipeline.MIN_UPPER_JOINT_EXCITATION_DEG,
        )
        self.assertFalse(result["validation"]["passed"])
        self.assertEqual(
            result["termination"]["reason"],
            "insufficient_internal_joint_excitation",
        )


if __name__ == "__main__":
    unittest.main()
