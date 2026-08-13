#!/usr/bin/env python3
"""Regression tests for the contact-derived RB5 wrist-F/T scene."""

from pathlib import Path
import itertools
import math
import sys
import unittest
import xml.etree.ElementTree as ET

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import simulate_drake_rb5_contact_ft_custom_object as contact  # noqa: E402
import visualize_drake_rb5_hammer_payload as rb5  # noqa: E402


class DrakeRb5ContactFtCustomObjectTest(unittest.TestCase):
    def test_household_profiles_match_gripper_and_joint_limits(self):
        jewelry = contact.HOUSEHOLD_PROFILES["jewelry_box"]
        lamp = contact.HOUSEHOLD_PROFILES["desk_lamp"]
        phantom = contact.HOUSEHOLD_PROFILES["phantom_v3"]
        self.assertEqual(jewelry["joint_limits_deg"], (120.0,))
        self.assertEqual(lamp["joint_limits_deg"], (120.0, 90.0))
        self.assertEqual(jewelry["part_names"][0], "base")
        self.assertEqual(lamp["part_names"][0], "arm")
        self.assertLessEqual(lamp["sizes_m"][0][1], 0.085)
        self.assertEqual(
            phantom["joint_angle_bounds_deg"],
            ((-133.0, 133.0), (-147.0, 149.0)),
        )
        self.assertAlmostEqual(sum(phantom["default_masses_kg"]), 1.0)
        self.assertEqual(
            jewelry["default_initial_angles_deg"], (105.0,)
        )
        self.assertEqual(
            lamp["default_initial_angles_deg"], (120.0, 0.0)
        )
        self.assertEqual(
            phantom["default_initial_angles_deg"], (0.0, 0.0)
        )
        for profile in (jewelry, lamp, phantom):
            self.assertNotIn("wrist_pitch_sequence_deg", profile)
            self.assertNotIn("wrist_roll_sequence_deg", profile)
        self.assertTrue(
            all(
                (contact.PHANTOM_V3_ASSET_DIR / mesh).is_file()
                for mesh in phantom["mesh_files"]
            )
        )

    def test_phantom_v3_builder_uses_safe_asymmetric_limits(self):
        builder = contact.DiagramBuilder()
        plant, _ = contact.AddMultibodyPlantSceneGraph(
            builder, contact.TIME_STEP_S
        )
        _, bodies, joints = contact._add_household_object(
            plant,
            "phantom_v3",
            include_visuals=False,
            part_masses_kg=None,
        )
        plant.Finalize()
        self.assertEqual(len(bodies), 3)
        self.assertEqual(
            tuple(
                (
                    round(math.degrees(joint.position_lower_limits()[0])),
                    round(math.degrees(joint.position_upper_limits()[0])),
                )
                for joint in joints
            ),
            ((-133, 133), (-147, 149)),
        )

    def test_household_full_joint_ranges_have_no_part_overlap(self):
        ranges = {
            "jewelry_box": ((range(0, 121, 10)),),
            "desk_lamp": (range(0, 121, 10), range(0, 91, 10)),
        }
        for profile, angle_ranges in ranges.items():
            builder = contact.DiagramBuilder()
            plant, scene_graph = contact.AddMultibodyPlantSceneGraph(
                builder, contact.TIME_STEP_S
            )
            parent, bodies, joints = contact._add_household_object(
                plant,
                profile,
                include_visuals=False,
                part_masses_kg=None,
            )
            plant.Finalize()
            diagram = builder.Build()
            context = diagram.CreateDefaultContext()
            plant_context = plant.GetMyMutableContextFromRoot(context)
            plant.SetFreeBodyPose(
                plant_context, parent, contact.RigidTransform([0.0, 0.0, 1.0])
            )
            inspector = scene_graph.model_inspector()
            geometry_ids = list(inspector.GetAllGeometryIds())
            minimum_clearance = math.inf
            for angles in itertools.product(*angle_ranges):
                for joint, angle in zip(joints, angles, strict=True):
                    joint.set_angle(
                        plant_context,
                        contact._internal_joint_angle_rad(
                            profile, joint.name(), angle
                        ),
                    )
                query = scene_graph.get_query_output_port().Eval(
                    scene_graph.GetMyContextFromRoot(context)
                )
                for a, b in itertools.combinations(geometry_ids, 2):
                    names = {
                        inspector.GetName(a),
                        inspector.GetName(b),
                    }
                    if profile == "desk_lamp" and names not in (
                        {
                            "desk_lamp_arm_model::desk_lamp_arm_collision",
                            "desk_lamp_head_model::desk_lamp_head_collision",
                        },
                        {
                            "desk_lamp_base_model::desk_lamp_base_collision",
                            "desk_lamp_arm_model::desk_lamp_arm_collision",
                        },
                        {
                            "desk_lamp_hinge_pedestal_collision",
                            "desk_lamp_arm_model::desk_lamp_arm_collision",
                        },
                    ):
                        continue
                    minimum_clearance = min(
                        minimum_clearance,
                        query.ComputeSignedDistancePairClosestPoints(a, b).distance,
                    )
            self.assertGreaterEqual(minimum_clearance, 0.0, profile)

            if profile == "desk_lamp":
                upper = joints[1]
                for requested in (0.0, 45.0, 90.0):
                    upper.set_angle(
                        plant_context,
                        contact._internal_joint_angle_rad(
                            profile, upper.name(), requested
                        ),
                    )
                    folded_axis = -(
                        plant.EvalBodyPoseInWorld(
                            plant_context, bodies[0]
                        ).rotation()
                        @ contact.RotationMatrix.MakeYRotation(
                            math.radians(
                                contact.DESK_LAMP_LOWER_ZERO_OFFSET_DEG
                            )
                        )
                    ).matrix()[:, 0]
                    head_axis = -plant.EvalBodyPoseInWorld(
                        plant_context, bodies[1]
                    ).rotation().matrix()[:, 0]
                    physical = math.degrees(
                        math.acos(np.clip(folded_axis @ head_axis, -1.0, 1.0))
                    )
                    self.assertAlmostEqual(physical, requested, places=5)

    def test_household_free_hinges_settle_under_gravity(self):
        cases = {
            "jewelry_box": ((105.0,), (120.0,)),
            "desk_lamp": ((90.0, 90.0), None),
        }
        for profile, (initial, expected) in cases.items():
            builder = contact.DiagramBuilder()
            plant, _ = contact.AddMultibodyPlantSceneGraph(
                builder, contact.TIME_STEP_S
            )
            parent, _, joints = contact._add_household_object(
                plant,
                profile,
                include_visuals=False,
                part_masses_kg=None,
            )
            plant.WeldFrames(
                plant.world_frame(),
                parent.body_frame(),
                contact.RigidTransform([0.0, 0.0, 1.0]),
            )
            plant.Finalize()
            diagram = builder.Build()
            simulator = contact.Simulator(diagram)
            context = simulator.get_mutable_context()
            plant_context = plant.GetMyMutableContextFromRoot(context)
            for joint, angle in zip(joints, initial, strict=True):
                joint.set_angle(
                    plant_context,
                    contact._internal_joint_angle_rad(
                        profile, joint.name(), angle
                    ),
                )
            simulator.AdvanceTo(5.0)
            openings = [
                (
                    -math.degrees(joint.get_angle(plant_context))
                    - contact.DESK_LAMP_LOWER_ZERO_OFFSET_DEG
                    if profile == "desk_lamp" and "lower" in joint.name()
                    else -math.degrees(joint.get_angle(plant_context))
                )
                for joint in joints
            ]
            self.assertTrue(all(np.isfinite(openings)))
            self.assertTrue(
                all(
                    abs(math.degrees(joint.get_angular_rate(plant_context))) < 0.1
                    for joint in joints
                )
            )
            if expected is not None:
                self.assertTrue(np.allclose(openings, expected, atol=0.1))
            else:
                self.assertTrue(-1e-3 <= openings[0] <= 120.0 + 1e-3)
                self.assertTrue(-1e-3 <= openings[1] <= 90.0 + 1e-3)
                self.assertFalse(np.allclose(openings, initial, atol=1.0))

    def test_rejects_aft200_overload(self):
        force = [[contact.AFT200_FORCE_LIMIT_N + 1.0, 0.0, 0.0]]
        torque = [[0.0, 0.0, 0.0]]
        with self.assertRaisesRegex(RuntimeError, "safe force/torque"):
            contact._filter_static_wrench_samples(force, torque)

    def test_rejects_object_over_pgc_payload_limit(self):
        with self.assertRaisesRegex(ValueError, "exceeds the PGC payload"):
            contact.simulate(2, part_masses_kg=(1.6, 1.5))

    def test_rejects_desk_lamp_base_lighter_than_moving_links(self):
        with self.assertRaisesRegex(
            ValueError, "base mass must be at least the combined mass"
        ):
            contact.simulate(
                3,
                object_profile="desk_lamp",
                part_masses_kg=(0.4, 0.4, 0.5),
                initial_opening_angles_deg=(120.0, 0.0),
            )

    def test_movable_pgc_keeps_two_independent_prismatic_joints(self):
        _, _, pgc_path, _ = rb5.validate_htd_source(rb5.DEFAULT_HTD_ROOT)
        root = ET.fromstring(
            rb5.pgc_movable_urdf_string(
                pgc_path,
                "test_pgc",
                include_visuals=False,
            )
        )
        joints = {
            joint.get("name"): joint for joint in root.findall("joint")
        }
        for name in ("finger1_joint", "finger2_joint"):
            self.assertEqual(joints[name].get("type"), "prismatic")
            self.assertIsNone(joints[name].find("mimic"))

    def test_two_and_three_link_objects_follow_one_grasp_trajectory(self):
        for part_count, masses, pitches in (
            (2, (0.8, 0.4), (-10.0, 0.0)),
            (3, (0.8, 0.4, 0.25), (-10.0, 0.0)),
        ):
            result = contact.simulate(
                part_count,
                initial_opening_angle_deg=180.0,
                opening_angle_deg=180.0,
                grasp_offset_m=contact.PARENT_END_GRASP_OFFSET_M,
                pgc_controller_kp=3000.0,
                part_masses_kg=masses,
                wrist_pitch_sequence_deg=pitches,
                free_hinges=True,
            )
            self.assertEqual(len(result.holds), len(pitches))
            self.assertEqual(result.initial_contact_count, 0)
            self.assertGreaterEqual(result.final_contact_count, 2)
            self.assertTrue(
                any("table" in pair for pair in result.contact_pairs_after_close)
            )
            self.assertFalse(
                any(
                    "finger" in pair and "table" in pair
                    for pair in result.contact_pairs_after_close
                )
            )
            self.assertTrue(
                all(
                    any(f"finger{index}" in pair for pair in result.contact_pairs_after_close)
                    for index in (1, 2)
                )
            )
            self.assertTrue(result.passed_grasp_translation)
            self.assertLessEqual(
                result.grasp_relative_rotation_drift_deg,
                contact.MAX_FREE_GRASP_ROTATION_DRIFT_DEG,
            )
            self.assertTrue(result.passed_lift)
            self.assertTrue(all(hold.ft_stationary for hold in result.holds))
            self.assertTrue(all(hold.contact_free for hold in result.holds))


if __name__ == "__main__":
    unittest.main()
