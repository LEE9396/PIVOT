#!/usr/bin/env python3
"""Dynamic RB5 + movable PGC + contact-derived F/T custom-object test."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from pydrake.all import (
    AddContactMaterial,
    AddMultibodyPlantSceneGraph,
    AddRigidHydroelasticProperties,
    AngleAxis,
    BasicVector,
    Box,
    CameraInfo,
    ClippingRange,
    ColorRenderCamera,
    ConstantVectorSource,
    ContactModel,
    CoulombFriction,
    DiagramBuilder,
    DiscreteContactApproximation,
    DoorHinge,
    DoorHingeConfig,
    DepthRange,
    DepthRenderCamera,
    FixedOffsetFrame,
    InverseKinematics,
    InverseDynamics,
    JacobianWrtVariable,
    JointIndex,
    LeafSystem,
    MatrixGain,
    MakeRenderEngineVtk,
    Mesh,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Parser,
    PiecewisePolynomial,
    ProximityProperties,
    RevoluteJoint,
    RenderCameraCore,
    RenderEngineVtkParams,
    RigidTransform,
    RgbdSensor,
    RotationMatrix,
    Simulator,
    Solve,
    SpatialVelocity,
    SpatialInertia,
    StackedTrajectory,
    StartMeshcat,
    TrajectorySource,
    UnitInertia,
)


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

import run_custom_cad_configuration_information_experiment as study  # noqa: E402
import visualize_drake_custom_cad_configuration_information as cad_visual  # noqa: E402
import visualize_drake_full_lab_configuration_information as lab  # noqa: E402
import visualize_drake_rb5_hammer_payload as rb5  # noqa: E402


DEFAULT_OUTPUT = (
    ROOT
    / "progress/artifacts/2026-07-28/"
    "drake_rb5_contact_ft_custom_object.json"
)
TIME_STEP_S = 0.001
SIMULATION_END_S = 15.0
APPROACH_START_S = 0.5
APPROACH_END_S = 3.0
GRIPPER_CLOSE_START_S = 3.0
GRIPPER_CLOSE_END_S = 4.5
LIFT_START_S = 5.0
LIFT_END_S = 9.0
ORIENT_START_S = 10.5
ORIENT_END_S = 16.5
ARM_CONTROLLER_KP = 2500.0
ARM_CONTROLLER_KD = 140.0
ARM_EFFORT_LIMIT_NM = 160.0
OBJECT_LIFT_M = 0.45
OBJECT_RETRACT_M = 0.30
PREGRASP_DISTANCE_M = 0.08
OBJECT_TABLE_CLEARANCE_M = 0.0
FOUNDATIONPOSE_INITIAL_HOLD_S = 5.0
WRIST_ROTATION_DURATION_S = 1.0
ADAPTIVE_WRIST_ROTATION_DURATION_S = 3.0
ADAPTIVE_WRIST_STEP_DEG = (-30.0, -15.0, 0.0, 15.0, 30.0)
ADAPTIVE_WRIST_LIMIT_DEG = 90.0
ADAPTIVE_COLLISION_CLEARANCE_M = 0.010
INITIAL_FINGER_POSITION_M = 0.00565
HELD_OBJECT_CENTER_Y_M = 0.12
HELD_OBJECT_CENTER_X_M = 0.0
PARENT_END_GRASP_OFFSET_M = 0.020
GRASP_OBJECT_CENTER_ABOVE_TCP_M = 0.040
PGC_FINAL_POSITION_M = 0.025
PGC_MAX_PAYLOAD_KG = 3.0
MAX_GRASP_TRANSLATION_DRIFT_M = 0.006
MAX_GRASP_ROTATION_DRIFT_DEG = 5.0
MAX_FREE_GRASP_ROTATION_DRIFT_DEG = 20.0
FT_HOLD_WINDOW_S = 0.5
FT_HOLD_SAMPLE_COUNT = 500
AFT200_FORCE_RANGE_N = 200.0
AFT200_TORQUE_RANGE_NM = 15.0
AFT200_FORCE_LIMIT_N = 300.0
AFT200_TORQUE_LIMIT_NM = 25.0
AFT200_FORCE_RESOLUTION_N = 0.15
AFT200_TORQUE_RESOLUTION_NM = 0.015
AFT200_SAMPLE_RATE_HZ = 1000
# Serial-number-specific values stay neutral until the physical calibration.
AFT200_CALIBRATION_MATRIX = np.eye(6)
AFT200_BIAS = np.zeros(6)
AFT200_NOISE_STD = np.zeros(6)
FREE_HINGE_DAMPING = 0.08
DESK_LAMP_SETTLE_DURATION_S = 12.0
DESK_LAMP_LOWER_ZERO_OFFSET_DEG = 0.0
DESK_LAMP_TOOL_TILT_DEG = -45.0
DESK_LAMP_ARM_GRASP_FRACTION = 0.10
IK_INCLUDE_NOMINAL_SEED = True
OBJECT_FRICTION = CoulombFriction(1.1, 0.9)
GRASP_SURFACE_FRICTION = CoulombFriction(5.0, 4.0)
TABLE_FRICTION = CoulombFriction(0.85, 0.72)
DEFAULT_PART_MASSES_KG = (0.8, 0.4, 0.25, 0.2)
MINIMUM_PART_MASS_KG = 0.1
MINIMUM_OBJECT_MASS_KG = 1.0
PHANTOM_V3_ASSET_DIR = ROOT / "robot_learning/assets/phantom_v3"
HOUSEHOLD_PROFILES = {
    "jewelry_box": {
        "part_count": 2,
        "joint_limits_deg": (120.0,),
        "part_names": ("base", "lid"),
        "sizes_m": ((0.28, 0.20, 0.045), (0.28, 0.20, 0.015)),
        "colors": ((0.18, 0.32, 0.55, 1.0), (0.62, 0.28, 0.58, 1.0)),
        "default_masses_kg": (0.8, 0.4),
        "default_initial_angles_deg": (105.0,),
        "pgc_controller_kp": 6000.0,
    },
    "desk_lamp": {
        "part_count": 3,
        "joint_limits_deg": (120.0, 90.0),
        "part_names": ("arm", "head", "base"),
        "sizes_m": ((0.40, 0.035, 0.035), (0.283, 0.055, 0.025), (0.30, 0.20, 0.035)),
        "colors": ((0.05, 0.62, 0.58, 1.0), (0.95, 0.65, 0.08, 1.0), (0.08, 0.28, 0.68, 1.0)),
        "default_masses_kg": (0.4, 0.25, 0.8),
        "default_initial_angles_deg": (120.0, 0.0),
    },
    "phantom_v3": {
        "part_count": 3,
        "joint_limits_deg": (133.0, 149.0),
        "joint_angle_bounds_deg": ((-133.0, 133.0), (-147.0, 149.0)),
        "part_names": ("root", "elbow", "tip"),
        "sizes_m": (
            (0.172, 0.036, 0.018),
            (0.110, 0.030, 0.024),
            (0.085, 0.030, 0.028),
        ),
        "colors": (
            (0.05, 0.62, 0.35, 1.0),
            (0.95, 0.35, 0.05, 1.0),
            (0.42, 0.22, 0.85, 1.0),
        ),
        "default_masses_kg": (0.40, 0.33, 0.27),
        "default_initial_angles_deg": (0.0, 0.0),
        "pgc_controller_kp": 6000.0,
        "volumes_m3": (
            82.76094066049584e-6,
            46.501072490713806e-6,
            41.31595498814226e-6,
        ),
        "centroids_body_m": (
            (0.05982472, 0.00007177, 0.00077157),
            (0.05944421, 0.00280704, -0.00123153),
            (0.04154190, -0.00310785, 0.00005978),
        ),
        "mesh_files": (
            "v3_part0_root.obj",
            "v3_part1_elbow.obj",
            "v3_part2_tip.obj",
        ),
        "joint_origins_m": ((0.150, 0.0, 0.0), (0.110, 0.0, 0.0)),
        "joint_axes": ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    },
}


def _rigid_hydroelastic_properties(
    friction: CoulombFriction,
) -> ProximityProperties:
    properties = ProximityProperties()
    AddContactMaterial(
        properties,
        dissipation=1.0,
        point_stiffness=1.0e7,
        friction=friction,
    )
    AddRigidHydroelasticProperties(0.008, properties)
    return properties


@dataclass(frozen=True)
class Scenario:
    diagram: object
    plant: object
    scene_graph: object
    robot: rb5.ArmRecord
    pgc_joints: tuple[object, object]
    parent: object
    grasped_body: object
    object_bodies: tuple[object, ...]
    object_joints: tuple[RevoluteJoint, ...]
    sensor_mount_joint_index: int
    initial_finger_position_m: float
    q_lower: tuple[float, ...]
    q_lift: tuple[float, ...]
    grasp_offset_m: float
    initial_object_pose: RigidTransform
    simulation_end_s: float
    hold_end_times_s: tuple[float, ...]
    orient_dynamic_times_s: tuple[float, ...]
    wrist_dynamic_times_s: tuple[tuple[float, ...], ...]
    robot_source: object
    adaptive_wrist: bool
    rgbd_cameras: tuple[tuple[str, object, np.ndarray, RigidTransform], ...]
    object_profile: str | None


@dataclass(frozen=True)
class StaticHoldMeasurement:
    hold_end_s: float
    wrist_pitch_deg: float
    ft_reaction_force: tuple[float, float, float]
    ft_reaction_torque: tuple[float, float, float]
    known_tool_wrench: tuple[float, float, float, float, float, float]
    ft_sample_std: tuple[float, float, float, float, float, float]
    ft_mean_standard_error: tuple[float, float, float, float, float, float]
    ft_block_means: tuple[
        tuple[float, float, float, float, float, float], ...
    ]
    ft_inlier_count: int
    ft_stationary: bool
    contact_free: bool
    contact_pairs: tuple[str, ...]
    actual_opening_angles_deg: tuple[float, ...]
    opening_angle_span_deg: tuple[float, ...]
    maximum_abs_joint_velocity_deg_s: tuple[float, ...]
    maximum_abs_joint_acceleration_deg_s2: tuple[float, ...]
    joints_stationary: bool
    gravity_sensor_m_s2: tuple[float, float, float]
    part_com_sensor_m: tuple[tuple[float, float, float], ...]
    part_rotations_sensor: tuple[
        tuple[tuple[float, float, float], ...], ...
    ]
    joint_origins_sensor_m: tuple[tuple[float, float, float], ...]
    joint_axes_sensor: tuple[tuple[float, float, float], ...]
    sensor_world_pose: tuple[tuple[float, float, float, float], ...]
    grasp_relative_pose: tuple[tuple[float, float, float, float], ...]


@dataclass(frozen=True)
class DynamicLiftMeasurement:
    time_s: float
    ft_reaction_force: tuple[float, float, float]
    ft_reaction_torque: tuple[float, float, float]
    mass_regressor: tuple[tuple[float, ...], ...]
    known_tool_wrench: tuple[float, float, float, float, float, float]
    sensor_rotation_world_to_sensor: tuple[
        tuple[float, float, float], ...
    ]
    contact_free: bool
    contact_pairs: tuple[str, ...]
    actual_opening_angles_deg: tuple[float, ...]


@dataclass(frozen=True)
class ContactFtResult:
    part_count: int
    grasp_offset_m: float
    simulation_end_s: float
    parent_lift_m: float
    tcp_lift_m: float
    parent_tcp_lift_ratio: float
    finger_positions_after_close_m: tuple[float, float]
    initial_contact_count: int
    initial_contact_pairs: tuple[str, ...]
    contact_count_after_close: int
    contact_pairs_after_close: tuple[str, ...]
    grasp_relative_translation_drift_m: float
    grasp_relative_rotation_drift_deg: float
    maximum_internal_joint_drift_deg: float
    final_finger_positions_m: tuple[float, float]
    final_contact_count: int
    ft_reaction_force: tuple[float, float, float]
    ft_reaction_torque: tuple[float, float, float]
    ft_sample_std: tuple[float, float, float, float, float, float]
    ft_mean_standard_error: tuple[float, float, float, float, float, float]
    ft_block_means: tuple[tuple[float, float, float, float, float, float], ...]
    ft_raw_sample_count: int
    ft_inlier_count: int
    ft_stationary: bool
    actual_opening_angles_deg: tuple[float, ...]
    gravity_sensor_m_s2: tuple[float, float, float]
    part_com_sensor_m: tuple[tuple[float, float, float], ...]
    part_com_offsets_body_m: tuple[tuple[float, float, float], ...]
    joint_origin_sensor_m: tuple[float, float, float]
    joint_axis_sensor: tuple[float, float, float]
    dynamic_lift_measurements: tuple[DynamicLiftMeasurement, ...]
    holds: tuple[StaticHoldMeasurement, ...]
    adaptive_actions: tuple[dict[str, object], ...]
    passed_lift: bool
    passed_grasp_translation: bool
    passed_grasp_rotation: bool
    passed_joint_stability: bool

    @property
    def passed(self) -> bool:
        return all(
            (
                self.passed_lift,
                self.passed_grasp_translation,
                self.passed_grasp_rotation,
                self.passed_joint_stability,
            )
        )


def _dummy_assembly() -> rb5.HammerAssembly:
    mass = 0.001
    inertia = SpatialInertia(
        mass,
        np.zeros(3),
        UnitInertia.SolidSphere(0.001),
    )
    central = (
        mass * UnitInertia.SolidSphere(0.001)
    ).CopyToFullMatrix3()
    return rb5.HammerAssembly(
        spatial_inertia=inertia,
        mass_kg=mass,
        com_h_m=(0.0, 0.0, 0.0),
        central_inertia_h_kg_m2=tuple(
            tuple(float(value) for value in row) for row in central
        ),
        part_masses_kg=(mass, 0.0, 0.0, 0.0),
        effective_densities_kg_m3=(1.0, 0.0, 0.0, 0.0),
    )


def _ignore(*_args) -> None:
    return None


def _part_mass(
    index: int,
    part_masses_kg: tuple[float, ...] | None = None,
) -> float:
    if part_masses_kg is not None:
        return float(part_masses_kg[index])
    return DEFAULT_PART_MASSES_KG[index]


def _box_inertia(mass: float, length: float, com) -> SpatialInertia:
    return SpatialInertia.MakeFromCentralInertia(
        mass,
        np.asarray(com),
        mass
        * UnitInertia.SolidBox(
            length,
            study.BOX_DEPTH_M,
            study.BOX_HEIGHT_M,
        ),
    )


def _sized_box_inertia(mass: float, size, com) -> SpatialInertia:
    return SpatialInertia.MakeFromCentralInertia(
        mass,
        np.asarray(com),
        mass * UnitInertia.SolidBox(*size),
    )


def _add_upright_desk_lamp(
    plant,
    *,
    include_visuals: bool,
    part_masses_kg: tuple[float, ...] | None,
    part_com_offsets_body_m: tuple[
        tuple[float, float, float], ...
    ] | None = None,
) -> tuple[object, tuple[object, ...], tuple[RevoluteJoint, ...]]:
    profile = HOUSEHOLD_PROFILES["desk_lamp"]
    masses = part_masses_kg or profile["default_masses_kg"]
    arm_size, head_size, base_size = profile["sizes_m"]
    hinge_gap = 0.003
    pedestal_height = 0.07
    hinge_x = -base_size[0] / 2.0 + arm_size[2] / 2.0
    arm_link_rotation = RotationMatrix()
    arm_com = arm_link_rotation @ np.array(
        [arm_size[0] / 2.0, 0.0, 0.0]
    )
    head_com = np.array([-head_size[0] / 2.0, 0.0, 0.0])
    inertia_coms = part_com_offsets_body_m or (
        tuple(arm_com),
        tuple(head_com),
        (0.0, 0.0, 0.0),
    )
    base = plant.AddRigidBody(
        "desk_lamp_base",
        plant.AddModelInstance("desk_lamp_base_model"),
        _sized_box_inertia(masses[2], base_size, inertia_coms[2]),
    )
    arm = plant.AddRigidBody(
        "desk_lamp_arm",
        plant.AddModelInstance("desk_lamp_arm_model"),
        _sized_box_inertia(masses[0], arm_size, inertia_coms[0]),
    )
    head = plant.AddRigidBody(
        "desk_lamp_head",
        plant.AddModelInstance("desk_lamp_head_model"),
        _sized_box_inertia(masses[1], head_size, inertia_coms[1]),
    )
    lower_parent = plant.AddFrame(
        FixedOffsetFrame(
            "desk_lamp_lower_hinge_parent",
            base.body_frame(),
            RigidTransform(
                [
                    hinge_x,
                    0.0,
                    base_size[2] / 2.0
                    + pedestal_height
                    + arm_size[2] / 2.0
                    + hinge_gap,
                ]
            ),
        )
    )
    lower = plant.AddJoint(
        RevoluteJoint(
            "desk_lamp_lower_hinge",
            lower_parent,
            arm.body_frame(),
            [0.0, 1.0, 0.0],
            damping=FREE_HINGE_DAMPING,
        )
    )
    lower.set_position_limits(
        np.array(
            [
                -math.radians(
                    120.0 + DESK_LAMP_LOWER_ZERO_OFFSET_DEG
                )
            ]
        ),
        np.array([-math.radians(DESK_LAMP_LOWER_ZERO_OFFSET_DEG)]),
    )
    upper_parent = plant.AddFrame(
        FixedOffsetFrame(
            "desk_lamp_upper_hinge_parent",
            arm.body_frame(),
            RigidTransform(
                arm_link_rotation,
                arm_link_rotation
                @ np.array(
                    [
                        arm_size[0],
                        0.0,
                        -(arm_size[2] / 2.0
                        + head_size[2] / 2.0
                        + hinge_gap),
                    ]
                ),
            ),
        )
    )
    upper = plant.AddJoint(
        RevoluteJoint(
            "desk_lamp_upper_hinge",
            upper_parent,
            head.body_frame(),
            [0.0, 1.0, 0.0],
            damping=FREE_HINGE_DAMPING,
        )
    )
    upper.set_position_limits(
        np.array([-math.radians(90.0)]),
        np.array([0.0]),
    )
    for body, center, size, name, color in (
        (head, head_com, head_size, "head", profile["colors"][1]),
        (base, np.zeros(3), base_size, "base", profile["colors"][2]),
    ):
        if include_visuals:
            plant.RegisterVisualGeometry(
                body,
                RigidTransform(center),
                Box(*size),
                f"desk_lamp_{name}_visual",
                np.asarray(color),
            )
        plant.RegisterCollisionGeometry(
            body,
            RigidTransform(center),
            Box(*size),
            f"desk_lamp_{name}_collision",
            _rigid_hydroelastic_properties(
                GRASP_SURFACE_FRICTION if body is arm else OBJECT_FRICTION
            ),
        )
    if include_visuals:
        plant.RegisterVisualGeometry(
            arm,
            RigidTransform(arm_link_rotation, arm_com),
            Box(*arm_size),
            "desk_lamp_arm_visual",
            np.asarray(profile["colors"][0]),
        )
        for index, (center, size, color) in enumerate(
            (
                (
                    (0.12, 0.012, arm_size[2] / 2.0 + 0.002),
                    (0.070, 0.008, 0.004),
                    (0.95, 0.20, 0.12, 1.0),
                ),
                (
                    (0.29, -0.010, arm_size[2] / 2.0 + 0.002),
                    (0.025, 0.012, 0.004),
                    (0.95, 0.82, 0.12, 1.0),
                ),
            )
        ):
            plant.RegisterVisualGeometry(
                arm,
                RigidTransform(center),
                Box(*size),
                f"desk_lamp_arm_pose_marker_{index}",
                np.asarray(color),
            )
    plant.RegisterCollisionGeometry(
        arm,
        RigidTransform(arm_link_rotation, arm_com),
        Box(*arm_size),
        "desk_lamp_arm_collision",
        _rigid_hydroelastic_properties(GRASP_SURFACE_FRICTION),
    )
    pedestal_pose = RigidTransform(
        [hinge_x, 0.0, base_size[2] / 2.0 + pedestal_height / 2.0]
    )
    if include_visuals:
        plant.RegisterVisualGeometry(
            base,
            pedestal_pose,
            Box(arm_size[2], 0.05, pedestal_height),
            "desk_lamp_hinge_pedestal_visual",
            np.asarray(profile["colors"][2]),
        )
    plant.RegisterCollisionGeometry(
        base,
        pedestal_pose,
        Box(arm_size[2], 0.05, pedestal_height),
        "desk_lamp_hinge_pedestal_collision",
        _rigid_hydroelastic_properties(OBJECT_FRICTION),
    )
    if include_visuals:
        plant.RegisterVisualGeometry(
            head,
            RigidTransform(
                head_com + np.array([0.0, 0.0, -head_size[2] / 2.0 - 0.003])
            ),
            Box(0.25, 0.035, 0.004),
            "desk_lamp_light_panel",
            np.array([1.0, 0.92, 0.55, 1.0]),
        )
        plant.RegisterVisualGeometry(
            head,
            RigidTransform(
                (-0.075, 0.020, head_size[2] / 2.0 + 0.002)
            ),
            Box(0.055, 0.010, 0.004),
            "desk_lamp_head_pose_marker",
            np.array([0.80, 0.12, 0.72, 1.0]),
        )
        plant.RegisterVisualGeometry(
            base,
            RigidTransform(
                (0.075, 0.055, base_size[2] / 2.0 + 0.002)
            ),
            Box(0.085, 0.018, 0.004),
            "desk_lamp_base_pose_marker",
            np.array([0.12, 0.90, 0.82, 1.0]),
        )
    return base, (arm, head, base), (lower, upper)


def _add_jewelry_box(
    plant,
    *,
    include_visuals: bool,
    part_masses_kg: tuple[float, ...] | None,
    part_com_offsets_body_m: tuple[tuple[float, float, float], ...] | None,
) -> tuple[object, tuple[object, ...], tuple[RevoluteJoint, ...]]:
    profile = HOUSEHOLD_PROFILES["jewelry_box"]
    masses = part_masses_kg or profile["default_masses_kg"]
    base_size, lid_size = profile["sizes_m"]
    hinge_gap = 0.003
    lid_com = np.array([0.0, -base_size[1] / 2.0, hinge_gap + lid_size[2] / 2.0])
    inertia_coms = part_com_offsets_body_m or (
        (0.0, 0.0, 0.0),
        tuple(lid_com),
    )
    base = plant.AddRigidBody(
        "jewelry_box_base",
        plant.AddModelInstance("jewelry_box_base_model"),
        _sized_box_inertia(masses[0], base_size, inertia_coms[0]),
    )
    lid = plant.AddRigidBody(
        "jewelry_box_lid",
        plant.AddModelInstance("jewelry_box_lid_model"),
        _sized_box_inertia(masses[1], lid_size, inertia_coms[1]),
    )
    hinge_parent = plant.AddFrame(
        FixedOffsetFrame(
            "jewelry_box_hinge_parent",
            base.body_frame(),
            RigidTransform([0.0, base_size[1] / 2.0, base_size[2] / 2.0]),
        )
    )
    hinge = plant.AddJoint(
        RevoluteJoint(
            "jewelry_box_hinge",
            hinge_parent,
            lid.body_frame(),
            [1.0, 0.0, 0.0],
            damping=FREE_HINGE_DAMPING,
        )
    )
    hinge.set_position_limits(
        np.array([-math.radians(120.0)]),
        np.array([0.0]),
    )
    for body, center, size, name, color in (
        (base, np.zeros(3), base_size, "base", profile["colors"][0]),
        (lid, lid_com, lid_size, "lid", profile["colors"][1]),
    ):
        if include_visuals:
            plant.RegisterVisualGeometry(
                body,
                RigidTransform(center),
                Box(*size),
                f"jewelry_box_{name}_visual",
                np.asarray(color),
            )
        plant.RegisterCollisionGeometry(
            body,
            RigidTransform(center),
            Box(*size),
            f"jewelry_box_{name}_collision",
            _rigid_hydroelastic_properties(
                GRASP_SURFACE_FRICTION if body is base else OBJECT_FRICTION
            ),
        )
    if include_visuals:
        plant.RegisterVisualGeometry(
            base,
            RigidTransform([0.0, 0.0, base_size[2] / 2.0 + 0.004]),
            Box(0.245, 0.165, 0.006),
            "jewelry_box_insert",
            np.array([0.78, 0.80, 0.86, 1.0]),
        )
        for body, center, size, name, color in (
            (
                base,
                (0.070, -0.055, 0.0315),
                (0.050, 0.018, 0.003),
                "base",
                (0.95, 0.72, 0.12, 1.0),
            ),
            (
                lid,
                (-0.075, -0.055, 0.0195),
                (0.045, 0.014, 0.003),
                "lid",
                (0.10, 0.72, 0.82, 1.0),
            ),
        ):
            plant.RegisterVisualGeometry(
                body,
                RigidTransform(center),
                Box(*size),
                f"jewelry_box_{name}_pose_marker",
                np.asarray(color),
            )
    return base, (base, lid), (hinge,)


def _add_phantom_v3(
    plant,
    *,
    include_visuals: bool,
    part_masses_kg: tuple[float, ...] | None,
    part_com_offsets_body_m: tuple[
        tuple[float, float, float], ...
    ] | None,
) -> tuple[object, tuple[object, ...], tuple[RevoluteJoint, ...]]:
    profile = HOUSEHOLD_PROFILES["phantom_v3"]
    masses = part_masses_kg or profile["default_masses_kg"]
    centers = part_com_offsets_body_m or profile["centroids_body_m"]
    bodies = tuple(
        plant.AddRigidBody(
            f"phantom_v3_{name}",
            plant.AddModelInstance(f"phantom_v3_{name}_model"),
            _sized_box_inertia(mass, size, center),
        )
        for name, mass, size, center in zip(
            profile["part_names"],
            masses,
            profile["sizes_m"],
            centers,
            strict=True,
        )
    )
    for index, (body, mesh_file) in enumerate(
        zip(bodies, profile["mesh_files"], strict=True)
    ):
        mesh = Mesh(str(PHANTOM_V3_ASSET_DIR / mesh_file), 0.001)
        if include_visuals:
            plant.RegisterVisualGeometry(
                body,
                RigidTransform(),
                mesh,
                f"phantom_v3_{profile['part_names'][index]}_visual",
                np.asarray(profile["colors"][index]),
            )
        plant.RegisterCollisionGeometry(
            body,
            RigidTransform(),
            mesh,
            f"phantom_v3_{profile['part_names'][index]}_collision",
            _rigid_hydroelastic_properties(
                GRASP_SURFACE_FRICTION if index == 0 else OBJECT_FRICTION
            ),
        )
    joints = []
    for index, (origin, axis, bounds) in enumerate(
        zip(
            profile["joint_origins_m"],
            profile["joint_axes"],
            profile["joint_angle_bounds_deg"],
            strict=True,
        ),
        start=1,
    ):
        parent_frame = plant.AddFrame(
            FixedOffsetFrame(
                f"phantom_v3_joint{index}_parent",
                bodies[index - 1].body_frame(),
                RigidTransform(origin),
            )
        )
        joint = plant.AddJoint(
            RevoluteJoint(
                f"phantom_v3_joint{index}",
                parent_frame,
                bodies[index].body_frame(),
                axis,
                damping=FREE_HINGE_DAMPING,
            )
        )
        joint.set_position_limits(
            np.radians([bounds[0]]),
            np.radians([bounds[1]]),
        )
        joints.append(joint)
    return bodies[0], bodies, tuple(joints)


def _add_household_object(
    plant,
    object_profile: str,
    *,
    include_visuals: bool,
    part_masses_kg: tuple[float, ...] | None,
    part_com_offsets_body_m: tuple[
        tuple[float, float, float], ...
    ] | None = None,
) -> tuple[object, tuple[object, ...], tuple[RevoluteJoint, ...]]:
    if object_profile == "desk_lamp":
        return _add_upright_desk_lamp(
            plant,
            include_visuals=include_visuals,
            part_masses_kg=part_masses_kg,
            part_com_offsets_body_m=part_com_offsets_body_m,
        )
    if object_profile == "jewelry_box":
        return _add_jewelry_box(
            plant,
            include_visuals=include_visuals,
            part_masses_kg=part_masses_kg,
            part_com_offsets_body_m=part_com_offsets_body_m,
        )
    if object_profile == "phantom_v3":
        return _add_phantom_v3(
            plant,
            include_visuals=include_visuals,
            part_masses_kg=part_masses_kg,
            part_com_offsets_body_m=part_com_offsets_body_m,
        )
    profile = HOUSEHOLD_PROFILES[object_profile]
    bodies = []
    joints = []
    parent_frame = None
    previous_size = None
    for index, (name, size, color) in enumerate(
        zip(
            profile["part_names"],
            profile["sizes_m"],
            profile["colors"],
            strict=True,
        )
    ):
        model = plant.AddModelInstance(f"{object_profile}_{name}_model")
        center = (
            np.zeros(3)
            if index == 0
            else np.array(
                [
                    study.HINGE_GAP_M / 2.0 + size[0] / 2.0,
                    0.0,
                    -size[2] / 2.0,
                ]
            )
        )
        body = plant.AddRigidBody(
            f"{object_profile}_{name}",
            model,
            _sized_box_inertia(
                _part_mass(index, part_masses_kg),
                size,
                center,
            ),
        )
        bodies.append(body)
        if include_visuals:
            plant.RegisterVisualGeometry(
                body,
                RigidTransform(center),
                Box(*size),
                f"{object_profile}_{name}_visual",
                np.asarray(color),
            )
            marker_size = (0.06, min(size[1] + 0.002, 0.04), 0.002)
            plant.RegisterVisualGeometry(
                body,
                RigidTransform(center + np.array([0.0, 0.0, size[2] / 2.0 + 0.001])),
                Box(*marker_size),
                f"{object_profile}_{name}_marker",
                np.array(([0.95, 0.72, 0.12, 1.0] if index % 2 == 0 else [0.10, 0.72, 0.82, 1.0])),
            )
        plant.RegisterCollisionGeometry(
            body,
            RigidTransform(center),
            Box(*size),
            f"{object_profile}_{name}_collision",
            _rigid_hydroelastic_properties(OBJECT_FRICTION),
        )
        if index:
            joint = plant.AddJoint(
                RevoluteJoint(
                    f"{object_profile}_hinge_{index}",
                    parent_frame,
                    body.body_frame(),
                    [0.0, -1.0, 0.0],
                    damping=FREE_HINGE_DAMPING,
                )
            )
            limit_deg = profile["joint_limits_deg"][index - 1]
            joint.set_position_limits(
                np.array([-math.pi]),
                np.array([-math.radians(180.0 - limit_deg)]),
            )
            joints.append(joint)
        if index < profile["part_count"] - 1:
            offset = (
                size[0] + study.HINGE_GAP_M
                if index
                else size[0] / 2.0 + study.HINGE_GAP_M / 2.0
            )
            parent_frame = plant.AddFrame(
                FixedOffsetFrame(
                    f"{object_profile}_hinge_{index + 1}_parent",
                    body.body_frame(),
                    RigidTransform(
                        [
                            offset,
                            0.0,
                            size[2] / 2.0 + 0.010
                            if index == 0
                            else 0.0,
                        ]
                    ),
                )
            )
    return bodies[0], tuple(bodies), tuple(joints)


def _internal_joint_angle_rad(
    object_profile: str | None,
    joint_name: str,
    opening_angle_deg: float,
) -> float:
    angle = math.radians(opening_angle_deg)
    if object_profile == "desk_lamp":
        return (
            -math.radians(
                opening_angle_deg + DESK_LAMP_LOWER_ZERO_OFFSET_DEG
            )
            if "lower" in joint_name
            else -angle
        )
    if object_profile == "jewelry_box":
        return -angle
    if object_profile == "phantom_v3":
        return angle
    return -math.radians(180.0 - opening_angle_deg)


def _payload_object_transform(grasp_offset_m: float) -> RigidTransform:
    return RigidTransform(
        RotationMatrix.MakeYRotation(-math.pi / 2.0),
        [0.0, 0.0, grasp_offset_m - GRASP_OBJECT_CENTER_ABOVE_TCP_M],
    )


def _add_free_custom_object(
    plant,
    part_count: int,
    *,
    include_visuals: bool,
    simple_render_visuals: bool = False,
    opening_angle_deg: float = 180.0,
    part_masses_kg: tuple[float, ...] | None = None,
    part_com_offsets_body_m: tuple[tuple[float, float, float], ...] | None = None,
    free_hinges: bool = False,
    object_profile: str | None = None,
) -> tuple[object, tuple[object, ...], tuple[RevoluteJoint, ...]]:
    if object_profile is not None:
        return _add_household_object(
            plant,
            object_profile,
            include_visuals=include_visuals,
            part_masses_kg=part_masses_kg,
            part_com_offsets_body_m=part_com_offsets_body_m,
        )
    parent_model = plant.AddModelInstance("contact_custom_parent_model")
    parent = plant.AddRigidBody(
        "contact_custom_parent",
        parent_model,
        _box_inertia(
            _part_mass(0, part_masses_kg),
            study.PARENT_LENGTH_M,
            np.zeros(3),
        ),
    )
    if include_visuals:
        if simple_render_visuals:
            plant.RegisterVisualGeometry(
                parent,
                RigidTransform(
                    [
                        (
                            study.GRASP_NECK_LENGTH_M
                            + study.GRASP_FLANGE_LENGTH_M
                        )
                        / 2.0,
                        0.0,
                        0.0,
                    ]
                ),
                Box(
                    study.PARENT_LENGTH_M
                    - study.GRASP_NECK_LENGTH_M
                    - study.GRASP_FLANGE_LENGTH_M,
                    study.BOX_DEPTH_M,
                    study.BOX_HEIGHT_M,
                ),
                "foundationpose_parent_main",
                np.array([0.08, 0.24, 0.62, 1.0]),
            )
            plant.RegisterVisualGeometry(
                parent,
                RigidTransform(
                    [
                        -study.PARENT_LENGTH_M / 2.0
                        + study.GRASP_FLANGE_LENGTH_M
                        + study.GRASP_NECK_LENGTH_M / 2.0,
                        0.0,
                        0.0,
                    ]
                ),
                Box(
                    study.GRASP_NECK_LENGTH_M,
                    study.GRASP_NECK_DEPTH_M,
                    study.BOX_HEIGHT_M,
                ),
                "foundationpose_parent_grasp_neck",
                np.array([0.08, 0.24, 0.62, 1.0]),
            )
            plant.RegisterVisualGeometry(
                parent,
                RigidTransform(
                    [
                        -study.PARENT_LENGTH_M / 2.0
                        + study.GRASP_FLANGE_LENGTH_M / 2.0,
                        0.0,
                        0.0,
                    ]
                ),
                Box(
                    study.GRASP_FLANGE_LENGTH_M,
                    study.BOX_DEPTH_M,
                    study.BOX_HEIGHT_M,
                ),
                "foundationpose_parent_grasp_flange",
                np.array([0.08, 0.24, 0.62, 1.0]),
            )
            plant.RegisterVisualGeometry(
                parent,
                RigidTransform([0.030, 0.012, 0.0101]),
                Box(0.025, 0.008, 0.0002),
                "foundationpose_parent_marker",
                np.array([0.92, 0.20, 0.12, 1.0]),
            )
            plant.RegisterVisualGeometry(
                parent,
                RigidTransform([0.020, 0.008, 0.0101]),
                Box(0.008, 0.012, 0.0002),
                "foundationpose_parent_marker_short",
                np.array([0.92, 0.20, 0.12, 1.0]),
            )
        else:
            cad_visual._register_parent_visual(plant, parent, None)
    plant.RegisterCollisionGeometry(
        parent,
        RigidTransform(
            [
                (
                    study.GRASP_NECK_LENGTH_M
                    + study.GRASP_FLANGE_LENGTH_M
                )
                / 2.0,
                0.0,
                0.0,
            ]
        ),
        Box(
            study.PARENT_LENGTH_M
            - study.GRASP_NECK_LENGTH_M
            - study.GRASP_FLANGE_LENGTH_M,
            study.BOX_DEPTH_M,
            study.BOX_HEIGHT_M,
        ),
        "contact_parent_main",
        _rigid_hydroelastic_properties(OBJECT_FRICTION),
    )
    plant.RegisterCollisionGeometry(
        parent,
        RigidTransform(
            [
                -study.PARENT_LENGTH_M / 2.0
                + study.GRASP_FLANGE_LENGTH_M
                + study.GRASP_NECK_LENGTH_M / 2.0,
                0.0,
                0.0,
            ]
        ),
        Box(
            study.GRASP_NECK_LENGTH_M,
            study.GRASP_NECK_DEPTH_M,
            study.BOX_HEIGHT_M,
        ),
        "contact_parent_grasp_neck",
        _rigid_hydroelastic_properties(OBJECT_FRICTION),
    )
    plant.RegisterCollisionGeometry(
        parent,
        RigidTransform(
            [
                -study.PARENT_LENGTH_M / 2.0
                + study.GRASP_FLANGE_LENGTH_M / 2.0,
                0.0,
                0.0,
            ]
        ),
        Box(
            study.GRASP_FLANGE_LENGTH_M,
            study.BOX_DEPTH_M,
            study.BOX_HEIGHT_M,
        ),
        "contact_parent_grasp_flange",
        _rigid_hydroelastic_properties(OBJECT_FRICTION),
    )
    bodies = [parent]
    joints = []
    parent_frame = plant.AddFrame(
        FixedOffsetFrame(
            "contact_hinge_1_parent",
            parent.body_frame(),
            RigidTransform(
                [
                    study.PARENT_LENGTH_M / 2.0
                    + study.HINGE_GAP_M / 2.0,
                    0.0,
                    study.BOX_HEIGHT_M / 2.0,
                ]
            ),
        )
    )
    for child_index in range(1, part_count):
        child_colors = (
            np.array([0.08, 0.58, 0.46, 1.0]),
            np.array([0.86, 0.44, 0.08, 1.0]),
            np.array([0.55, 0.22, 0.75, 1.0]),
        )
        marker_colors = (
            np.array([0.92, 0.80, 0.10, 1.0]),
            np.array([0.08, 0.75, 0.86, 1.0]),
            np.array([0.92, 0.25, 0.70, 1.0]),
        )
        marker_side = -1.0 if child_index % 2 else 1.0
        center = np.array(
            [
                study.HINGE_GAP_M / 2.0
                + study.CHILD_LENGTH_M / 2.0,
                0.0,
                -study.BOX_HEIGHT_M / 2.0,
            ]
        )
        model = plant.AddModelInstance(
            f"contact_custom_child_{child_index}_model"
        )
        child = plant.AddRigidBody(
            f"contact_custom_child_{child_index}",
            model,
            _box_inertia(
                _part_mass(child_index, part_masses_kg),
                study.CHILD_LENGTH_M,
                center,
            ),
        )
        bodies.append(child)
        if include_visuals:
            if simple_render_visuals:
                plant.RegisterVisualGeometry(
                    child,
                    RigidTransform(center),
                    Box(
                        study.CHILD_LENGTH_M,
                        study.BOX_DEPTH_M,
                        study.BOX_HEIGHT_M,
                    ),
                    f"foundationpose_child_{child_index}",
                    child_colors[child_index - 1],
                )
                plant.RegisterVisualGeometry(
                    child,
                    RigidTransform(
                        center
                        + np.array(
                            [0.015 * marker_side, 0.012 * marker_side, 0.0101]
                        )
                    ),
                    Box(0.025, 0.008, 0.0002),
                    f"foundationpose_child_{child_index}_marker",
                    marker_colors[child_index - 1],
                )
                plant.RegisterVisualGeometry(
                    child,
                    RigidTransform(
                        center
                        + np.array(
                            [0.005 * marker_side, 0.008 * marker_side, 0.0101]
                        )
                    ),
                    Box(0.008, 0.012, 0.0002),
                    f"foundationpose_child_{child_index}_marker_short",
                    marker_colors[child_index - 1],
                )
            else:
                cad_visual._register_child_visual(
                    plant,
                    child,
                    child_index,
                )
        plant.RegisterCollisionGeometry(
            child,
            RigidTransform(center),
            Box(
                study.CHILD_LENGTH_M,
                study.BOX_DEPTH_M,
                study.BOX_HEIGHT_M,
            ),
            f"contact_child_{child_index}_box",
            _rigid_hydroelastic_properties(OBJECT_FRICTION),
        )
        joint = plant.AddJoint(
            RevoluteJoint(
                f"contact_hinge_{child_index}",
                parent_frame,
                child.body_frame(),
                [0.0, 1.0, 0.0],
                damping=FREE_HINGE_DAMPING if free_hinges else 0.01,
            )
        )
        joint.set_position_limits(
            np.array([-math.pi]),
            np.array([-math.radians(180.0 - opening_angle_deg)]),
        )
        joints.append(joint)
        if not free_hinges:
            hinge_config = DoorHingeConfig()
            hinge_config.spring_zero_angle_rad = -math.radians(
                180.0 - opening_angle_deg
            )
            hinge_config.spring_constant = 30.0
            hinge_config.dynamic_friction_torque = 8.8
            hinge_config.static_friction_torque = 9.6
            hinge_config.viscous_friction = 0.01
            hinge_config.catch_width = 0.0
            hinge_config.catch_torque = 0.0
            plant.AddForceElement(DoorHinge(joint, hinge_config))
        if child_index < part_count - 1:
            parent_frame = plant.AddFrame(
                FixedOffsetFrame(
                    f"contact_hinge_{child_index + 1}_parent",
                    child.body_frame(),
                    RigidTransform(
                        [
                            study.HINGE_GAP_M + study.CHILD_LENGTH_M,
                            0.0,
                            0.0,
                        ]
                    ),
                )
            )
    return parent, tuple(bodies), tuple(joints)


def _set_robot_positions(context, robot, values) -> None:
    for joint, value in zip(robot.joints, values, strict=True):
        joint.set_angle(context, float(value))
        joint.set_angular_rate(context, 0.0)


def _integrate_translation_path(
    plant,
    context,
    robot,
    initial_q: np.ndarray,
    delta_xyz_m: np.ndarray,
    *,
    samples: int = 80,
) -> np.ndarray:
    current = np.asarray(initial_q, dtype=float).copy()
    values = [current.copy()]
    for _ in range(samples):
        _set_robot_positions(context, robot, current)
        jacobian = plant.CalcJacobianSpatialVelocity(
            context,
            JacobianWrtVariable.kV,
            robot.payload.body_frame(),
            np.zeros(3),
            plant.world_frame(),
            plant.world_frame(),
        )
        columns = np.column_stack(
            [jacobian[:, joint.velocity_start()] for joint in robot.joints]
        )
        desired = np.concatenate(
            (np.zeros(3), np.asarray(delta_xyz_m, dtype=float) / samples)
        )
        delta_q, *_ = np.linalg.lstsq(columns, desired, rcond=None)
        current += np.clip(delta_q, -0.012, 0.012)
        values.append(current.copy())
    return np.asarray(values)


def _integrate_pose_path(
    plant,
    context,
    robot,
    initial_q: np.ndarray,
    target_pose: RigidTransform,
    *,
    samples: int = 240,
) -> np.ndarray:
    """Solve the endpoint exactly, then let the cubic command interpolate."""
    target_q = _solve_robot_pose(
        plant,
        context,
        robot,
        np.asarray(initial_q, dtype=float),
        target_pose,
    )
    values = np.linspace(initial_q, target_q, samples + 1)
    _set_robot_positions(context, robot, target_q)
    final_pose = plant.EvalBodyPoseInWorld(context, robot.payload)
    position_error = np.linalg.norm(
        final_pose.translation() - target_pose.translation()
    )
    rotation_error = AngleAxis(
        (target_pose.rotation() @ final_pose.rotation().inverse()).matrix()
    ).angle()
    if position_error > 0.005 or rotation_error > math.radians(2.0):
        raise RuntimeError(
            "TCP pose path is outside the reachable workspace: "
            f"position_error={position_error:.4f} m, "
            f"rotation_error={math.degrees(rotation_error):.2f} deg"
        )
    return values


def _solve_robot_pose(
    plant,
    context,
    robot,
    initial_q: np.ndarray,
    target_pose: RigidTransform,
) -> np.ndarray:
    """Solve the six-axis arm pose while keeping non-robot states fixed."""
    initial_q = np.asarray(initial_q, dtype=float)
    ik = InverseKinematics(plant, context)
    q = ik.q()
    target = target_pose.translation()
    tolerance = 1e-3
    ik.AddPositionConstraint(
        robot.payload.body_frame(),
        np.zeros(3),
        plant.world_frame(),
        target - tolerance,
        target + tolerance,
    )
    ik.AddOrientationConstraint(
        plant.world_frame(),
        target_pose.rotation(),
        robot.payload.body_frame(),
        RotationMatrix(),
        1e-2,
    )
    robot_indices = np.array(
        [joint.position_start() for joint in robot.joints],
        dtype=int,
    )
    ik.prog().AddQuadraticErrorCost(
        np.eye(len(robot_indices)),
        initial_q,
        q[robot_indices],
    )
    initial_all = plant.GetPositions(context)
    initial_all[robot_indices] = initial_q
    non_robot_indices = np.setdiff1d(
        np.arange(plant.num_positions()),
        robot_indices,
    )
    ik.prog().AddBoundingBoxConstraint(
        initial_all[non_robot_indices],
        initial_all[non_robot_indices],
        q[non_robot_indices],
    )
    lower = plant.GetPositionLowerLimits()
    upper = plant.GetPositionUpperLimits()
    finite_lower = np.where(np.isfinite(lower), lower, -math.pi)
    finite_upper = np.where(np.isfinite(upper), upper, math.pi)
    seeds = [initial_all.copy()] if IK_INCLUDE_NOMINAL_SEED else []
    rng = np.random.default_rng(8421)
    scored_seeds = []
    for _ in range(6000):
        seed = initial_all.copy()
        seed[robot_indices] = rng.uniform(
            finite_lower[robot_indices],
            finite_upper[robot_indices],
        )
        _set_robot_positions(context, robot, seed[robot_indices])
        pose = plant.EvalBodyPoseInWorld(context, robot.payload)
        rotation = pose.rotation().matrix()
        score = (
            8.0 * np.linalg.norm(pose.translation() - target)
            + 20.0 * (1.0 + rotation[2, 2])
            + 8.0 * abs(rotation[2, 1])
        )
        scored_seeds.append((score, seed))
    seeds.extend(
        seed for _, seed in sorted(scored_seeds, key=lambda item: item[0])[:50]
    )
    result = None
    for seed in seeds:
        ik.prog().SetInitialGuess(q, seed)
        candidate = Solve(ik.prog())
        if candidate.is_success():
            result = candidate
            break
    if result is None:
        best_seed = min(scored_seeds, key=lambda item: item[0])[1]
        _set_robot_positions(context, robot, best_seed[robot_indices])
        best_pose = plant.EvalBodyPoseInWorld(context, robot.payload)
        raise RuntimeError(
            "Could not solve horizontal PGC grasp pose; "
            f"best_sample_position={best_pose.translation().tolist()}, "
            "best_sample_tool_z="
            f"{best_pose.rotation().matrix()[:, 2].tolist()}"
        )
    return result.GetSolution(q)[robot_indices]


def _desired_trajectory(
    breaks: np.ndarray,
    positions: np.ndarray,
) -> StackedTrajectory:
    position = PiecewisePolynomial.CubicShapePreserving(
        breaks,
        positions.T,
        zero_end_point_derivatives=True,
    )
    desired = StackedTrajectory()
    desired.Append(position)
    desired.Append(position.MakeDerivative())
    return desired


def _desired_source(
    breaks: np.ndarray,
    positions: np.ndarray,
) -> TrajectorySource:
    return TrajectorySource(_desired_trajectory(breaks, positions))


class _MutableDesiredSource(LeafSystem):
    def __init__(self, breaks: np.ndarray, positions: np.ndarray):
        super().__init__()
        self._trajectory = _desired_trajectory(breaks, positions)
        self.DeclareVectorOutputPort(
            "desired_state",
            BasicVector(2 * positions.shape[1]),
            self._calc_output,
        )

    def set_trajectory(
        self,
        breaks: np.ndarray,
        positions: np.ndarray,
    ) -> None:
        self._trajectory = _desired_trajectory(breaks, positions)

    def _calc_output(self, context, output) -> None:
        output.SetFromVector(
            self._trajectory.value(context.get_time()).reshape(-1)
        )


def _camera_pose(position: np.ndarray, target: np.ndarray) -> RigidTransform:
    forward = target - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    return RigidTransform(
        RotationMatrix(np.column_stack((right, down, forward))),
        position,
    )


def _add_rgbd_cameras(builder, scene_graph, camera_configs):
    renderer = "foundationpose_renderer"
    scene_graph.AddRenderer(
        renderer,
        MakeRenderEngineVtk(RenderEngineVtkParams()),
    )
    records = []
    for config in camera_configs:
        width, height = config["resolution"]
        calibration = config["depth_intrinsics"]
        focal_x = float(calibration["fx"])
        focal_y = float(calibration["fy"])
        center_x = float(calibration["cx"])
        center_y = float(calibration["cy"])
        intrinsics = CameraInfo(
            width,
            height,
            focal_x,
            focal_y,
            center_x,
            center_y,
        )
        depth_range = config["depth_range_m"]
        core = RenderCameraCore(
            renderer,
            intrinsics,
            ClippingRange(*depth_range),
            RigidTransform(),
        )
        x_world_camera = _camera_pose(
            np.asarray(config["position_xyz_m"], dtype=float),
            np.asarray(config["look_at_xyz_m"], dtype=float),
        )
        sensor = builder.AddSystem(
            RgbdSensor(
                scene_graph.world_frame_id(),
                x_world_camera,
                ColorRenderCamera(core, False),
                DepthRenderCamera(core, DepthRange(*depth_range)),
            )
        )
        builder.Connect(
            scene_graph.get_query_output_port(),
            sensor.query_object_input_port(),
        )
        records.append(
            (
                config["id"],
                sensor,
                np.array(
                    [
                        [focal_x, 0.0, center_x],
                        [0.0, focal_y, center_y],
                        [0.0, 0.0, 1.0],
                    ]
                ),
                x_world_camera,
            )
        )
    return tuple(records)


def build_scenario(
    part_count: int,
    *,
    meshcat=None,
    opening_angle_deg: float = 180.0,
    initial_opening_angle_deg: float = 180.0,
    initial_opening_angles_deg: tuple[float, ...] | None = None,
    initial_wrist_pitch_deg: float = 0.0,
    wrist_pitch_deg: float = 0.0,
    simulation_end_s: float = SIMULATION_END_S,
    pgc_controller_kp: float = 3000.0,
    part_masses_kg: tuple[float, ...] | None = None,
    part_com_offsets_body_m: tuple[tuple[float, float, float], ...] | None = None,
    wrist_pitch_sequence_deg: tuple[float, ...] | None = None,
    wrist_roll_sequence_deg: tuple[float, ...] | None = None,
    render_rgbd: bool = False,
    free_hinges: bool = True,
    grasp_offset_m: float = PARENT_END_GRASP_OFFSET_M,
    object_profile: str | None = None,
    wrist_joints_only: bool = False,
    adaptive_wrist: bool = False,
) -> Scenario:
    with lab.LAB_CONFIG.open(encoding="utf-8") as stream:
        lab_config = json.load(stream)
    rb5_description, urdf_path, pgc_urdf_path, drake_assets = (
        rb5.validate_htd_source(rb5.DEFAULT_HTD_ROOT)
    )
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(
        builder,
        time_step=TIME_STEP_S,
    )
    plant.set_discrete_contact_approximation(
        DiscreteContactApproximation.kSimilar
    )
    plant.set_contact_model(ContactModel.kHydroelasticWithFallback)
    plant.set_penetration_allowance(0.0002)
    plant.mutable_gravity_field().set_gravity_vector([0.0, 0.0, -9.81])
    parser = Parser(plant)
    parser.package_map().Add("rbpodo_description", str(rb5_description))
    parser.package_map().Add("htd", str(rb5.DEFAULT_HTD_ROOT.resolve()))
    variant = rb5.VariantSpec(
        name=f"contact_custom_{part_count}_link",
        left_insert_mass_kg=rb5.LIGHT_INSERT_MASS_KG,
        right_insert_mass_kg=rb5.LIGHT_INSERT_MASS_KG,
        base_y_m=float(lab_config["robot"]["base_xyz_m"][1]),
    )
    robot = rb5.add_rb5_with_payload(
        plant,
        parser,
        urdf_path,
        pgc_urdf_path,
        drake_assets,
        variant,
        include_visuals=meshcat is not None or render_rgbd,
        assembly_override=_dummy_assembly(),
        controller_kp=ARM_CONTROLLER_KP,
        controller_kd=ARM_CONTROLLER_KD,
        actuator_effort_limit_nm=ARM_EFFORT_LIMIT_NM,
        payload_body_name="contact_tcp_marker",
        payload_attachment=RigidTransform(
            [0.0, 0.0, 0.145]
        ),
        payload_visual_registrar=_ignore,
        payload_collision_registrar=_ignore,
        pgc_movable=True,
        pgc_controller_kp=pgc_controller_kp,
        pgc_controller_kd=80.0,
        base_pose=RigidTransform(
            RotationMatrix.MakeZRotation(
                math.radians(
                    float(lab_config["robot"]["base_rpy_deg"][2])
                )
            ),
            lab_config["robot"]["base_xyz_m"],
        ),
    )
    parent, object_bodies, object_joints = _add_free_custom_object(
        plant,
        part_count,
        include_visuals=meshcat is not None or render_rgbd,
        simple_render_visuals=True,
        opening_angle_deg=opening_angle_deg,
        part_masses_kg=part_masses_kg,
        part_com_offsets_body_m=part_com_offsets_body_m,
        free_hinges=free_hinges,
        object_profile=object_profile,
    )
    grasped_body = (
        object_bodies[1]
        if object_profile == "jewelry_box"
        else object_bodies[0]
    )
    table = lab_config["table"]
    table_size = np.asarray(table["size_xyz_m"])
    table_center_z = float(table["top_height_m"]) - table_size[2] / 2.0
    plant.RegisterCollisionGeometry(
        plant.world_body(),
        RigidTransform([*table["center_xy_m"], table_center_z]),
        Box(*table_size),
        "contact_lab_table",
        _rigid_hydroelastic_properties(TABLE_FRICTION),
    )
    if meshcat is not None or render_rgbd:
        lab._register_lab_visuals(plant, lab_config)
    plant.Finalize()
    rgbd_cameras = (
        _add_rgbd_cameras(builder, scene_graph, lab_config["cameras"])
        if render_rgbd
        else ()
    )

    pgc_joints = tuple(
        plant.GetJointByName(name, robot.pgc_model_instance)
        for name in ("finger1_joint", "finger2_joint")
    )
    temporary_context = plant.CreateDefaultContext()
    _set_robot_positions(
        temporary_context,
        robot,
        lab.LAB_ARM_POSITION_RAD,
    )
    initial_angles = initial_opening_angles_deg or (
        initial_opening_angle_deg,
    ) * len(object_joints)
    for joint, angle_deg in zip(object_joints, initial_angles, strict=True):
        joint.set_angle(
            temporary_context,
            _internal_joint_angle_rad(
                object_profile, joint.name(), angle_deg
            ),
        )
    robot_grasp_rotation = RotationMatrix(
        np.column_stack(
            (
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
            )
        )
    )
    robot_grasp_rotation = (
        RotationMatrix(
            AngleAxis(
                math.radians(initial_wrist_pitch_deg),
                robot_grasp_rotation.matrix()[:, 1],
            )
        )
        @ robot_grasp_rotation
    )
    if object_profile == "desk_lamp":
        tilt = math.radians(DESK_LAMP_TOOL_TILT_DEG)
        robot_grasp_rotation = RotationMatrix(
            np.column_stack(
                (
                    [-math.cos(tilt), 0.0, -math.sin(tilt)],
                    [0.0, 1.0, 0.0],
                    [math.sin(tilt), 0.0, -math.cos(tilt)],
                )
            )
        )
    profile = (
        HOUSEHOLD_PROFILES[object_profile]
        if object_profile is not None
        else None
    )
    grasp_point_y = (
        0.0
        if object_profile in ("desk_lamp", "phantom_v3")
        else -profile["sizes_m"][0][1] / 2.0
        if profile is not None
        else 0.0
    )
    object_rotation = (
        RotationMatrix()
        if object_profile == "desk_lamp"
        else
        RotationMatrix.MakeZRotation(math.pi / 2.0)
        if profile is not None
        else robot_grasp_rotation
    )
    x_parent_grasped = plant.CalcRelativeTransform(
        temporary_context,
        parent.body_frame(),
        grasped_body.body_frame(),
    )
    grasped_rotation_world = (
        object_rotation @ x_parent_grasped.rotation()
    )
    if object_profile == "desk_lamp":
        axes = grasped_rotation_world.matrix()
        robot_grasp_rotation = RotationMatrix(
            np.column_stack((-axes[:, 0], axes[:, 2], axes[:, 1]))
        )
    object_payload_rotation = (
        grasped_rotation_world.inverse() @ robot_grasp_rotation
    )
    jewelry_grasp_point = np.array(
        [
            0.0,
            -profile["sizes_m"][1][1] / 2.0,
            0.003 + profile["sizes_m"][1][2] / 2.0,
        ]
    ) if object_profile == "jewelry_box" else None
    x_object_payload = RigidTransform(
        object_payload_rotation,
        (
            (
                np.array(
                    [
                        profile["sizes_m"][0][0]
                        * DESK_LAMP_ARM_GRASP_FRACTION,
                        0.0,
                        0.0,
                    ]
                )
                + 0.055 * object_payload_rotation.matrix()[:, 2]
            )
            if object_profile == "desk_lamp"
            else [
                -0.011,
                0.0,
                -grasp_offset_m,
            ]
            if object_profile == "phantom_v3"
            else jewelry_grasp_point
            + 0.043 * object_payload_rotation.matrix()[:, 2]
            if object_profile == "jewelry_box"
            else [
                0.0,
                grasp_point_y,
                -grasp_offset_m
                if profile is not None
                else grasp_offset_m,
            ]
        ),
    )
    x_payload_object = x_object_payload.inverse()
    object_center = np.array(
        [
            HELD_OBJECT_CENTER_X_M,
            (
                HELD_OBJECT_CENTER_Y_M
            ),
            float(table["top_height_m"])
            + (
                profile["sizes_m"][2][2] / 2.0
                if object_profile == "desk_lamp"
                else profile["sizes_m"][0][2] / 2.0
                if profile is not None
                else study.BOX_HEIGHT_M / 2.0
            )
            + OBJECT_TABLE_CLEARANCE_M,
        ]
    )
    initial_object_pose = RigidTransform(object_rotation, object_center)
    grasp_pose = (
        initial_object_pose
        @ x_parent_grasped
        @ x_payload_object.inverse()
    )
    q_grasp = _solve_robot_pose(
        plant,
        temporary_context,
        robot,
        lab.LAB_ARM_POSITION_RAD,
        grasp_pose,
    )
    pregrasp_pose = RigidTransform(
        grasp_pose.rotation(),
        grasp_pose.translation()
        - grasp_pose.rotation().matrix()[:, 2]
        * (
            PREGRASP_DISTANCE_M
        ),
    )
    q_pregrasp = _solve_robot_pose(
        plant,
        temporary_context,
        robot,
        q_grasp,
        pregrasp_pose,
    )
    approach_path = _integrate_translation_path(
        plant,
        temporary_context,
        robot,
        q_pregrasp,
        grasp_pose.translation() - pregrasp_pose.translation(),
        samples=40,
    )
    clearance_delta = (
        np.array([0.0, 0.0, 0.15])
        if object_profile == "desk_lamp"
        else np.array([0.0, 0.0, 0.08])
    )
    lift_clearance = _integrate_translation_path(
        plant,
        temporary_context,
        robot,
        q_grasp,
        clearance_delta,
        samples=40,
    )
    object_lift_m = 0.20 if object_profile == "desk_lamp" else OBJECT_LIFT_M
    lift_path = _integrate_translation_path(
        plant,
        temporary_context,
        robot,
        lift_clearance[-1],
        np.array(
            [0.0, -OBJECT_RETRACT_M, object_lift_m]
            - clearance_delta
        ),
        samples=80,
    )
    lift_path = np.vstack((lift_clearance, lift_path[1:]))
    q_lift = lift_path[-1]

    robot_breaks = [0.0, APPROACH_START_S]
    robot_positions = [q_pregrasp, q_pregrasp]
    approach_times = np.linspace(
        APPROACH_START_S, APPROACH_END_S, len(approach_path)
    )
    robot_breaks.extend(approach_times[1:].tolist())
    robot_positions.extend(approach_path[1:].tolist())
    robot_breaks.append(LIFT_START_S)
    robot_positions.append(q_grasp)
    lift_times = np.linspace(LIFT_START_S, LIFT_END_S, len(lift_path))
    robot_breaks.extend(lift_times[1:].tolist())
    robot_positions.extend(lift_path[1:].tolist())
    _set_robot_positions(temporary_context, robot, q_lift)
    lift_pose = plant.EvalBodyPoseInWorld(
        temporary_context,
        robot.payload,
    )
    # Static gravity identification does not need the old ±20° shake.
    orient_path = np.repeat(q_lift[None, :], 121, axis=0)
    robot_breaks.append(ORIENT_START_S)
    robot_positions.append(q_lift)
    orient_times = np.linspace(
        ORIENT_START_S,
        ORIENT_END_S,
        len(orient_path),
    )
    robot_breaks.extend(orient_times[1:].tolist())
    robot_positions.extend(orient_path[1:].tolist())
    q_lift = orient_path[-1]
    hold_end_times = []
    wrist_dynamic_times = []
    pitch_sequence = (
        wrist_pitch_sequence_deg
        if wrist_pitch_sequence_deg is not None
        else ((wrist_pitch_deg,) if wrist_pitch_deg else ())
    )
    roll_sequence = (
        wrist_roll_sequence_deg
        if wrist_roll_sequence_deg is not None
        else (0.0,) * len(pitch_sequence)
    )
    if len(roll_sequence) != len(pitch_sequence):
        raise ValueError("wrist roll and pitch sequences must have equal length")
    if adaptive_wrist:
        if pitch_sequence:
            raise ValueError(
                "adaptive wrist control cannot use a fixed wrist sequence"
            )
        initial_hold_end_s = ORIENT_END_S + FOUNDATIONPOSE_INITIAL_HOLD_S
        robot_breaks.append(initial_hold_end_s)
        robot_positions.append(q_lift)
        hold_end_times.append(initial_hold_end_s)
        wrist_dynamic_times.append(())
        simulation_end_s = initial_hold_end_s
    elif pitch_sequence:
        initial_hold_end_s = ORIENT_END_S + FOUNDATIONPOSE_INITIAL_HOLD_S
        robot_breaks.append(initial_hold_end_s)
        robot_positions.append(q_lift)
        _set_robot_positions(temporary_context, robot, q_lift)
        lift_pose = plant.EvalBodyPoseInWorld(
            temporary_context,
            robot.payload,
        )
        pitch_axis = lift_pose.rotation().matrix()[:, 1]
        roll_axis = lift_pose.rotation().matrix()[:, 0]
        current_q = q_lift
        current_time = initial_hold_end_s
        previous_target = None
        for pitch_deg, roll_deg in zip(
            pitch_sequence,
            roll_sequence,
            strict=True,
        ):
            rotation_start_s = current_time + 0.2
            rotation_end_s = rotation_start_s + WRIST_ROTATION_DURATION_S
            wrist_dynamic_times.append(
                tuple(
                    rotation_start_s
                    + fraction * WRIST_ROTATION_DURATION_S
                    for fraction in (0.25, 0.5, 0.75)
                )
            )
            hold_end_s = (
                rotation_end_s
                + FT_HOLD_WINDOW_S
                + (
                    4.0
                    if previous_target == (pitch_deg, roll_deg)
                    else DESK_LAMP_SETTLE_DURATION_S
                    if object_profile in ("desk_lamp", "phantom_v3")
                    else 4.0
                    if free_hinges
                    else 2.0
                )
            )
            if object_profile == "desk_lamp" and wrist_joints_only:
                target_q = q_lift.copy()
                target_q[-2] += math.radians(pitch_deg)
                target_q[-1] += math.radians(roll_deg)
                for joint, position in zip(
                    robot.joints[-2:],
                    target_q[-2:],
                    strict=True,
                ):
                    if not (
                        joint.position_lower_limits()[0]
                        <= position
                        <= joint.position_upper_limits()[0]
                    ):
                        raise ValueError(
                            f"{joint.name()} wrist target exceeds limits"
                        )
                rotation_path = np.linspace(current_q, target_q, 121)
            else:
                _set_robot_positions(temporary_context, robot, current_q)
                target_pose = RigidTransform(
                    RotationMatrix(
                        AngleAxis(math.radians(roll_deg), roll_axis)
                    )
                    @ RotationMatrix(
                        AngleAxis(math.radians(pitch_deg), pitch_axis)
                    )
                    @ lift_pose.rotation(),
                    lift_pose.translation(),
                )
                rotation_path = _integrate_pose_path(
                    plant,
                    temporary_context,
                    robot,
                    current_q,
                    target_pose,
                    samples=120,
                )
            robot_breaks.append(rotation_start_s)
            robot_positions.append(current_q)
            rotation_times = np.linspace(
                rotation_start_s,
                rotation_end_s,
                len(rotation_path),
            )
            robot_breaks.extend(rotation_times[1:].tolist())
            robot_positions.extend(rotation_path[1:].tolist())
            robot_breaks.append(hold_end_s)
            robot_positions.append(rotation_path[-1])
            hold_end_times.append(hold_end_s)
            current_q = rotation_path[-1]
            current_time = hold_end_s
            previous_target = (pitch_deg, roll_deg)
        q_lift = current_q
        simulation_end_s = hold_end_times[-1]
    else:
        robot_breaks.append(simulation_end_s)
        robot_positions.append(q_lift)
        hold_end_times.append(simulation_end_s)
        wrist_dynamic_times.append(())
    source_type = _MutableDesiredSource if adaptive_wrist else _desired_source
    robot_source = builder.AddSystem(
        source_type(
            np.asarray(robot_breaks),
            np.asarray(robot_positions),
        )
    )
    builder.Connect(
        robot_source.get_output_port(),
        plant.get_desired_state_input_port(robot.model_instance),
    )

    gripper_breaks = np.array(
        [
            0.0,
            GRIPPER_CLOSE_START_S,
            GRIPPER_CLOSE_END_S,
            simulation_end_s,
        ]
    )
    initial_finger_position_m = (
        0.0 if object_profile == "jewelry_box" else INITIAL_FINGER_POSITION_M
    )
    gripper_positions = np.array(
        [
            [initial_finger_position_m, initial_finger_position_m],
            [initial_finger_position_m, initial_finger_position_m],
            [PGC_FINAL_POSITION_M, PGC_FINAL_POSITION_M],
            [PGC_FINAL_POSITION_M, PGC_FINAL_POSITION_M],
        ]
    )
    gripper_source = builder.AddSystem(
        _desired_source(gripper_breaks, gripper_positions)
    )
    builder.Connect(
        gripper_source.get_output_port(),
        plant.get_desired_state_input_port(robot.pgc_model_instance),
    )

    actuation_matrix = plant.MakeActuationMatrix()
    gravity = builder.AddSystem(
        InverseDynamics(
            plant,
            InverseDynamics.kGravityCompensation,
        )
    )
    gravity_projection = builder.AddSystem(
        MatrixGain(actuation_matrix.T)
    )
    builder.Connect(plant.get_state_output_port(), gravity.get_input_port())
    builder.Connect(
        gravity.get_output_port(),
        gravity_projection.get_input_port(),
    )
    builder.Connect(
        gravity_projection.get_output_port(),
        plant.get_actuation_input_port(),
    )
    if meshcat is not None:
        MeshcatVisualizer.AddToBuilder(
            builder,
            scene_graph,
            meshcat,
            MeshcatVisualizerParams(publish_period=1.0 / 30.0),
        )
        lab._set_camera_sight_lines(meshcat, lab_config["cameras"])
    sensor_joint_index = None
    for index in range(plant.num_joints()):
        joint = plant.get_joint(JointIndex(index))
        if joint.child_body().name() == "bracket":
            sensor_joint_index = int(index)
            break
    if sensor_joint_index is None:
        raise RuntimeError("Could not find link6-to-AFT bracket weld joint")
    return Scenario(
        diagram=builder.Build(),
        plant=plant,
        scene_graph=scene_graph,
        robot=robot,
        pgc_joints=pgc_joints,
        parent=parent,
        grasped_body=grasped_body,
        object_bodies=object_bodies,
        object_joints=object_joints,
        sensor_mount_joint_index=sensor_joint_index,
        initial_finger_position_m=initial_finger_position_m,
        q_lower=tuple(float(value) for value in q_pregrasp),
        q_lift=tuple(float(value) for value in q_lift),
        grasp_offset_m=grasp_offset_m,
        initial_object_pose=initial_object_pose,
        simulation_end_s=simulation_end_s,
        hold_end_times_s=tuple(hold_end_times),
        orient_dynamic_times_s=tuple(
            np.linspace(ORIENT_START_S, ORIENT_END_S, 121)
        ),
        wrist_dynamic_times_s=tuple(wrist_dynamic_times),
        robot_source=robot_source,
        adaptive_wrist=adaptive_wrist,
        rgbd_cameras=rgbd_cameras,
        object_profile=object_profile,
    )


def _rotation_angle_deg(rotation) -> float:
    cosine = np.clip((np.trace(rotation.matrix()) - 1.0) / 2.0, -1.0, 1.0)
    return math.degrees(math.acos(cosine))


def _opening_angle_deg(scenario: Scenario, joint, context) -> float:
    angle = math.degrees(joint.get_angle(context))
    if scenario.object_profile == "desk_lamp":
        return (
            -angle - DESK_LAMP_LOWER_ZERO_OFFSET_DEG
            if "lower" in joint.name()
            else -angle
        )
    if scenario.object_profile == "jewelry_box":
        return -angle
    if scenario.object_profile == "phantom_v3":
        return angle
    return 180.0 + angle


def _pair_has_object(pair: str) -> bool:
    return any(
        name in pair
        for name in (
            "contact_custom_parent_model",
            "contact_custom_child_",
            "desk_lamp_",
            "jewelry_box_",
            "phantom_v3_",
        )
    )


def _geometry_label(inspector, geometry_id) -> str:
    return (
        f"{inspector.GetName(inspector.GetFrameId(geometry_id))}/"
        f"{inspector.GetName(geometry_id)}"
    )


def _contact_pairs(contact_results, inspector) -> tuple[str, ...]:
    pairs = []
    for index in range(contact_results.num_point_pair_contacts()):
        pair = contact_results.point_pair_contact_info(index).point_pair()
        pairs.append(
            f"{_geometry_label(inspector, pair.id_A)} <-> "
            f"{_geometry_label(inspector, pair.id_B)}"
        )
    for index in range(contact_results.num_hydroelastic_contacts()):
        surface = contact_results.hydroelastic_contact_info(
            index
        ).contact_surface()
        pairs.append(
            f"{_geometry_label(inspector, surface.id_M())} <-> "
            f"{_geometry_label(inspector, surface.id_N())}"
        )
    return tuple(sorted(set(pairs)))


def _clearance_pair_allowed(
    scenario: Scenario,
    inspector,
    geometry_a,
    geometry_b,
) -> bool:
    try:
        body_a = scenario.plant.GetBodyFromFrameId(
            inspector.GetFrameId(geometry_a)
        )
        body_b = scenario.plant.GetBodyFromFrameId(
            inspector.GetFrameId(geometry_b)
        )
    except RuntimeError:
        return False
    index_a, index_b = int(body_a.index()), int(body_b.index())
    if index_a == index_b:
        return True
    object_indices = {
        int(body.index()) for body in scenario.object_bodies
    }
    adjacent = {
        frozenset(
            (
                int(joint.parent_body().index()),
                int(joint.child_body().index()),
            )
        )
        for joint in scenario.object_joints
    }
    if index_a in object_indices and index_b in object_indices:
        return frozenset((index_a, index_b)) in adjacent
    gripper_indices = {
        int(index)
        for index in scenario.plant.GetBodyIndices(
            scenario.robot.pgc_model_instance
        )
    }
    if index_a in gripper_indices and index_b in gripper_indices:
        return True
    grasped = int(scenario.grasped_body.index())
    return (
        index_a == grasped and index_b in gripper_indices
    ) or (
        index_b == grasped and index_a in gripper_indices
    )


def _candidate_path_clearance(
    scenario: Scenario,
    root_context,
    path: np.ndarray,
    required_clearance_m: float,
) -> tuple[bool, float, str | None]:
    test_context = root_context.Clone()
    plant_context = scenario.plant.GetMyMutableContextFromRoot(test_context)
    actual_context = scenario.plant.GetMyContextFromRoot(root_context)
    x_payload_parent = scenario.plant.CalcRelativeTransform(
        actual_context,
        scenario.robot.payload.body_frame(),
        scenario.parent.body_frame(),
    )
    joint_angles = [
        joint.get_angle(actual_context) for joint in scenario.object_joints
    ]
    scene_graph_context = scenario.scene_graph.GetMyContextFromRoot(
        test_context
    )
    inspector = scenario.scene_graph.model_inspector()
    minimum = required_clearance_m
    indices = np.unique(
        np.linspace(0, len(path) - 1, min(len(path), 25), dtype=int)
    )
    for index in indices:
        _set_robot_positions(plant_context, scenario.robot, path[index])
        x_world_payload = scenario.plant.EvalBodyPoseInWorld(
            plant_context,
            scenario.robot.payload,
        )
        scenario.plant.SetFreeBodyPose(
            plant_context,
            scenario.parent,
            x_world_payload @ x_payload_parent,
        )
        for joint, angle in zip(
            scenario.object_joints,
            joint_angles,
            strict=True,
        ):
            joint.set_angle(plant_context, angle)
            joint.set_angular_rate(plant_context, 0.0)
        query = scenario.scene_graph.get_query_output_port().Eval(
            scene_graph_context
        )
        for pair in query.ComputeSignedDistancePairwiseClosestPoints(
            required_clearance_m
        ):
            if _clearance_pair_allowed(
                scenario,
                inspector,
                pair.id_A,
                pair.id_B,
            ):
                continue
            distance = float(pair.distance)
            minimum = min(minimum, distance)
            if distance < required_clearance_m:
                return (
                    False,
                    distance,
                    f"{_geometry_label(inspector, pair.id_A)} <-> "
                    f"{_geometry_label(inspector, pair.id_B)}",
                )
    return True, minimum, None


def _adaptive_wrist_candidates(
    scenario: Scenario,
    root_context,
    hold: StaticHoldMeasurement,
    current_pitch_roll_deg: tuple[float, float],
    visited: set[tuple[float, float]],
    required_clearance_m: float,
) -> list[dict[str, object]]:
    plant_context = scenario.plant.GetMyContextFromRoot(root_context)
    current_q = np.array(
        [
            joint.get_angle(plant_context)
            for joint in scenario.robot.joints
        ]
    )
    current_pose = scenario.plant.EvalBodyPoseInWorld(
        plant_context,
        scenario.robot.payload,
    )
    roll_axis = current_pose.rotation().matrix()[:, 0]
    pitch_axis = current_pose.rotation().matrix()[:, 1]
    sensor_joint = scenario.plant.get_joint(
        JointIndex(scenario.sensor_mount_joint_index)
    )
    candidates = []
    for pitch_step in ADAPTIVE_WRIST_STEP_DEG:
        for roll_step in ADAPTIVE_WRIST_STEP_DEG:
            repeat_hold = pitch_step == 0.0 and roll_step == 0.0
            pitch = current_pitch_roll_deg[0] + pitch_step
            roll = current_pitch_roll_deg[1] + roll_step
            action = {
                "wrist_pitch_deg": pitch,
                "wrist_roll_deg": roll,
                "pitch_step_deg": pitch_step,
                "roll_step_deg": roll_step,
                "minimum_clearance_m": None,
                "safe": False,
                "rejection_reason": None,
                "previously_visited": (pitch, roll) in visited,
            }
            if (
                abs(pitch) > ADAPTIVE_WRIST_LIMIT_DEG
                or abs(roll) > ADAPTIVE_WRIST_LIMIT_DEG
            ):
                action["rejection_reason"] = "adaptive_wrist_limit"
                candidates.append(action)
                continue
            planning_context = scenario.plant.CreateDefaultContext()
            _set_robot_positions(
                planning_context,
                scenario.robot,
                current_q,
            )
            if repeat_hold:
                path = np.repeat(current_q[None, :], 2, axis=0)
            else:
                target_pose = RigidTransform(
                    RotationMatrix(
                        AngleAxis(math.radians(roll_step), roll_axis)
                    )
                    @ RotationMatrix(
                        AngleAxis(math.radians(pitch_step), pitch_axis)
                    )
                    @ current_pose.rotation(),
                    current_pose.translation(),
                )
                try:
                    path = _integrate_pose_path(
                        scenario.plant,
                        planning_context,
                        scenario.robot,
                        current_q,
                        target_pose,
                        samples=61,
                    )
                except RuntimeError:
                    action["rejection_reason"] = "ik_failed"
                    candidates.append(action)
                    continue
            if repeat_hold and hold.contact_free:
                safe, clearance, pair = (
                    True,
                    required_clearance_m,
                    None,
                )
            else:
                safe, clearance, pair = _candidate_path_clearance(
                    scenario,
                    root_context,
                    path,
                    required_clearance_m,
                )
            action["minimum_clearance_m"] = clearance
            if not safe:
                action["rejection_reason"] = f"clearance:{pair}"
                candidates.append(action)
                continue
            _set_robot_positions(
                planning_context,
                scenario.robot,
                path[-1],
            )
            x_world_sensor = scenario.plant.CalcRelativeTransform(
                planning_context,
                scenario.plant.world_frame(),
                sensor_joint.frame_on_child(),
            )
            gravity = x_world_sensor.rotation().inverse().multiply(
                np.array([0.0, 0.0, -9.81])
            )
            positions = np.asarray(hold.part_com_sensor_m)
            regressor = np.column_stack(
                [
                    np.concatenate(
                        (-gravity, -np.cross(position, gravity))
                    )
                    for position in positions
                ]
            )
            action.update(
                {
                    "safe": True,
                    "mass_regressor": regressor,
                    "movement_cost": float(
                        np.linalg.norm(path[-1] - current_q)
                    ),
                    "_path": path,
                }
            )
            candidates.append(action)
    return candidates


def _sample_dynamic_lift(
    scenario: Scenario,
    plant_context,
    time_s: float,
    inspector,
) -> DynamicLiftMeasurement:
    sensor_joint = scenario.plant.get_joint(
        JointIndex(scenario.sensor_mount_joint_index)
    )
    x_world_sensor = scenario.plant.CalcRelativeTransform(
        plant_context,
        scenario.plant.world_frame(),
        sensor_joint.frame_on_child(),
    )
    sensor_rotation = x_world_sensor.rotation().inverse().matrix()
    sensor_position = x_world_sensor.translation()
    gravity = np.array([0.0, 0.0, -9.81])
    columns = []
    for body in scenario.object_bodies:
        x_world_body = scenario.plant.EvalBodyPoseInWorld(
            plant_context,
            body,
        )
        velocity = body.EvalSpatialVelocityInWorld(plant_context)
        acceleration = body.EvalSpatialAccelerationInWorld(plant_context)
        com_world = x_world_body.multiply(
            body.default_spatial_inertia().get_com()
        )
        specific_force = (
            body.CalcCenterOfMassTranslationalAccelerationInWorld(
                plant_context
            )
            - gravity
        )
        body_inertia = body.default_spatial_inertia()
        unit_inertia_body = (
            body_inertia.Shift(body_inertia.get_com())
            .CalcRotationalInertia()
            .CopyToFullMatrix3()
            / body_inertia.get_mass()
        )
        rotation = x_world_body.rotation().matrix()
        unit_inertia_world = rotation @ unit_inertia_body @ rotation.T
        omega = velocity.rotational()
        torque = (
            np.cross(com_world - sensor_position, specific_force)
            + unit_inertia_world @ acceleration.rotational()
            + np.cross(omega, unit_inertia_world @ omega)
        )
        columns.append(
            np.concatenate(
                (
                    sensor_rotation @ specific_force,
                    sensor_rotation @ torque,
                )
            )
        )
    known_tool_wrench = _known_tool_wrench(
        scenario,
        plant_context,
        sensor_position,
        sensor_rotation,
    )
    reaction = scenario.plant.get_reaction_forces_output_port().Eval(
        plant_context
    )[scenario.sensor_mount_joint_index]
    pairs = _contact_pairs(
        scenario.plant.get_contact_results_output_port().Eval(plant_context),
        inspector,
    )
    return DynamicLiftMeasurement(
        time_s=time_s,
        ft_reaction_force=tuple(
            float(value) for value in reaction.translational()
        ),
        ft_reaction_torque=tuple(
            float(value) for value in reaction.rotational()
        ),
        mass_regressor=tuple(
            tuple(float(value) for value in row)
            for row in np.column_stack(columns)
        ),
        known_tool_wrench=tuple(
            float(value) for value in known_tool_wrench
        ),
        sensor_rotation_world_to_sensor=tuple(
            tuple(float(value) for value in row)
            for row in sensor_rotation
        ),
        contact_free=not any(
            "world/contact_lab_table" in pair
            and _pair_has_object(pair)
            for pair in pairs
        ),
        contact_pairs=pairs,
        actual_opening_angles_deg=tuple(
            _opening_angle_deg(scenario, joint, plant_context)
            for joint in scenario.object_joints
        ),
    )


def _known_tool_wrench(
    scenario: Scenario,
    plant_context,
    sensor_position: np.ndarray,
    sensor_rotation: np.ndarray,
) -> np.ndarray:
    gravity = np.array([0.0, 0.0, -9.81])
    known_tool_wrench_world = np.zeros(6)
    for model_instance in (
        scenario.robot.aft_model_instance,
        scenario.robot.pgc_model_instance,
        scenario.robot.payload.model_instance(),
    ):
        for body_index in scenario.plant.GetBodyIndices(model_instance):
            body = scenario.plant.get_body(body_index)
            inertia = body.default_spatial_inertia()
            mass = float(inertia.get_mass())
            if mass == 0.0:
                continue
            pose = scenario.plant.EvalBodyPoseInWorld(plant_context, body)
            velocity = body.EvalSpatialVelocityInWorld(plant_context)
            acceleration = body.EvalSpatialAccelerationInWorld(plant_context)
            com_world = pose.multiply(inertia.get_com())
            force = mass * (
                body.CalcCenterOfMassTranslationalAccelerationInWorld(
                    plant_context
                )
                - gravity
            )
            central_inertia = (
                inertia.Shift(inertia.get_com())
                .CalcRotationalInertia()
                .CopyToFullMatrix3()
            )
            rotation = pose.rotation().matrix()
            central_inertia_world = (
                rotation @ central_inertia @ rotation.T
            )
            omega = velocity.rotational()
            torque = (
                np.cross(com_world - sensor_position, force)
                + central_inertia_world @ acceleration.rotational()
                + np.cross(
                    omega,
                    central_inertia_world @ omega,
                )
            )
            known_tool_wrench_world += np.concatenate((force, torque))
    return np.concatenate(
        (
            sensor_rotation @ known_tool_wrench_world[:3],
            sensor_rotation @ known_tool_wrench_world[3:],
        )
    )


def _filter_static_wrench_samples(
    force_samples: list[np.ndarray],
    torque_samples: list[np.ndarray],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
    bool,
]:
    samples = np.column_stack((force_samples, torque_samples))
    safety_limits = np.array(
        [AFT200_FORCE_LIMIT_N] * 3 + [AFT200_TORQUE_LIMIT_NM] * 3
    )
    if np.any(np.abs(samples) > safety_limits):
        raise RuntimeError("AFT200 safe force/torque limit exceeded")
    samples = samples @ AFT200_CALIBRATION_MATRIX.T + AFT200_BIAS
    if np.any(AFT200_NOISE_STD):
        samples += np.random.default_rng(20260728).normal(
            0.0,
            AFT200_NOISE_STD,
            samples.shape,
        )
    resolution = np.array(
        [AFT200_FORCE_RESOLUTION_N] * 3
        + [AFT200_TORQUE_RESOLUTION_NM] * 3
    )
    limits = np.array(
        [AFT200_FORCE_RANGE_N] * 3 + [AFT200_TORQUE_RANGE_NM] * 3
    )
    samples = np.round(np.clip(samples, -limits, limits) / resolution) * resolution
    median = np.median(samples, axis=0)
    mad = np.median(np.abs(samples - median), axis=0)
    scale = np.maximum(1.4826 * mad, 1e-9)
    inliers = np.all(np.abs(samples - median) <= 3.5 * scale, axis=1)
    if np.count_nonzero(inliers) < len(samples) // 2:
        inliers[:] = True
    filtered = samples[inliers]
    midpoint = len(filtered) // 2
    first, second = filtered[:midpoint], filtered[midpoint:]
    half_delta = np.abs(first.mean(axis=0) - second.mean(axis=0))
    standard_error = np.sqrt(
        first.var(axis=0) / len(first)
        + second.var(axis=0) / len(second)
    )
    stationary = bool(
        np.all(
            half_delta
            <= 3.0 * standard_error
            + np.array((0.20, 0.20, 0.20, 0.02, 0.02, 0.02))
        )
    )
    mean = filtered.mean(axis=0)
    std = filtered.std(axis=0, ddof=1) if len(filtered) > 1 else np.zeros(6)
    block_means = np.asarray(
        [block.mean(axis=0) for block in np.array_split(filtered, 10)]
    )
    mean_standard_error = block_means.std(axis=0, ddof=1) / math.sqrt(
        len(block_means)
    )
    return (
        mean[:3],
        mean[3:],
        std,
        mean_standard_error,
        block_means,
        int(np.count_nonzero(inliers)),
        stationary,
    )


def _sample_static_hold(
    simulator,
    scenario: Scenario,
    plant_context,
    hold_end_s: float,
    wrist_pitch_deg: float,
) -> StaticHoldMeasurement:
    if not math.isclose(
        FT_HOLD_SAMPLE_COUNT / FT_HOLD_WINDOW_S,
        AFT200_SAMPLE_RATE_HZ,
    ):
        raise RuntimeError("AFT200 hold sampling must run at 1000 Hz")
    force_samples = []
    torque_samples = []
    opening_samples = []
    angular_rate_samples = []
    hold_start_s = hold_end_s - FT_HOLD_WINDOW_S
    simulator.AdvanceTo(hold_start_s)
    for sample_time in np.linspace(
        hold_start_s + FT_HOLD_WINDOW_S / FT_HOLD_SAMPLE_COUNT,
        hold_end_s,
        FT_HOLD_SAMPLE_COUNT,
    ):
        simulator.AdvanceTo(float(sample_time))
        sample = scenario.plant.get_reaction_forces_output_port().Eval(
            plant_context
        )[scenario.sensor_mount_joint_index]
        force_samples.append(sample.translational())
        torque_samples.append(sample.rotational())
        opening_samples.append(
            [
                _opening_angle_deg(scenario, joint, plant_context)
                for joint in scenario.object_joints
            ]
        )
        angular_rate_samples.append(
            [
                math.degrees(joint.get_angular_rate(plant_context))
                for joint in scenario.object_joints
            ]
        )
    (
        force,
        torque,
        std,
        mean_standard_error,
        block_means,
        inlier_count,
        stationary,
    ) = _filter_static_wrench_samples(force_samples, torque_samples)
    sensor_joint = scenario.plant.get_joint(
        JointIndex(scenario.sensor_mount_joint_index)
    )
    x_world_sensor = scenario.plant.CalcRelativeTransform(
        plant_context,
        scenario.plant.world_frame(),
        sensor_joint.frame_on_child(),
    )
    x_sensor_world = x_world_sensor.inverse()
    known_tool_wrench = _known_tool_wrench(
        scenario,
        plant_context,
        x_world_sensor.translation(),
        x_world_sensor.rotation().inverse().matrix(),
    )
    gravity_sensor = x_world_sensor.rotation().inverse().multiply(
        np.array([0.0, 0.0, -9.81])
    )
    part_com_sensor = []
    part_rotations_sensor = []
    for body in scenario.object_bodies:
        x_world_body = scenario.plant.EvalBodyPoseInWorld(
            plant_context,
            body,
        )
        com_world = x_world_body.multiply(
            body.default_spatial_inertia().get_com()
        )
        part_com_sensor.append(
            tuple(float(value) for value in x_sensor_world.multiply(com_world))
        )
        part_rotations_sensor.append(
            tuple(
                tuple(float(value) for value in row)
                for row in (
                    x_world_sensor.rotation().inverse().matrix()
                    @ x_world_body.rotation().matrix()
                )
            )
        )
    joint_origins_sensor = []
    joint_axes_sensor = []
    for joint in scenario.object_joints:
        x_world_joint = scenario.plant.CalcRelativeTransform(
            plant_context,
            scenario.plant.world_frame(),
            joint.frame_on_parent(),
        )
        joint_origins_sensor.append(
            tuple(
                float(value)
                for value in x_sensor_world.multiply(
                    x_world_joint.translation()
                )
            )
        )
        joint_axes_sensor.append(
            tuple(
                float(value)
                for value in (
                    x_world_sensor.rotation().inverse().matrix()
                    @ x_world_joint.rotation().matrix()
                    @ joint.revolute_axis()
                )
            )
        )
    opening_samples = np.asarray(opening_samples)
    opening_span = np.ptp(opening_samples, axis=0)
    angular_rate_samples = np.asarray(angular_rate_samples)
    maximum_velocity = np.max(np.abs(angular_rate_samples), axis=0)
    maximum_acceleration = np.max(
        np.abs(
            np.diff(angular_rate_samples, axis=0)
            * (FT_HOLD_SAMPLE_COUNT / FT_HOLD_WINDOW_S)
        ),
        axis=0,
    )
    contact_pairs = _contact_pairs(
        scenario.plant.get_contact_results_output_port().Eval(plant_context),
        scenario.scene_graph.model_inspector(),
    )
    contact_free = not any(
        "world/contact_lab_table" in pair
        and _pair_has_object(pair)
        for pair in contact_pairs
    )
    return StaticHoldMeasurement(
        hold_end_s=hold_end_s,
        wrist_pitch_deg=wrist_pitch_deg,
        ft_reaction_force=tuple(float(value) for value in force),
        ft_reaction_torque=tuple(float(value) for value in torque),
        known_tool_wrench=tuple(
            float(value) for value in known_tool_wrench
        ),
        ft_sample_std=tuple(float(value) for value in std),
        ft_mean_standard_error=tuple(
            float(value) for value in mean_standard_error
        ),
        ft_block_means=tuple(
            tuple(float(value) for value in block)
            for block in block_means
        ),
        ft_inlier_count=inlier_count,
        ft_stationary=stationary,
        contact_free=contact_free,
        contact_pairs=contact_pairs,
        actual_opening_angles_deg=tuple(
            _opening_angle_deg(scenario, joint, plant_context)
            for joint in scenario.object_joints
        ),
        opening_angle_span_deg=tuple(float(value) for value in opening_span),
        maximum_abs_joint_velocity_deg_s=tuple(
            float(value) for value in maximum_velocity
        ),
        maximum_abs_joint_acceleration_deg_s2=tuple(
            float(value) for value in maximum_acceleration
        ),
        joints_stationary=bool(np.all(opening_span <= 1.0)),
        gravity_sensor_m_s2=tuple(float(value) for value in gravity_sensor),
        part_com_sensor_m=tuple(part_com_sensor),
        part_rotations_sensor=tuple(part_rotations_sensor),
        joint_origins_sensor_m=tuple(joint_origins_sensor),
        joint_axes_sensor=tuple(joint_axes_sensor),
        sensor_world_pose=tuple(
            tuple(float(value) for value in row)
            for row in x_world_sensor.GetAsMatrix4()
        ),
        grasp_relative_pose=tuple(
            tuple(float(value) for value in row)
            for row in scenario.plant.CalcRelativeTransform(
                plant_context,
                scenario.robot.payload.body_frame(),
                scenario.grasped_body.body_frame(),
            ).GetAsMatrix4()
        ),
    )


def _save_foundationpose_frame(
    scenario: Scenario,
    root_context,
    plant_context,
    output_dir: Path,
    frame_index: int,
    time_s: float,
    metadata: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    body_labels = metadata["body_labels"]
    tracking_bodies = (
        (
            scenario.object_bodies[2],
            scenario.object_bodies[0],
            scenario.object_bodies[1],
        )
        if scenario.object_profile == "desk_lamp"
        else scenario.object_bodies
    )
    frame = {
        "index": frame_index,
        "time_s": time_s,
        "actual_opening_deg": _opening_angle_deg(
            scenario, scenario.object_joints[0], plant_context
        ),
        "actual_opening_angles_deg": [
            _opening_angle_deg(scenario, joint, plant_context)
            for joint in scenario.object_joints
        ],
        "encoder_parent_world_pose": (
            scenario.plant.EvalBodyPoseInWorld(
                plant_context,
                scenario.robot.payload,
            )
            @ RigidTransform([0.0, 0.0, -scenario.grasp_offset_m])
        ).GetAsMatrix4().tolist(),
        "cameras": {},
    }
    for name, sensor, intrinsics, x_world_camera in scenario.rgbd_cameras:
        sensor_context = sensor.GetMyContextFromRoot(root_context)
        color = np.asarray(
            sensor.color_image_output_port().Eval(sensor_context).data
        )[:, :, :3]
        depth = np.asarray(
            sensor.depth_image_32F_output_port().Eval(sensor_context).data
        )[:, :, 0]
        labels = np.asarray(
            sensor.label_image_output_port().Eval(sensor_context).data
        )[:, :, 0]
        camera_dir = output_dir / name
        for subdir in ("rgb", "depth_m"):
            (camera_dir / subdir).mkdir(parents=True, exist_ok=True)
        for link in body_labels:
            (camera_dir / f"{link}_mask").mkdir(
                parents=True,
                exist_ok=True,
            )
        stem = f"{frame_index:04d}"
        Image.fromarray(color).save(camera_dir / "rgb" / f"{stem}.png")
        np.save(
            camera_dir / "depth_m" / f"{stem}.npy",
            depth.astype(np.float32),
        )
        np.savetxt(camera_dir / "K.txt", intrinsics)
        for link, label in body_labels.items():
            Image.fromarray(
                np.where(labels == label, 255, 0).astype(np.uint8)
            ).save(camera_dir / f"{link}_mask" / f"{stem}.png")
        x_camera_world = x_world_camera.inverse()
        gt = {}
        for link, body in zip(
            body_labels,
            tracking_bodies,
            strict=True,
        ):
            x_world_body = scenario.plant.EvalBodyPoseInWorld(
                plant_context,
                body,
            )
            gt[link] = (
                x_camera_world @ x_world_body
            ).GetAsMatrix4().tolist()
        metadata["cameras"].setdefault(
            name,
            {
                "K": intrinsics.tolist(),
                "X_world_camera": x_world_camera.GetAsMatrix4().tolist(),
            },
        )
        frame["cameras"][name] = {
            "gt_object_in_camera": gt,
            "visible_pixels": {
                link: int(np.count_nonzero(labels == label))
                for link, label in body_labels.items()
            },
            "edge_margin_pixels": {
                link: (
                    min(
                        int(np.min(np.nonzero(labels == label)[1])),
                        int(np.min(np.nonzero(labels == label)[0])),
                        int(labels.shape[1] - 1 - np.max(
                            np.nonzero(labels == label)[1]
                        )),
                        int(labels.shape[0] - 1 - np.max(
                            np.nonzero(labels == label)[0]
                        )),
                    )
                    if np.any(labels == label)
                    else -1
                )
                for link, label in body_labels.items()
            },
            "render_labels": np.unique(labels).astype(int).tolist(),
        }
    metadata["frames"].append(frame)


def simulate(
    part_count: int,
    *,
    meshcat=None,
    grasp_offset_m: float | None = None,
    opening_angle_deg: float = 180.0,
    initial_opening_angle_deg: float = 180.0,
    initial_opening_angles_deg: tuple[float, ...] | None = None,
    initial_wrist_pitch_deg: float = 0.0,
    wrist_pitch_deg: float = 0.0,
    grasp_object: bool = True,
    simulation_end_s: float | None = None,
    pgc_controller_kp: float = 3000.0,
    part_masses_kg: tuple[float, ...] | None = None,
    part_com_offsets_body_m: tuple[tuple[float, float, float], ...] | None = None,
    foundationpose_capture_dir: Path | None = None,
    foundationpose_capture_fps: float = 10.0,
    wrist_pitch_sequence_deg: tuple[float, ...] | None = None,
    wrist_roll_sequence_deg: tuple[float, ...] | None = None,
    free_hinges: bool = True,
    object_profile: str | None = None,
    wrist_joints_only: bool = False,
    adaptive_action_selector=None,
    adaptive_max_holds: int = 12,
    adaptive_collision_clearance_m: float = ADAPTIVE_COLLISION_CLEARANCE_M,
) -> ContactFtResult:
    if adaptive_action_selector is not None and (
        wrist_pitch_sequence_deg is not None
        or wrist_roll_sequence_deg is not None
        or wrist_pitch_deg
    ):
        raise ValueError(
            "adaptive control cannot use a fixed wrist trajectory"
        )
    if adaptive_max_holds < 1:
        raise ValueError("adaptive_max_holds must be positive")
    if adaptive_collision_clearance_m < 0.0:
        raise ValueError("adaptive collision clearance must be nonnegative")
    if (
        adaptive_action_selector is not None
        and foundationpose_capture_dir is not None
    ):
        raise ValueError(
            "adaptive RGB-D capture requires the online camera controller"
        )
    if object_profile is not None:
        profile = HOUSEHOLD_PROFILES[object_profile]
        if part_count != profile["part_count"]:
            raise ValueError(
                f"{object_profile} requires {profile['part_count']} parts"
            )
        opening_limits = profile["joint_limits_deg"]
        opening_bounds = profile.get(
            "joint_angle_bounds_deg",
            tuple((0.0, limit) for limit in opening_limits),
        )
    else:
        opening_limits = (opening_angle_deg,) * (part_count - 1)
        opening_bounds = tuple(
            (0.0, limit) for limit in opening_limits
        )
    initial_angles = initial_opening_angles_deg or (
        initial_opening_angle_deg,
    ) * (part_count - 1)
    if len(initial_angles) != part_count - 1:
        raise ValueError("one initial opening angle is required per hinge")
    if any(
        not lower <= angle <= upper
        for angle, (lower, upper) in zip(
            initial_angles,
            opening_bounds,
            strict=True,
        )
    ):
        raise ValueError(
            "initial opening angles must be within their joint limits"
        )
    resolved_masses = tuple(
        _part_mass(index, part_masses_kg)
        for index in range(part_count)
    )
    if any(mass < MINIMUM_PART_MASS_KG for mass in resolved_masses):
        raise ValueError(
            f"every object part mass must be at least "
            f"{MINIMUM_PART_MASS_KG:.1f} kg"
        )
    if part_com_offsets_body_m is not None and (
        len(part_com_offsets_body_m) != part_count
        or any(len(com) != 3 for com in part_com_offsets_body_m)
        or not np.isfinite(part_com_offsets_body_m).all()
    ):
        raise ValueError("one finite body-frame COM is required per part")
    if sum(resolved_masses) > PGC_MAX_PAYLOAD_KG:
        raise ValueError(
            f"total object mass {sum(resolved_masses):.3f} kg exceeds "
            f"the PGC payload limit {PGC_MAX_PAYLOAD_KG:.1f} kg"
        )
    if sum(resolved_masses) < MINIMUM_OBJECT_MASS_KG:
        raise ValueError(
            f"total object mass must be at least "
            f"{MINIMUM_OBJECT_MASS_KG:.1f} kg"
        )
    if (
        object_profile == "desk_lamp"
        and resolved_masses[2] < sum(resolved_masses[:2])
    ):
        raise ValueError(
            "desk lamp base mass must be at least the combined mass "
            "of links 2 and 3"
        )
    if (
        object_profile == "jewelry_box"
        and resolved_masses[0] < resolved_masses[1]
    ):
        raise ValueError(
            "jewelry box base mass must be at least the lid mass"
        )
    if grasp_offset_m is None:
        grasp_offset_m = PARENT_END_GRASP_OFFSET_M
    if simulation_end_s is None:
        simulation_end_s = (
            7.0 if wrist_pitch_deg else SIMULATION_END_S
        )
    scenario = build_scenario(
        part_count,
        meshcat=meshcat,
        opening_angle_deg=opening_angle_deg,
        initial_opening_angle_deg=initial_opening_angle_deg,
        initial_opening_angles_deg=initial_angles,
        initial_wrist_pitch_deg=initial_wrist_pitch_deg,
        wrist_pitch_deg=wrist_pitch_deg,
        simulation_end_s=simulation_end_s,
        pgc_controller_kp=pgc_controller_kp,
        part_masses_kg=resolved_masses,
        part_com_offsets_body_m=part_com_offsets_body_m,
        wrist_pitch_sequence_deg=wrist_pitch_sequence_deg,
        wrist_roll_sequence_deg=wrist_roll_sequence_deg,
        free_hinges=free_hinges,
        render_rgbd=foundationpose_capture_dir is not None,
        grasp_offset_m=grasp_offset_m,
        object_profile=object_profile,
        wrist_joints_only=wrist_joints_only,
        adaptive_wrist=adaptive_action_selector is not None,
    )
    simulator = Simulator(scenario.diagram)
    context = simulator.get_mutable_context()
    plant_context = scenario.plant.GetMyMutableContextFromRoot(context)
    _set_robot_positions(
        plant_context,
        scenario.robot,
        scenario.q_lower,
    )
    for joint in scenario.pgc_joints:
        joint.set_translation(
            plant_context,
            scenario.initial_finger_position_m,
        )
        joint.set_translation_rate(plant_context, 0.0)
    for joint, angle_deg in zip(
        scenario.object_joints,
        initial_angles,
        strict=True,
    ):
        joint.set_angle(
            plant_context,
            _internal_joint_angle_rad(
                object_profile, joint.name(), angle_deg
            ),
        )
        joint.set_angular_rate(plant_context, 0.0)
    x_world_tcp = scenario.plant.EvalBodyPoseInWorld(
        plant_context,
        scenario.robot.payload,
    )
    object_pose = (
        scenario.initial_object_pose
        if grasp_object
        else RigidTransform([4.0, 0.0, 1.0])
    )
    scenario.plant.SetFreeBodyPose(
        plant_context,
        scenario.parent,
        object_pose,
    )
    scenario.plant.SetFreeBodySpatialVelocity(
        plant_context,
        scenario.parent,
        SpatialVelocity.Zero(),
    )
    initial_parent_z = float(
        scenario.plant.EvalBodyPoseInWorld(
            plant_context,
            scenario.grasped_body,
        ).translation()[2]
    )
    initial_tcp_z = float(x_world_tcp.translation()[2])
    initial_joint_angles = np.array(
        [
            joint.get_angle(plant_context)
            for joint in scenario.object_joints
        ]
    )
    simulator.Initialize()
    initial_hold_end_s = (
        scenario.hold_end_times_s[0]
        if scenario.adaptive_wrist
        else min(
            ORIENT_END_S + FOUNDATIONPOSE_INITIAL_HOLD_S,
            scenario.hold_end_times_s[0] - FT_HOLD_WINDOW_S,
        )
    )
    capture_metadata = None
    capture_index = 0
    next_capture_s = 0.0
    capture_period_s = 0.0
    if foundationpose_capture_dir is not None:
        if foundationpose_capture_fps <= 0.0:
            raise ValueError("foundationpose_capture_fps must be positive")
        capture_period_s = 1.0 / foundationpose_capture_fps
        tracking_bodies = (
            (
                scenario.object_bodies[2],
                scenario.object_bodies[0],
                scenario.object_bodies[1],
            )
            if object_profile == "desk_lamp"
            else scenario.object_bodies
        )
        capture_metadata = {
            "schema": "drake-foundationpose-rgbd-sequence-v1",
            "opening_limit_deg": opening_angle_deg,
            "joint_opening_limits_deg": list(opening_limits),
            "joint_angle_bounds_deg": [
                list(bounds) for bounds in opening_bounds
            ],
            "object_profile": object_profile or "generic",
            "initial_opening_deg": initial_opening_angle_deg,
            "initial_opening_angles_deg": list(initial_angles),
            "initial_tracking_hold_end_s": initial_hold_end_s,
            "hold_end_times_s": list(scenario.hold_end_times_s),
            "body_labels": {
                (
                    "parent" if index == 0 else f"child_{index}"
                ): int(body.index())
                for index, body in enumerate(tracking_bodies)
            },
            "mass_order_links": (
                ["child_1", "child_2", "parent"]
                if object_profile == "desk_lamp"
                else [
                    "parent",
                    *(
                        f"child_{index}"
                        for index in range(1, part_count)
                    ),
                ]
            ),
            "cameras": {},
            "frames": [],
        }
    if meshcat is not None:
        meshcat.StartRecording()
    initial_contacts = scenario.plant.get_contact_results_output_port().Eval(
        plant_context
    )
    inspector = scenario.scene_graph.model_inspector()
    initial_contact_count = int(
        initial_contacts.num_point_pair_contacts()
        + initial_contacts.num_hydroelastic_contacts()
    )
    initial_contact_pairs = _contact_pairs(initial_contacts, inspector)
    simulator.set_target_realtime_rate(1.0 if meshcat is not None else 0.0)
    if capture_metadata is not None:
        while next_capture_s <= GRIPPER_CLOSE_END_S + 0.2:
            simulator.AdvanceTo(next_capture_s)
            _save_foundationpose_frame(
                scenario,
                context,
                plant_context,
                foundationpose_capture_dir,
                capture_index,
                next_capture_s,
                capture_metadata,
            )
            capture_index += 1
            next_capture_s += capture_period_s
    simulator.AdvanceTo(GRIPPER_CLOSE_END_S + 0.2)
    finger_positions_after_close = tuple(
        float(joint.get_translation(plant_context))
        for joint in scenario.pgc_joints
    )
    close_contacts = scenario.plant.get_contact_results_output_port().Eval(
        plant_context
    )
    contact_count_after_close = int(
        close_contacts.num_point_pair_contacts()
        + close_contacts.num_hydroelastic_contacts()
    )
    contact_pairs_after_close = _contact_pairs(close_contacts, inspector)
    # Approach, close, lift, and settle before the wrist trajectory.
    dynamic_times = np.append(
        np.linspace(LIFT_START_S + 0.4, LIFT_END_S - 0.1, 4),
        ORIENT_START_S - 0.2,
    )
    lift_sample_end_s = float(dynamic_times[-1])
    dynamic_lift_measurements = []
    dynamic_index = 0
    while dynamic_index < len(dynamic_times) or (
        capture_metadata is not None and next_capture_s <= lift_sample_end_s
    ):
        dynamic_time = (
            dynamic_times[dynamic_index]
            if dynamic_index < len(dynamic_times)
            else math.inf
        )
        capture_time = (
            next_capture_s
            if capture_metadata is not None
            and next_capture_s <= lift_sample_end_s
            else math.inf
        )
        event_time = min(dynamic_time, capture_time)
        simulator.AdvanceTo(float(event_time))
        if capture_time <= dynamic_time:
            _save_foundationpose_frame(
                scenario,
                context,
                plant_context,
                foundationpose_capture_dir,
                capture_index,
                capture_time,
                capture_metadata,
            )
            capture_index += 1
            next_capture_s += capture_period_s
        if dynamic_time <= capture_time:
            dynamic_lift_measurements.append(
                _sample_dynamic_lift(
                    scenario,
                    plant_context,
                    float(dynamic_time),
                    inspector,
                )
            )
            dynamic_index += 1
    simulator.AdvanceTo(lift_sample_end_s)
    if capture_metadata is not None:
        capture_limit_s = (
            initial_hold_end_s - FT_HOLD_WINDOW_S
            if scenario.adaptive_wrist
            else initial_hold_end_s
        )
        while next_capture_s <= capture_limit_s:
            simulator.AdvanceTo(next_capture_s)
            _save_foundationpose_frame(
                scenario,
                context,
                plant_context,
                foundationpose_capture_dir,
                capture_index,
                next_capture_s,
                capture_metadata,
            )
            capture_index += 1
            next_capture_s += capture_period_s
    if capture_metadata is None:
        for dynamic_time in scenario.orient_dynamic_times_s:
            simulator.AdvanceTo(dynamic_time)
            dynamic_lift_measurements.append(
                _sample_dynamic_lift(
                    scenario,
                    plant_context,
                    dynamic_time,
                    inspector,
                )
            )
    adaptive_actions = []
    if scenario.adaptive_wrist:
        holds = [
            _sample_static_hold(
                simulator,
                scenario,
                plant_context,
                initial_hold_end_s,
                0.0,
            )
        ]
        current_pitch_roll = (0.0, 0.0)
        visited = {current_pitch_roll}
        while True:
            candidates = (
                _adaptive_wrist_candidates(
                    scenario,
                    context,
                    holds[-1],
                    current_pitch_roll,
                    visited,
                    adaptive_collision_clearance_m,
                )
                if len(holds) < adaptive_max_holds
                else []
            )
            public_candidates = [
                {
                    key: (
                        value.tolist()
                        if isinstance(value, np.ndarray)
                        else value
                    )
                    for key, value in candidate.items()
                    if key != "_path"
                }
                for candidate in candidates
            ]
            selected_index = adaptive_action_selector(
                tuple(holds),
                tuple(public_candidates),
            )
            if selected_index is None:
                break
            if not 0 <= int(selected_index) < len(candidates):
                raise RuntimeError("adaptive selector returned invalid action")
            selected = candidates[int(selected_index)]
            if not selected["safe"]:
                raise RuntimeError("adaptive selector returned unsafe action")
            current_time = simulator.get_context().get_time()
            rotation_start_s = current_time + 0.2
            rotation_end_s = (
                rotation_start_s + ADAPTIVE_WRIST_ROTATION_DURATION_S
            )
            path = selected["_path"]
            current_q = np.array(
                [
                    joint.get_angle(plant_context)
                    for joint in scenario.robot.joints
                ]
            )
            rotation_times = np.linspace(
                rotation_start_s,
                rotation_end_s,
                len(path),
            )
            settle_duration_s = (
                DESK_LAMP_SETTLE_DURATION_S
                if object_profile in ("desk_lamp", "phantom_v3")
                else 4.0
            )
            hold_end_s = (
                rotation_end_s + settle_duration_s + FT_HOLD_WINDOW_S
            )
            scenario.robot_source.set_trajectory(
                np.asarray(
                    [
                        current_time,
                        rotation_start_s,
                        *rotation_times[1:],
                        hold_end_s,
                    ]
                ),
                np.asarray(
                    [
                        current_q,
                        current_q,
                        *path[1:],
                        path[-1],
                    ]
                ),
            )
            for dynamic_time in (
                rotation_start_s
                + ADAPTIVE_WRIST_ROTATION_DURATION_S
                * np.array((0.25, 0.5, 0.75))
            ):
                simulator.AdvanceTo(float(dynamic_time))
                dynamic_lift_measurements.append(
                    _sample_dynamic_lift(
                        scenario,
                        plant_context,
                        float(dynamic_time),
                        inspector,
                    )
                )
            holds.append(
                _sample_static_hold(
                    simulator,
                    scenario,
                    plant_context,
                    hold_end_s,
                    float(selected["wrist_pitch_deg"]),
                )
            )
            current_pitch_roll = (
                float(selected["wrist_pitch_deg"]),
                float(selected["wrist_roll_deg"]),
            )
            visited.add(current_pitch_roll)
            adaptive_actions.append(
                {
                    "after_hold": len(holds) - 1,
                    "selected_index": int(selected_index),
                    "selected": public_candidates[int(selected_index)],
                    "candidates": public_candidates,
                }
            )
    else:
        simulator.AdvanceTo(initial_hold_end_s)
        pitch_sequence = (
            wrist_pitch_sequence_deg
            if wrist_pitch_sequence_deg is not None
            else (wrist_pitch_deg,)
        )
        holds = []
        for hold_end, pitch in zip(
            scenario.hold_end_times_s,
            pitch_sequence,
            strict=True,
        ):
            hold_index = len(holds)
            if capture_metadata is None:
                for dynamic_time in scenario.wrist_dynamic_times_s[hold_index]:
                    simulator.AdvanceTo(dynamic_time)
                    dynamic_lift_measurements.append(
                        _sample_dynamic_lift(
                            scenario,
                            plant_context,
                            dynamic_time,
                            inspector,
                        )
                    )
            if capture_metadata is not None:
                while (
                    next_capture_s
                    < hold_end - FT_HOLD_WINDOW_S - 1e-6
                ):
                    frame_time = max(
                        next_capture_s,
                        simulator.get_context().get_time(),
                    )
                    simulator.AdvanceTo(frame_time)
                    _save_foundationpose_frame(
                        scenario,
                        context,
                        plant_context,
                        foundationpose_capture_dir,
                        capture_index,
                        frame_time,
                        capture_metadata,
                    )
                    capture_index += 1
                    next_capture_s = frame_time + capture_period_s
            holds.append(
                _sample_static_hold(
                    simulator,
                    scenario,
                    plant_context,
                    hold_end,
                    pitch,
                )
            )
            if capture_metadata is not None:
                _save_foundationpose_frame(
                    scenario,
                    context,
                    plant_context,
                    foundationpose_capture_dir,
                    capture_index,
                    hold_end,
                    capture_metadata,
                )
                capture_index += 1
                next_capture_s = hold_end + capture_period_s
    holds = tuple(holds)
    final_hold = holds[-1]
    grasp_reference = RigidTransform(np.asarray(holds[0].grasp_relative_pose))
    final_parent_pose = scenario.plant.EvalBodyPoseInWorld(
        plant_context,
        scenario.grasped_body,
    )
    final_relative = scenario.plant.CalcRelativeTransform(
        plant_context,
        scenario.robot.payload.body_frame(),
        scenario.grasped_body.body_frame(),
    )
    delta_relative = grasp_reference.inverse() @ final_relative
    final_joint_angles = np.array(
        [
            joint.get_angle(plant_context)
            for joint in scenario.object_joints
        ]
    )
    sensor_force = np.asarray(final_hold.ft_reaction_force)
    sensor_torque = np.asarray(final_hold.ft_reaction_torque)
    sensor_std = np.asarray(final_hold.ft_sample_std)
    sensor_mean_standard_error = np.asarray(
        final_hold.ft_mean_standard_error
    )
    sensor_block_means = np.asarray(final_hold.ft_block_means)
    sensor_inliers = final_hold.ft_inlier_count
    sensor_stationary = final_hold.ft_stationary
    sensor_joint = scenario.plant.get_joint(
        JointIndex(scenario.sensor_mount_joint_index)
    )
    x_world_sensor = scenario.plant.CalcRelativeTransform(
        plant_context,
        scenario.plant.world_frame(),
        sensor_joint.frame_on_child(),
    )
    x_sensor_world = x_world_sensor.inverse()
    gravity_sensor = x_world_sensor.rotation().inverse().multiply(
        np.array([0.0, 0.0, -9.81])
    )
    part_com_sensor = []
    for body in scenario.object_bodies:
        x_world_body = scenario.plant.EvalBodyPoseInWorld(
            plant_context,
            body,
        )
        com_world = x_world_body.multiply(
            body.default_spatial_inertia().get_com()
        )
        part_com_sensor.append(
            tuple(float(value) for value in x_sensor_world.multiply(com_world))
        )
    first_joint = scenario.object_joints[0]
    x_world_joint = scenario.plant.CalcRelativeTransform(
        plant_context,
        scenario.plant.world_frame(),
        first_joint.frame_on_parent(),
    )
    joint_origin_sensor = x_sensor_world.multiply(
        x_world_joint.translation()
    )
    joint_axis_sensor = (
        x_world_sensor.rotation().inverse().matrix()
        @ x_world_joint.rotation().matrix()
        @ np.array([0.0, 1.0, 0.0])
    )
    contact_results = scenario.plant.get_contact_results_output_port().Eval(
        plant_context
    )
    if foundationpose_capture_dir is not None:
        (foundationpose_capture_dir / "metadata.json").write_text(
            json.dumps(capture_metadata, indent=2) + "\n",
            encoding="utf-8",
        )
    translation_drift = float(
        np.linalg.norm(delta_relative.translation())
    )
    rotation_drift = _rotation_angle_deg(delta_relative.rotation())
    joint_drift = float(
        np.max(np.abs(final_joint_angles - initial_joint_angles))
        if final_joint_angles.size
        else 0.0
    )
    parent_lift = float(final_parent_pose.translation()[2] - initial_parent_z)
    final_tcp_z = float(
        scenario.plant.EvalBodyPoseInWorld(
            plant_context,
            scenario.robot.payload,
        ).translation()[2]
    )
    tcp_lift = final_tcp_z - initial_tcp_z
    final_contact_count = int(
        contact_results.num_point_pair_contacts()
        + contact_results.num_hydroelastic_contacts()
    )
    result = ContactFtResult(
        part_count=part_count,
        grasp_offset_m=grasp_offset_m,
        simulation_end_s=float(simulator.get_context().get_time()),
        parent_lift_m=parent_lift,
        tcp_lift_m=tcp_lift,
        parent_tcp_lift_ratio=parent_lift / max(tcp_lift, 1e-9),
        finger_positions_after_close_m=finger_positions_after_close,
        initial_contact_count=initial_contact_count,
        initial_contact_pairs=initial_contact_pairs,
        contact_count_after_close=contact_count_after_close,
        contact_pairs_after_close=contact_pairs_after_close,
        grasp_relative_translation_drift_m=translation_drift,
        grasp_relative_rotation_drift_deg=rotation_drift,
        maximum_internal_joint_drift_deg=math.degrees(joint_drift),
        final_finger_positions_m=tuple(
            float(joint.get_translation(plant_context))
            for joint in scenario.pgc_joints
        ),
        final_contact_count=final_contact_count,
        ft_reaction_force=tuple(
            float(value) for value in sensor_force
        ),
        ft_reaction_torque=tuple(
            float(value) for value in sensor_torque
        ),
        ft_sample_std=tuple(float(value) for value in sensor_std),
        ft_mean_standard_error=tuple(
            float(value) for value in sensor_mean_standard_error
        ),
        ft_block_means=tuple(
            tuple(float(value) for value in block)
            for block in sensor_block_means
        ),
        ft_raw_sample_count=FT_HOLD_SAMPLE_COUNT,
        ft_inlier_count=sensor_inliers,
        ft_stationary=sensor_stationary,
        actual_opening_angles_deg=tuple(
            (
                (
                    -math.degrees(value)
                    - (
                        DESK_LAMP_LOWER_ZERO_OFFSET_DEG
                        if "lower" in joint.name()
                        else 0.0
                    )
                )
                if object_profile == "desk_lamp"
                else -math.degrees(value)
                if object_profile == "jewelry_box"
                else math.degrees(value)
                if object_profile == "phantom_v3"
                else 180.0 + math.degrees(value)
            )
            for joint, value in zip(
                scenario.object_joints,
                final_joint_angles,
                strict=True,
            )
        ),
        gravity_sensor_m_s2=tuple(float(value) for value in gravity_sensor),
        part_com_sensor_m=tuple(part_com_sensor),
        part_com_offsets_body_m=tuple(
            tuple(
                float(value)
                for value in body.default_spatial_inertia().get_com()
            )
            for body in scenario.object_bodies
        ),
        joint_origin_sensor_m=tuple(
            float(value) for value in joint_origin_sensor
        ),
        joint_axis_sensor=tuple(float(value) for value in joint_axis_sensor),
        dynamic_lift_measurements=tuple(dynamic_lift_measurements),
        holds=holds,
        adaptive_actions=tuple(adaptive_actions),
        passed_lift=(
            parent_lift >= 0.08 and final_contact_count >= 2
        ),
        passed_grasp_translation=(
            translation_drift <= MAX_GRASP_TRANSLATION_DRIFT_M
        ),
        passed_grasp_rotation=(
            rotation_drift <= MAX_GRASP_ROTATION_DRIFT_DEG
        ),
        passed_joint_stability=(
            free_hinges or math.degrees(joint_drift) <= 1.5
        ),
    )
    if meshcat is not None:
        meshcat.StopRecording()
        meshcat.PublishRecording()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part-count", type=int, choices=(2, 3, 4), default=2)
    parser.add_argument("--grasp-offset-m", type=float)
    parser.add_argument("--opening-angle-deg", type=float, default=180.0)
    parser.add_argument("--initial-opening-angle-deg", type=float, default=180.0)
    parser.add_argument(
        "--initial-opening-angles-deg",
        type=float,
        nargs="+",
    )
    parser.add_argument("--initial-wrist-pitch-deg", type=float, default=0.0)
    parser.add_argument("--free-hinges", action="store_true")
    parser.add_argument(
        "--part-com-offsets-body-m",
        type=float,
        nargs="+",
        help="body-frame COM triples, one x y z triple per part",
    )
    parser.add_argument(
        "--object-profile",
        choices=tuple(HOUSEHOLD_PROFILES),
    )
    parser.add_argument(
        "--wrist-pitch-sequence-deg",
        type=float,
        nargs="+",
    )
    parser.add_argument(
        "--wrist-roll-sequence-deg",
        type=float,
        nargs="+",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--wrist-joints-only", action="store_true")
    args = parser.parse_args()
    meshcat = StartMeshcat() if args.live else None
    result = simulate(
        args.part_count,
        meshcat=meshcat,
        grasp_offset_m=args.grasp_offset_m,
        opening_angle_deg=args.opening_angle_deg,
        initial_opening_angle_deg=args.initial_opening_angle_deg,
        initial_opening_angles_deg=(
            tuple(args.initial_opening_angles_deg)
            if args.initial_opening_angles_deg
            else None
        ),
        initial_wrist_pitch_deg=args.initial_wrist_pitch_deg,
        wrist_pitch_sequence_deg=(
            tuple(args.wrist_pitch_sequence_deg)
            if args.wrist_pitch_sequence_deg
            else None
        ),
        wrist_roll_sequence_deg=(
            tuple(args.wrist_roll_sequence_deg)
            if args.wrist_roll_sequence_deg
            else None
        ),
        free_hinges=args.free_hinges,
        object_profile=args.object_profile,
        wrist_joints_only=args.wrist_joints_only,
        part_com_offsets_body_m=(
            tuple(
                tuple(args.part_com_offsets_body_m[index : index + 3])
                for index in range(0, len(args.part_com_offsets_body_m), 3)
            )
            if args.part_com_offsets_body_m
            else None
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                **asdict(result),
                "passed": result.passed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({**asdict(result), "passed": result.passed}, indent=2))
    if meshcat is not None:
        print(f"Meshcat: {meshcat.web_url()}", flush=True)
        input("Press Enter to close the contact simulation... ")


if __name__ == "__main__":
    main()
