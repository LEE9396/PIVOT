#!/usr/bin/env python3
"""Run the latest-HTD discrete RB5 hammer inertia-sensitivity gate."""

import argparse
from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np
from pydrake.all import (
    AddCompliantHydroelasticProperties,
    AddContactMaterial,
    AddMultibodyPlantSceneGraph,
    Box,
    CoulombFriction,
    Cylinder,
    DiagramBuilder,
    DiscreteContactApproximation,
    InverseDynamics,
    JointActuatorIndex,
    Mesh,
    MeshcatAnimation,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    ModelInstanceIndex,
    MultibodyPlant,
    Parser,
    PdControllerGains,
    PiecewisePolynomial,
    ProximityProperties,
    Quaternion,
    RevoluteJoint,
    RigidBody,
    RigidTransform,
    Role,
    RotationalInertia,
    RotationMatrix,
    Simulator,
    SpatialInertia,
    StartMeshcat,
    TrajectorySource,
    UnitInertia,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HTD_ROOT = WORKSPACE_ROOT / "third_party/HTD"
HAMMER_ASSET_ROOT = (
    Path(__file__).resolve().parents[1] / "assets/modular_hammer_proxy_v0"
)
HAMMER_MANIFEST = HAMMER_ASSET_ROOT / "manifest.json"
HTD_COMMIT = "59b41ed98e822a69ea8839896de0513d534e8757"
HTD_SCENE_RELATIVE = Path("drake_rb5_scene.py")
HTD_SCENE_SHA256 = (
    "56adb54d6e0aa54146dfc49183524725dd8cda6222b3eedfc2934b69eb27de25"
)
RB5_URDF_SHA256 = (
    "38c357aaca3ea0ee2284eeb673920579b40ffaed1a40b5b77f02087516487598"
)
RB5_DESCRIPTION_RELATIVE = Path("assets/rbpodo_description")
RB5_URDF_RELATIVE = RB5_DESCRIPTION_RELATIVE / "robots/rb5_850e.urdf"
PGC_URDF_RELATIVE = Path("assets/pgc_140_50/pgc_140_50.urdf")
PGC_URDF_SHA256 = (
    "57e877069542fb11fbb0286a6ba549e07ef251bad96f7d0f3bf9ef89eba8f44d"
)
DRAKE_ASSETS_RELATIVE = Path("assets/drake")
RB5_COLLISION_RELATIVE = DRAKE_ASSETS_RELATIVE / "rb5_collision"
RB5_VISUAL_RELATIVE = DRAKE_ASSETS_RELATIVE / "rb5_visual"
RB5_MESH_LINK_NAMES = tuple(f"link{index}.obj" for index in range(7))
PGC_VISUAL_OVERRIDES = {
    "base_link.STL": "pgc_base_link.obj",
    "finger1_Link.STL": "pgc_finger1.obj",
    "finger2_Link.STL": "pgc_finger2.obj",
}

ARM_JOINT_NAMES = (
    "base",
    "shoulder",
    "elbow",
    "wrist1",
    "wrist2",
    "wrist3",
)
INITIAL_ARM_POSITION_RAD = np.array([0.0, -0.65, 1.35, -0.70, 0.0, 0.0])
PLANT_TIME_STEP_S = 0.001
CONTACT_PENETRATION_ALLOWANCE_M = 0.0001
ARM_KP = 200.0
ARM_KD = 30.0
ARM_MAX_TORQUE_NM = 80.0
COLLISION_GEOMETRIES_PER_ARM = 16

# HALDER SIMPLEX 3029.040-inspired four-part proxy. The two insert masses
# follow the nominal 3202.040 (65 g) and 3209.040 (120 g) data. The handle
# and housing split is a pre-measurement proxy whose sum closes the nominal
# 700 g mixed assembly; replace both values after destructive GT measurement.
HANDLE_MASS_KG = 0.160
HOUSING_MASS_KG = 0.355
LIGHT_INSERT_MASS_KG = 0.065
HEAVY_INSERT_MASS_KG = 0.120

HANDLE_RADIUS_M = 0.014
HANDLE_LENGTH_M = 0.290
HANDLE_CENTER_H_M = np.array([0.0, 0.0, HANDLE_LENGTH_M / 2.0])
HEAD_CENTER_Z_M = 0.305
HOUSING_RADIUS_M = 0.024
HOUSING_LENGTH_M = 0.050
INSERT_RADIUS_M = 0.020
INSERT_LENGTH_M = 0.030
INSERT_CENTER_X_M = (HOUSING_LENGTH_M + INSERT_LENGTH_M) / 2.0
HAMMER_REFERENCE_POINT_H_M = np.array(
    [
        HOUSING_LENGTH_M / 2.0 + INSERT_LENGTH_M,
        0.0,
        HEAD_CENTER_Z_M,
    ]
)

HANDLE_COLOR = (0.64, 0.38, 0.17, 1.0)
HOUSING_COLOR = (0.18, 0.20, 0.23, 1.0)
LIGHT_INSERT_COLOR = (0.035, 0.04, 0.045, 1.0)
HEAVY_INSERT_COLOR = (0.72, 0.75, 0.79, 1.0)

AFT200_DIAMETER_M = 0.080
AFT200_SENSOR_HEIGHT_M = 0.0215
AFT200_BRACKET_HEIGHT_M = 0.0523 - AFT200_SENSOR_HEIGHT_M
AFT200_SENSOR_MASS_KG = 0.236
AFT200_BRACKET_MASS_KG = 0.399
AFT200_SENSOR_COM_M = np.array([0.0009204, 0.0007747, 0.0003665])
AFT200_SENSOR_INERTIA_KG_M2 = (9.43117e-05, 1.538887e-04, 2.314693e-04)
AFT200_SENSOR_PRINCIPAL_AXES = (-0.3936979, 0.0012072, 0.0005299, 0.9192389)
PGC_BASE_MASS_KG = 0.971444
PGC_FINGER_MASS_KG = 0.014278
PGC_ASSEMBLY_MASS_KG = PGC_BASE_MASS_KG + 2.0 * PGC_FINGER_MASS_KG
PGC_TCP_Z_M = 0.125
PGC_BASE_COLLISION_SIZE_M = np.array([0.0751, 0.0751, 0.100])
PGC_FINGER_COLLISION_SIZE_M = np.array([0.0115, 0.020, 0.0565])
PGC_FINGER_COLLISION_CENTER_M = np.array([-0.00425, -0.0073, 0.01225])
GRIPPER_FRICTION = CoulombFriction(1.5, 1.2)
MOVABLE_PAD_FRICTION = CoulombFriction(3.0, 2.5)
PAYLOAD_FRICTION = CoulombFriction(0.8, 0.6)
END_EFFECTOR_MASS_KG = (
    AFT200_SENSOR_MASS_KG + AFT200_BRACKET_MASS_KG + PGC_ASSEMBLY_MASS_KG
)

ANIMATION_END_S = 4.20
HOLD_CHECK_TIME_S = 0.19
DESIRED_WRIST_BREAKS_S = (0.00, 0.20, 1.20, 2.20, 3.20, ANIMATION_END_S)
DESIRED_WRIST_TARGETS_RAD = (0.00, 0.90, -0.90, 0.90, 0.00, 0.00)
SAMPLE_TIMES_S = (
    HOLD_CHECK_TIME_S,
    0.20,
    0.30,
    0.40,
    0.60,
    1.00,
    1.20,
    1.225,
    2.40,
    3.40,
    4.20,
)
TRAJECTORY_SAMPLE_TIMES_S = tuple(
    round(float(value), 9)
    for value in np.linspace(0.0, ANIMATION_END_S, 169)
)
SENSITIVITY_TIME_S = 1.225
STOP_BUTTON = "Stop RB5 hammer demo"

VARIANT_DISPLAY = (
    ("경량–경량 · 645 g", "#0072B2"),
    ("중량–중량 · 755 g", "#6A3D9A"),
    ("우측 중량 · 700 g", "#D55E00"),
)


@dataclass(frozen=True)
class VariantSpec:
    name: str
    left_insert_mass_kg: float
    right_insert_mass_kg: float
    base_y_m: float


@dataclass(frozen=True)
class HammerAssembly:
    spatial_inertia: SpatialInertia
    mass_kg: float
    com_h_m: tuple[float, float, float]
    central_inertia_h_kg_m2: tuple[tuple[float, float, float], ...]
    part_masses_kg: tuple[float, float, float, float]
    effective_densities_kg_m3: tuple[float, float, float, float]


VARIANTS = (
    VariantSpec(
        "light_light_symmetric",
        LIGHT_INSERT_MASS_KG,
        LIGHT_INSERT_MASS_KG,
        -1.25,
    ),
    VariantSpec(
        "heavy_heavy_symmetric",
        HEAVY_INSERT_MASS_KG,
        HEAVY_INSERT_MASS_KG,
        0.00,
    ),
    VariantSpec(
        "right_heavy_asymmetric",
        LIGHT_INSERT_MASS_KG,
        HEAVY_INSERT_MASS_KG,
        1.25,
    ),
)


@dataclass(frozen=True)
class ArmRecord:
    spec: VariantSpec
    assembly: HammerAssembly
    model_instance: ModelInstanceIndex
    aft_model_instance: ModelInstanceIndex
    pgc_model_instance: ModelInstanceIndex
    joints: tuple[RevoluteJoint, ...]
    link6: RigidBody
    payload: RigidBody


@dataclass(frozen=True)
class SimulationResult:
    plant_time_step_s: float
    actuator_count: int
    actuator_effort_limits_nm: tuple[float, ...]
    actuator_pd_gains: tuple[tuple[float, float], ...]
    collision_geometry_count: int
    max_collision_penetration_m: float
    rb5_position_counts: tuple[int, ...]
    pgc_position_counts: tuple[int, ...]
    end_effector_masses_kg: tuple[float, ...]
    task_axis_alignments: tuple[float, ...]
    max_hold_errors_rad: tuple[float, ...]
    hammer_masses_kg: tuple[float, ...]
    hammer_coms_h_m: tuple[tuple[float, float, float], ...]
    hammer_task_inertias_kg_m2: tuple[float, ...]
    hammer_ixz_kg_m2: tuple[float, ...]
    wrist_joint_inertias_kg_m2: tuple[float, ...]
    wrist_angles_by_time_rad: dict[float, tuple[float, ...]]
    hammer_tips_at_sensitivity_m: tuple[tuple[float, float, float], ...]
    trajectory_times_s: tuple[float, ...]
    wrist_trajectories_rad: tuple[tuple[float, ...], ...]
    hammer_reference_trajectories_m: tuple[
        tuple[tuple[float, float, float], ...], ...
    ]


def validate_htd_source(htd_root: Path) -> tuple[Path, Path, Path, Path]:
    htd_root = htd_root.resolve()
    rb5_description = htd_root / RB5_DESCRIPTION_RELATIVE
    urdf_path = htd_root / RB5_URDF_RELATIVE
    pgc_urdf_path = htd_root / PGC_URDF_RELATIVE
    drake_assets = htd_root / DRAKE_ASSETS_RELATIVE
    scene_path = htd_root / HTD_SCENE_RELATIVE
    if not scene_path.is_file():
        raise FileNotFoundError(f"HTD Drake scene not found: {scene_path}")
    actual_scene_hash = hashlib.sha256(scene_path.read_bytes()).hexdigest()
    if actual_scene_hash != HTD_SCENE_SHA256:
        raise RuntimeError(
            "HTD Drake scene hash mismatch: "
            f"expected {HTD_SCENE_SHA256}, got {actual_scene_hash}"
        )
    if not urdf_path.is_file():
        raise FileNotFoundError(
            f"HTD RB5 URDF not found: {urdf_path}. "
            "Clone https://github.com/Yuseong-Cheon/HTD at "
            f"commit {HTD_COMMIT}."
        )
    actual_hash = hashlib.sha256(urdf_path.read_bytes()).hexdigest()
    if actual_hash != RB5_URDF_SHA256:
        raise RuntimeError(
            "HTD RB5 URDF hash mismatch: "
            f"expected {RB5_URDF_SHA256}, got {actual_hash}"
        )
    if not pgc_urdf_path.is_file():
        raise FileNotFoundError(f"HTD PGC URDF not found: {pgc_urdf_path}")
    actual_pgc_hash = hashlib.sha256(pgc_urdf_path.read_bytes()).hexdigest()
    if actual_pgc_hash != PGC_URDF_SHA256:
        raise RuntimeError(
            "HTD PGC URDF hash mismatch: "
            f"expected {PGC_URDF_SHA256}, got {actual_pgc_hash}"
        )
    for mesh_name in PGC_VISUAL_OVERRIDES.values():
        mesh_path = drake_assets / mesh_name
        if not mesh_path.is_file():
            raise FileNotFoundError(f"HTD converted PGC mesh not found: {mesh_path}")
    for mesh_directory in (RB5_COLLISION_RELATIVE, RB5_VISUAL_RELATIVE):
        for mesh_name in RB5_MESH_LINK_NAMES:
            mesh_path = htd_root / mesh_directory / mesh_name
            if not mesh_path.is_file():
                raise FileNotFoundError(f"HTD converted RB5 mesh not found: {mesh_path}")
    aft_visual = drake_assets / "aft200_visual.obj"
    if not aft_visual.is_file():
        raise FileNotFoundError(f"HTD AFT200 visual not found: {aft_visual}")
    return rb5_description, urdf_path, pgc_urdf_path, drake_assets


def rb5_urdf_string(
    urdf_path: Path,
    model_name: str,
    include_visuals: bool,
) -> str:
    """Return the pinned RB5 model with HTD's Drake-ready OBJ geometry."""
    root = ET.parse(urdf_path).getroot()
    root.set("name", model_name)
    for link in root.findall("link"):
        if not include_visuals:
            for visual in list(link.findall("visual")):
                link.remove(visual)
    for mesh in root.iter("mesh"):
        raw = mesh.get("filename")
        mesh_name = f"{Path(raw).stem}.obj"
        if "/collision/" in raw:
            mesh.set(
                "filename",
                f"package://htd/{RB5_COLLISION_RELATIVE.as_posix()}/{mesh_name}",
            )
        elif "/visual/" in raw:
            mesh.set(
                "filename",
                f"package://htd/{RB5_VISUAL_RELATIVE.as_posix()}/{mesh_name}",
            )
    for ros2_control in list(root.findall("ros2_control")):
        root.remove(ros2_control)
    return ET.tostring(root, encoding="unicode")


def pgc_fixed_urdf_string(
    pgc_urdf_path: Path,
    model_name: str,
    include_visuals: bool,
    joint_position_m: float = 0.0,
) -> str:
    """Return the PGC gripper as a rigid closed-jaw assembly."""
    if not 0.0 <= joint_position_m <= 0.025:
        raise ValueError("PGC joint_position_m must be within [0, 0.025]")
    root = ET.parse(pgc_urdf_path).getroot()
    root.set("name", model_name)
    for link in root.findall("link"):
        for collision in list(link.findall("collision")):
            link.remove(collision)
        if not include_visuals:
            for visual in list(link.findall("visual")):
                link.remove(visual)
    for joint in root.findall("joint"):
        if joint_position_m > 0.0:
            origin = joint.find("origin")
            if origin is None:
                raise ValueError(f"PGC joint {joint.get('name')} has no origin")
            xyz = np.fromstring(origin.get("xyz"), sep=" ")
            if joint.get("name") == "finger1_joint":
                xyz[1] -= joint_position_m
            elif joint.get("name") == "finger2_joint":
                xyz[1] += joint_position_m
            origin.set("xyz", " ".join(f"{value:.9g}" for value in xyz))
        joint.set("type", "fixed")
        for child_name in ("axis", "limit", "mimic"):
            for child in list(joint.findall(child_name)):
                joint.remove(child)
    if include_visuals:
        for mesh in root.iter("mesh"):
            source_name = Path(mesh.get("filename")).name
            target_name = PGC_VISUAL_OVERRIDES[source_name]
            mesh.set(
                "filename",
                f"package://htd/assets/drake/{target_name}",
            )
    return ET.tostring(root, encoding="unicode")


def pgc_movable_urdf_string(
    pgc_urdf_path: Path,
    model_name: str,
    include_visuals: bool,
) -> str:
    """Return PGC with independent movable finger joints for contact tests."""
    root = ET.parse(pgc_urdf_path).getroot()
    root.set("name", model_name)
    for link in root.findall("link"):
        for collision in list(link.findall("collision")):
            link.remove(collision)
        if not include_visuals:
            for visual in list(link.findall("visual")):
                link.remove(visual)
    for joint in root.findall("joint"):
        for mimic in list(joint.findall("mimic")):
            joint.remove(mimic)
    if include_visuals:
        for mesh in root.iter("mesh"):
            source_name = Path(mesh.get("filename")).name
            target_name = PGC_VISUAL_OVERRIDES[source_name]
            mesh.set(
                "filename",
                f"package://htd/assets/drake/{target_name}",
            )
    return ET.tostring(root, encoding="unicode")


def cylinder_volume(radius_m: float, length_m: float) -> float:
    return float(np.pi * radius_m**2 * length_m)


def cylinder_central_inertia(
    mass_kg: float,
    radius_m: float,
    length_m: float,
    axis: str,
) -> RotationalInertia:
    axial = 0.5 * mass_kg * radius_m**2
    transverse = mass_kg * (3.0 * radius_m**2 + length_m**2) / 12.0
    if axis == "x":
        return RotationalInertia(axial, transverse, transverse)
    if axis == "z":
        return RotationalInertia(transverse, transverse, axial)
    raise ValueError(f"Unsupported cylinder axis: {axis}")


def cylinder_part_spatial_inertia(
    mass_kg: float,
    com_h_m: np.ndarray,
    radius_m: float,
    length_m: float,
    axis: str,
) -> SpatialInertia:
    return SpatialInertia.MakeFromCentralInertia(
        mass_kg,
        com_h_m,
        cylinder_central_inertia(mass_kg, radius_m, length_m, axis),
    )


def make_hammer_assembly(spec: VariantSpec) -> HammerAssembly:
    part_masses = (
        HANDLE_MASS_KG,
        HOUSING_MASS_KG,
        spec.left_insert_mass_kg,
        spec.right_insert_mass_kg,
    )
    part_inertias = (
        cylinder_part_spatial_inertia(
            HANDLE_MASS_KG,
            HANDLE_CENTER_H_M,
            HANDLE_RADIUS_M,
            HANDLE_LENGTH_M,
            "z",
        ),
        cylinder_part_spatial_inertia(
            HOUSING_MASS_KG,
            np.array([0.0, 0.0, HEAD_CENTER_Z_M]),
            HOUSING_RADIUS_M,
            HOUSING_LENGTH_M,
            "x",
        ),
        cylinder_part_spatial_inertia(
            spec.left_insert_mass_kg,
            np.array([-INSERT_CENTER_X_M, 0.0, HEAD_CENTER_Z_M]),
            INSERT_RADIUS_M,
            INSERT_LENGTH_M,
            "x",
        ),
        cylinder_part_spatial_inertia(
            spec.right_insert_mass_kg,
            np.array([INSERT_CENTER_X_M, 0.0, HEAD_CENTER_Z_M]),
            INSERT_RADIUS_M,
            INSERT_LENGTH_M,
            "x",
        ),
    )
    total = part_inertias[0]
    for part_inertia in part_inertias[1:]:
        total += part_inertia
    if not total.IsPhysicallyValid():
        raise ValueError(f"Invalid composed hammer inertia for {spec.name}")

    com_h_m = np.asarray(total.get_com())
    central_inertia = (
        total.Shift(com_h_m)
        .CalcRotationalInertia()
        .CopyToFullMatrix3()
    )
    part_volumes = (
        cylinder_volume(HANDLE_RADIUS_M, HANDLE_LENGTH_M),
        cylinder_volume(HOUSING_RADIUS_M, HOUSING_LENGTH_M),
        cylinder_volume(INSERT_RADIUS_M, INSERT_LENGTH_M),
        cylinder_volume(INSERT_RADIUS_M, INSERT_LENGTH_M),
    )
    return HammerAssembly(
        spatial_inertia=total,
        mass_kg=float(total.get_mass()),
        com_h_m=tuple(float(value) for value in com_h_m),
        central_inertia_h_kg_m2=tuple(
            tuple(float(value) for value in row) for row in central_inertia
        ),
        part_masses_kg=part_masses,
        effective_densities_kg_m3=tuple(
            mass / volume
            for mass, volume in zip(part_masses, part_volumes, strict=True)
        ),
    )


def validate_hammer_manifest() -> None:
    if not HAMMER_MANIFEST.is_file():
        raise FileNotFoundError(f"Hammer manifest not found: {HAMMER_MANIFEST}")
    manifest = json.loads(HAMMER_MANIFEST.read_text(encoding="utf-8"))
    for mesh_relative in ("meshes/handle.obj", "meshes/housing.obj", "meshes/insert.obj"):
        mesh_path = HAMMER_ASSET_ROOT / mesh_relative
        if not mesh_path.is_file() or mesh_path.stat().st_size == 0:
            raise FileNotFoundError(f"Hammer part mesh not found: {mesh_path}")
    for spec in VARIANTS:
        assembly = make_hammer_assembly(spec)
        stored = manifest["configurations"][spec.name]
        if not np.isclose(stored["mass_kg"], assembly.mass_kg, atol=1e-9):
            raise RuntimeError(f"Manifest mass drift for {spec.name}")
        if not np.allclose(stored["com_H_m"], assembly.com_h_m, atol=1e-9):
            raise RuntimeError(f"Manifest CoM drift for {spec.name}")
        if not np.allclose(
            stored["central_inertia_H_kg_m2"],
            assembly.central_inertia_h_kg_m2,
            atol=1e-9,
        ):
            raise RuntimeError(f"Manifest inertia drift for {spec.name}")


def insert_color(insert_mass_kg: float) -> np.ndarray:
    if np.isclose(insert_mass_kg, LIGHT_INSERT_MASS_KG):
        return np.asarray(LIGHT_INSERT_COLOR)
    if np.isclose(insert_mass_kg, HEAVY_INSERT_MASS_KG):
        return np.asarray(HEAVY_INSERT_COLOR)
    raise ValueError(f"Unknown insert mass: {insert_mass_kg}")


def register_modular_hammer_visuals(
    plant: MultibodyPlant,
    hammer: RigidBody,
    spec: VariantSpec,
) -> None:
    head_rotation = RotationMatrix.MakeYRotation(np.pi / 2.0)
    plant.RegisterVisualGeometry(
        hammer,
        RigidTransform(HANDLE_CENTER_H_M),
        Cylinder(HANDLE_RADIUS_M, HANDLE_LENGTH_M),
        f"{spec.name}_handle_visual",
        np.asarray(HANDLE_COLOR),
    )
    plant.RegisterVisualGeometry(
        hammer,
        RigidTransform(
            head_rotation,
            [0.0, 0.0, HEAD_CENTER_Z_M],
        ),
        Cylinder(HOUSING_RADIUS_M, HOUSING_LENGTH_M),
        f"{spec.name}_housing_visual",
        np.asarray(HOUSING_COLOR),
    )
    for side, center_x_m, insert_mass_kg in (
        ("left", -INSERT_CENTER_X_M, spec.left_insert_mass_kg),
        ("right", INSERT_CENTER_X_M, spec.right_insert_mass_kg),
    ):
        plant.RegisterVisualGeometry(
            hammer,
            RigidTransform(
                head_rotation,
                [center_x_m, 0.0, HEAD_CENTER_Z_M],
            ),
            Cylinder(INSERT_RADIUS_M, INSERT_LENGTH_M),
            f"{spec.name}_{side}_insert_visual",
            insert_color(insert_mass_kg),
        )


def register_modular_hammer_collisions(
    plant: MultibodyPlant,
    hammer: RigidBody,
    spec: VariantSpec,
) -> None:
    """Register the four visible hammer parts as collision safety geometry."""
    head_rotation = RotationMatrix.MakeYRotation(np.pi / 2.0)
    plant.RegisterCollisionGeometry(
        hammer,
        RigidTransform(HANDLE_CENTER_H_M),
        Cylinder(HANDLE_RADIUS_M, HANDLE_LENGTH_M),
        f"{spec.name}_handle_collision",
        PAYLOAD_FRICTION,
    )
    plant.RegisterCollisionGeometry(
        hammer,
        RigidTransform(head_rotation, [0.0, 0.0, HEAD_CENTER_Z_M]),
        Cylinder(HOUSING_RADIUS_M, HOUSING_LENGTH_M),
        f"{spec.name}_housing_collision",
        PAYLOAD_FRICTION,
    )
    for side, center_x_m in (
        ("left", -INSERT_CENTER_X_M),
        ("right", INSERT_CENTER_X_M),
    ):
        plant.RegisterCollisionGeometry(
            hammer,
            RigidTransform(
                head_rotation,
                [center_x_m, 0.0, HEAD_CENTER_Z_M],
            ),
            Cylinder(INSERT_RADIUS_M, INSERT_LENGTH_M),
            f"{spec.name}_{side}_insert_collision",
            PAYLOAD_FRICTION,
        )


def add_rb5_with_payload(
    plant: MultibodyPlant,
    parser: Parser,
    urdf_path: Path,
    pgc_urdf_path: Path,
    drake_assets: Path,
    spec: VariantSpec,
    include_visuals: bool,
    assembly_override: HammerAssembly | None = None,
    controller_kp: float = ARM_KP,
    controller_kd: float = ARM_KD,
    actuator_effort_limit_nm: float = ARM_MAX_TORQUE_NM,
    payload_body_name: str = "hammer",
    payload_attachment: RigidTransform | None = None,
    payload_visual_registrar: Callable[
        [MultibodyPlant, RigidBody, VariantSpec], None
    ]
    | None = None,
    payload_collision_registrar: Callable[
        [MultibodyPlant, RigidBody, VariantSpec], None
    ]
    | None = None,
    pgc_joint_position_m: float = 0.0,
    pgc_movable: bool = False,
    pgc_controller_kp: float = 6000.0,
    pgc_controller_kd: float = 80.0,
    base_pose: RigidTransform | None = None,
) -> ArmRecord:
    model_instance = parser.AddModelsFromString(
        rb5_urdf_string(urdf_path, spec.name, include_visuals),
        "urdf",
    )[0]
    plant.WeldFrames(
        plant.world_frame(),
        plant.GetFrameByName("link0", model_instance),
        (
            base_pose
            if base_pose is not None
            else RigidTransform([0.0, spec.base_y_m, 0.0])
        ),
    )

    joints = tuple(
        plant.GetJointByName(name, model_instance) for name in ARM_JOINT_NAMES
    )
    for joint in joints:
        actuator = plant.AddJointActuator(
            f"{spec.name}_{joint.name()}_motor",
            joint,
            actuator_effort_limit_nm,
        )
        if controller_kp > 0.0 or controller_kd > 0.0:
            actuator.set_controller_gains(
                PdControllerGains(p=controller_kp, d=controller_kd)
            )

    aft_model = plant.AddModelInstance(f"{spec.name}_aft200")
    bracket_inertia = SpatialInertia(
        AFT200_BRACKET_MASS_KG,
        np.zeros(3),
        UnitInertia.SolidCylinder(
            AFT200_DIAMETER_M / 2.0,
            AFT200_BRACKET_HEIGHT_M,
            [0.0, 0.0, 1.0],
        ),
    )
    sensor_principal_inertia = RotationalInertia(
        *AFT200_SENSOR_INERTIA_KG_M2
    )
    sensor_principal_rotation = RotationMatrix(
        Quaternion(AFT200_SENSOR_PRINCIPAL_AXES)
    )
    sensor_inertia = SpatialInertia.MakeFromCentralInertia(
        AFT200_SENSOR_MASS_KG,
        AFT200_SENSOR_COM_M,
        sensor_principal_inertia.ReExpress(sensor_principal_rotation),
    )
    bracket = plant.AddRigidBody("bracket", aft_model, bracket_inertia)
    sensor = plant.AddRigidBody("sensor", aft_model, sensor_inertia)
    if include_visuals:
        plant.RegisterVisualGeometry(
            bracket,
            RigidTransform(),
            Cylinder(AFT200_DIAMETER_M / 2.0, AFT200_BRACKET_HEIGHT_M),
            f"{spec.name}_aft_bracket_visual",
            np.array([0.28, 0.28, 0.30, 1.0]),
        )
        plant.RegisterVisualGeometry(
            sensor,
            RigidTransform(),
            Mesh(drake_assets / "aft200_visual.obj"),
            f"{spec.name}_aft_sensor_visual",
            np.array([0.12, 0.12, 0.14, 1.0]),
        )
    plant.RegisterCollisionGeometry(
        bracket,
        RigidTransform(),
        Cylinder(AFT200_DIAMETER_M / 2.0, AFT200_BRACKET_HEIGHT_M),
        f"{spec.name}_aft_bracket_collision",
        PAYLOAD_FRICTION,
    )
    plant.RegisterCollisionGeometry(
        sensor,
        RigidTransform(),
        Cylinder(AFT200_DIAMETER_M / 2.0, AFT200_SENSOR_HEIGHT_M),
        f"{spec.name}_aft_sensor_collision",
        PAYLOAD_FRICTION,
    )

    link6 = plant.GetBodyByName("link6", model_instance)
    x_link6_joint = RigidTransform(
        RotationMatrix(Quaternion(0.7071068, 0.7071068, 0.0, 0.0)),
        [0.0, -0.0967, 0.0],
    )
    x_bracket_joint = RigidTransform(
        [0.0, 0.0, -AFT200_BRACKET_HEIGHT_M / 2.0]
    )
    x_link6_bracket = x_link6_joint @ x_bracket_joint.inverse()
    x_bracket_sensor = RigidTransform(
        [
            0.0,
            0.0,
            (AFT200_BRACKET_HEIGHT_M + AFT200_SENSOR_HEIGHT_M) / 2.0,
        ]
    )
    x_sensor_pgc = RigidTransform([0.0, 0.0, AFT200_SENSOR_HEIGHT_M / 2.0])
    plant.WeldFrames(link6.body_frame(), bracket.body_frame(), x_link6_bracket)
    plant.WeldFrames(bracket.body_frame(), sensor.body_frame(), x_bracket_sensor)

    pgc_model_instance = parser.AddModelsFromString(
        (
            pgc_movable_urdf_string(
                pgc_urdf_path,
                f"{spec.name}_pgc",
                include_visuals,
            )
            if pgc_movable
            else pgc_fixed_urdf_string(
                pgc_urdf_path,
                f"{spec.name}_pgc",
                include_visuals,
                joint_position_m=pgc_joint_position_m,
            )
        ),
        "urdf",
    )[0]
    if pgc_movable:
        for finger_joint_name in ("finger1_joint", "finger2_joint"):
            finger_joint = plant.GetJointByName(
                finger_joint_name,
                pgc_model_instance,
            )
            finger_actuator = plant.AddJointActuator(
                f"{spec.name}_{finger_joint_name}_motor",
                finger_joint,
                140.0,
            )
            finger_actuator.set_controller_gains(
                PdControllerGains(
                    p=pgc_controller_kp,
                    d=pgc_controller_kd,
                )
            )
    plant.RegisterCollisionGeometry(
        plant.GetBodyByName("base_link", pgc_model_instance),
        RigidTransform([0.0, 0.0, 0.0415]),
        Box(*PGC_BASE_COLLISION_SIZE_M),
        f"{spec.name}_pgc_base_collision",
        GRIPPER_FRICTION,
    )
    for finger_name in ("finger1_link", "finger2_link"):
        collision_shapes = (
            (
                RigidTransform(PGC_FINGER_COLLISION_CENTER_M),
                Box(*PGC_FINGER_COLLISION_SIZE_M),
                "collision",
            ),
        )
        for transform, shape, suffix in collision_shapes:
            contact_properties = GRIPPER_FRICTION
            if pgc_movable:
                contact_properties = ProximityProperties()
                AddContactMaterial(
                    contact_properties,
                    dissipation=2.0,
                    point_stiffness=1.0e6,
                    friction=MOVABLE_PAD_FRICTION,
                )
                AddCompliantHydroelasticProperties(
                    0.004,
                    5.0e6,
                    contact_properties,
                )
            plant.RegisterCollisionGeometry(
                plant.GetBodyByName(finger_name, pgc_model_instance),
                transform,
                shape,
                f"{spec.name}_{finger_name}_{suffix}",
                contact_properties,
            )
    plant.WeldFrames(
        sensor.body_frame(),
        plant.GetFrameByName("base_link", pgc_model_instance),
        x_sensor_pgc,
    )

    assembly = assembly_override or make_hammer_assembly(spec)
    payload_model = plant.AddModelInstance(
        f"{spec.name}_{payload_body_name}_payload"
    )
    payload = plant.AddRigidBody(
        payload_body_name,
        payload_model,
        assembly.spatial_inertia,
    )
    plant.WeldFrames(
        plant.GetFrameByName("base_link", pgc_model_instance),
        payload.body_frame(),
        payload_attachment
        or RigidTransform(
            RotationMatrix.MakeXRotation(np.pi / 2.0),
            [0.0, 0.0, PGC_TCP_Z_M],
        ),
    )
    if include_visuals:
        (payload_visual_registrar or register_modular_hammer_visuals)(
            plant,
            payload,
            spec,
        )
    (payload_collision_registrar or register_modular_hammer_collisions)(
        plant,
        payload,
        spec,
    )

    return ArmRecord(
        spec=spec,
        assembly=assembly,
        model_instance=model_instance,
        aft_model_instance=aft_model,
        pgc_model_instance=pgc_model_instance,
        joints=joints,
        link6=link6,
        payload=payload,
    )


def desired_state_source() -> TrajectorySource:
    position_samples = []
    for wrist_target in DESIRED_WRIST_TARGETS_RAD:
        targets = INITIAL_ARM_POSITION_RAD.copy()
        targets[-1] += wrist_target
        position_samples.append(targets)
    positions_by_break = np.stack(position_samples, axis=1)
    desired_states = np.vstack(
        (positions_by_break, np.zeros_like(positions_by_break))
    )
    return TrajectorySource(
        PiecewisePolynomial.ZeroOrderHold(DESIRED_WRIST_BREAKS_S, desired_states)
    )


def build_scenario(htd_root: Path, meshcat=None):
    validate_hammer_manifest()
    rb5_description, urdf_path, pgc_urdf_path, drake_assets = (
        validate_htd_source(htd_root)
    )
    include_visuals = meshcat is not None

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(
        builder,
        time_step=PLANT_TIME_STEP_S,
    )
    plant.set_discrete_contact_approximation(DiscreteContactApproximation.kSimilar)
    plant.set_penetration_allowance(CONTACT_PENETRATION_ALLOWANCE_M)
    plant.mutable_gravity_field().set_gravity_vector([0.0, 0.0, -9.81])
    parser = Parser(plant)
    parser.package_map().Add("rbpodo_description", str(rb5_description))
    parser.package_map().Add("htd", str(htd_root.resolve()))
    arm_records = tuple(
        add_rb5_with_payload(
            plant,
            parser,
            urdf_path,
            pgc_urdf_path,
            drake_assets,
            spec,
            include_visuals,
        )
        for spec in VARIANTS
    )
    plant.Finalize()
    actuation_matrix = plant.MakeActuationMatrix()
    if not np.allclose(
        actuation_matrix,
        np.eye(plant.num_velocities()),
        atol=0.0,
    ):
        raise RuntimeError("RB5 actuator order is not identity in generalized forces")

    gravity_compensation = builder.AddSystem(
        InverseDynamics(plant, InverseDynamics.kGravityCompensation)
    )
    builder.Connect(
        plant.get_state_output_port(),
        gravity_compensation.get_input_port(0),
    )
    builder.Connect(
        gravity_compensation.get_output_port(0),
        plant.get_actuation_input_port(),
    )
    for record in arm_records:
        desired_state = builder.AddSystem(desired_state_source())
        desired_state.set_name(f"{record.spec.name}_desired_state")
        builder.Connect(
            desired_state.get_output_port(),
            plant.get_desired_state_input_port(record.model_instance),
        )

    if meshcat is not None:
        MeshcatVisualizer.AddToBuilder(
            builder,
            scene_graph,
            meshcat,
            MeshcatVisualizerParams(publish_period=1.0 / 64.0),
        )
    return builder.Build(), plant, scene_graph, arm_records


def simulate(htd_root: Path, meshcat=None) -> SimulationResult:
    diagram, plant, scene_graph, arm_records = build_scenario(htd_root, meshcat)
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyMutableContextFromRoot(context)
    for record in arm_records:
        for joint, initial_position in zip(
            record.joints,
            INITIAL_ARM_POSITION_RAD,
            strict=True,
        ):
            joint.set_angle(plant_context, initial_position)
            joint.set_angular_rate(plant_context, 0.0)

    mass_matrix = plant.CalcMassMatrixViaInverseDynamics(plant_context)
    wrist_joint_inertias = tuple(
        float(
            mass_matrix[
                record.joints[-1].position_start(),
                record.joints[-1].position_start(),
            ]
        )
        for record in arm_records
    )
    task_axis_alignments = tuple(
        float(
            abs(
                np.dot(
                    plant.CalcRelativeTransform(
                        plant_context,
                        record.link6.body_frame(),
                        record.payload.body_frame(),
                    ).rotation().matrix()
                    @ np.array([0.0, 1.0, 0.0]),
                    np.array([0.0, 1.0, 0.0]),
                )
            )
        )
        for record in arm_records
    )

    collision_geometry_count = scene_graph.model_inspector().NumGeometriesWithRole(
        Role.kProximity
    )
    actuators = tuple(
        plant.get_joint_actuator(JointActuatorIndex(index))
        for index in range(plant.num_actuators())
    )
    actuator_effort_limits = tuple(
        float(actuator.effort_limit()) for actuator in actuators
    )
    actuator_pd_gains = tuple(
        (
            float(actuator.get_controller_gains().p),
            float(actuator.get_controller_gains().d),
        )
        for actuator in actuators
    )

    simulator = Simulator(diagram, context)
    simulator.Initialize()
    if meshcat is not None:
        meshcat.StartRecording(frames_per_second=64.0)

    wrist_angles_by_time = {}
    wrist_trajectory_by_time = []
    hammer_reference_trajectory_by_time = []
    hammer_tips = None
    max_hold_errors = None
    max_collision_penetration_m = 0.0
    evaluation_times = tuple(
        sorted(set(SAMPLE_TIMES_S + TRAJECTORY_SAMPLE_TIMES_S))
    )
    regression_sample_times = set(SAMPLE_TIMES_S)
    trajectory_sample_times = set(TRAJECTORY_SAMPLE_TIMES_S)
    for sample_time in evaluation_times:
        simulator.AdvanceTo(sample_time)
        plant_context = plant.GetMyContextFromRoot(simulator.get_context())
        query = scene_graph.get_query_output_port().Eval(
            scene_graph.GetMyContextFromRoot(simulator.get_context())
        )
        max_collision_penetration_m = max(
            max_collision_penetration_m,
            max(
                (
                    float(pair.depth)
                    for pair in query.ComputePointPairPenetration()
                ),
                default=0.0,
            ),
        )
        wrist_angles = tuple(
            float(record.joints[-1].get_angle(plant_context))
            for record in arm_records
        )
        if sample_time in regression_sample_times:
            wrist_angles_by_time[sample_time] = wrist_angles
        if sample_time in trajectory_sample_times:
            wrist_trajectory_by_time.append(wrist_angles)
            hammer_reference_trajectory_by_time.append(
                tuple(
                    tuple(
                        np.asarray(
                            plant.CalcPointsPositions(
                                plant_context,
                                record.payload.body_frame(),
                                HAMMER_REFERENCE_POINT_H_M,
                                plant.world_frame(),
                            )
                        ).reshape(3)
                        - np.array([0.0, record.spec.base_y_m, 0.0])
                    )
                    for record in arm_records
                )
            )
        if np.isclose(sample_time, HOLD_CHECK_TIME_S):
            max_hold_errors = tuple(
                float(
                    np.max(
                        np.abs(
                            np.array(
                                [
                                    joint.get_angle(plant_context)
                                    for joint in record.joints
                                ]
                            )
                            - INITIAL_ARM_POSITION_RAD
                        )
                    )
                )
                for record in arm_records
            )
        if np.isclose(sample_time, SENSITIVITY_TIME_S):
            hammer_tips = tuple(
                tuple(
                    np.asarray(
                        plant.CalcPointsPositions(
                            plant_context,
                            record.payload.body_frame(),
                            HAMMER_REFERENCE_POINT_H_M,
                            plant.world_frame(),
                        )
                    ).reshape(3)
                    - np.array([0.0, record.spec.base_y_m, 0.0])
                )
                for record in arm_records
            )

    if meshcat is not None:
        meshcat.StopRecording()
        animation = meshcat.get_mutable_recording()
        animation.set_autoplay(True)
        animation.set_loop_mode(MeshcatAnimation.LoopMode.kLoopRepeat)
        animation.set_repetitions(1000)
        meshcat.PublishRecording()

    if hammer_tips is None:
        raise RuntimeError("Sensitivity sample was not recorded")
    if max_hold_errors is None:
        raise RuntimeError("Gravity hold sample was not recorded")
    wrist_trajectories = np.asarray(wrist_trajectory_by_time).T
    hammer_reference_trajectories = np.asarray(
        hammer_reference_trajectory_by_time
    ).transpose(1, 0, 2)
    return SimulationResult(
        plant_time_step_s=plant.time_step(),
        actuator_count=plant.num_actuators(),
        actuator_effort_limits_nm=actuator_effort_limits,
        actuator_pd_gains=actuator_pd_gains,
        collision_geometry_count=collision_geometry_count,
        max_collision_penetration_m=max_collision_penetration_m,
        rb5_position_counts=tuple(
            plant.num_positions(record.model_instance) for record in arm_records
        ),
        pgc_position_counts=tuple(
            plant.num_positions(record.pgc_model_instance) for record in arm_records
        ),
        end_effector_masses_kg=tuple(
            float(
                plant.CalcTotalMass(
                    plant_context,
                    [record.aft_model_instance, record.pgc_model_instance],
                )
            )
            for record in arm_records
        ),
        task_axis_alignments=task_axis_alignments,
        max_hold_errors_rad=max_hold_errors,
        hammer_masses_kg=tuple(
            record.assembly.mass_kg for record in arm_records
        ),
        hammer_coms_h_m=tuple(
            record.assembly.com_h_m for record in arm_records
        ),
        hammer_task_inertias_kg_m2=tuple(
            record.assembly.central_inertia_h_kg_m2[1][1]
            for record in arm_records
        ),
        hammer_ixz_kg_m2=tuple(
            record.assembly.central_inertia_h_kg_m2[0][2]
            for record in arm_records
        ),
        wrist_joint_inertias_kg_m2=wrist_joint_inertias,
        wrist_angles_by_time_rad=wrist_angles_by_time,
        hammer_tips_at_sensitivity_m=hammer_tips,
        trajectory_times_s=TRAJECTORY_SAMPLE_TIMES_S,
        wrist_trajectories_rad=tuple(
            tuple(float(value) for value in trajectory)
            for trajectory in wrist_trajectories
        ),
        hammer_reference_trajectories_m=tuple(
            tuple(
                tuple(float(value) for value in point)
                for point in trajectory
            )
            for trajectory in hammer_reference_trajectories
        ),
    )


def validate_result(result: SimulationResult) -> tuple[float, float]:
    trajectory_times = np.asarray(result.trajectory_times_s)
    wrist_trajectories = np.asarray(result.wrist_trajectories_rad)
    hammer_trajectories = np.asarray(result.hammer_reference_trajectories_m)
    if trajectory_times.shape != (len(TRAJECTORY_SAMPLE_TIMES_S),):
        raise AssertionError(f"Unexpected trajectory time shape: {trajectory_times.shape}")
    if wrist_trajectories.shape != (len(VARIANTS), len(trajectory_times)):
        raise AssertionError(
            f"Unexpected wrist trajectory shape: {wrist_trajectories.shape}"
        )
    if hammer_trajectories.shape != (
        len(VARIANTS),
        len(trajectory_times),
        3,
    ):
        raise AssertionError(
            f"Unexpected hammer trajectory shape: {hammer_trajectories.shape}"
        )
    if not np.all(np.isfinite(wrist_trajectories)) or not np.all(
        np.isfinite(hammer_trajectories)
    ):
        raise AssertionError("Trajectory output contains non-finite values")

    if not np.isclose(result.plant_time_step_s, PLANT_TIME_STEP_S, atol=0.0):
        raise AssertionError(
            f"Unexpected discrete time step: {result.plant_time_step_s}"
        )
    expected_actuator_count = len(VARIANTS) * len(ARM_JOINT_NAMES)
    if result.actuator_count != expected_actuator_count:
        raise AssertionError(f"Unexpected actuator count: {result.actuator_count}")
    if not np.allclose(
        result.actuator_effort_limits_nm,
        ARM_MAX_TORQUE_NM,
        atol=0.0,
    ):
        raise AssertionError(
            f"Unexpected actuator limits: {result.actuator_effort_limits_nm}"
        )
    if not np.allclose(
        result.actuator_pd_gains,
        (ARM_KP, ARM_KD),
        atol=0.0,
    ):
        raise AssertionError(
            f"Unexpected native PD gains: {result.actuator_pd_gains}"
        )
    expected_collision_count = len(VARIANTS) * COLLISION_GEOMETRIES_PER_ARM
    if result.collision_geometry_count != expected_collision_count:
        raise AssertionError(
            "Unexpected collision geometry count: "
            f"{result.collision_geometry_count} != {expected_collision_count}"
        )
    if result.max_collision_penetration_m > 0.0:
        raise AssertionError(
            "Free-space trajectory entered collision: "
            f"{result.max_collision_penetration_m} m"
        )

    if result.rb5_position_counts != (6, 6, 6):
        raise AssertionError(f"Unexpected RB5 position counts: {result.rb5_position_counts}")
    if result.pgc_position_counts != (0, 0, 0):
        raise AssertionError(
            f"PGC fixed-assembly position counts failed: {result.pgc_position_counts}"
        )
    if not np.allclose(
        result.end_effector_masses_kg,
        END_EFFECTOR_MASS_KG,
        atol=1e-12,
    ):
        raise AssertionError(
            "AFT200+PGC model mass mismatch: "
            f"{result.end_effector_masses_kg}"
        )
    if min(result.task_axis_alignments) < 0.999999:
        raise AssertionError(
            f"Hammer task axis is not aligned to wrist3: {result.task_axis_alignments}"
        )
    if max(result.max_hold_errors_rad) > 1e-6:
        raise AssertionError(
            f"Gravity-compensated hold drifted: {result.max_hold_errors_rad}"
        )

    expected_masses = (0.645, 0.755, 0.700)
    if not np.allclose(result.hammer_masses_kg, expected_masses, atol=1e-12):
        raise AssertionError(
            f"Modular hammer mass closure failed: {result.hammer_masses_kg}"
        )
    hammer_coms = np.asarray(result.hammer_coms_h_m)
    if not np.allclose(hammer_coms[:2, 0], 0.0, atol=1e-12):
        raise AssertionError(f"Symmetric hammer CoM shifted: {hammer_coms}")
    if hammer_coms[2, 0] <= 0.003:
        raise AssertionError(
            f"Right-heavy hammer CoM did not shift right: {hammer_coms[2]}"
        )
    if not np.allclose(result.hammer_ixz_kg_m2[:2], 0.0, atol=1e-12):
        raise AssertionError(
            f"Symmetric hammer has nonzero Ixz: {result.hammer_ixz_kg_m2}"
        )
    if abs(result.hammer_ixz_kg_m2[2]) <= 5e-5:
        raise AssertionError(
            f"Right-heavy hammer Ixz is not observable: {result.hammer_ixz_kg_m2}"
        )
    hammer_task_inertias = result.hammer_task_inertias_kg_m2
    if not (
        hammer_task_inertias[0]
        < hammer_task_inertias[2]
        < hammer_task_inertias[1]
    ):
        raise AssertionError(
            f"Hammer task inertia ordering failed: {hammer_task_inertias}"
        )

    wrist_inertias = result.wrist_joint_inertias_kg_m2
    if not wrist_inertias[0] < wrist_inertias[2] < wrist_inertias[1]:
        raise AssertionError(f"Wrist inertia ordering failed: {wrist_inertias}")

    wrist_angles = result.wrist_angles_by_time_rad[SENSITIVITY_TIME_S]
    symmetric_bounds = sorted((wrist_angles[0], wrist_angles[1]))
    if not symmetric_bounds[0] < wrist_angles[2] < symmetric_bounds[1]:
        raise AssertionError(
            "Asymmetric wrist response did not remain between the two "
            f"symmetric configurations: {wrist_angles}"
        )
    wrist_gap_deg = float(np.rad2deg(abs(wrist_angles[0] - wrist_angles[1])))
    if wrist_gap_deg <= 0.25:
        raise AssertionError(f"Wrist gap too small: {wrist_gap_deg} deg")

    low_tip = np.asarray(result.hammer_tips_at_sensitivity_m[0])
    high_tip = np.asarray(result.hammer_tips_at_sensitivity_m[1])
    tip_gap_m = float(np.linalg.norm(low_tip - high_tip))
    if tip_gap_m <= 0.0025:
        raise AssertionError(f"Hammer-tip gap too small: {tip_gap_m} m")
    sensitivity_index = int(
        np.argmin(np.abs(trajectory_times - SENSITIVITY_TIME_S))
    )
    dense_tip_gap_m = float(
        np.linalg.norm(
            hammer_trajectories[0, sensitivity_index]
            - hammer_trajectories[1, sensitivity_index]
        )
    )
    if not np.isclose(dense_tip_gap_m, tip_gap_m, atol=1e-8):
        raise AssertionError(
            "Dense trajectory and sensitivity sample disagree: "
            f"{dense_tip_gap_m} vs {tip_gap_m}"
        )
    return wrist_gap_deg, tip_gap_m


def print_result(
    result: SimulationResult,
    wrist_gap_deg: float,
    tip_gap_m: float,
) -> None:
    wrist_angles = result.wrist_angles_by_time_rad[SENSITIVITY_TIME_S]
    print("DRAKE_RB5_HAMMER_PAYLOAD_TEST=PASS", flush=True)
    print(f"htd_commit={HTD_COMMIT}", flush=True)
    print("robot=HTD RB5-850E,6-DOF", flush=True)
    print("end_effector=AFT200+fixed PGC-140-50", flush=True)
    print(
        f"end_effector_mass_kg={result.end_effector_masses_kg[0]:.6f}",
        flush=True,
    )
    print("payload=4-part modular hammer rigidly welded at PGC tcp", flush=True)
    print("parts=handle,housing,left_insert,right_insert", flush=True)
    print(
        "common=robot,end-effector,hammer geometry,native discrete PD,trajectory",
        flush=True,
    )
    print("variable=left/right light-or-heavy insert composition", flush=True)
    print("configurations=light-light,heavy-heavy,light-heavy-right", flush=True)
    print(
        "physics=discrete,kSimilar,collision-enabled,free-space-no-contact",
        flush=True,
    )
    print(f"time_step_s={result.plant_time_step_s:.6f}", flush=True)
    print(
        f"native_pd=kp:{ARM_KP:.1f},kd:{ARM_KD:.1f},"
        f"effort_limit_nm:{ARM_MAX_TORQUE_NM:.1f}",
        flush=True,
    )
    print(
        f"collision_geometries={result.collision_geometry_count}",
        flush=True,
    )
    print(
        "max_collision_penetration_m="
        f"{result.max_collision_penetration_m:.12f}",
        flush=True,
    )
    print("gravity=enabled,configuration gravity-compensation", flush=True)
    print(
        "max_hold_error_rad="
        f"{max(result.max_hold_errors_rad):.12f}",
        flush=True,
    )
    print(
        "min_task_axis_alignment="
        f"{min(result.task_axis_alignments):.12f}",
        flush=True,
    )
    print(
        "name,mass_kg,com_x_mm,com_z_mm,Iyy_com_kg_m2,"
        "Ixz_com_kg_m2,wrist_joint_inertia_kg_m2,q_sensitivity_rad",
        flush=True,
    )
    for (
        spec,
        mass_kg,
        com_h_m,
        task_inertia,
        ixz,
        joint_inertia,
        wrist_angle,
    ) in zip(
        VARIANTS,
        result.hammer_masses_kg,
        result.hammer_coms_h_m,
        result.hammer_task_inertias_kg_m2,
        result.hammer_ixz_kg_m2,
        result.wrist_joint_inertias_kg_m2,
        wrist_angles,
        strict=True,
    ):
        print(
            f"{spec.name},{mass_kg:.6f},{com_h_m[0] * 1000.0:.6f},"
            f"{com_h_m[2] * 1000.0:.6f},{task_inertia:.9f},"
            f"{ixz:.9f},{joint_inertia:.6f},{wrist_angle:.9f}",
            flush=True,
        )
    print(
        f"light_light_to_heavy_heavy_wrist_gap_deg={wrist_gap_deg:.9f}",
        flush=True,
    )
    print(
        f"light_light_to_heavy_heavy_reference_gap_m={tip_gap_m:.9f}",
        flush=True,
    )
    print(
        "right_heavy_com_shift_mm="
        f"{result.hammer_coms_h_m[2][0] * 1000.0:.9f}",
        flush=True,
    )


def run_test_only(htd_root: Path = DEFAULT_HTD_ROOT) -> None:
    result = simulate(htd_root)
    repeated = simulate(htd_root)
    wrist_gap_deg, tip_gap_m = validate_result(result)
    validate_result(repeated)
    if not np.allclose(
        result.wrist_joint_inertias_kg_m2,
        repeated.wrist_joint_inertias_kg_m2,
        atol=1e-12,
    ):
        raise AssertionError("Mass matrix result is not deterministic")
    for sample_time in SAMPLE_TIMES_S:
        if not np.allclose(
            result.wrist_angles_by_time_rad[sample_time],
            repeated.wrist_angles_by_time_rad[sample_time],
            atol=1e-10,
        ):
            raise AssertionError(
                f"Trajectory result is not deterministic at {sample_time} s"
            )
    if not np.allclose(
        result.wrist_trajectories_rad,
        repeated.wrist_trajectories_rad,
        atol=1e-10,
    ):
        raise AssertionError("Dense wrist trajectories are not deterministic")
    if not np.allclose(
        result.hammer_reference_trajectories_m,
        repeated.hammer_reference_trajectories_m,
        atol=1e-10,
    ):
        raise AssertionError("Dense hammer trajectories are not deterministic")
    print_result(result, wrist_gap_deg, tip_gap_m)
    print(f"trajectory_samples={len(result.trajectory_times_s)}", flush=True)
    print("deterministic_repeat=PASS", flush=True)


def render_hammer_configuration_card(output_path: Path) -> None:
    """Render the three four-part hammer configurations without a server."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    figure, axes = plt.subplots(1, 3, figsize=(13.5, 6.3), layout="constrained")
    for axis, spec in zip(axes, VARIANTS, strict=True):
        assembly = make_hammer_assembly(spec)
        com_x_m, _, com_z_m = assembly.com_h_m

        axis.add_patch(
            Rectangle(
                (-HANDLE_RADIUS_M, 0.0),
                2.0 * HANDLE_RADIUS_M,
                HANDLE_LENGTH_M,
                facecolor=HANDLE_COLOR,
                edgecolor="white",
                linewidth=1.2,
            )
        )
        axis.add_patch(
            Rectangle(
                (-HOUSING_LENGTH_M / 2.0, HEAD_CENTER_Z_M - HOUSING_RADIUS_M),
                HOUSING_LENGTH_M,
                2.0 * HOUSING_RADIUS_M,
                facecolor=HOUSING_COLOR,
                edgecolor="white",
                linewidth=1.2,
            )
        )
        for side, center_x_m, insert_mass_kg in (
            ("L", -INSERT_CENTER_X_M, spec.left_insert_mass_kg),
            ("R", INSERT_CENTER_X_M, spec.right_insert_mass_kg),
        ):
            axis.add_patch(
                Rectangle(
                    (
                        center_x_m - INSERT_LENGTH_M / 2.0,
                        HEAD_CENTER_Z_M - INSERT_RADIUS_M,
                    ),
                    INSERT_LENGTH_M,
                    2.0 * INSERT_RADIUS_M,
                    facecolor=insert_color(insert_mass_kg),
                    edgecolor="#f5f5f5",
                    linewidth=1.4,
                )
            )
            axis.text(
                center_x_m,
                HEAD_CENTER_Z_M,
                side,
                ha="center",
                va="center",
                color="white" if np.isclose(insert_mass_kg, LIGHT_INSERT_MASS_KG) else "black",
                fontsize=10,
                fontweight="bold",
            )
        axis.plot(
            com_x_m,
            com_z_m,
            marker="*",
            color="#d81b60",
            markeredgecolor="white",
            markeredgewidth=0.8,
            markersize=14,
            zorder=10,
        )
        axis.axvline(0.0, color="#9aa0aa", linestyle=":", linewidth=1.0)
        axis.annotate(
            f"CoM x = {com_x_m * 1000.0:.2f} mm",
            xy=(com_x_m, com_z_m),
            xytext=(0.0, 0.235),
            arrowprops={"arrowstyle": "->", "color": "#d81b60"},
            ha="center",
            fontsize=9,
            color="#a20f49",
        )
        axis.text(
            0.0,
            0.105,
            "P1",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="white",
        )
        axis.text(
            0.0,
            HEAD_CENTER_Z_M + 0.002,
            "P2",
            ha="center",
            va="center",
            fontsize=8,
            color="white",
        )
        axis.set_title(
            f"{spec.name.replace('_', ' ')}\n"
            f"M={assembly.mass_kg * 1000.0:.0f} g | "
            f"Ixz={assembly.central_inertia_h_kg_m2[0][2]:+.2e}",
            fontsize=11,
            fontweight="bold",
        )
        axis.set(xlim=(-0.075, 0.075), ylim=(-0.005, 0.35))
        axis.set_aspect("equal")
        axis.axis("off")

    figure.suptitle(
        "Four-Part Modular Hammer: Three Requested Configurations",
        fontsize=17,
        fontweight="bold",
    )
    figure.supxlabel(
        "Black: light 65 g | Silver: heavy 120 g | Magenta star: assembly CoM | Proxy v0; GT pending",
        fontsize=9.5,
        color="#3d424c",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, facecolor="white")
    plt.close(figure)
    print(f"hammer_configuration_card={output_path.resolve()}", flush=True)


def render_trajectory_comparison(
    result: SimulationResult,
    output_path: Path,
) -> None:
    """Render task-space and wrist trajectory differences for all variants."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    korean_font_candidates = (
        (
            Path("/mnt/c/Windows/Fonts/NanumGothic.ttf"),
            Path("/mnt/c/Windows/Fonts/NanumGothicBold.ttf"),
        ),
        (
            Path("/mnt/c/Windows/Fonts/malgun.ttf"),
            Path("/mnt/c/Windows/Fonts/malgunbd.ttf"),
        ),
    )
    selected_font = None
    for regular_font, bold_font in korean_font_candidates:
        if regular_font.is_file():
            font_manager.fontManager.addfont(regular_font)
            if bold_font.is_file():
                font_manager.fontManager.addfont(bold_font)
            selected_font = regular_font
            break
    if selected_font is None:
        raise RuntimeError(
            "Korean plot font not found. Expected NanumGothic or Malgun Gothic "
            "under /mnt/c/Windows/Fonts."
        )
    plt.rcParams["font.family"] = font_manager.FontProperties(
        fname=selected_font
    ).get_name()
    plt.rcParams["axes.unicode_minus"] = False

    times = np.asarray(result.trajectory_times_s)
    wrist_deg = np.rad2deg(np.asarray(result.wrist_trajectories_rad))
    paths_m = np.asarray(result.hammer_reference_trajectories_m)
    sensitivity_index = int(np.argmin(np.abs(times - SENSITIVITY_TIME_S)))

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(14.0, 9.2),
    )
    figure.subplots_adjust(
        left=0.065,
        right=0.985,
        top=0.905,
        bottom=0.105,
        hspace=0.30,
        wspace=0.22,
    )
    path_axis, wrist_axis, separation_axis, delta_axis = axes.flat

    for variant_index, (label, color) in enumerate(VARIANT_DISPLAY):
        path = paths_m[variant_index]
        path_axis.plot(
            path[:, 0],
            path[:, 2],
            color=color,
            linewidth=2.3,
            label=label,
        )
        point = path[sensitivity_index]
        path_axis.scatter(
            point[0],
            point[2],
            s=64,
            color=color,
            edgecolor="white",
            linewidth=1.0,
            zorder=5,
        )
        wrist_axis.plot(
            times,
            wrist_deg[variant_index],
            color=color,
            linewidth=2.2,
            label=label,
        )

    desired_deg = np.rad2deg(np.asarray(DESIRED_WRIST_TARGETS_RAD))
    wrist_axis.step(
        DESIRED_WRIST_BREAKS_S,
        desired_deg,
        where="post",
        color="#30343B",
        linewidth=1.5,
        linestyle="--",
        label="동일한 6번 관절(wrist3) 목표각",
    )

    light_path = paths_m[0]
    for variant_index in (2, 1):
        label, color = VARIANT_DISPLAY[variant_index]
        separation_mm = (
            np.linalg.norm(paths_m[variant_index] - light_path, axis=1)
            * 1000.0
        )
        separation_axis.plot(
            times,
            separation_mm,
            color=color,
            linewidth=2.3,
            label=f"{label} / 경량–경량 대비",
        )
        separation_axis.scatter(
            times[sensitivity_index],
            separation_mm[sensitivity_index],
            s=58,
            color=color,
            edgecolor="white",
            linewidth=1.0,
            zorder=5,
        )

        wrist_delta_deg = np.abs(
            wrist_deg[variant_index] - wrist_deg[0]
        )
        delta_axis.plot(
            times,
            wrist_delta_deg,
            color=color,
            linewidth=2.3,
            label=f"{label} / 경량–경량 대비",
        )
        delta_axis.scatter(
            times[sensitivity_index],
            wrist_delta_deg[sensitivity_index],
            s=58,
            color=color,
            edgecolor="white",
            linewidth=1.0,
            zorder=5,
        )

    start_point = paths_m[0, 0]
    path_axis.scatter(
        start_point[0],
        start_point[2],
        marker="s",
        s=58,
        color="#20242A",
        label="공통 시작점",
        zorder=6,
    )
    path_axis.annotate(
        f"원형 표식: t = {SENSITIVITY_TIME_S:.3f} s",
        xy=(0.02, 0.03),
        xycoords="axes fraction",
        fontsize=9,
        color="#4A4F58",
    )
    path_axis.set_title("A  망치 기준점의 공간 궤적", loc="left", fontweight="bold")
    path_axis.set_xlabel("공통 좌표계 x (m)")
    path_axis.set_ylabel("공통 좌표계 z (m)")
    path_axis.set_aspect("equal", adjustable="datalim")
    path_axis.grid(alpha=0.25)
    path_axis.legend(fontsize=8.5, loc="best")

    wrist_axis.set_title(
        "B  동일 명령에 대한 6번 관절(wrist3) 응답",
        loc="left",
        fontweight="bold",
    )
    wrist_axis.set_xlabel("시간 (s)")
    wrist_axis.set_ylabel("6번 관절(wrist3) 각도 (°)")
    wrist_axis.text(
        0.018,
        0.035,
        "wrist3: RB5의 6번째이자 마지막 회전 관절\n"
        "그리퍼와 망치가 연결된 말단 관절",
        transform=wrist_axis.transAxes,
        fontsize=9.2,
        color="#343840",
        va="bottom",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#B8BDC5",
            "alpha": 0.92,
        },
    )
    wrist_axis.grid(alpha=0.25)
    wrist_axis.legend(fontsize=8.5, loc="best")

    separation_axis.set_title(
        "C  경량–경량 대비 작업공간 위치 차이",
        loc="left",
        fontweight="bold",
    )
    separation_axis.set_xlabel("시간 (s)")
    separation_axis.set_ylabel("기준점 거리 (mm)")
    separation_axis.grid(alpha=0.25)
    separation_axis.legend(fontsize=8.5, loc="best")

    delta_axis.set_title(
        "D  경량–경량 대비 6번 관절(wrist3) 각도 차이",
        loc="left",
        fontweight="bold",
    )
    delta_axis.set_xlabel("시간 (s)")
    delta_axis.set_ylabel("6번 관절 절대 각도 차이 (°)")
    delta_axis.grid(alpha=0.25)
    delta_axis.legend(fontsize=8.5, loc="best")

    for axis in (wrist_axis, separation_axis, delta_axis):
        axis.axvline(
            SENSITIVITY_TIME_S,
            color="#70757E",
            linewidth=1.0,
            linestyle=":",
        )
        axis.set_xlim(0.0, ANIMATION_END_S)

    tip_gap_mm = float(
        np.linalg.norm(
            paths_m[0, sensitivity_index] - paths_m[1, sensitivity_index]
        )
        * 1000.0
    )
    wrist_gap_deg = float(
        abs(wrist_deg[0, sensitivity_index] - wrist_deg[1, sensitivity_index])
    )
    figure.suptitle(
        "동일한 RB5 명령, 다른 파트 조합 → 달라지는 궤적",
        fontsize=17,
        fontweight="bold",
        y=0.975,
        va="top",
    )
    figure.supxlabel(
        "HTD RB5 + AFT200 + 고정 PGC | 1 ms native discrete PD | "
        "외부 물체 접촉 없음 | RL 없음 | "
        f"{SENSITIVITY_TIME_S:.3f} s에서 경량–경량 / 중량–중량 차이 = "
        f"{tip_gap_mm:.1f} mm, "
        f"{wrist_gap_deg:.2f}° | 프록시 v0: 실제–시뮬레이션 정확도 주장이 아님",
        fontsize=9.5,
        color="#3D424C",
        y=0.018,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, facecolor="white")
    plt.close(figure)
    print(f"trajectory_comparison={output_path.resolve()}", flush=True)
    print(f"trajectory_samples={len(times)}", flush=True)
    print(
        f"trajectory_ll_hh_gap_at_sensitivity_mm={tip_gap_mm:.6f}",
        flush=True,
    )
    print(
        f"wrist_ll_hh_gap_at_sensitivity_deg={wrist_gap_deg:.6f}",
        flush=True,
    )


def serve_visualization(htd_root: Path) -> None:
    meshcat = StartMeshcat()
    meshcat.Delete()
    meshcat.SetProperty("/Background", "top_color", [0.95, 0.97, 1.0])
    meshcat.SetProperty("/Background", "bottom_color", [0.78, 0.84, 0.92])
    meshcat.SetCameraPose(
        camera_in_world=np.array([2.20, -4.20, 1.80]),
        target_in_world=np.array([0.10, 0.00, 0.55]),
    )
    result = simulate(htd_root, meshcat)
    wrist_gap_deg, tip_gap_m = validate_result(result)
    print_result(result, wrist_gap_deg, tip_gap_m)

    meshcat.AddButton(STOP_BUTTON)
    print("DRAKE_RB5_HAMMER_MESHCAT=READY", flush=True)
    print(f"url={meshcat.web_url()}", flush=True)
    print("legend=left:light-light, center:heavy-heavy, right:left-light/right-heavy", flush=True)
    print("stop=click the Meshcat button or press Ctrl+C", flush=True)
    try:
        while meshcat.GetButtonClicks(STOP_BUTTON) < 1:
            time.sleep(0.25)
    except KeyboardInterrupt:
        pass
    finally:
        meshcat.DeleteButton(STOP_BUTTON)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--htd-root",
        type=Path,
        default=DEFAULT_HTD_ROOT,
        help="Pinned HTD checkout root.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--test-only",
        action="store_true",
        help="Run assertions without starting Meshcat.",
    )
    mode.add_argument(
        "--render-hammer-card",
        type=Path,
        metavar="PNG",
        help="Render the three modular hammer configurations to a PNG.",
    )
    mode.add_argument(
        "--render-trajectory-comparison",
        type=Path,
        metavar="PNG",
        help="Render task-space and wrist trajectory differences to a PNG.",
    )
    args = parser.parse_args()
    if args.test_only:
        run_test_only(args.htd_root)
    elif args.render_hammer_card is not None:
        validate_hammer_manifest()
        render_hammer_configuration_card(args.render_hammer_card)
    elif args.render_trajectory_comparison is not None:
        result = simulate(args.htd_root)
        validate_result(result)
        render_trajectory_comparison(result, args.render_trajectory_comparison)
    else:
        serve_visualization(args.htd_root)


if __name__ == "__main__":
    main()
