#!/usr/bin/env python3
"""Drake view of the Lab RB5 + AFT200 + DH PGC setup."""

import argparse
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from pydrake.all import (
    AddCompliantHydroelasticProperties,
    AddContactMaterial,
    AddMultibodyPlantSceneGraph,
    AddRigidHydroelasticProperties,
    BodyIndex,
    Box,
    CameraInfo,
    ClippingRange,
    ColorRenderCamera,
    CoulombFriction,
    Cylinder,
    DepthRange,
    DepthRenderCamera,
    DiagramBuilder,
    DiscreteContactApproximation,
    FixedOffsetFrame,
    JointActuatorIndex,
    JointIndex,
    Mesh,
    Meshcat,
    MeshcatVisualizer,
    MakeRenderEngineVtk,
    Parser,
    PdControllerGains,
    ProximityProperties,
    Quaternion,
    RenderCameraCore,
    RenderEngineVtkParams,
    RigidTransform,
    Role,
    RgbdSensor,
    RotationalInertia,
    RotationMatrix,
    SpatialInertia,
    SpatialVelocity,
    Sphere,
    Simulator,
    UnitInertia,
)


HERE = Path(__file__).resolve().parent
BUNDLE_ASSETS = HERE / "assets"
LAB = HERE if BUNDLE_ASSETS.is_dir() else HERE.parent
RB5_DESCRIPTION = (
    BUNDLE_ASSETS / "rbpodo_description"
    if BUNDLE_ASSETS.is_dir()
    else LAB / "pipline/repo_chain/rb5_work/rbpodo_ros2/rbpodo_description"
)
RB5_URDF = RB5_DESCRIPTION / "robots/rb5_850e.urdf"
D435I_URDF = (
    BUNDLE_ASSETS / "d435i/d435i.urdf"
    if BUNDLE_ASSETS.is_dir()
    else HERE / "rb5_preview/d435i/d435i.urdf"
)
PGC_DIR = BUNDLE_ASSETS / "pgc_140_50" if BUNDLE_ASSETS.is_dir() else HERE / "rb5_preview/pgc_140_50"
PGC_URDF = PGC_DIR / "pgc_140_50.urdf"
DRAKE_ASSETS = BUNDLE_ASSETS / "drake" if BUNDLE_ASSETS.is_dir() else HERE / "rb5_preview/drake"
AFT200_OBJ = DRAKE_ASSETS / "aft200_visual.obj"
D435I_VISUAL = DRAKE_ASSETS / "d435_visual.obj"

ARM_JOINT_NAMES = ("base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3")
DEFAULT_ARM_POSE = (0.0, -0.65, 1.35, -0.70, 0.0, 0.0)
RB5_POSITION = np.array([0.14, 0.10, 0.0])
RB5_BASE_X_MAX = 0.086413
BASE_TABLE_CLEARANCE_MIN = 0.020

AFT200_DIAMETER = 0.080
AFT200_SENSOR_HEIGHT = 0.0215
AFT200_BRACKET_HEIGHT = 0.0523 - AFT200_SENSOR_HEIGHT
AFT200_SENSOR_MASS = 0.236
AFT200_BRACKET_MASS = 0.399
AFT200_SENSOR_COM = np.array([0.0009204, 0.0007747, 0.0003665])
AFT200_SENSOR_INERTIA = (9.43117e-05, 1.538887e-04, 2.314693e-04)
AFT200_SENSOR_PRINCIPAL_AXES = (-0.3936979, 0.0012072, 0.0005299, 0.9192389)
AFT200_FORCE_NOMINAL = 200.0
AFT200_TORQUE_NOMINAL = 15.0

PGC_JAW_TRAVEL = 0.025
PGC_TCP_Z = 0.125
PGC_BASE_COLLISION_SIZE = np.array([0.0751, 0.0751, 0.100])
PGC_FINGER_COLLISION_SIZE = np.array([0.0115, 0.020, 0.0565])
PGC_FINGER_COLLISION_CENTER = np.array([-0.00425, -0.0073, 0.01225])
GRIPPER_STATIC_FRICTION = 1.5
GRIPPER_DYNAMIC_FRICTION = 1.2
TABLE_CENTER = np.array([0.70, 0.12, 0.15])
TABLE_SIZE = np.array([0.90, 1.00, 0.30])
FLOOR_SIZE = np.array([4.0, 4.0, 0.02])
TEST_BOX_SIZE = np.array([0.06, 0.06, 0.06])
TEST_BOX_MASS = 0.20
TEST_BOX_START = np.array([0.90, -0.15, 0.55])
FLOOR_TEST_START = np.array([-0.40, -0.50, 0.50])
ARM_KP = 200.0
ARM_KD = 30.0
ARM_MAX_TORQUE = 80.0
GRIPPER_KP = 8000.0
GRIPPER_KD = 100.0
GRIPPER_MAX_FORCE = 140.0
CONTACT_PENETRATION_ALLOWANCE = 0.0001
CONTACT_RESOLUTION_HINT = 0.01
CONTACT_MODULUS = 1e8
COLLISION_TEST_POSE = np.array([0.6255, 1.9861, 1.3784, -1.3740, -0.9992, 1.8678])
SELF_COLLISION_TEST_POSE = np.array(
    [-1.257809, -0.485524, -2.962152, -2.359501, 1.071521, 0.924350]
)
SAFE_ARM_POSES = {
    "default": np.array(DEFAULT_ARM_POSE),
    "upright": np.array([0.0, -1.57, 1.57, 0.0, 0.0, 0.0]),
    "left": np.array([-0.5, -0.65, 1.35, -0.70, 0.0, 0.0]),
    "compact": np.array([0.0, -1.0, 1.7, -0.7, 0.5, 0.0]),
}

D435I_POSITION = np.array([1.28, 0.12, 0.78])
D435I_LOOK_AT = np.array([0.62, 0.12, 0.34])
D435I_RESOLUTION = (848, 480)
D435I_FREQUENCY = 30
D435I_DEPTH_FOV_DEG = (87.0, 58.0)
D435I_DEPTH_MIN_M = 0.195
D435I_DEPTH_MAX_M = 10.0
D435I_DEPTH_SCALE_M = 0.001
D435I_STEREO_BASELINE_M = 0.050
D435I_DEPTH_TO_COLOR_M = (0.0, 0.015, 0.0)
D435I_FX = D435I_RESOLUTION[0] / (2.0 * np.tan(np.deg2rad(D435I_DEPTH_FOV_DEG[0]) / 2.0))
D435I_FY = D435I_RESOLUTION[1] / (2.0 * np.tan(np.deg2rad(D435I_DEPTH_FOV_DEG[1]) / 2.0))
D435I_CX = D435I_RESOLUTION[0] / 2.0
D435I_CY = D435I_RESOLUTION[1] / 2.0

PGC_VISUALS = {
    (PGC_DIR / "meshes/base_link.STL").resolve(): DRAKE_ASSETS / "pgc_base_link.obj",
    (PGC_DIR / "meshes/finger1_Link.STL").resolve(): DRAKE_ASSETS / "pgc_finger1.obj",
    (PGC_DIR / "meshes/finger2_Link.STL").resolve(): DRAKE_ASSETS / "pgc_finger2.obj",
}


def lab_uri(path):
    return f"package://lab/{path.resolve().relative_to(LAB).as_posix()}"


def prepared_urdf(path, mesh_overrides=None):
    """Make local mesh paths portable and reuse RB5 visuals as convex collision meshes."""
    root = ET.parse(path).getroot()
    mesh_overrides = mesh_overrides or {}
    if path.resolve() == PGC_URDF.resolve():
        for link in root.findall("link"):
            for collision in list(link.findall("collision")):
                link.remove(collision)
    for joint in root.findall("joint"):
        mimic = joint.find("mimic")
        if mimic is not None:
            joint.remove(mimic)
    for mesh in root.iter("mesh"):
        raw = mesh.get("filename")
        if raw.startswith("package://"):
            if "/collision/" in raw:
                mesh.set("filename", lab_uri(DRAKE_ASSETS / "rb5_collision" / f"{Path(raw).stem}.obj"))
            elif "/visual/" in raw:
                mesh.set("filename", lab_uri(DRAKE_ASSETS / "rb5_visual" / f"{Path(raw).stem}.obj"))
            continue
        source = (path.parent / raw).resolve()
        target = mesh_overrides.get(source, source)
        if not target.is_file():
            raise FileNotFoundError(target)
        mesh.set("filename", lab_uri(target))
    return ET.tostring(root, encoding="unicode")


def look_at_quaternion(position, target):
    direction = np.asarray(target) - np.asarray(position)
    direction /= np.linalg.norm(direction)
    yaw = np.arctan2(direction[1], direction[0])
    elevation = np.arctan2(direction[2], np.hypot(direction[0], direction[1]))
    cy, sy = np.cos(yaw / 2.0), np.sin(yaw / 2.0)
    cp, sp = np.cos(-elevation / 2.0), np.sin(-elevation / 2.0)
    return np.array([cy * cp, -sy * sp, cy * sp, sy * cp])


def contact_properties(friction, compliant=False):
    properties = ProximityProperties()
    AddContactMaterial(20.0, None, friction, properties)
    if compliant:
        AddCompliantHydroelasticProperties(
            CONTACT_RESOLUTION_HINT, CONTACT_MODULUS, properties
        )
    else:
        AddRigidHydroelasticProperties(CONTACT_RESOLUTION_HINT, properties)
    return properties


def build_scene(
    physics=False,
    time_step=0.001,
    test_box_size=TEST_BOX_SIZE,
    test_box_mass=TEST_BOX_MASS,
    calibration_target=False,
    with_rgbd=True,
):
    for path in (RB5_URDF, D435I_URDF, PGC_URDF, AFT200_OBJ, D435I_VISUAL, *PGC_VISUALS.values()):
        if not path.is_file():
            raise FileNotFoundError(path)

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(
        builder, time_step=time_step if physics else 0.0
    )
    if physics:
        plant.set_discrete_contact_approximation(DiscreteContactApproximation.kSimilar)
        plant.set_penetration_allowance(CONTACT_PENETRATION_ALLOWANCE)
    parser = Parser(plant)
    parser.package_map().Add("lab", str(LAB))
    parser.package_map().Add("rbpodo_description", str(RB5_DESCRIPTION))

    rb5 = parser.AddModelsFromString(prepared_urdf(RB5_URDF), "urdf")[0]
    pgc = parser.AddModelsFromString(prepared_urdf(PGC_URDF, PGC_VISUALS), "urdf")[0]
    d435i = parser.AddModelsFromString(
        prepared_urdf(
            D435I_URDF,
            {
                (D435I_URDF.parent / "realsense2_description/meshes/d435.dae").resolve(): D435I_VISUAL
            },
        ),
        "urdf",
    )[0]
    if physics:
        for name in ARM_JOINT_NAMES:
            actuator = plant.AddJointActuator(
                f"{name}_motor", plant.GetJointByName(name, rb5), ARM_MAX_TORQUE
            )
            actuator.set_controller_gains(PdControllerGains(p=ARM_KP, d=ARM_KD))
        for name in ("finger1_joint", "finger2_joint"):
            actuator = plant.AddJointActuator(
                f"{name}_motor", plant.GetJointByName(name, pgc), GRIPPER_MAX_FORCE
            )
            actuator.set_controller_gains(PdControllerGains(p=GRIPPER_KP, d=GRIPPER_KD))

    gripper_friction = CoulombFriction(GRIPPER_STATIC_FRICTION, GRIPPER_DYNAMIC_FRICTION)
    plant.RegisterCollisionGeometry(
        plant.GetBodyByName("base_link", pgc),
        RigidTransform([0.0, 0.0, 0.0415]),
        Box(*PGC_BASE_COLLISION_SIZE),
        "pgc_base_collision",
        gripper_friction,
    )
    # ponytail: pad boxes avoid false convex-hull closure; replace with measured PGC collision geometry when available.
    for name in ("finger1_link", "finger2_link"):
        plant.RegisterCollisionGeometry(
            plant.GetBodyByName(name, pgc),
            RigidTransform(PGC_FINGER_COLLISION_CENTER),
            Box(*PGC_FINGER_COLLISION_SIZE),
            f"{name}_collision",
            gripper_friction,
        )

    plant.WeldFrames(
        plant.world_frame(),
        plant.GetFrameByName("link0", rb5),
        RigidTransform(RB5_POSITION),
    )

    aft = plant.AddModelInstance("aft200")
    bracket_inertia = SpatialInertia(
        AFT200_BRACKET_MASS,
        np.zeros(3),
        UnitInertia.SolidCylinder(
            AFT200_DIAMETER / 2.0,
            AFT200_BRACKET_HEIGHT,
            [0.0, 0.0, 1.0],
        ),
    )
    I_sensor_principal = RotationalInertia(*AFT200_SENSOR_INERTIA)
    R_BPrincipal = RotationMatrix(Quaternion(AFT200_SENSOR_PRINCIPAL_AXES))
    sensor_inertia = SpatialInertia.MakeFromCentralInertia(
        AFT200_SENSOR_MASS,
        AFT200_SENSOR_COM,
        I_sensor_principal.ReExpress(R_BPrincipal),
    )
    bracket = plant.AddRigidBody("bracket", aft, bracket_inertia)
    sensor = plant.AddRigidBody("sensor", aft, sensor_inertia)
    plant.RegisterVisualGeometry(
        bracket,
        RigidTransform(),
        Cylinder(AFT200_DIAMETER / 2.0, AFT200_BRACKET_HEIGHT),
        "bracket_visual",
        np.array([0.28, 0.28, 0.30, 1.0]),
    )
    plant.RegisterCollisionGeometry(
        bracket,
        RigidTransform(),
        Cylinder(AFT200_DIAMETER / 2.0, AFT200_BRACKET_HEIGHT),
        "bracket_collision",
        CoulombFriction(0.8, 0.6),
    )
    plant.RegisterVisualGeometry(
        sensor,
        RigidTransform(),
        Mesh(AFT200_OBJ),
        "sensor_visual",
        np.array([0.12, 0.12, 0.14, 1.0]),
    )
    plant.RegisterCollisionGeometry(
        sensor,
        RigidTransform(),
        Cylinder(AFT200_DIAMETER / 2.0, AFT200_SENSOR_HEIGHT),
        "sensor_collision",
        CoulombFriction(0.8, 0.6),
    )

    X_link6_joint = RigidTransform(
        RotationMatrix(Quaternion(0.7071068, 0.7071068, 0.0, 0.0)),
        [0.0, -0.0967, 0.0],
    )
    X_bracket_joint = RigidTransform([0.0, 0.0, -AFT200_BRACKET_HEIGHT / 2.0])
    X_link6_bracket = X_link6_joint @ X_bracket_joint.inverse()
    X_bracket_sensor = RigidTransform(
        [0.0, 0.0, (AFT200_BRACKET_HEIGHT + AFT200_SENSOR_HEIGHT) / 2.0]
    )
    X_sensor_pgc = RigidTransform([0.0, 0.0, AFT200_SENSOR_HEIGHT / 2.0])
    plant.WeldFrames(plant.GetFrameByName("link6", rb5), bracket.body_frame(), X_link6_bracket)
    plant.WeldFrames(bracket.body_frame(), sensor.body_frame(), X_bracket_sensor)
    sensor_pgc_joint = plant.WeldFrames(
        sensor.body_frame(), plant.GetFrameByName("base_link", pgc), X_sensor_pgc
    )
    pgc_tcp = plant.AddFrame(
        FixedOffsetFrame(
            "tcp",
            plant.GetFrameByName("base_link", pgc),
            RigidTransform([0.0, 0.0, PGC_TCP_Z]),
            pgc,
        )
    )

    q_camera = look_at_quaternion(D435I_POSITION, D435I_LOOK_AT)
    X_world_camera = RigidTransform(RotationMatrix(Quaternion(q_camera)), D435I_POSITION)
    plant.WeldFrames(
        plant.world_frame(),
        plant.GetFrameByName("base_link", d435i),
        X_world_camera,
    )

    table_model = plant.AddModelInstance("lab_table")
    table = plant.AddRigidBody("table", table_model)
    plant.RegisterVisualGeometry(
        table,
        RigidTransform(),
        Box(*TABLE_SIZE),
        "table_visual",
        np.array([0.34, 0.22, 0.12, 1.0]),
    )
    plant.RegisterCollisionGeometry(
        table,
        RigidTransform(),
        Box(*TABLE_SIZE),
        "table_collision",
        contact_properties(CoulombFriction(0.9, 0.7)),
    )
    plant.WeldFrames(plant.world_frame(), table.body_frame(), RigidTransform(TABLE_CENTER))

    plant.RegisterVisualGeometry(
        plant.world_body(),
        RigidTransform([0.0, 0.0, -FLOOR_SIZE[2] / 2.0]),
        Box(*FLOOR_SIZE),
        "floor_visual",
        np.array([0.18, 0.20, 0.22, 1.0]),
    )
    plant.RegisterCollisionGeometry(
        plant.world_body(),
        RigidTransform([0.0, 0.0, -FLOOR_SIZE[2] / 2.0]),
        Box(*FLOOR_SIZE),
        "floor_collision",
        contact_properties(CoulombFriction(0.9, 0.7)),
    )

    test_box = None
    test_box_size = np.asarray(test_box_size, dtype=float)
    if physics:
        test_model = plant.AddModelInstance("test_object")
        test_box = plant.AddRigidBody(
            "box",
            test_model,
            SpatialInertia(
                test_box_mass,
                np.zeros(3),
                UnitInertia.SolidBox(*test_box_size),
            ),
        )
        shape = Box(*test_box_size)
        plant.RegisterVisualGeometry(
            test_box, RigidTransform(), shape, "box_visual", np.array([0.85, 0.18, 0.08, 1.0])
        )
        plant.RegisterCollisionGeometry(
            test_box,
            RigidTransform(),
            shape,
            "box_collision",
            contact_properties(CoulombFriction(0.8, 0.6), compliant=True),
        )
    calibration_sphere = None
    if calibration_target:
        target_model = plant.AddModelInstance("d435i_calibration_target")
        calibration_sphere = plant.AddRigidBody(
            "sphere",
            target_model,
            SpatialInertia(0.01, np.zeros(3), UnitInertia.SolidSphere(0.03)),
        )
        plant.RegisterVisualGeometry(
            calibration_sphere,
            RigidTransform(),
            Sphere(0.03),
            "calibration_sphere_visual",
            np.array([1.0, 0.05, 0.05, 1.0]),
        )
    plant.Finalize()

    rgbd = None
    if with_rgbd:
        renderer_name = "d435i_renderer"
        scene_graph.AddRenderer(renderer_name, MakeRenderEngineVtk(RenderEngineVtkParams()))
        intrinsics = CameraInfo(
            *D435I_RESOLUTION, D435I_FX, D435I_FY, D435I_CX, D435I_CY
        )
        core = RenderCameraCore(
            renderer_name,
            intrinsics,
            ClippingRange(0.1, D435I_DEPTH_MAX_M),
            RigidTransform(),
        )
        rgbd = builder.AddSystem(
            RgbdSensor(
                plant.GetBodyFrameIdOrThrow(
                    plant.GetFrameByName("camera_depth_optical_frame", d435i).body().index()
                ),
                RigidTransform(),
                ColorRenderCamera(core, False),
                DepthRenderCamera(core, DepthRange(D435I_DEPTH_MIN_M, D435I_DEPTH_MAX_M)),
            )
        )
        builder.Connect(scene_graph.get_query_output_port(), rgbd.query_object_input_port())

    return builder, plant, scene_graph, {
        "rb5": rb5,
        "pgc": pgc,
        "d435i": d435i,
        "bracket": bracket,
        "sensor": sensor,
        "sensor_pgc_joint": sensor_pgc_joint,
        "table": table,
        "test_box": test_box,
        "test_box_size": test_box_size,
        "test_box_mass": test_box_mass,
        "calibration_sphere": calibration_sphere,
        "rgbd": rgbd,
        "pgc_tcp": pgc_tcp,
        "X_link6_bracket": X_link6_bracket,
        "X_bracket_sensor": X_bracket_sensor,
        "X_sensor_pgc": X_sensor_pgc,
        "X_world_camera": X_world_camera,
    }


def set_configuration(plant, context, models, arm=DEFAULT_ARM_POSE, jaw=0.0):
    arm, jaw = clip_command(plant, models, arm, jaw)
    for name, value in zip(ARM_JOINT_NAMES, arm):
        plant.GetJointByName(name, models["rb5"]).set_angle(context, value)
    plant.GetJointByName("finger1_joint", models["pgc"]).set_translation(context, jaw)
    plant.GetJointByName("finger2_joint", models["pgc"]).set_translation(context, jaw)
    return arm, jaw


def clip_command(plant, models, arm, jaw):
    arm = np.asarray(arm, dtype=float)
    if arm.shape != (6,) or not np.isfinite(arm).all() or not np.isfinite(jaw):
        raise ValueError("arm must contain six finite values and jaw must be finite")
    lower = np.array(
        [plant.GetJointByName(name, models["rb5"]).position_lower_limit() for name in ARM_JOINT_NAMES]
    )
    upper = np.array(
        [plant.GetJointByName(name, models["rb5"]).position_upper_limit() for name in ARM_JOINT_NAMES]
    )
    finger = plant.GetJointByName("finger1_joint", models["pgc"])
    return np.clip(arm, lower, upper), float(
        np.clip(jaw, finger.position_lower_limit(), finger.position_upper_limit())
    )


def set_motor_command(plant, context, models, arm, jaw):
    arm, jaw = clip_command(plant, models, arm, jaw)
    gravity_compensation = -plant.CalcGravityGeneralizedForces(context)[:8]
    plant.get_desired_state_input_port(models["rb5"]).FixValue(
        context, np.r_[arm, np.zeros(6)]
    )
    plant.get_desired_state_input_port(models["pgc"]).FixValue(
        context, np.r_[jaw, jaw, 0.0, 0.0]
    )
    plant.get_actuation_input_port(models["rb5"]).FixValue(context, gravity_compensation[:6])
    plant.get_actuation_input_port(models["pgc"]).FixValue(context, gravity_compensation[6:])
    return arm, jaw


def collision_pairs(root_context, plant, scene_graph, models):
    query = scene_graph.get_query_output_port().Eval(
        scene_graph.GetMyContextFromRoot(root_context)
    )
    inspector = scene_graph.model_inspector()
    robot_models = {models["rb5"], models["pgc"], models["sensor"].model_instance()}
    internal, environment = [], []
    for pair in query.ComputePointPairPenetration():
        body_a = plant.GetBodyFromFrameId(inspector.GetFrameId(pair.id_A))
        body_b = plant.GetBodyFromFrameId(inspector.GetFrameId(pair.id_B))
        item = (body_a.name(), body_b.name(), float(pair.depth))
        target = (
            internal
            if body_a.model_instance() in robot_models and body_b.model_instance() in robot_models
            else environment
        )
        target.append(item)
    return internal, environment


def preview_command(scratch_root, plant, scene_graph, models, arm, jaw):
    context = plant.GetMyMutableContextFromRoot(scratch_root)
    arm, jaw = set_configuration(plant, context, models, arm, jaw)
    internal, _ = collision_pairs(scratch_root, plant, scene_graph, models)
    return arm, jaw, internal


def run_model_checks(plant):
    moving_zero_mass = []
    for index in range(1, plant.num_bodies()):
        body = plant.get_body(BodyIndex(index))
        inertia = body.default_spatial_inertia()
        assert inertia.IsPhysicallyValid()
        if body.default_mass() == 0.0:
            welded = plant.GetBodiesWeldedTo(body)
            if not any(
                other.index() == plant.world_body().index() or other.default_mass() > 0.0
                for other in welded
            ):
                moving_zero_mass.append(body.name())
    assert not moving_zero_mass, f"movable zero-mass bodies: {moving_zero_mass}"

    limited_joints = 0
    for index in range(plant.num_joints()):
        joint = plant.get_joint(JointIndex(index))
        if not joint.num_positions():
            continue
        if joint.type_name() == "quaternion_floating":
            continue
        lower, upper = joint.position_lower_limits(), joint.position_upper_limits()
        velocity_lower, velocity_upper = (
            joint.velocity_lower_limits(),
            joint.velocity_upper_limits(),
        )
        assert np.isfinite(np.r_[lower, upper, velocity_lower, velocity_upper]).all()
        assert np.all(lower < upper) and np.all(velocity_lower < velocity_upper)
        limited_joints += 1

    for index in range(plant.num_actuators()):
        actuator = plant.get_joint_actuator(JointActuatorIndex(index))
        gains = actuator.get_controller_gains()
        assert np.isfinite([actuator.effort_limit(), gains.p, gains.d]).all()
        assert actuator.effort_limit() > 0.0 and gains.p > 0.0 and gains.d >= 0.0
    print(
        f"[DrakeRB5] MODEL PASS bodies={plant.num_bodies() - 1} "
        f"limited_joints={limited_joints} actuators={plant.num_actuators()}"
    )


def run_checks(diagram, root_context, plant, scene_graph, models):
    context = plant.GetMyMutableContextFromRoot(root_context)
    set_configuration(plant, context, models, jaw=PGC_JAW_TRAVEL)
    assert plant.num_positions(models["rb5"]) == 6
    assert plant.num_positions(models["pgc"]) == 2
    assert plant.num_positions() == 8
    run_model_checks(plant)
    collision_geometry_count = scene_graph.model_inspector().NumGeometriesWithRole(Role.kProximity)
    assert collision_geometry_count == 15
    for geometry_id in scene_graph.model_inspector().GetAllGeometryIds():
        properties = scene_graph.model_inspector().GetProximityProperties(geometry_id)
        if properties is None:
            continue
        friction = properties.GetProperty("material", "coulomb_friction")
        assert 0.0 <= friction.dynamic_friction() <= friction.static_friction()
    for jaw in (0.0, PGC_JAW_TRAVEL):
        set_configuration(plant, context, models, jaw=jaw)
        query = scene_graph.get_query_output_port().Eval(scene_graph.GetMyContextFromRoot(root_context))
        assert max((pair.depth for pair in query.ComputePointPairPenetration()), default=0.0) < 1e-6

    np.testing.assert_allclose(
        plant.CalcRelativeTransform(
            context, plant.GetFrameByName("link6", models["rb5"]), models["bracket"].body_frame()
        ).GetAsMatrix4(),
        models["X_link6_bracket"].GetAsMatrix4(),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        plant.CalcRelativeTransform(context, models["bracket"].body_frame(), models["sensor"].body_frame()).translation(),
        models["X_bracket_sensor"].translation(),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        plant.CalcRelativeTransform(
            context, models["sensor"].body_frame(), plant.GetFrameByName("base_link", models["pgc"])
        ).translation(),
        models["X_sensor_pgc"].translation(),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        plant.CalcRelativeTransform(
            context, plant.GetFrameByName("base_link", models["pgc"]), models["pgc_tcp"]
        ).translation(),
        [0.0, 0.0, PGC_TCP_Z],
        atol=1e-10,
    )
    np.testing.assert_allclose(
        plant.CalcRelativeTransform(
            context, plant.world_frame(), plant.GetFrameByName("base_link", models["d435i"])
        ).GetAsMatrix4(),
        models["X_world_camera"].GetAsMatrix4(),
        atol=1e-10,
    )
    direction = (D435I_LOOK_AT - D435I_POSITION) / np.linalg.norm(D435I_LOOK_AT - D435I_POSITION)
    X_world_optical = plant.CalcRelativeTransform(
        context,
        plant.world_frame(),
        plant.GetFrameByName("camera_depth_optical_frame", models["d435i"]),
    )
    np.testing.assert_allclose(X_world_optical.rotation().matrix()[:, 2], direction, atol=1e-10)
    assert np.isclose(TABLE_CENTER[2] + TABLE_SIZE[2] / 2.0, 0.300)
    base_table_clearance = TABLE_CENTER[0] - TABLE_SIZE[0] / 2.0 - (
        RB5_POSITION[0] + RB5_BASE_X_MAX
    )
    assert base_table_clearance >= BASE_TABLE_CLEARANCE_MIN
    assert np.isclose(AFT200_SENSOR_MASS + AFT200_BRACKET_MASS, 0.635)
    assert (AFT200_FORCE_NOMINAL, AFT200_TORQUE_NOMINAL) == (200.0, 15.0)
    assert (D435I_RESOLUTION, D435I_FREQUENCY) == ((848, 480), 30)
    assert (D435I_DEPTH_SCALE_M, D435I_STEREO_BASELINE_M, D435I_DEPTH_TO_COLOR_M) == (
        0.001,
        0.050,
        (0.0, 0.015, 0.0),
    )
    assert (D435I_DEPTH_MIN_M, D435I_DEPTH_MAX_M) == (0.195, 10.0)
    assert np.isclose(D435I_FX, 446.803118054254)
    assert np.isclose(D435I_FY, 432.9709424472207)
    color = models["rgbd"].color_image_output_port().Eval(
        models["rgbd"].GetMyContextFromRoot(root_context)
    )
    depth = models["rgbd"].depth_image_32F_output_port().Eval(
        models["rgbd"].GetMyContextFromRoot(root_context)
    )
    depth_data = np.asarray(depth.data).reshape(D435I_RESOLUTION[1], D435I_RESOLUTION[0])
    valid_depth = np.isfinite(depth_data) & (depth_data >= D435I_DEPTH_MIN_M)
    assert (color.width(), color.height()) == D435I_RESOLUTION
    assert (depth.width(), depth.height()) == D435I_RESOLUTION
    assert valid_depth.sum() > 1000
    diagram.ForcedPublish(root_context)
    print(
        "[DrakeRB5] PASS "
        f"positions={plant.num_positions()} collisions={collision_geometry_count} "
        f"base_clearance={base_table_clearance:.3f}m table_top=0.300m "
        f"PGC={PGC_JAW_TRAVEL * 1000:.0f}mm/jaw "
        f"D435i_K=({D435I_FX:.3f},{D435I_FY:.3f},{D435I_CX:.0f},{D435I_CY:.0f})"
    )
    print(
        f"[DrakeRB5] RGBD PASS color={color.width()}x{color.height()} "
        f"valid_depth={valid_depth.sum()}"
    )


def run_physics_check(simulator, plant, scene_graph, models):
    context = plant.GetMyContextFromRoot(simulator.get_context())
    assert plant.num_actuated_dofs() == 8
    run_model_checks(plant)
    clipped_arm, clipped_jaw = set_motor_command(
        plant, context, models, np.full(6, 100.0), 100.0
    )
    expected_upper = np.array(
        [plant.GetJointByName(name, models["rb5"]).position_upper_limit() for name in ARM_JOINT_NAMES]
    )
    np.testing.assert_allclose(clipped_arm, expected_upper)
    assert clipped_jaw == PGC_JAW_TRAVEL
    np.testing.assert_allclose(
        plant.get_desired_state_input_port(models["rb5"]).Eval(context)[:6],
        expected_upper,
    )
    for _ in range(150):
        set_motor_command(plant, context, models, DEFAULT_ARM_POSE, 0.0)
        simulator.AdvanceTo(simulator.get_context().get_time() + 0.01)
    hold_error = float(np.max(np.abs(plant.GetPositions(context)[:6] - DEFAULT_ARM_POSE)))
    gravity_torque = -plant.CalcGravityGeneralizedForces(context)[:6]
    assert hold_error < 0.01
    assert np.max(np.abs(gravity_torque)) < ARM_MAX_TORQUE
    assert np.linalg.norm(plant.GetVelocities(context)[:8]) < 0.05
    print(
        f"[DrakeRB5] GRAVITY PASS hold_error={hold_error:.4f}rad "
        f"max_compensation={np.max(np.abs(gravity_torque)):.1f}Nm"
    )
    reaction = plant.get_reaction_forces_output_port().Eval(context)[models["sensor_pgc_joint"].index()]
    force = reaction.translational()
    torque = reaction.rotational()
    joint = models["sensor_pgc_joint"]
    pgc_mass = plant.CalcTotalMass(context, [models["pgc"]])
    pgc_com = plant.CalcCenterOfMassPositionInWorld(context, [models["pgc"]])
    X_world_joint = plant.CalcRelativeTransform(
        context, plant.world_frame(), joint.frame_on_child()
    )
    gravity_world = np.array([0.0, 0.0, -9.81 * pgc_mass])
    expected_force = X_world_joint.rotation().inverse().multiply(-gravity_world)
    expected_torque = X_world_joint.rotation().inverse().multiply(
        -np.cross(pgc_com - X_world_joint.translation(), gravity_world)
    )
    force_error = np.linalg.norm(force - expected_force)
    torque_error = np.linalg.norm(torque - expected_torque)
    assert np.isfinite(force).all() and np.isfinite(torque).all()
    assert np.linalg.norm(force) < AFT200_FORCE_NOMINAL
    assert np.linalg.norm(torque) < AFT200_TORQUE_NOMINAL
    assert force_error <= 0.02 * np.linalg.norm(expected_force)
    assert torque_error <= max(0.02, 0.05 * np.linalg.norm(expected_torque))
    print(
        f"[DrakeRB5] AFT200 PASS force={np.linalg.norm(force):.2f}N "
        f"torque={np.linalg.norm(torque):.3f}Nm "
        f"error={force_error:.3g}N/{torque_error:.3g}Nm"
    )
    pose = plant.GetFreeBodyPose(context, models["test_box"])
    expected_z = TABLE_CENTER[2] + TABLE_SIZE[2] / 2.0 + models["test_box_size"][2] / 2.0
    assert np.isclose(pose.translation()[2], expected_z, atol=0.002)
    print(f"[DrakeRB5] PHYSICS PASS box_z={pose.translation()[2]:.3f}m")

    contact_speeds = []
    for step in range(300):
        set_motor_command(plant, context, models, COLLISION_TEST_POSE, 0.0)
        simulator.AdvanceTo(simulator.get_context().get_time() + 0.01)
        if step >= 200:
            contact_speeds.append(np.max(np.abs(plant.GetVelocities(context)[:8])))
    query = scene_graph.get_query_output_port().Eval(scene_graph.GetMyContextFromRoot(simulator.get_context()))
    inspector = scene_graph.model_inspector()
    table_depths = []
    for contact in query.ComputePointPairPenetration():
        frames = inspector.GetName(inspector.GetFrameId(contact.id_A)) + inspector.GetName(
            inspector.GetFrameId(contact.id_B)
        )
        if "rb5_850e" in frames and "lab_table" in frames:
            table_depths.append(contact.depth)
    assert table_depths and max(table_depths) < 0.005
    assert max(contact_speeds) < 0.01
    print(f"[DrakeRB5] MOTOR CONTACT PASS max_penetration={max(table_depths) * 1000:.1f}mm")

    set_configuration(plant, context, models)
    plant.SetVelocities(context, np.zeros(plant.num_velocities()))
    for _ in range(150):
        set_motor_command(plant, context, models, DEFAULT_ARM_POSE, PGC_JAW_TRAVEL)
        simulator.AdvanceTo(simulator.get_context().get_time() + 0.01)
    jaw = plant.GetPositions(context)[6:8]
    max_speed = np.max(np.abs(plant.GetVelocities(context)[:8]))
    np.testing.assert_allclose(jaw, PGC_JAW_TRAVEL, atol=1e-5)
    assert max_speed < 0.01
    print(f"[DrakeRB5] CONTROL PASS contact_speed={max(contact_speeds):.5f}rad/s jaw_speed={max_speed:.2e}rad/s")

    plant.SetFreeBodyPose(context, models["test_box"], RigidTransform(FLOOR_TEST_START))
    plant.SetVelocities(context, np.zeros(plant.num_velocities()))
    for _ in range(150):
        set_motor_command(plant, context, models, DEFAULT_ARM_POSE, 0.0)
        simulator.AdvanceTo(simulator.get_context().get_time() + 0.01)
    floor_z = plant.GetFreeBodyPose(context, models["test_box"]).translation()[2]
    assert np.isclose(floor_z, models["test_box_size"][2] / 2.0, atol=0.002)
    print(f"[DrakeRB5] FLOOR PASS box_z={floor_z:.3f}m")


def make_physics_scene(time_step=0.001, size=TEST_BOX_SIZE, mass=TEST_BOX_MASS):
    builder, plant, scene_graph, models = build_scene(
        True,
        time_step=time_step,
        test_box_size=size,
        test_box_mass=mass,
        with_rgbd=False,
    )
    diagram = builder.Build()
    simulator = Simulator(diagram)
    context = plant.GetMyMutableContextFromRoot(simulator.get_mutable_context())
    set_configuration(plant, context, models)
    plant.SetFreeBodyPose(context, models["test_box"], RigidTransform(TEST_BOX_START))
    set_motor_command(plant, context, models, DEFAULT_ARM_POSE, 0.0)
    return diagram, simulator, plant, scene_graph, models


def advance_command(simulator, plant, scene_graph, models, duration, arm):
    root = simulator.get_mutable_context()
    context = plant.GetMyMutableContextFromRoot(root)
    deadline = root.get_time() + duration
    max_error = max_speed = max_table_depth = 0.0
    while root.get_time() < deadline - 1e-12:
        set_motor_command(plant, context, models, arm, 0.0)
        simulator.AdvanceTo(min(deadline, root.get_time() + 0.05))
        max_error = max(
            max_error, float(np.max(np.abs(plant.GetPositions(context)[:6] - arm)))
        )
        max_speed = max(max_speed, float(np.max(np.abs(plant.GetVelocities(context)[:8]))))
        _, environment = collision_pairs(root, plant, scene_graph, models)
        max_table_depth = max(
            max_table_depth,
            max(
                (
                    depth
                    for body_a, body_b, depth in environment
                    if "table" in (body_a, body_b) and "box" not in (body_a, body_b)
                ),
                default=0.0,
            ),
        )
    return max_error, max_speed, max_table_depth


def run_collision_audit():
    builder, plant, scene_graph, models = build_scene(False, with_rgbd=False)
    diagram = builder.Build()
    root = diagram.CreateDefaultContext()
    context = plant.GetMyMutableContextFromRoot(root)
    run_model_checks(plant)
    for name, pose in SAFE_ARM_POSES.items():
        set_configuration(plant, context, models, pose, 0.0)
        internal, environment = collision_pairs(root, plant, scene_graph, models)
        assert not internal and not environment, f"safe pose {name} collides: {internal + environment}"

    set_configuration(plant, context, models, SELF_COLLISION_TEST_POSE, 0.0)
    internal, environment = collision_pairs(root, plant, scene_graph, models)
    assert internal and not environment

    rng = np.random.default_rng(0)
    lower = np.array(
        [plant.GetJointByName(name, models["rb5"]).position_lower_limit() for name in ARM_JOINT_NAMES]
    )
    upper = np.array(
        [plant.GetJointByName(name, models["rb5"]).position_upper_limit() for name in ARM_JOINT_NAMES]
    )
    self_hits = environment_hits = 0
    for _ in range(250):
        set_configuration(plant, context, models, rng.uniform(lower, upper), 0.0)
        internal, environment = collision_pairs(root, plant, scene_graph, models)
        self_hits += bool(internal)
        environment_hits += bool(environment)
    assert self_hits and environment_hits
    print(
        f"[DrakeRB5] COLLISION AUDIT PASS safe={len(SAFE_ARM_POSES)} "
        f"samples=250 self_hits={self_hits} environment_hits={environment_hits}"
    )


def run_long_stability_check():
    _, simulator, plant, scene_graph, models = make_physics_scene()
    hold_error, hold_speed, _ = advance_command(
        simulator, plant, scene_graph, models, 60.0, np.asarray(DEFAULT_ARM_POSE)
    )
    assert hold_error < 0.01 and hold_speed < 0.05
    advance_command(simulator, plant, scene_graph, models, 3.0, COLLISION_TEST_POSE)
    _, contact_speed, penetration = advance_command(
        simulator, plant, scene_graph, models, 60.0, COLLISION_TEST_POSE
    )
    assert 0.0 < penetration < 0.005 and contact_speed < 0.01
    print(
        f"[DrakeRB5] 60S STABILITY PASS hold={hold_error:.4f}rad/{hold_speed:.4f}rad/s "
        f"contact={penetration * 1000:.2f}mm/{contact_speed:.5f}rad/s"
    )


def run_repeatability_check():
    final_z, penetrations = [], []
    for _ in range(10):
        _, simulator, plant, scene_graph, models = make_physics_scene()
        advance_command(
            simulator, plant, scene_graph, models, 1.5, np.asarray(DEFAULT_ARM_POSE)
        )
        root = simulator.get_context()
        context = plant.GetMyContextFromRoot(root)
        final_z.append(plant.GetFreeBodyPose(context, models["test_box"]).translation()[2])
        _, environment = collision_pairs(root, plant, scene_graph, models)
        penetrations.append(
            max(
                (
                    depth
                    for body_a, body_b, depth in environment
                    if {body_a, body_b} == {"box", "table"}
                ),
                default=0.0,
            )
        )
    assert np.ptp(final_z) < 1e-5 and np.ptp(penetrations) < 1e-5
    print(
        f"[DrakeRB5] REPEATABILITY PASS runs=10 "
        f"z_range={np.ptp(final_z):.2g}m penetration_range={np.ptp(penetrations):.2g}m"
    )


def run_timestep_check():
    results = []
    expected_z = TABLE_CENTER[2] + TABLE_SIZE[2] / 2.0 + TEST_BOX_SIZE[2] / 2.0
    for time_step in (0.0005, 0.001, 0.002):
        _, simulator, plant, scene_graph, models = make_physics_scene(time_step=time_step)
        advance_command(
            simulator, plant, scene_graph, models, 1.5, np.asarray(DEFAULT_ARM_POSE)
        )
        context = plant.GetMyContextFromRoot(simulator.get_context())
        box_z = plant.GetFreeBodyPose(context, models["test_box"]).translation()[2]
        assert np.isclose(box_z, expected_z, atol=0.002)
        advance_command(simulator, plant, scene_graph, models, 3.0, COLLISION_TEST_POSE)
        _, speed, penetration = advance_command(
            simulator, plant, scene_graph, models, 1.0, COLLISION_TEST_POSE
        )
        assert 0.0 < penetration < 0.005 and speed < 0.01
        results.append(f"{time_step * 1000:g}ms:{penetration * 1000:.2f}mm")
    print(f"[DrakeRB5] TIMESTEP PASS {' '.join(results)}")


def run_object_sweep():
    cases = (
        ("light", 0.02, (0.02, 0.02, 0.02), 0.0),
        ("baseline", 0.20, (0.06, 0.06, 0.06), 0.0),
        ("heavy", 2.0, (0.08, 0.08, 0.08), 0.0),
        ("thin", 0.05, (0.04, 0.04, 0.005), 0.0),
        ("fast", 0.20, (0.02, 0.02, 0.02), -2.0),
    )
    table_top = TABLE_CENTER[2] + TABLE_SIZE[2] / 2.0
    results = []
    for name, mass, size, vertical_speed in cases:
        _, simulator, plant, _, models = make_physics_scene(size=size, mass=mass)
        context = plant.GetMyMutableContextFromRoot(simulator.get_mutable_context())
        if vertical_speed:
            plant.SetFreeBodySpatialVelocity(
                context,
                models["test_box"],
                SpatialVelocity(w=np.zeros(3), v=[0.0, 0.0, vertical_speed]),
            )
        min_bottom = np.inf
        deadline = simulator.get_context().get_time() + 3.0
        while simulator.get_context().get_time() < deadline - 1e-12:
            set_motor_command(plant, context, models, DEFAULT_ARM_POSE, 0.0)
            simulator.AdvanceTo(min(deadline, simulator.get_context().get_time() + 0.01))
            z = plant.GetFreeBodyPose(context, models["test_box"]).translation()[2]
            min_bottom = min(min_bottom, z - size[2] / 2.0)
        final_z = plant.GetFreeBodyPose(context, models["test_box"]).translation()[2]
        final_speed = np.linalg.norm(
            models["test_box"].EvalSpatialVelocityInWorld(context).translational()
        )
        assert min_bottom >= table_top - 0.005
        assert np.isclose(final_z, table_top + size[2] / 2.0, atol=0.002)
        assert final_speed < 0.01
        results.append(f"{name}:{final_z:.4f}m")
    print(f"[DrakeRB5] OBJECT SWEEP PASS {' '.join(results)}")


def run_camera_precision_check():
    builder, plant, _, models = build_scene(False, calibration_target=True)
    diagram = builder.Build()
    root = diagram.CreateDefaultContext()
    context = plant.GetMyMutableContextFromRoot(root)
    set_configuration(plant, context, models)
    plant.SetFreeBodyPose(context, models["calibration_sphere"], RigidTransform([0.0, 0.0, -2.0]))
    sensor_context = models["rgbd"].GetMyContextFromRoot(root)

    def center_depth():
        image = models["rgbd"].depth_image_32F_output_port().Eval(sensor_context)
        data = np.asarray(image.data).reshape(D435I_RESOLUTION[1], D435I_RESOLUTION[0])
        return float(data[int(D435I_CY), int(D435I_CX)])

    background = center_depth()
    X_world_optical = plant.CalcRelativeTransform(
        context,
        plant.world_frame(),
        plant.GetFrameByName("camera_depth_optical_frame", models["d435i"]),
    )
    target_distance = 0.5
    center = (
        X_world_optical.translation()
        + X_world_optical.rotation().matrix()[:, 2] * target_distance
    )
    plant.SetFreeBodyPose(context, models["calibration_sphere"], RigidTransform(center))
    measured = center_depth()
    expected = target_distance - 0.03
    assert abs(measured - expected) <= 0.005
    assert measured < background
    print(
        f"[DrakeRB5] D435I PRECISION PASS expected={expected:.3f}m "
        f"measured={measured:.3f}m background={background:.3f}m"
    )


def run_full_checks():
    started = time.monotonic()
    run_collision_audit()
    run_long_stability_check()
    run_repeatability_check()
    run_timestep_check()
    run_object_sweep()
    run_camera_precision_check()
    print(f"[DrakeRB5] FULL PASS elapsed={time.monotonic() - started:.1f}s")


def safe_slider_command(
    meshcat, scratch_root, plant, scene_graph, models, last_arm, last_jaw
):
    candidate_arm = [meshcat.GetSliderValue(name) for name in ARM_JOINT_NAMES]
    candidate_jaw = meshcat.GetSliderValue("PGC jaw travel (mm)") / 1000.0
    arm, jaw, internal = preview_command(
        scratch_root, plant, scene_graph, models, candidate_arm, candidate_jaw
    )
    if not internal:
        return arm, jaw
    for name, value in zip(ARM_JOINT_NAMES, last_arm):
        meshcat.SetSliderValue(name, float(value))
    meshcat.SetSliderValue("PGC jaw travel (mm)", last_jaw * 1000.0)
    links = ", ".join(sorted({f"{a}-{b}" for a, b, _ in internal}))
    print(
        f"[DrakeRB5] SELF-COLLISION REJECTED links={links} "
        f"max_penetration={max(depth for _, _, depth in internal) * 1000:.1f}mm"
    )
    return np.asarray(last_arm), last_jaw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate the assembly and exit")
    parser.add_argument(
        "--full-check", action="store_true", help="Run long stability, repeatability, and sensor checks"
    )
    parser.add_argument("--physics", action="store_true", help="Drop a test box onto the table using contact physics")
    args = parser.parse_args()

    if args.full_check:
        run_full_checks()
        return

    builder, plant, scene_graph, models = build_scene(args.physics)
    meshcat = None
    if not args.check:
        meshcat = Meshcat()
        meshcat.Delete()
        MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    diagram = builder.Build()
    simulator = Simulator(diagram)
    root_context = simulator.get_mutable_context()
    plant_context = plant.GetMyMutableContextFromRoot(root_context)
    set_configuration(plant, plant_context, models)
    if args.physics:
        plant.SetFreeBodyPose(plant_context, models["test_box"], RigidTransform(TEST_BOX_START))
        set_motor_command(plant, plant_context, models, DEFAULT_ARM_POSE, 0.0)

    if args.check:
        if args.physics:
            run_physics_check(simulator, plant, scene_graph, models)
        else:
            run_checks(diagram, root_context, plant, scene_graph, models)
        return

    scratch_root = root_context.Clone()
    last_arm, last_jaw = np.asarray(DEFAULT_ARM_POSE), 0.0

    for name, value in zip(ARM_JOINT_NAMES, DEFAULT_ARM_POSE):
        joint = plant.GetJointByName(name, models["rb5"])
        meshcat.AddSlider(
            name,
            float(joint.position_lower_limits()[0]),
            float(joint.position_upper_limits()[0]),
            0.01,
            value,
        )
    meshcat.AddSlider("PGC jaw travel (mm)", 0.0, 25.0, 0.1, 0.0)

    if args.physics:
        print(f"[DrakeRB5] Meshcat: {meshcat.web_url()}")
        print("[DrakeRB5] Physics demo; Ctrl-C to exit")
        simulator.set_target_realtime_rate(1.0)
        try:
            while True:
                arm, jaw = safe_slider_command(
                    meshcat,
                    scratch_root,
                    plant,
                    scene_graph,
                    models,
                    last_arm,
                    last_jaw,
                )
                last_arm, last_jaw = arm, jaw
                set_motor_command(plant, plant_context, models, arm, jaw)
                simulator.AdvanceTo(simulator.get_context().get_time() + 0.01)
        except KeyboardInterrupt:
            return

    print(f"[DrakeRB5] Meshcat: {meshcat.web_url()}")
    print("[DrakeRB5] Ctrl-C to exit")
    try:
        while True:
            arm, jaw = safe_slider_command(
                meshcat,
                scratch_root,
                plant,
                scene_graph,
                models,
                last_arm,
                last_jaw,
            )
            last_arm, last_jaw = arm, jaw
            set_configuration(plant, plant_context, models, arm=arm, jaw=jaw)
            diagram.ForcedPublish(root_context)
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
