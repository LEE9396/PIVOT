"""램프를 잡은 로봇 씬을 PNG 로 찍어 눈으로 확인한다."""
import numpy as np
from pathlib import Path
from PIL import Image
from pydrake.geometry import (ClippingRange, ColorRenderCamera, DepthRange,
                              DepthRenderCamera, MakeRenderEngineVtk,
                              RenderCameraCore, RenderEngineVtkParams)
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.sensors import CameraInfo, RgbdSensor

import density_id_objects as obj, desk_lamp as lamp, robot_scene as rs

spec = lamp.build_spec()
limits = rs.parse_joint_range(spec, None)
obj.set_measurement_averaging(); rho = obj.bind_object(spec)

builder = DiagramBuilder()
scene = rs.build_scene(spec, rho, limits, builder=builder, include_visuals=True)
plant, sg = scene["plant"], scene["scene_graph"]
sg.AddRenderer("vtk", MakeRenderEngineVtk(RenderEngineVtkParams()))
info = CameraInfo(900, 700, np.pi / 4)
core = RenderCameraCore("vtk", info, ClippingRange(0.05, 10.0), RigidTransform())
color = ColorRenderCamera(core, False)
depth = DepthRenderCamera(core, DepthRange(0.05, 10.0))

def look_at(eye, target):
    z = np.asarray(target) - np.asarray(eye); z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0]); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return RigidTransform(RotationMatrix(np.column_stack([x, y, z])), np.asarray(eye))

views = {"front": ((1.6, -1.5, 1.9), (0.15, -0.6, 1.9)),
         "side":  ((-1.3, -1.4, 2.1), (0.15, -0.6, 2.0))}
sensors = {}
for name, (eye, tgt) in views.items():
    s = builder.AddSystem(RgbdSensor(plant.GetBodyFrameIdOrThrow(
        plant.world_body().index()), look_at(eye, tgt), color, depth))
    builder.Connect(sg.get_query_output_port(), s.query_object_input_port())
    sensors[name] = s
diagram = builder.Build()
root = diagram.CreateDefaultContext()
ctx = plant.GetMyContextFromRoot(root)
plant.SetPositions(ctx, plant.GetPositions(ctx))

out = Path("/tmp/claude-1000/-home-junhyeoklee-Desktop/7f1a95d1-e992-4dbb-b36c-81c46ea30c51/scratchpad")
for name, s in sensors.items():
    img = s.color_image_output_port().Eval(s.GetMyContextFromRoot(root))
    Image.fromarray(img.data[:, :, :3]).save(out / f"lamp_{name}.png")
    print(f"saved lamp_{name}.png")
print("색:  link_2 어두운 청회색   link_3 밝은 회색   link_1 금색")
