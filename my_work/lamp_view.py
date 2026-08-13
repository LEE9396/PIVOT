"""실물 스캔 스탠드 램프의 탐색 과정을 Drake 로 본다.

desk_lamp.py 가 배달물을 ObjectSpec 으로 바꿔 놓았고, 여기서는 그 물체로
탐색 폐루프를 돌리며 매 단계를 Meshcat 에 띄운다.

  1) 다음에 잴 관절각을 연속 최적화로 고른다
  2) 그 자세로 램프를 접는다              <- 화면
  3) 중력 3방향으로 기울여 렌치를 잰다     <- 화면 (물체가 대신 회전한다)
  4) 질량 추정과 불확실성을 갱신한다
  5) 목표에 닿을 때까지 반복

박스가 아니라 **스캔한 실제 메시**를 그린다. Drake 의 Mesh 도형이 .stl 을
안 받으므로 .obj 로 한 번 바꿔서 쓴다.

실행:
  ../robot_learning/scripts/run_drake_env.sh python lamp_view.py
  브라우저에서 http://localhost:7000 을 연 뒤 터미널에서 Enter
"""

import argparse
import time
from pathlib import Path

import numpy as np
from pydrake.geometry import Cylinder, Mesh, Rgba, StartMeshcat
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.multibody.tree import (FixedOffsetFrame, RevoluteJoint,
                                    SpatialInertia, UnitInertia)
from pydrake.systems.framework import DiagramBuilder
from pydrake.visualization import AddDefaultVisualization

import angle_aware as aa
import density_id_drake as alg
import density_id_objects as obj
import design_core as dc
import desk_lamp as lamp
import mesh_props as mp

MM = 1e-3
CACHE = Path(__file__).resolve().parent / "cache_lamp_obj"


def ensure_obj():
    """화면용 .obj 경로. 정점 색이 있으면 색까지 들어간 파일이 나온다."""
    return {name: info["visual"] for name, info in lamp.ensure_obj().items()}


def build_lamp_plant(spec, densities, builder):
    """램프를 띄운 plant. 베이스가 자유롭게 떠 있어 중력 방향을 바꿔 보인다.

    몸체 프레임 원점 = 그 파트의 도심 (회귀행렬이 그렇게 읽는다).
    메시는 스캔 좌표에 있으므로 -도심 만큼 옮겨서 붙인다.
    """
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    geometry = lamp.part_geometry()
    frame_origin, _ = lamp.read_urdf()
    obj_files = ensure_obj()

    bodies = {}
    for part, rho in zip(spec.parts, densities):
        info = geometry[part.name]
        bodies[part.name] = plant.AddRigidBody(
            part.name,
            SpatialInertia(rho * part.volume_m3, np.zeros(3),
                           UnitInertia.SolidBox(*[d * MM for d in part.bbox_mm])))
        plant.RegisterVisualGeometry(
            bodies[part.name],
            RigidTransform(-info["centroid"]),          # 스캔좌표 -> 도심 기준
            Mesh(str(obj_files[part.name]), 1.0),
            f"{part.name}_visual", part.color)

    parts = {p.name: p for p in spec.parts}
    for joint in spec.joints:
        origin = np.array(joint.origin_in_parent_link_mm)
        on_parent = (origin
                     - np.array(parts[joint.parent].bbox_center_in_link_mm)) * MM
        on_child = (-np.array(parts[joint.child].bbox_center_in_link_mm)) * MM
        plant.AddJoint(RevoluteJoint(
            joint.name,
            plant.AddFrame(FixedOffsetFrame(
                f"{joint.name}_parent", bodies[joint.parent].body_frame(),
                RigidTransform(on_parent))),
            plant.AddFrame(FixedOffsetFrame(
                f"{joint.name}_child", bodies[joint.child].body_frame(),
                RigidTransform(on_child))),
            joint.axis, damping=0.0))
    plant.Finalize()
    return plant, bodies


def rotation_taking(a, b):
    """벡터 a 를 b 로 보내는 회전."""
    a = np.asarray(a, float) / np.linalg.norm(a)
    b = np.asarray(b, float) / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(a @ b)
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3) if c > 0 else -np.eye(3) + 2 * np.outer(a, a)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)


class Viewer:
    def __init__(self, spec, meshcat, densities):
        self.spec = spec
        self.meshcat = meshcat
        builder = DiagramBuilder()
        self.plant, self.bodies = build_lamp_plant(spec, densities, builder)
        AddDefaultVisualization(builder, meshcat)
        self.diagram = builder.Build()
        self.context = self.diagram.CreateDefaultContext()
        self.plant_context = self.plant.GetMyMutableContextFromRoot(self.context)
        self.base = self.bodies[spec.parts[0].name]
        self._gravity_arrow()

    def _gravity_arrow(self):
        self.meshcat.SetObject("/gravity/shaft", Cylinder(0.006, 0.30),
                               Rgba(0.85, 0.15, 0.15, 1.0))
        self.meshcat.SetTransform("/gravity/shaft",
                                  RigidTransform(RotationMatrix(),
                                                 [-0.28, 0.0, 0.20]))

    def show(self, theta, g_hat=(0.0, 0.0, -1.0)):
        """중력은 늘 월드 -z. 물체가 대신 기운다 (로봇이 손목을 돌리는 것)."""
        R = RotationMatrix(rotation_taking(g_hat, [0.0, 0.0, -1.0]))
        self.plant.SetFreeBodyPose(self.plant_context, self.base,
                                   RigidTransform(R, [0.0, 0.0, 0.0]))
        for joint, value in zip(self.spec.joints, np.atleast_1d(theta)):
            self.plant.GetJointByName(joint.name).set_angle(
                self.plant_context, float(value))
        self.diagram.ForcedPublish(self.context)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=0.01)
    ap.add_argument("--max-rounds", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pace", type=float, default=1.2,
                    help="한 화면을 보여주는 시간 [s]")
    ap.add_argument("--sweep", type=int, default=9,
                    help="후보를 훑어 보여주는 자세 수 (0 이면 생략)")
    ap.add_argument("--autostart", action="store_true")
    ap.add_argument("--exit-when-done", action="store_true",
                    help="탐색이 끝나면 창을 닫고 종료한다 (검사용)")
    args = ap.parse_args()

    spec = lamp.build_spec()
    lamp.report(spec)

    obj.set_measurement_averaging()
    rho_gt = obj.bind_object(spec)
    volumes = np.array([p.volume_m3 for p in spec.parts])
    gt_mass = np.array([lamp.GROUND_TRUTH[p.name]["mass_kg"] for p in spec.parts])
    obj.apply_weight_prior(spec, float(gt_mass.sum()))

    meshcat = StartMeshcat()
    print(f"\n  화면 주소 {meshcat.web_url()}")
    viewer = Viewer(spec, meshcat, rho_gt)
    viewer.show(np.zeros(len(spec.joints)))
    if not args.autostart:
        input("  브라우저에서 위 주소를 연 뒤 Enter 를 누르세요... ")

    g_dirs = dc.CANONICAL_TRIAD
    rng = np.random.default_rng(args.seed)
    Sigma, rho_hat = alg.SIGMA0.copy(), alg.MU0.copy()
    bounds = [j.limits_rad for j in spec.joints]
    blocks, rounds = [], []

    for index in range(1, args.max_rounds + 1):
        print(f"\n── round {index}")
        if args.sweep:
            grid = np.linspace(0, 1, args.sweep)
            for s in grid:                       # 후보 공간을 훑어 보여준다
                theta = np.array([lo + s * (hi - lo) for lo, hi in bounds])
                viewer.show(theta)
                time.sleep(args.pace / 6)

        theta, _ = dc.continuous_best(
            bounds, lambda t: dc.utility(t, rho_hat, Sigma, g_dirs, "D", 0.05),
            n_starts=8, seed=args.seed + index)
        theta = np.atleast_1d(theta)
        print(f"  추천 관절각 {np.round(np.degrees(theta), 1)} deg")
        viewer.show(theta)
        time.sleep(args.pace)

        sigma = np.sqrt(np.diag(aa.angle_covariance(theta, 0.05)))
        actual = theta + rng.normal(0.0, sigma)
        measured = actual + rng.normal(0.0, sigma)
        print(f"  FoundationPose 측정 {np.round(np.degrees(measured), 1)} deg")

        for g_hat in g_dirs:                     # 중력 3방향
            viewer.show(measured, g_hat)
            print(f"    중력 {np.round(g_hat, 0)} 방향으로 기울여 측정")
            time.sleep(args.pace)
        viewer.show(measured)

        y = alg.measure(actual, g_dirs=list(g_dirs), rng=rng)
        A = dc.regressor(measured, g_dirs)
        R = dc.effective_cov(measured, rho_hat, g_dirs, 0.05)
        blocks.append((A, y, R)); rounds.append((measured, y))
        Sigma = dc.posterior(Sigma, A, R)
        rho_wls = dc.wls_map(blocks, alg.MU0, alg.SIGMA0, alg.RHO_BOUNDS)
        rho_hat, _ = dc.tls_map(rounds, alg.MU0, alg.SIGMA0, alg.RHO_BOUNDS,
                                g_dirs, rho_init=rho_wls, rel_error=0.05)
        inflate = dc.residual_inflation(blocks, rho_hat)
        half = dc.half_width(Sigma, rho_hat, inflate=inflate)
        worst = dc.stopping_width(half, len(spec.parts))

        mass = rho_hat * volumes
        print(f"  갱신  95% 반폭 {100*worst:.2f}% / 목표 {100*args.target:.2f}%"
              + (f"  (잔차팽창 x{inflate:.2f})" if inflate > 1.005 else ""))
        for part, m, g, hw in zip(spec.parts, mass, gt_mass, half):
            label = lamp.GROUND_TRUTH[part.name]["label"]
            print(f"    {label:<12} 추정 {1000*m:7.1f} +/-{1000*m*hw:5.1f} g"
                  f"   [GT {1000*g:6.1f} g  오차 {100*abs(m-g)/g:5.2f}%]")
        if worst <= args.target:
            print(f"\n목표 불확실성 도달 — {index} 라운드에서 정지")
            break
    else:
        print(f"\n{args.max_rounds} 라운드까지 돌았으나 목표 미달")

    print("\n" + "=" * 62)
    print("최종 (질량으로 채점 — 스캔 부피가 재료 부피와 달라 밀도는 유효값)")
    for part, m, g in zip(spec.parts, rho_hat * volumes, gt_mass):
        label = lamp.GROUND_TRUTH[part.name]["label"]
        print(f"  {label:<12} 추정 {1000*m:7.1f} g   GT {1000*g:6.1f} g"
              f"   오차 {100*abs(m-g)/g:5.2f}%")
    print("=" * 62)
    if args.exit_when_done:
        return
    print("\n창을 닫으려면 Ctrl+C")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
