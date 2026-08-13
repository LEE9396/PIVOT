"""알고리즘이 자세를 탐색·선택·측정하는 과정을 Drake Meshcat 으로 재생한다.

라운드마다 세 단계를 보여준다.
  1) 탐색 : 후보 격자를 훑는다. 관절이 고정되지 않는 자세는 건너뛴다.
  2) 선택 : 정보이득이 가장 큰 자세에서 멈춘다.
  3) 측정 : 중력 3방향으로 물체를 다시 세운다(로봇이 손목을 돌리는 동작).
            물체가 기울고 중력 화살표는 항상 아래를 가리킨다.

계산은 density_id_drake 의 함수를 그대로 부르므로, 화면에 보이는 선택은
알고리즘이 실제로 하는 선택과 같다.

실행:
    cd ~/Desktop/HTD-main/my_work
    ../robot_learning/scripts/run_drake_env.sh python explore_view.py --object 3link
    ../robot_learning/scripts/run_drake_env.sh python explore_view.py --object 2link
"""

import argparse
import time

import numpy as np
from pydrake.geometry import Cylinder, Meshcat, Rgba, StartMeshcat
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.systems.framework import DiagramBuilder
from pydrake.visualization import AddDefaultVisualization

import density_id_drake as alg
import density_id_objects as obj

MM = 1e-3


def rotation_taking(a, b):
    """벡터 a 를 b 로 보내는 회전행렬."""
    a = np.asarray(a, dtype=float) / np.linalg.norm(a)
    b = np.asarray(b, dtype=float) / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3) if c > 0 else -np.eye(3) + 2 * np.outer(a, a) * 0
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def build_floating(spec, densities, builder):
    """base 를 고정하지 않고 띄운다. 중력 방향을 바꿔 보이기 위함."""
    from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
    from pydrake.multibody.tree import (
        FixedOffsetFrame,
        RevoluteJoint,
        SpatialInertia,
        UnitInertia,
    )
    from pydrake.geometry import Box

    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    parts = {p.name: p for p in spec.parts}
    bodies = {}
    for part, rho in zip(spec.parts, densities):
        dims_m = tuple(d * MM for d in part.bbox_mm)
        bodies[part.name] = plant.AddRigidBody(
            part.name,
            SpatialInertia(rho * part.volume_m3, np.zeros(3),
                           UnitInertia.SolidBox(*dims_m)),
        )
        obj.register_part_visual(plant, bodies[part.name], part, dims_m)
    # 힌지도 부위 하나다. 자식 링크의 핀 축(+도심 오프셋)에 용접한다.
    n_part = len(spec.parts)
    for index, (joint, volume, _) in enumerate(obj.hinge_bodies(spec)):
        rho = densities[n_part + index]
        dims_m = obj.hinge_dims_m(volume)
        name = f"{joint.name}_hinge"
        bodies[name] = plant.AddRigidBody(
            name, SpatialInertia(rho * volume, np.zeros(3),
                                 UnitInertia.SolidBox(*dims_m)))
        plant.RegisterVisualGeometry(bodies[name], RigidTransform(),
                                     Box(*dims_m), f"{name}_visual",
                                     (0.15, 0.15, 0.18, 1.0))
        on_child = ((-np.array(parts[joint.child].bbox_center_in_link_mm)
                     + np.array(joint.hinge_com_offset_mm)) * MM)
        plant.WeldFrames(
            plant.AddFrame(FixedOffsetFrame(
                f"{name}_mount", bodies[joint.child].body_frame(),
                RigidTransform(on_child))),
            bodies[name].body_frame())

    for joint in spec.joints:
        origin = np.array(joint.origin_in_parent_link_mm)
        on_parent = (origin
                     - np.array(parts[joint.parent].bbox_center_in_link_mm)) * MM
        child_origin = np.array(getattr(joint, "origin_in_child_link_mm", None)
                                or (0.0, 0.0, 0.0))
        on_child = (child_origin
                    - np.array(parts[joint.child].bbox_center_in_link_mm)) * MM
        plant.AddJoint(
            RevoluteJoint(
                joint.name,
                plant.AddFrame(FixedOffsetFrame(
                    f"{joint.name}_parent", bodies[joint.parent].body_frame(),
                    RigidTransform(on_parent))),
                plant.AddFrame(FixedOffsetFrame(
                    f"{joint.name}_child", bodies[joint.child].body_frame(),
                    RigidTransform(on_child))),
                joint.axis, damping=0.0,
            )
        )
    plant.Finalize()
    return plant, bodies


class Player:
    def __init__(self, spec, meshcat, plant, bodies, context, diagram):
        self.spec = spec
        self.meshcat = meshcat
        self.plant = plant
        self.bodies = bodies
        self.context = context
        self.plant_context = plant.GetMyMutableContextFromRoot(context)
        self.diagram = diagram
        self.base = bodies[spec.parts[0].name]
        self.X_SB = RigidTransform(
            np.array(spec.base_bbox_center_in_sensor_mm) * MM
        )
        self._draw_gravity_arrow()

    def _draw_gravity_arrow(self):
        """중력 방향은 항상 월드 -z. 물체가 대신 기운다."""
        self.meshcat.SetObject("/gravity/shaft", Cylinder(0.004, 0.22),
                               Rgba(0.85, 0.15, 0.15, 1.0))
        self.meshcat.SetTransform(
            "/gravity/shaft",
            RigidTransform(RotationMatrix(), [-0.16, 0.0, 0.16]),
        )

    def show(self, theta, g_hat=(0.0, 0.0, -1.0)):
        R = RotationMatrix(rotation_taking(g_hat, [0.0, 0.0, -1.0]))
        X_WS = RigidTransform(R, [0.0, 0.0, 0.12])
        self.plant.SetFreeBodyPose(self.plant_context, self.base,
                                   X_WS @ self.X_SB)
        for joint, value in zip(self.spec.joints, np.atleast_1d(theta)):
            self.plant.GetJointByName(joint.name).set_angle(
                self.plant_context, float(value)
            )
        self.diagram.ForcedPublish(self.context)


def run(spec, hinge, safety, density_scale, n_rounds, pace):
    rho_gt = obj.bind_object(spec, hinge=hinge, safety=safety,
                             density_scale=density_scale)
    candidates = alg.candidate_grid()
    grid = obj.full_grid(spec)
    if not candidates:
        print("관절이 고정되는 자세가 없다. 힌지 토크나 밀도를 조정해야 한다.")
        return

    meshcat = StartMeshcat()
    builder = DiagramBuilder()
    plant, bodies = build_floating(spec, rho_gt, builder)
    AddDefaultVisualization(builder, meshcat)
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    player = Player(spec, meshcat, plant, bodies, context, diagram)

    print(f"\n{spec.label}")
    print(f"힌지 {hinge.label}, 유지토크 {hinge.holding_torque_nm} N·m,"
          f" 안전배수 {safety}")
    print(f"후보 격자 {len(grid)}개 중 관절이 고정되는 자세 {len(candidates)}개\n")
    print(f"Meshcat: {meshcat.web_url()}")
    player.show(candidates[0])
    input("브라우저에서 위 주소를 연 뒤 Enter를 누르세요...")

    Sigma = alg.SIGMA0.copy()
    A_all = np.empty((0, alg.P))
    y_all = np.empty(0)
    rng = np.random.default_rng(1)

    for round_index in range(n_rounds):
        print(f"\n===== round {round_index + 1} =====")

        # 1) 탐색 — 후보를 훑으며 정보이득을 계산한다.
        print("[탐색] 후보 자세를 훑는다")
        gains = []
        for theta in candidates:
            gain = alg.info_gain(alg.regressor(theta), Sigma)
            gains.append(gain)
            player.show(theta)
            time.sleep(pace)
        gains = np.array(gains)
        order = np.argsort(gains)[::-1]
        for rank in range(min(3, len(order))):
            deg = np.round(np.degrees(candidates[order[rank]]), 0)
            print(f"   {rank + 1}위 q={deg}  정보이득 {gains[order[rank]]:.3f}")

        # 2) 선택
        theta = candidates[int(np.argmax(gains))]
        torque = obj.max_joint_torques(spec, [theta], rho_gt)[0]
        print(f"[선택] q={np.round(np.degrees(theta), 0)}"
              f"  관절토크 {np.round(torque, 3)} N·m"
              f" (한계 {hinge.holding_torque_nm / safety:.3f})")
        player.show(theta)
        time.sleep(1.0)

        # 3) 측정 — 중력 3방향으로 물체를 다시 세운다.
        print("[측정] 중력 3방향으로 자세를 바꾼다")
        for g_hat in alg.G_DIRS:
            player.show(theta, g_hat)
            time.sleep(0.9)
        player.show(theta)

        A = alg.regressor(theta)
        y = alg.measure(theta, rng=rng)
        A_all = np.vstack([A_all, A])
        y_all = np.concatenate([y_all, y])
        Sigma = alg.posterior_covariance(Sigma, A)
        rho_hat = alg.constrained_map(A_all, y_all)

        print(f"[추정] {'part':<13}{'GT':>8}{'추정':>10}{'오차':>9}")
        for part, gt, est in zip(spec.parts, rho_gt, rho_hat):
            print(f"       {part.name:<13}{gt:>8.0f}{est:>10.1f}"
                  f"{100 * abs(est - gt) / gt:>8.2f}%")
        print(f"       RMSE {np.sqrt(np.mean((rho_hat - rho_gt) ** 2)):.1f}"
              f" kg/m^3,  사후 최대 표준편차"
              f" {np.sqrt(np.linalg.eigvalsh(Sigma).max()):.1f}")

    input("\n끝났습니다. Enter를 누르면 종료합니다...")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--object", choices=tuple(obj.OBJECTS), default="2link")
    parser.add_argument("--hinge-torque", type=float,
                        default=obj.MG_PLASTIC_DEFAULT_NM)
    parser.add_argument("--safety", type=float, default=obj.DEFAULT_SAFETY)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--pace", type=float, default=0.08,
                        help="후보 하나를 보여주는 시간 [s]")
    parser.add_argument("--no-auto-scale", action="store_true")
    args = parser.parse_args()

    spec = obj.OBJECTS[args.object]
    hinge = obj.Hinge(obj.HINGES["mg_plastic"].label, args.hinge_torque,
                      obj.HINGES["mg_plastic"].note)
    scale = 1.0
    if not args.no_auto_scale:
        scale = min(1.0, obj.max_feasible_density_scale(spec, hinge, args.safety))
    run(spec, hinge, args.safety, scale, args.rounds, args.pace)


if __name__ == "__main__":
    main()
