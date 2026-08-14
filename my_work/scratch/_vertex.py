"""검토: '메시 꼭짓점이 안 부딪히면 충돌 없음' 이 성립하는가.

결론부터: 성립하지 않는다. 반례를 실제로 만들어 보인다.
"""
import numpy as np
from pydrake.geometry import Box, SceneGraph
from pydrake.math import RigidTransform, RotationMatrix, RollPitchYaw
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph, CoulombFriction
from pydrake.multibody.tree import SpatialInertia, UnitInertia
from pydrake.systems.framework import DiagramBuilder


def box_vertices(dims, X):
    """상자 8 꼭짓점의 월드 좌표."""
    hx, hy, hz = np.array(dims) / 2.0
    local = np.array([[sx*hx, sy*hy, sz*hz]
                      for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    return (X.rotation().matrix() @ local.T).T + X.translation()


def point_in_box(point, dims, X):
    local = X.rotation().matrix().T @ (point - X.translation())
    return bool(np.all(np.abs(local) <= np.array(dims) / 2.0 + 1e-12))


def scene(dims_a, X_a, dims_b, X_b):
    """두 상자를 놓고 Drake 로 실제 최소거리를 잰다."""
    # 둘 다 월드에 용접하면 Drake 가 같은 강체 묶음으로 보고 쌍을 걸러낸다.
    # 자유 물체로 두고 자세를 직접 지정해야 실제 거리가 나온다.
    builder = DiagramBuilder()
    plant, sg = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    bodies = {}
    for name, dims in (("A", dims_a), ("B", dims_b)):
        model = plant.AddModelInstance(name)
        body = plant.AddRigidBody(
            name, model,
            SpatialInertia(1.0, np.zeros(3), UnitInertia.SolidBox(*dims)))
        plant.RegisterCollisionGeometry(body, RigidTransform(), Box(*dims),
                                        f"{name}_c", CoulombFriction(1, 1))
        bodies[name] = body
    plant.Finalize()
    diagram = builder.Build()
    root = diagram.CreateDefaultContext()
    context = plant.GetMyContextFromRoot(root)
    plant.SetFreeBodyPose(context, bodies["A"], X_a)
    plant.SetFreeBodyPose(context, bodies["B"], X_b)
    query = plant.get_geometry_query_input_port().Eval(context)
    pairs = query.ComputeSignedDistancePairwiseClosestPoints(1.0)
    return min((p.distance for p in pairs), default=np.inf)


print("반례 1) 십자로 겹친 두 막대")
print("  A: x 방향으로 긴 막대,  B: y 방향으로 긴 막대. 둘 다 원점 중심.")
dims_a = (0.40, 0.02, 0.02)          # x 로 김
dims_b = (0.02, 0.40, 0.02)          # y 로 김
X = RigidTransform()
va = box_vertices(dims_a, X)
vb = box_vertices(dims_b, X)
in_b = [point_in_box(p, dims_b, X) for p in va]
in_a = [point_in_box(p, dims_a, X) for p in vb]
print(f"  A 의 꼭짓점 8개 중 B 안에 있는 것: {sum(in_b)}")
print(f"  B 의 꼭짓점 8개 중 A 안에 있는 것: {sum(in_a)}")
print(f"  -> 꼭짓점 검사 판정: {'충돌' if (sum(in_b)+sum(in_a)) else '충돌 없음'}")
print(f"  Drake 실제 최소거리: {scene(dims_a, X, dims_b, X):+.4f} m"
      f"  (음수 = 파고들어 있음)")

print("\n반례 2) 얇은 판이 틈을 가로지름")
dims_a = (0.30, 0.30, 0.004)         # 넓고 얇은 판
dims_b = (0.02, 0.02, 0.30)          # 가는 기둥
Xa = RigidTransform()
Xb = RigidTransform([0.0, 0.0, 0.0])
va = box_vertices(dims_a, Xa); vb = box_vertices(dims_b, Xb)
print(f"  판의 꼭짓점 중 기둥 안: {sum(point_in_box(p, dims_b, Xb) for p in va)}")
print(f"  기둥의 꼭짓점 중 판 안: {sum(point_in_box(p, dims_a, Xa) for p in vb)}")
print(f"  Drake 실제 최소거리: {scene(dims_a, Xa, dims_b, Xb):+.4f} m")

print("\n반례 3) 모서리끼리 스치는 경우 (회전한 상자)")
dims_a = (0.20, 0.20, 0.20)
dims_b = (0.20, 0.20, 0.20)
Xa = RigidTransform()
Xb = RigidTransform(RotationMatrix(RollPitchYaw(0.6, 0.5, 0.4)),
                    [0.18, 0.18, 0.18])
va = box_vertices(dims_a, Xa); vb = box_vertices(dims_b, Xb)
print(f"  A 꼭짓점 중 B 안: {sum(point_in_box(p, dims_b, Xb) for p in va)}")
print(f"  B 꼭짓점 중 A 안: {sum(point_in_box(p, dims_a, Xa) for p in vb)}")
print(f"  Drake 실제 최소거리: {scene(dims_a, Xa, dims_b, Xb):+.4f} m")

print("\n비용 비교 (앞선 측정에서)")
print("  IK 재풀이   3497 ms/후보  (격자 25점)")
print("  충돌 질의    170 ms/후보  (격자 25점, Drake 정확 계산)")
print("  -> 정확한 충돌 질의가 이미 IK 의 1/20 이다. 꼭짓점 근사로 더 줄일")
print("     여지가 거의 없고, 줄여봐야 틀린 답을 얻는다.")
