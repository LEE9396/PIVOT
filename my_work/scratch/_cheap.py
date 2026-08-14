"""제안 검토: IK 재풀이 대신 '팔 고정 + 충돌만 확인' 으로 충분한가.

핵심 주장
---------
IK 가 거는 제약은 세 가지다.
  (a) 중력 방향   : 센서 프레임이 월드 -z 를 g_hat 으로 본다
  (b) 작업공간    : 센서 원점이 상자 안에 있다
  (c) 최소거리    : 아무 것도 안 부딪힌다

센서 프레임은 **잡힌 base part 에 용접**돼 있고, base part 는 그리퍼에
용접돼 있다. 그러니 (a)(b) 는 팔 관절각만으로 정해지고 **물체 관절각 theta
와 무관**하다. theta 에 의존하는 것은 (c) 충돌뿐이다.

맞다면: theta 가 흔들려도 팔 자세는 그대로 두고 충돌만 다시 보면 된다.
IK 최적화(비싸고 초기추측에 흔들림)를 충돌 질의(싸고 결정적)로 바꿀 수 있다.
"""

import sys as _sys, pathlib as _pathlib
# 이 폴더는 my_work 밖이라 형제 모듈이 안 보인다. run_drake_env.sh 가
# PYTHONPATH 를 지우므로 (ROS 오염 제거) 환경변수로는 못 넣는다.
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import time
import numpy as np
import angle_aware as aa, density_id_drake as alg, density_id_objects as obj
import desk_lamp as lamp, robot_scene as rs

spec = lamp.build_spec()
limits = rs.parse_joint_range(spec, None)
obj.set_measurement_averaging(); rho = obj.bind_object(spec)
ck = rs.PoseChecker(spec, densities=rho, joint_limits_rad=limits,
                    min_distance_m=0.006)
lows = np.array([lo for lo, _ in limits]); highs = np.array([hi for _, hi in limits])
rng = np.random.default_rng(0)

# --- 주장 검증 1: 팔을 고정한 채 theta 만 바꾸면 센서 프레임이 안 움직이는가
theta0 = np.deg2rad([-87.4, 41.8])
ck._last_solution = None
arm_q = ck.solve(theta0, alg.G_DIRS[0])
assert arm_q is not None
X0 = None
worst = 0.0
for _ in range(30):
    th = np.clip(theta0 + rng.normal(0, np.deg2rad(8.0), 2), lows, highs)
    q = np.asarray(arm_q).copy()
    for j, v in zip(ck.object_joints, th):
        q[j.position_start()] = v
    ck.plant.SetPositions(ck.context, q)
    X = ck.plant.CalcRelativeTransform(ck.context, ck.plant.world_frame(),
                                       ck.sensor_frame)
    if X0 is None:
        X0 = X
    else:
        worst = max(worst, float(np.linalg.norm(X.translation() - X0.translation())),
                    float(np.linalg.norm(X.rotation().matrix() - X0.rotation().matrix())))
print(f"주장 1) 팔 고정 + theta 흔들기 -> 센서 프레임 변화 최대 {worst:.3e}")
print(f"        (0 이면 중력방향·작업공간 제약은 theta 와 무관하다는 뜻)\n")

# --- 주장 검증 2: 판정이 일치하는가, 그리고 얼마나 싼가
def probes(theta, n_side):
    sig = np.sqrt(np.diag(aa.angle_covariance(theta, 0.05)))
    axes = [np.linspace(t - 1.96*s, t + 1.96*s, n_side)
            for t, s in zip(theta, sig)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 2)
    return np.clip(grid, lows, highs)

def by_ik(theta, n_side):
    for p in probes(theta, n_side):
        if any(v is None for v in ck.solutions_for(p).values()):
            return False
    return True

def by_collision(theta, n_side):
    """팔 자세는 명목 각도에서 한 번만 풀고, 이후엔 충돌만 본다."""
    ck._last_solution = None
    arms = ck.solutions_for(theta)
    if any(v is None for v in arms.values()):
        return False
    for p in probes(theta, n_side):
        for arm in arms.values():
            if not ck.arm_pose_is_clear(arm, p):
                return False
    return True

rng2 = np.random.default_rng(1)
cases = [np.clip(lows + (highs - lows) * rng2.random(2), lows, highs)
         for _ in range(25)]
cases.append(theta0)

for n_side in (2, 5):
    t0 = time.perf_counter(); a = [by_ik(c, n_side) for c in cases]
    t_ik = time.perf_counter() - t0
    t0 = time.perf_counter(); b = [by_collision(c, n_side) for c in cases]
    t_col = time.perf_counter() - t0
    agree = sum(x == y for x, y in zip(a, b))
    print(f"격자 {n_side}x{n_side} ({n_side**2}점)")
    print(f"  IK 재풀이   통과 {sum(a):>2}/{len(cases)}   {t_ik:6.2f}s"
          f"  ({1000*t_ik/len(cases):.0f} ms/후보)")
    print(f"  충돌만      통과 {sum(b):>2}/{len(cases)}   {t_col:6.2f}s"
          f"  ({1000*t_col/len(cases):.0f} ms/후보)   {t_ik/max(t_col,1e-9):.0f}배 빠름")
    print(f"  판정 일치   {agree}/{len(cases)}")
    diff = [(np.round(np.degrees(c),1), x, y)
            for c, x, y in zip(cases, a, b) if x != y]
    for c, x, y in diff[:5]:
        print(f"    {c}  IK={'O' if x else '.'}  충돌={'O' if y else '.'}")
    print()
