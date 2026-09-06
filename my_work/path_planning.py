"""팔 관절공간에서 충돌 없는 경로를 찾는 RRT-Connect.

왜 필요한가
-----------
IK 는 **자세** 하나가 충돌 없음을 보장할 뿐이다. 두 자세를 직선 보간으로
이으면 그 사이는 아무도 확인하지 않는다. 실제로 확인해 보면 물체 끝 링크가
테이블을 62 mm 관통하는 구간이 나온다.

무엇을 하는가
-------------
팔 6 자유도 공간에서 시작 자세와 목표 자세를 잇는 경로를 찾는다. 물체
관절각과 손가락은 고정한 채 팔만 움직인다. 판정 기준은 IK 와 같은
"모든 충돌쌍의 최소거리 >= min_distance" 이므로, 경로 위 모든 점이
자세 검사와 같은 안전 수준을 만족한다.

RRT-Connect: 시작과 목표에서 트리를 하나씩 키우고 서로를 향해 뻗다가
만나면 끝낸다. 찾은 뒤에는 지름길 다듬기로 불필요한 우회를 걷어낸다.
"""

import numpy as np

DEFAULT_STEP_RAD = 0.20          # 트리를 한 번에 뻗는 거리
DEFAULT_EDGE_RES_RAD = 0.05      # 간선 검사 해상도 (아래 resolution_for 가 이긴다)

# 팔이 1 rad 돌 때 로봇·물체의 어떤 점도 이보다 멀리 못 움직인다.
#   RB5-850E 도달 0.85 m + AFT200·그리퍼 0.2 m + 물체 0.35 m
MAX_REACH_M = 1.4


def resolution_for(min_distance_m, max_reach_m=MAX_REACH_M):
    """검사 간격을 안전 여유에서 유도한다.

    왜 상수로 두면 안 되나
    ----------------------
    경로는 직선 구간을 잘라 그 점들에서만 충돌을 본다. 사이는 안 본다.
    간격이 0.05 rad 이면 팔 끝이 **4.5 cm 씩 건너뛴다.** 그런데 지키겠다는
    간격은 1~2 cm 다. 눈을 감았다 떴다 하며 걷는 것과 같아서, 카메라 지지봉
    (지름 3.2 cm) 이 눈 감은 구간에 통째로 들어가면 한 번도 안 보인다.

    표본 사이에 점이 최대 s 만큼 움직이고 표본에서 여유가 d 이상이면,
    사이에서 여유는 d - s/2 이상이다. 그러므로

        s <= 2d  이어야 관통이 없음을 보장
        s <=  d  이어야 여유의 절반이라도 지켜짐

    여기서는 뒤쪽을 쓴다. 여유 20 mm 기준 0.014 rad 이고, 예전 0.05 보다
    3.5 배 촘촘하다. 계산이 그만큼 늘지만 **보장하지 못하는 값을 보장한다고
    말하는 것보다 낫다.**
    """
    return max(float(min_distance_m) / float(max_reach_m), 1e-3)
DEFAULT_MAX_ITERS = 4000
DEFAULT_GOAL_BIAS = 0.10

# 계획기는 요구 간격을 **그대로** 지킨다.
#
# 예전에는 여기서 문턱을 10% 낮췄다. IK 가 최소거리 제약을 경계에 딱 붙여
# 풀어서, 계획기가 같은 값을 요구하면 시작·목표 자세가 부동소수점 차이로
# 탈락했기 때문이다. 그런데 그 방법은 **보장하는 값 자체를 깎는다** —
# 10 mm 를 요구해도 경로는 9 mm 까지 파고들 수 있었다.
#
# 지금은 반대로 IK 쪽이 IK_SLACK_M 만큼 더 요구한다(robot_scene). 만드는
# 쪽이 여유를 갖고 나오므로 계획기는 문턱을 깎을 이유가 없다.


def _required_clearance(model_a, model_b, arm_model, min_distance_m):
    return (0.0 if arm_model is not None
            and model_a == arm_model and model_b == arm_model
            else min_distance_m)


def collision_free(plant, query, min_distance_m, arm_model=None):
    """RB5 자체는 관통만 금지하고, 외부 형상에는 안전 여유를 적용한다."""
    inspector = query.inspector()
    for pair in query.ComputeSignedDistancePairwiseClosestPoints(min_distance_m):
        models = tuple(plant.GetBodyFromFrameId(inspector.GetFrameId(gid))
                       .model_instance() for gid in (pair.id_A, pair.id_B))
        required = _required_clearance(*models, arm_model, min_distance_m)
        if pair.distance < required:
            return False
    return True


def _collision_policy_self_test():
    assert _required_clearance(1, 1, 1, 0.01) == 0.0
    assert _required_clearance(1, 2, 1, 0.01) == 0.01
    print("충돌 여유 정책 자기검사 통과")


class ArmPathPlanner:
    """물체 관절각을 고정한 채 팔 경로를 계획한다."""

    def __init__(self, plant, context, arm_joints, min_distance_m,
                 fixed_positions, edge_resolution_rad=None,
                 step_rad=DEFAULT_STEP_RAD, seed=0, pose_is_valid=None):
        self.plant = plant
        self.context = context
        self.arm_joints = arm_joints
        self.indices = [joint.position_start() for joint in arm_joints]
        self.min_distance_m = min_distance_m
        self.fixed = np.asarray(fixed_positions, dtype=float).copy()
        self.edge_resolution = (resolution_for(min_distance_m)
                                if edge_resolution_rad is None
                                else edge_resolution_rad)
        self.step = step_rad
        self.rng = np.random.default_rng(seed)
        self.query_port = plant.get_geometry_query_input_port()
        self.pose_is_valid = pose_is_valid
        self.arm_model = arm_joints[0].model_instance()

        lower, upper = [], []
        for joint in arm_joints:
            lower.append(joint.position_lower_limits()[0])
            upper.append(joint.position_upper_limits()[0])
        self.lower = np.array(lower)
        self.upper = np.array(upper)
        self.checks = 0
        self._paths = {}

    def set_fixed(self, positions):
        """물체 관절각이 바뀌면 계획기가 보는 고정 자세도 갱신해야 한다."""
        positions = np.asarray(positions, dtype=float)
        if not np.array_equal(self.fixed, positions):
            self._paths.clear()
        self.fixed = positions.copy()

    # ------------------------------------------------------------------
    def full_q(self, arm_q):
        q = self.fixed.copy()
        for index, value in zip(self.indices, arm_q):
            q[index] = value
        return q

    def valid(self, arm_q):
        """IK 와 같은 기준: 모든 충돌쌍의 최소거리가 문턱 이상인가."""
        self.checks += 1
        if np.any(arm_q < self.lower) or np.any(arm_q > self.upper):
            return False
        self.plant.SetPositions(self.context, self.full_q(arm_q))
        query = self.query_port.Eval(self.context)
        clear = collision_free(self.plant, query, self.min_distance_m,
                               self.arm_model)
        return clear and (self.pose_is_valid is None or self.pose_is_valid())

    def edge_valid(self, a, b):
        """두 자세를 잇는 직선을 해상도 단위로 쪼개어 전부 검사한다."""
        distance = float(np.linalg.norm(b - a))
        steps = max(int(np.ceil(distance / self.edge_resolution)), 1)
        for index in range(1, steps + 1):
            if not self.valid(a + (b - a) * (index / steps)):
                return False
        return True

    # ------------------------------------------------------------------
    def _steer(self, source, target):
        delta = target - source
        distance = float(np.linalg.norm(delta))
        if distance <= self.step:
            return target
        return source + delta * (self.step / distance)

    def _extend(self, tree, parents, target):
        """가장 가까운 마디에서 target 쪽으로 한 걸음. 새 마디 인덱스나 None."""
        nearest = int(np.argmin([np.linalg.norm(node - target)
                                 for node in tree]))
        candidate = self._steer(tree[nearest], target)
        if not self.edge_valid(tree[nearest], candidate):
            return None
        tree.append(candidate)
        parents.append(nearest)
        return len(tree) - 1

    def plan(self, start, goal, max_iters=DEFAULT_MAX_ITERS,
             goal_bias=DEFAULT_GOAL_BIAS):
        """RRT-Connect. 실패하면 None."""
        start = np.asarray(start, dtype=float)
        goal = np.asarray(goal, dtype=float)
        key = (start.tobytes(), goal.tobytes())
        if key in self._paths:
            return self._paths[key]
        reverse = (key[1], key[0])
        if reverse in self._paths:
            return self._paths[reverse][::-1]
        if not self.valid(start) or not self.valid(goal):
            return None
        if self.edge_valid(start, goal):        # 직선으로 되면 그게 최선
            self._paths[key] = [start, goal]
            return self._paths[key]

        tree_a, parents_a = [start], [-1]
        tree_b, parents_b = [goal], [-1]
        # 두 트리를 번갈아 키우느라 매 반복마다 a 와 b 를 맞바꾼다. 그래서
        # 연결이 성사된 순간의 tree_a 가 **출발 쪽이 아닐 수 있다.** 이걸
        # 안 따지면 경로가 거꾸로 나온다 — 로봇이 목표 자세로 순간이동한 뒤
        # 거꾸로 되짚어 와서 출발점에 멈춘다 (측정 자세에 도착하지 못한다).
        # 실제로 그렇게 돌고 있었다: 화면이 134도 건너뛰고, 이동이 끝난 뒤
        # 팔은 출발 자세에 있었다.
        a_is_start = True
        for _ in range(max_iters):
            if self.rng.random() < goal_bias:
                sample = tree_b[int(self.rng.integers(len(tree_b)))]
            else:
                sample = self.rng.uniform(self.lower, self.upper)

            new_a = self._extend(tree_a, parents_a, sample)
            if new_a is not None:
                # 반대쪽 트리가 새 마디를 향해 계속 뻗는다 (connect)
                while True:
                    new_b = self._extend(tree_b, parents_b, tree_a[new_a])
                    if new_b is None:
                        break
                    if np.linalg.norm(tree_b[new_b] - tree_a[new_a]) < 1e-9:
                        path_a = self._trace(tree_a, parents_a, new_a)
                        path_b = self._trace(tree_b, parents_b, new_b)
                        path = path_a + path_b[::-1][1:]
                        if not a_is_start:
                            path = path[::-1]
                        path = self.shortcut(path)
                        # 마지막으로 한 번 더 확인한다. 방향이 틀린 경로는
                        # 조용히 잘못 움직이므로 여기서 반드시 잡아야 한다.
                        if (np.linalg.norm(path[0] - start)
                                > np.linalg.norm(path[-1] - start)):
                            path = path[::-1]
                        self._paths[key] = path
                        return path
            tree_a, tree_b = tree_b, tree_a
            parents_a, parents_b = parents_b, parents_a
            a_is_start = not a_is_start
        return None

    @staticmethod
    def _trace(tree, parents, index):
        path = []
        while index != -1:
            path.append(tree[index])
            index = parents[index]
        return path[::-1]

    # ------------------------------------------------------------------
    def shortcut(self, path, rounds=200):
        """무작위 두 점을 직선으로 이을 수 있으면 사이를 걷어낸다."""
        path = [np.asarray(p, dtype=float) for p in path]
        for _ in range(rounds):
            if len(path) <= 2:
                break
            i = int(self.rng.integers(0, len(path) - 2))
            j = int(self.rng.integers(i + 2, len(path)))
            if self.edge_valid(path[i], path[j]):
                path = path[:i + 1] + path[j:]
        return path

    # ------------------------------------------------------------------
    @staticmethod
    def resample(path, count):
        """경로를 호길이 기준으로 균등 분할해 애니메이션용 점열로 만든다."""
        path = [np.asarray(p, dtype=float) for p in path]
        lengths = [0.0]
        for a, b in zip(path[:-1], path[1:]):
            lengths.append(lengths[-1] + float(np.linalg.norm(b - a)))
        total = lengths[-1]
        if total < 1e-12:
            return [path[0]] * count
        out = []
        for fraction in np.linspace(0.0, 1.0, count):
            distance = fraction * total
            index = int(np.searchsorted(lengths, distance, side="right")) - 1
            index = min(max(index, 0), len(path) - 2)
            span = lengths[index + 1] - lengths[index]
            local = 0.0 if span < 1e-12 else (distance - lengths[index]) / span
            out.append(path[index] + (path[index + 1] - path[index]) * local)
        return out

    def path_clearance(self, path, samples_per_edge=20):
        """계획한 경로 위의 최소거리. 검증용."""
        return self.path_closest_pair(path, samples_per_edge)[0]

    def path_closest_pair(self, path, samples_per_edge=20):
        """경로 위 최소거리와 **그때 맞닿은 두 물체 이름**.

        숫자만 돌려주면 "여유 13.29 mm" 를 보고도 그것이 진짜 위험인지
        모형의 허상인지 알 수가 없다. 실제로 그 13.29 mm 는 볼트로 붙어
        있어 절대 안 변하는 두 물건 사이 거리였고, 안전 여유를 올리면
        영원히 통과 못 하는 상태였다. 이름을 같이 찍으면 바로 보인다.
        """
        worst, who = np.inf, ("", "")
        inspector = None
        for a, b in zip(path[:-1], path[1:]):
            for s in np.linspace(0.0, 1.0, samples_per_edge):
                self.plant.SetPositions(self.context, self.full_q(a + (b - a) * s))
                query = self.query_port.Eval(self.context)
                pairs = query.ComputeSignedDistancePairwiseClosestPoints(0.05)
                if not pairs:
                    continue
                near = min(pairs, key=lambda p: p.distance)
                if near.distance < worst:
                    worst = near.distance
                    if inspector is None:
                        inspector = query.inspector()
                    who = tuple(
                        self.plant.GetBodyFromFrameId(
                            inspector.GetFrameId(gid)).name()
                        for gid in (near.id_A, near.id_B))
        return worst, who[0], who[1]


if __name__ == "__main__":
    _collision_policy_self_test()
