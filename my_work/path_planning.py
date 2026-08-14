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
DEFAULT_EDGE_RES_RAD = 0.05      # 간선 검사 해상도
DEFAULT_MAX_ITERS = 4000
DEFAULT_GOAL_BIAS = 0.10

# IK 는 최소거리 제약을 경계에 딱 붙여 푼다. 목표 자세가 정확히 6.0000 mm
# 로 나오므로, 계획기가 같은 값을 요구하면 부동소수점 차이로 탈락한다.
# 그래서 계획기 문턱을 조금 낮춰 IK 해가 항상 유효하도록 한다.
PLANNER_MARGIN_RATIO = 0.9


class ArmPathPlanner:
    """물체 관절각을 고정한 채 팔 경로를 계획한다."""

    def __init__(self, plant, context, arm_joints, min_distance_m,
                 fixed_positions, edge_resolution_rad=DEFAULT_EDGE_RES_RAD,
                 step_rad=DEFAULT_STEP_RAD, seed=0):
        self.plant = plant
        self.context = context
        self.arm_joints = arm_joints
        self.indices = [joint.position_start() for joint in arm_joints]
        # IK 해가 경계에 붙어 있으므로 계획기 문턱을 약간 낮춘다.
        self.min_distance_m = min_distance_m * PLANNER_MARGIN_RATIO
        self.required_distance_m = min_distance_m
        self.fixed = np.asarray(fixed_positions, dtype=float).copy()
        self.edge_resolution = edge_resolution_rad
        self.step = step_rad
        self.rng = np.random.default_rng(seed)
        self.query_port = plant.get_geometry_query_input_port()

        lower, upper = [], []
        for joint in arm_joints:
            lower.append(joint.position_lower_limits()[0])
            upper.append(joint.position_upper_limits()[0])
        self.lower = np.array(lower)
        self.upper = np.array(upper)
        self.checks = 0

    def set_fixed(self, positions):
        """물체 관절각이 바뀌면 계획기가 보는 고정 자세도 갱신해야 한다."""
        self.fixed = np.asarray(positions, dtype=float).copy()

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
        pairs = query.ComputeSignedDistancePairwiseClosestPoints(
            self.min_distance_m)
        return all(pair.distance >= self.min_distance_m for pair in pairs)

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
        if not self.valid(start) or not self.valid(goal):
            return None
        if self.edge_valid(start, goal):        # 직선으로 되면 그게 최선
            return [start, goal]

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
        worst = np.inf
        for a, b in zip(path[:-1], path[1:]):
            for s in np.linspace(0.0, 1.0, samples_per_edge):
                self.plant.SetPositions(self.context, self.full_q(a + (b - a) * s))
                query = self.query_port.Eval(self.context)
                pairs = query.ComputeSignedDistancePairwiseClosestPoints(0.05)
                if pairs:
                    worst = min(worst, min(p.distance for p in pairs))
        return worst
