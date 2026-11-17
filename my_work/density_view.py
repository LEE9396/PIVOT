"""탐색 결과 비교 화면 — 부위별 밀도를 메시 색으로 보여준다.

무엇을 보여주나
---------------
같은 물체를 나란히 두 벌(정답을 아는 시뮬레이션이면 세 벌) 그린다.

    [초기값]        저울 총무게만 아는 상태. 모든 부위가 같은 평균 밀도라
                    한 색으로 칠해진다.
    [탐색 결과]     라운드를 돌고 난 추정값. 부위마다 색이 갈린다.
    [정답]          시뮬레이션에서만. **채점용**이며, 탐색·정지 판단에는
                    절대 안 쓴다 (AGENTS.md 의 약속).

왜 색인가. 숫자표는 이미 터미널에 찍힌다. 사람이 알고 싶은 것은 "어느 부위가
무겁다고 나왔나" 인데, 그건 물체 그림 위에서 봐야 한 눈에 들어온다. 명도만
바뀌는 컬러맵은 인쇄·화면에서 차이가 안 보여서 **turbo(무지개)** 를 쓴다.
figures/desklamp_density_render.py 가 논문 그림에 쓴 것과 같은 컬러맵이다.

주의: 이 화면은 탐색이 **끝난 뒤에** 뜬다. 라운드 중에 정답 색을 띄우면
사람이 그걸 보고 다음 자세를 고르게 되고, 그러면 연구가 무의미해진다.

단독 실행 (색과 배치 확인용):
    cd ~/Desktop/PIVOT/my_work
    ../robot_learning/scripts/run_drake_env.sh python density_view.py \
        --object 3link
"""

import argparse
import time

import numpy as np
from pydrake.geometry import Box, Mesh, Rgba, Sphere, StartMeshcat
from pydrake.math import RigidTransform, RotationMatrix

import density_id_objects as obj

ROOT = "/density"
COLORBAR_STEPS = 24
# 컬러맵. figures/ 의 논문 그림과 같은 것을 쓴다.
COLORMAP = "turbo"


def color_ramp(value, lo, hi, alpha=1.0):
    """밀도 하나를 turbo 색으로. 정의역 밖은 양 끝 색으로 눌린다."""
    from matplotlib import colormaps

    span = max(float(hi) - float(lo), 1e-9)
    t = float(np.clip((float(value) - float(lo)) / span, 0.0, 1.0))
    r, g, b = colormaps[COLORMAP](t)[:3]
    return Rgba(float(r), float(g), float(b), float(alpha))


def color_range(columns, pad=0.05):
    """모든 열의 밀도를 한 정의역에 담는다.

    열마다 다른 정의역을 쓰면 색이 서로 다른 뜻이 되어 비교가 안 된다.
    """
    values = np.concatenate([np.asarray(c["rho"], dtype=float).ravel()
                             for c in columns])
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < 1e-6:                       # 초기값만 있는 경우
        lo, hi = lo * 0.9, hi * 1.1
    margin = pad * (hi - lo)
    return lo - margin, hi + margin


class DensityPanel:
    """부위별 밀도를 색으로 칠한 물체를 여러 벌 나란히 그리는 화면."""

    def __init__(self, spec, meshcat, theta_deg=None):
        self.spec = spec
        self.meshcat = meshcat
        self.n_part = len(spec.parts)
        # 자세는 밀도와 무관하므로 한 번만 푼다. 밀도는 아무 값이나 넣어도
        # 되지만(운동학만 쓴다) 0 은 관성이 0 이라 Finalize 가 싫어한다.
        # 길이는 부위 + 힌지 (body_table 의 행 수) 여야 한다.
        plant, bodies = obj.build_plant(
            spec, np.ones(len(obj.body_table(spec))))
        context = plant.CreateDefaultContext()
        theta = (None if theta_deg is None
                 else np.deg2rad(np.atleast_1d(theta_deg)).ravel())
        # 관절 수가 안 맞으면 기본 자세로 그린다. 결과 화면 하나 때문에
        # 세션 마지막에 예외를 내는 것이 더 나쁘다.
        if theta is not None and theta.size == plant.num_positions():
            plant.SetPositions(context, theta)
        elif theta is not None:
            print(f"  [밀도 화면] 관절각 {theta.size} 개가 물체 자유도"
                  f" {plant.num_positions()} 개와 달라 기본 자세로 그립니다")
        poses = {p.name: plant.EvalBodyPoseInWorld(context, bodies[p.name])
                 for p in spec.parts}
        # 화면 가운데에 오도록 물체의 대략 중심을 뺀다.
        origins = np.array([X.translation() for X in poses.values()])
        extents = np.array([np.asarray(p.bbox_mm) * 1e-3 for p in spec.parts])
        self.center = 0.5 * (origins.min(axis=0) + origins.max(axis=0))
        self.size = float(np.max(origins.max(axis=0) - origins.min(axis=0)
                                 + extents.max(axis=0)))
        self.poses = poses
        self._buttons = []
        # begin() 이 채운다. 라운드마다 갱신할 때 쓰는 상태다.
        self.prior = None
        self.gt = None
        self.target_rel = 0.05
        self.range = None
        self.spacing = 1.35 * self.size

    # ------------------------------------------------------------------
    def _draw_part(self, path, part, X_WP, rgba):
        """부위 하나를 한 가지 색으로 칠해 그린다.

        스캔 물체는 보통 정점 색 조각(visual_pieces)으로 그리지만, 여기서는
        **밀도가 색**이므로 원래 색을 버리고 메시 하나를 단색으로 칠한다.
        """
        X_mesh = RigidTransform(np.array(part.mesh_offset_m))
        if str(part.visual_mesh or "").lower().endswith(".gltf"):
            X_mesh = RigidTransform(RotationMatrix.MakeXRotation(-np.pi / 2),
                                    np.array(part.mesh_offset_m))
        if part.visual_mesh:
            self.meshcat.SetObject(path, Mesh(str(part.visual_mesh), 1.0), rgba)
            self.meshcat.SetTransform(path, X_WP @ X_mesh)
        elif part.visual_pieces:
            for index, (mesh_path, _) in enumerate(part.visual_pieces):
                node = f"{path}/piece_{index}"
                self.meshcat.SetObject(node, Mesh(str(mesh_path), 1.0), rgba)
                self.meshcat.SetTransform(node, X_WP @ X_mesh)
        else:
            dims = tuple(d * 1e-3 for d in part.bbox_mm)
            self.meshcat.SetObject(path, Box(*dims), rgba)
            self.meshcat.SetTransform(path, X_WP)

    def _draw_column(self, key, rho, lo, hi, offset_x):
        # 열은 x 로 벌린다. 카메라를 -y 쪽에 두므로 x 가 화면의 좌우다.
        # y 로 벌리면 열이 앞뒤로 겹쳐 보여 비교가 안 된다.
        shift = RigidTransform([offset_x - self.center[0], -self.center[1],
                                -self.center[2]])
        for part, value in zip(self.spec.parts, np.asarray(rho)[:self.n_part]):
            self._draw_part(f"{ROOT}/{key}/{part.name}", part,
                            shift @ self.poses[part.name],
                            color_ramp(value, lo, hi))

    def _draw_colorbar(self, lo, hi, x_offset):
        """색이 어떤 밀도를 뜻하는지 알려주는 막대.

        Meshcat 에는 글자를 놓을 방법이 없다. 그래서 눈금 숫자는 버튼 이름으로
        왼쪽 패널에 적는다 (이 저장소가 이미 쓰는 방식이다).
        """
        height = 0.6 * self.size
        step = height / COLORBAR_STEPS
        for index in range(COLORBAR_STEPS):
            t = (index + 0.5) / COLORBAR_STEPS
            node = f"{ROOT}/colorbar/{index}"
            self.meshcat.SetObject(node, Box(0.25 * step, 0.25 * step, step),
                                   color_ramp(lo + t * (hi - lo), lo, hi))
            self.meshcat.SetTransform(node, RigidTransform(
                [x_offset, 0.0, -0.5 * height + (index + 0.5) * step]))

    def _label(self, text):
        self.meshcat.AddButton(text)
        self._buttons.append(text)

    def clear(self):
        self.meshcat.Delete(ROOT)
        for name in self._buttons:
            self.meshcat.DeleteButton(name)
        self._buttons.clear()

    # ------------------------------------------------------------------
    # 라운드마다 갱신하는 실험용 화면 (창 4)
    # ------------------------------------------------------------------
    def begin(self, rho_prior, target_rel=0.05, gt=None, density_range=None):
        """왼쪽=초기(물 밀도), 오른쪽=탐색 후 두 벌을 세운다.

        오른쪽은 아직 결과가 없으므로 **초기값과 같은 색**으로 시작한다.
        라운드가 돌 때마다 update() 가 그 열만 다시 칠한다. 사람이 왼쪽과
        오른쪽을 나란히 두고 "얼마나 달라졌나" 를 보는 것이 목적이다.
        """
        self.clear()
        self.prior = np.asarray(rho_prior, dtype=float)
        self.target_rel = float(target_rel)
        self.gt = None if gt is None else np.asarray(gt, dtype=float)
        # 색 정의역은 처음에 정해 두고 라운드마다 바꾸지 않는다. 매번 바꾸면
        # 같은 색이 라운드마다 다른 밀도를 뜻하게 되어 비교가 안 된다.
        reference = self.prior if self.gt is None else np.concatenate(
            [self.prior, self.gt])
        self.range = (tuple(map(float, density_range)) if density_range is not None
                      else color_range([dict(rho=reference)], pad=0.6))
        self.spacing = 1.35 * self.size
        self._draw_column("before", self.prior, *self.range, -0.5 * self.spacing)
        self._draw_column("after", self.prior, *self.range, 0.5 * self.spacing)
        self._draw_colorbar(*self.range, 1.15 * self.spacing)
        self.meshcat.SetCameraPose([0.2 * self.spacing, -2.6 * self.spacing,
                                    0.8 * self.spacing], [0.0, 0.0, 0.0])
        self._labels(None, None, 0, False)
        return self

    def update(self, rho_hat, half_width_rel, round_index, converged):
        """탐색 결과를 오른쪽 열에 반영한다.

        half_width_rel 은 부위별 95 % 상대 반폭이다. 이 값이 목표보다 크면
        아직 모르는 것이고, 화면은 **각도를 다시 조정해 달라고** 말해야 한다.
        숫자만 띄우면 작업자는 언제 멈추는지 알 수 없다.
        """
        rho_hat = np.asarray(rho_hat, dtype=float)
        half = np.asarray(half_width_rel, dtype=float)
        self._draw_column("after", rho_hat, *self.range, 0.5 * self.spacing)
        self._draw_uncertainty(rho_hat, half)
        self._labels(rho_hat, half, round_index, converged)
        return self

    def _draw_uncertainty(self, rho_hat, half):
        """부위마다 불확실성 막대. 목표 안이면 초록, 밖이면 빨강.

        색(=밀도)만 보면 그 값을 믿어도 되는지 알 수 없다. 막대 길이가
        상대 반폭이고, 목표선을 넘으면 색이 바뀐다.
        """
        base = f"{ROOT}/uncertainty"
        self.meshcat.Delete(base)
        unit = 0.45 * self.size / max(self.target_rel * 3.0, 1e-6)
        for index, part in enumerate(self.spec.parts):
            value = float(np.atleast_1d(half)[index])
            length = max(min(value, self.target_rel * 3.0) * unit, 1e-4)
            inside = value <= self.target_rel
            color = (Rgba(0.15, 0.70, 0.30, 1.0) if inside
                     else Rgba(0.90, 0.25, 0.15, 1.0))
            x = 0.5 * self.spacing + (index - 0.5 * (self.n_part - 1)) \
                * 0.16 * self.size
            z = -0.62 * self.size
            node = f"{base}/{part.name}"
            self.meshcat.SetObject(
                node, Box(0.05 * self.size, 0.05 * self.size, length), color)
            self.meshcat.SetTransform(
                node, RigidTransform([x, 0.0, z + 0.5 * length]))
        # 목표선 — 이 높이를 넘으면 아직 모른다는 뜻이다.
        node = f"{base}/target"
        self.meshcat.SetObject(
            node, Box(0.6 * self.size, 0.02 * self.size, 0.004 * self.size),
            Rgba(0.1, 0.1, 0.1, 0.9))
        self.meshcat.SetTransform(node, RigidTransform(
            [0.5 * self.spacing, 0.0, -0.62 * self.size
             + self.target_rel * unit]))

    def _labels(self, rho_hat, half, round_index, converged):
        for name in list(self._buttons):
            self.meshcat.DeleteButton(name)
        self._buttons.clear()
        self._label(f"── 왼쪽 초기(물 {self.prior[0]:.0f}) │ 오른쪽 탐색 후 ──")
        if rho_hat is None:
            self._label("아직 탐색 전입니다 — 파지와 각도 조정을 끝내세요")
            return
        worst = float(np.max(half[:self.n_part]))
        self._label(f"라운드 {round_index} · 최대 반폭 {100*worst:.1f}%"
                    f" (목표 {100*self.target_rel:.1f}%)")
        for index, part in enumerate(self.spec.parts):
            value = float(rho_hat[index])
            rel = float(half[index])
            mark = "OK" if rel <= self.target_rel else "더 필요"
            row = (f"{part.name}: {self.prior[index]:.0f}"
                   f" -> {value:.0f} ±{100*rel:.1f}%  [{mark}]")
            if self.gt is not None:
                row += f"  (정답 {self.gt[index]:.0f})"
            self._label(row)
        if converged:
            self._label("목표 불확실성 도달 — 탐색을 멈춥니다")
        else:
            self._label("아직 목표에 못 미칩니다 —"
                        " 창 1 의 추천 각도로 다시 맞춰 주세요")

    # ------------------------------------------------------------------
    def show(self, columns, half_width=None):
        """columns = [dict(key, label, rho, note=None), ...] 를 나란히 그린다.

        half_width 를 주면 추정 열의 라벨에 95 % 반폭을 같이 적는다.
        """
        self.clear()
        lo, hi = color_range(columns)
        spacing = 1.35 * self.size
        first = -0.5 * spacing * (len(columns) - 1)
        for index, column in enumerate(columns):
            self._draw_column(column["key"], column["rho"], lo, hi,
                              first + index * spacing)
        # 컬러바는 맨 오른쪽 열 바로 옆에 세운다.
        self._draw_colorbar(lo, hi, first + (len(columns) - 0.4) * spacing)

        # ---- 왼쪽 패널의 글자 (버튼 이름을 라벨로 쓴다) ----
        self._label("── 부위별 밀도 [kg/m^3] ──")
        order = " | ".join(f"{c['label']}" for c in columns)
        self._label(f"왼쪽부터: {order}")
        for row, part in enumerate(self.spec.parts):
            cells = []
            for column in columns:
                value = float(np.asarray(column["rho"])[row])
                cell = f"{column['label']} {value:.0f}"
                if column.get("key") == "after" and half_width is not None:
                    cell += f"±{float(np.atleast_1d(half_width)[row]):.0f}"
                cells.append(cell)
            self._label(f"{part.name}: " + "  /  ".join(cells))
        gt = next((c for c in columns if c["key"] == "gt"), None)
        after = next((c for c in columns if c["key"] == "after"), None)
        if gt is not None and after is not None:
            errors = 100.0 * np.abs(np.asarray(after["rho"])[:self.n_part]
                                    - np.asarray(gt["rho"])[:self.n_part]) \
                / np.maximum(np.abs(np.asarray(gt["rho"])[:self.n_part]), 1e-9)
            self._label("정답 대비 오차: " + ", ".join(
                f"{p.name} {e:.2f}%" for p, e in zip(self.spec.parts, errors)))
        self._label(f"컬러바 {COLORMAP}: {lo:.0f} (파랑) ~ {hi:.0f} (빨강)")

        # 열을 한눈에 담도록 카메라를 -y 쪽으로 뺀다 (화면 좌우 = x).
        span = spacing * (len(columns) + 0.6)
        self.meshcat.SetCameraPose([0.15 * span, -0.95 * span, 0.30 * span],
                                   [0.1 * span, 0.0, 0.0])
        return self


def panel_columns(rho_prior, rho_hat, rho_gt=None):
    """show() 에 넣을 열 목록을 규약대로 만든다."""
    columns = [dict(key="before", label="초기", rho=rho_prior),
               dict(key="after", label="추정", rho=rho_hat)]
    if rho_gt is not None:
        columns.append(dict(key="gt", label="정답", rho=rho_gt))
    return columns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", default="3link",
                    choices=tuple(obj.OBJECTS) + ("desklamp",))
    ap.add_argument("--hold", type=float, default=600.0,
                    help="화면을 몇 초 열어 둘지")
    args = ap.parse_args()

    if args.object == "desklamp":
        import desk_lamp as lamp
        spec = lamp.build_spec()
    else:
        spec = obj.OBJECTS[args.object]

    rho_gt = np.array([p.rho_gt for p in spec.parts])
    # 초기값은 저울 총무게만 아는 상태 — 모든 부위가 평균 밀도 한 색.
    volume = np.array([p.volume_m3 for p in spec.parts])
    prior = np.full_like(rho_gt, float((rho_gt * volume).sum() / volume.sum()))
    # 추정값 자리에는 정답에 약간의 오차를 준 값을 넣어 색이 갈리는지 본다.
    rng = np.random.default_rng(0)
    rho_hat = rho_gt * (1.0 + 0.02 * rng.normal(size=rho_gt.size))

    meshcat = StartMeshcat()
    panel = DensityPanel(spec, meshcat)
    panel.show(panel_columns(prior, rho_hat, rho_gt),
               half_width=0.02 * rho_gt)
    print(f"\n  밀도 비교 화면   {meshcat.web_url()}")
    print(f"  {args.hold:.0f} 초 뒤 닫습니다 (Ctrl-C 로 즉시 종료)")
    try:
        time.sleep(args.hold)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
