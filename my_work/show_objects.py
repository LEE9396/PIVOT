"""3링크(실물 CAD)와 6링크(합성 사슬)를 한 Meshcat 화면에 나란히 띄운다.

논문 그림용 스크린샷을 찍기 위한 화면이다. 겉모습은 render_nlink_paper.py 의
Fig. 4 를 따른다: 회색-파랑 단계 색, 검은 모서리. 관절에는 실제 토크힌지
(명가철물 HC-TC3840, cache_hinge_obj/) 를 단다. 배경은 Drake 기본.

  ../robot_learning/scripts/run_drake_env.sh python show_objects.py
  ../robot_learning/scripts/run_drake_env.sh python show_objects.py --pose mid --gap 0.45

브라우저 주소는 터미널에 찍힌다.

패널
  - "<물체> jointN 각도" 슬라이더: 바꾸면 아래 "= NN°" 읽음표가 같이 바뀐다.
  - "<물체> <부위> 색상/채도/밝기": 부위 색. 기본값이 논문 그림의 회색 단계.
  - --colors colors.json ("3link/link0_base": [r,g,b]) 을 주면 슬라이더보다 우선.
  - --angles 3link=20,150 6link=20,150,60,130,35 처럼 시작 각도를 줄 수 있다.

6링크는 dynex/exp_nlink.py 와 같은 규칙(seed=1000)으로 만든다. 그래야 논문
표의 물체와 그림의 물체가 같다.
"""
import argparse
import colorsys
import json
import time
from pathlib import Path

import numpy as np
from pydrake.geometry import (Mesh, MeshcatVisualizer, MeshcatVisualizerParams, Rgba,
                              StartMeshcat)
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.systems.framework import DiagramBuilder

import density_id_objects as obj
import nlink

MM = 1e-3
POSES = {"extended": 0.0, "mid": 90.0, "folded": 180.0}

# render_nlink_paper.py 와 같은 값. 부위 순서대로 어두운 -> 밝은.
GREYS = ["#3f4a56", "#5d6b7a", "#7d8b99", "#9caab6", "#bcc7d0", "#dae1e7"]
EDGE = "#20262c"
HINGE_RGBA = Rgba(0.13, 0.13, 0.14, 1.0)         # 흑색도장

HERE = Path(__file__).resolve().parent
HINGE_DIR = HERE / "cache_hinge_obj"
# 힌지 좌표계(핀 축 = z, 판 바닥 = y -5.5). 부모 쪽 날개는 -x, 자식 쪽은 +x.
HINGE_LEAVES = {"parent": ["HC-TC3840_leaf_parent.obj"],
                "child": ["HC-TC3840_leaf_child_lower.obj",
                          "HC-TC3840_leaf_child_upper.obj"]}
HINGE_SEAT_MM = 5.5     # 판 바닥에서 핀 축까지. 링크 표면 위에 얹기 위한 값.

# 논문 그림의 시점: matplotlib view_init(elev=20, azim=-60) 과 같은 방향.
VIEW_ELEV_DEG, VIEW_AZIM_DEG = 20.0, -60.0


def hex_rgb(h):
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (1, 3, 5))


def view_direction():
    """물체에서 카메라를 향하는 단위벡터."""
    e, a = np.radians(VIEW_ELEV_DEG), np.radians(VIEW_AZIM_DEG)
    return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


def camera_right(toward_cam):
    """카메라 화면의 오른쪽 방향 (세계 좌표)."""
    right = np.cross(-toward_cam, [0.0, 0.0, 1.0])
    return right / np.linalg.norm(right)


def box_edges(dims_m):
    """상자 12 모서리의 시작점/끝점 (3xN)."""
    d = np.array(dims_m) / 2.0
    c = np.array([[sx * d[0], sy * d[1], sz * d[2]]
                  for sx, sy, sz in [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
                                     (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]])
    pairs = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    return c[[p[0] for p in pairs]].T, c[[p[1] for p in pairs]].T


class Shown:
    """물체 하나: plant 와 Meshcat 경로."""

    def __init__(self, builder, meshcat, spec, prefix, offset_m):
        # build_plant 는 base 를 세계 원점에 용접하므로, 두 물체를 떨어뜨리려면
        # plant 가 아니라 Meshcat 쪽 경로 전체를 옮긴다.
        self.spec, self.prefix, self.meshcat = spec, prefix, meshcat
        self.offset = np.array(offset_m, dtype=float)
        rho = [row["rho_gt"] for row in obj.body_table(spec)]
        self.plant, self.bodies = obj.build_plant(spec, rho, builder=builder)
        self.scene_graph = builder.GetSubsystemByName("scene_graph")
        self.plant.set_name(f"plant_{prefix}")
        self.scene_graph.set_name(f"scene_graph_{prefix}")
        params = MeshcatVisualizerParams()
        params.prefix = prefix
        MeshcatVisualizer.AddToBuilder(builder, self.scene_graph, meshcat, params)
        meshcat.SetTransform(f"/drake/{prefix}", RigidTransform(self.offset))
        self.parts = {p.name: p for p in spec.parts}

    # --- 경로 --------------------------------------------------------------
    def body_path(self, body_name):
        return f"/drake/{self.prefix}/{body_name}"

    def geometry_paths(self, part):
        """부위의 화면 형상이 Meshcat 에서 갖는 경로들 (색을 바꿀 자리)."""
        # MeshcatVisualizer 는 "prefix/몸체이름/형상이름" 으로 올린다 (HasPath 로 확인).
        body = self.bodies[part.name]
        inspector = self.scene_graph.model_inspector()
        return [f"{self.body_path(body.name())}/{inspector.GetName(gid)}"
                for gid in self.plant.GetVisualGeometriesForBody(body)]

    # --- 관절 기하 -----------------------------------------------------------
    def hinge_frame(self, joint):
        """힌지 좌표계를 부모/자식 몸체 프레임에 놓는 변환 두 개.

        핀은 링크 표면 위 5 mm 에 있고 힌지 판 바닥은 핀 아래 5.5 mm 이므로,
        판이 표면에 얹히도록 힌지를 표면 법선 쪽으로 0.5 mm 띄운다. 핀 축이
        관절 축과 0.5 mm 어긋나지만 화면에서는 보이지 않는다.
        각도 0 에서 자식 프레임은 부모 프레임과 같은 방향이므로 같은 회전을 쓴다.
        """
        parent = self.parts[joint.parent]
        origin = np.array(joint.origin_in_parent_link_mm, dtype=float)
        normal = origin - np.array(parent.bbox_center_in_link_mm)
        normal[0] = 0.0
        normal /= np.linalg.norm(normal)
        x_axis = np.array([1.0, 0.0, 0.0])
        R = RotationMatrix(np.column_stack([x_axis, normal, np.cross(x_axis, normal)]))
        lift = (HINGE_SEAT_MM - 5.0) * normal * MM
        on_parent = (origin - np.array(parent.bbox_center_in_link_mm)) * MM + lift
        on_child = self.joint_point_in_child(joint) + lift
        return RigidTransform(R, on_parent), RigidTransform(R, on_child)

    def joint_point_in_child(self, joint):
        """관절 축 위의 점 (자식 몸체 프레임, m)."""
        child = self.parts[joint.child]
        child_origin = np.array(joint.origin_in_child_link_mm or (0.0, 0.0, 0.0), dtype=float)
        return (child_origin - np.array(child.bbox_center_in_link_mm)) * MM

    # --- 장식: 모서리, 힌지 ------------------------------------------------------
    def add_edges(self):
        for part in self.spec.parts:
            start, end = box_edges([d * MM * 1.003 for d in part.bbox_mm])
            self.meshcat.SetLineSegments(f"{self.body_path(part.name)}/edges",
                                         start, end, 1.0, Rgba(*hex_rgb(EDGE), 1.0))

    def add_hinges(self):
        for joint in self.spec.joints:
            # 추정기용 상자 힌지는 숨기고 실제 힌지 메시를 단다.
            self.meshcat.SetProperty(self.body_path(f"{joint.name}_hinge"), "visible", False)
            X_parent, X_child = self.hinge_frame(joint)
            for side, X, body in (("parent", X_parent, joint.parent),
                                  ("child", X_child, joint.child)):
                for k, fname in enumerate(HINGE_LEAVES[side]):
                    path = f"{self.body_path(body)}/hinge_{joint.name}_{side}{k}"
                    self.meshcat.SetObject(path, Mesh(str(HINGE_DIR / fname), MM), HINGE_RGBA)
                    self.meshcat.SetTransform(path, X)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose", choices=POSES, default="mid", help="시작 자세")
    ap.add_argument("--gap", type=float, default=0.45,
                    help="두 물체 사이 간격 [m] (화면 오른쪽 방향)")
    ap.add_argument("--seed", type=int, default=1000, help="6링크 링크 길이 seed")
    ap.add_argument("--colors", type=Path, default=None,
                    help='"3link/link0_base": [r,g,b] (0~1) JSON. 슬라이더보다 우선한다')
    ap.add_argument("--no-hinge", action="store_true", help="실제 힌지 메시를 달지 않는다")
    ap.add_argument("--cam-dist", type=float, default=0.75, help="시작 카메라 거리 [m]")
    ap.add_argument("--angles", nargs="*", default=[], metavar="OBJ=DEG,DEG,...",
                    help="시작 관절각 [deg]. 예: 3link=20,150 6link=20,150,60,130,35 "
                         "(논문 Fig.4 첫 줄). --pose 보다 우선한다")
    a = ap.parse_args()
    start_deg = {}
    for item in a.angles:
        key, _, vals = item.partition("=")
        start_deg[key] = [float(v) for v in vals.split(",") if v]

    meshcat = StartMeshcat()
    meshcat.Delete()
    meshcat.DeleteAddedControls()
    # 배경·격자·축은 Drake 기본 그대로 둔다. 흰 배경이 필요하면 브라우저 패널에서
    # Background/Grid/Axes 를 끈다.

    toward_cam0 = view_direction()
    right0 = camera_right(toward_cam0)
    builder = DiagramBuilder()
    shown = [
        Shown(builder, meshcat, obj.THREE_LINK, "3link", (0.0, 0.0, 0.0)),
        Shown(builder, meshcat, nlink.make_spec(6, seed=a.seed), "6link", right0 * a.gap),
    ]
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    diagram.ForcedPublish(context)          # 몸체 경로가 먼저 생겨야 그 아래에 붙일 수 있다
    time.sleep(0.3)
    for s in shown:
        s.add_edges()
        if not a.no_hinge:
            s.add_hinges()

    target = right0 * (a.gap / 2.0) + np.array([0.1, 0.0, 0.0])
    meshcat.SetCameraPose(target + toward_cam0 * a.cam_dist, target)

    # --- 각도 슬라이더와 읽음표 -------------------------------------------
    # Meshcat 패널에는 글자만 놓는 칸이 없어서, 이름이 곧 내용인 버튼을 읽음표로
    # 쓴다. 값이 바뀌면 지우고 새 이름으로 다시 만든다.
    angle_sliders, readouts = [], {}
    for s in shown:
        given = start_deg.get(s.prefix, [])
        for j, joint in enumerate(s.spec.joints):
            lo, hi = np.degrees(joint.limits_rad)
            name = f"{s.prefix} {joint.name} 각도"
            deg0 = float(np.clip(given[j] if j < len(given) else POSES[a.pose], lo, hi))
            meshcat.AddSlider(name, lo, hi, 1.0, deg0)
            readouts[name] = f"  = {deg0:.0f}°  ({s.prefix} {joint.name})"
            meshcat.AddButton(readouts[name])
            angle_sliders.append((s, name))

    # --- 색 슬라이더 ------------------------------------------------------
    color_sliders = []
    for s in shown:
        for i, part in enumerate(s.spec.parts):
            h, l, sat = colorsys.rgb_to_hls(*hex_rgb(GREYS[i % len(GREYS)]))
            names = (f"{s.prefix} {part.name} 색상", f"{s.prefix} {part.name} 채도",
                     f"{s.prefix} {part.name} 밝기")
            meshcat.AddSlider(names[0], 0.0, 360.0, 1.0, h * 360.0)
            meshcat.AddSlider(names[1], 0.0, 1.0, 0.01, sat)
            meshcat.AddSlider(names[2], 0.0, 1.0, 0.01, l)
            color_sliders.append((s, part, names))

    def read_colors_file():
        if a.colors is None or not a.colors.exists():
            return {}, None
        try:
            return json.loads(a.colors.read_text()), a.colors.stat().st_mtime
        except (json.JSONDecodeError, OSError) as e:
            print(f"[colors] {a.colors} 읽기 실패: {e}")
            return {}, None

    state = {"angles": {}, "colors": {}, "file_mtime": None, "file": {}}

    def update_angles():
        changed = False
        for s in shown:
            q = []
            for s2, name in angle_sliders:
                if s2 is not s:
                    continue
                deg = meshcat.GetSliderValue(name)
                q.append(np.radians(deg))
                if state["angles"].get(name) != deg:
                    state["angles"][name] = deg
                    meshcat.DeleteButton(readouts[name])
                    readouts[name] = f"  = {deg:.0f}°  ({name.replace(' 각도', '')})"
                    meshcat.AddButton(readouts[name])
                    changed = True
            s.plant.SetPositions(s.plant.GetMyMutableContextFromRoot(context), np.array(q))
        diagram.ForcedPublish(context)
        if changed:
            print("각도 [deg]: " + "  ".join(
                f"{n.replace(' 각도', '')}={v:.0f}"
                for n, v in state["angles"].items()), flush=True)

    def update_colors():
        file_colors, mtime = read_colors_file()
        if mtime != state["file_mtime"]:
            state["file_mtime"], state["file"] = mtime, file_colors
            state["colors"].clear()          # 파일이 바뀌면 전부 다시 칠한다
        for s, part, names in color_sliders:
            key = f"{s.prefix}/{part.name}"
            if key in state["file"]:
                rgb = tuple(float(c) for c in state["file"][key][:3])
            elif part.name in state["file"]:
                rgb = tuple(float(c) for c in state["file"][part.name][:3])
            else:
                h, sat, l = (meshcat.GetSliderValue(n) for n in names)
                rgb = colorsys.hls_to_rgb(h / 360.0, l, sat)
            if state["colors"].get(key) != rgb:
                state["colors"][key] = rgb
                for path in s.geometry_paths(part):
                    meshcat.SetProperty(path, "color", [*rgb, 1.0])

    update_angles()
    time.sleep(0.5)                      # 형상이 먼저 올라간 뒤에 색을 입힌다
    update_colors()
    print(f"\nMeshcat: {meshcat.web_url()}")
    print("6링크 길이 [mm]:", [f"{p.bbox_mm[0]:.0f}" for p in shown[1].spec.parts],
          f"(seed={a.seed})")
    print("왼쪽 패널: 각도 슬라이더 -> '= NN°' 읽음표가 같이 바뀐다. "
          "색상/채도/밝기 슬라이더로 부위 색. 종료는 Ctrl+C.", flush=True)
    try:
        while True:
            update_angles()
            update_colors()
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
