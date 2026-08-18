"""바로잡은 부위 매핑 + 바로 세운 자세로 램프 3종을 렌더한다.

  link_1 = 베이스 (바닥), link_3 = 연결부, link_2 = Head
  위 방향 = 베이스 최대 평면(205.9 cm^2)의 법선

색은 turbo (무지개) — 명도만이 아니라 색상으로 차이가 보이게 해 달라는 요청.
배경은 테마와 무관한 고정 판이라 이미지는 한 벌만 만든다.
"""
import os, sys
import numpy as np
import trimesh
import open3d as o3d
from open3d.visualization import rendering
from matplotlib import colormaps

DRAKE = ("/home/junhyeoklee/Downloads/desk_lamp_minimal_sim_delivery_20260814-"
         "20260813T155839Z-1-001/desk_lamp_minimal_sim_delivery_20260814/"
         "desk_lamp_minimal_sim/drake")
OUT = os.path.dirname(os.path.abspath(__file__))

MESH = {"base": "link_1_head_rgb", "arm": "link_3_support_rgb", "head": "link_2_base_rgb"}
meshes = {p: trimesh.load(f"{DRAKE}/visuals/{fn}.gltf", force="mesh") for p, fn in MESH.items()}

# ---- 위 방향: 베이스의 최대 평면 법선 ---------------------------------------
m1 = meshes["base"]
n, a = m1.face_normals, m1.area_faces
best = None
for i in np.argsort(-a)[:600]:
    sel = (n @ n[i]) > 0.985
    w = a[sel].sum()
    if best is None or w > best[0]:
        best = (w, n[i])
up = best[1]
if (np.asarray(meshes["head"].vertices) @ up).mean() < (np.asarray(m1.vertices) @ up).mean():
    up = -up
z = up / np.linalg.norm(up)
x = np.cross(z, [0, 0, 1.0]); x /= np.linalg.norm(x)
y = np.cross(z, x)
R = np.vstack([x, y, z])

geo = {}
allv = []
for p, m in meshes.items():
    v = (R @ np.asarray(m.vertices).T).T
    nv = (R @ np.asarray(m.vertex_normals).T).T
    geo[p] = (v, np.asarray(m.faces), nv)
    allv.append(v)
allv = np.vstack(allv)
center = (allv.min(0) + allv.max(0)) / 2.0
extent = allv.max(0) - allv.min(0)
for p in geo:
    v, f, nv = geo[p]
    geo[p] = (v - center, f, nv)

# ---- turbo 컬러맵 -----------------------------------------------------------
DMIN, DMAX = 300.0, 1050.0
TURBO = colormaps["turbo"]


def ramp(d):
    t = np.clip((d - DMIN) / (DMAX - DMIN), 0, 1)
    return np.array(TURBO(float(t))[:3])


# 유효밀도 [kg/m^3] = 부위 질량 / 그 링크의 충돌메시 부피
COLUMNS = {
    "before": {"base": 1000.0, "arm": 1000.0, "head": 1000.0},
    "after":  {"base": 897.8,  "arm": 372.3,  "head": 507.4},
    "gt":     {"base": 897.9,  "arm": 373.4,  "head": 507.0},
}

LIGHT = np.array([-0.30, -0.80, 0.52]); LIGHT /= np.linalg.norm(LIGHT)
W, H = 1000, 1250
AZ = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
EL = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
TAG = sys.argv[3] if len(sys.argv) > 3 else ""

renderer = rendering.OffscreenRenderer(W, H)


def build(col, dens):
    sc = renderer.scene
    sc.clear_geometry()
    sc.view.set_post_processing(False)   # 톤매핑 끄면 unlit 색이 그대로 나온다
    for p, (v, f, nv) in geo.items():
        base_col = ramp(dens[p])
        lam = np.clip(nv @ LIGHT, 0, 1)
        # 색-값 대응이 깨지지 않게 음영은 72~100% 범위로만
        shade = 0.72 + 0.28 * (0.25 + 0.75 * lam)
        vc = np.clip(base_col[None, :] * shade[:, None], 0, 1)
        g = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),
                                      o3d.utility.Vector3iVector(f))
        g.vertex_colors = o3d.utility.Vector3dVector(vc)
        g.compute_vertex_normals()
        mat = rendering.MaterialRecord(); mat.shader = "defaultUnlit"
        sc.add_geometry(f"{col}_{p}", g, mat)

    r = float(np.linalg.norm(extent)) * 0.56
    aa, ee = np.radians(AZ), np.radians(EL)
    eye = np.array([np.sin(aa) * np.cos(ee), -np.cos(aa) * np.cos(ee), np.sin(ee)]) * 10.0
    sc.camera.look_at([0, 0, 0], eye.tolist(), [0, 0, 1])
    asp = W / H
    sc.camera.set_projection(rendering.Camera.Projection.Ortho,
                             -r * asp, r * asp, -r, r, 0.1, 40.0)
    sc.set_background([0, 0, 0, 1])
    blk = np.asarray(renderer.render_to_image()).astype(np.float64) / 255.0
    sc.set_background([1, 1, 1, 1])
    wht = np.asarray(renderer.render_to_image()).astype(np.float64) / 255.0
    alpha = np.clip(1.0 - (wht - blk), 0, 1).max(axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        rgb = np.where(alpha[..., None] > 1e-3, blk / alpha[..., None], 0.0)
    rgba = np.concatenate([np.clip(rgb, 0, 1), alpha[..., None]], axis=2)
    from PIL import Image
    Image.fromarray((rgba * 255).round().astype(np.uint8)).save(f"{OUT}/lamp2_{col}{TAG}.png")
    print("wrote", f"lamp2_{col}{TAG}.png")


for c, d in COLUMNS.items():
    build(c, d)

stops = []
for k in range(11):
    c = (ramp(DMIN + (DMAX - DMIN) * k / 10.0) * 255).round().astype(int)
    stops.append("#%02x%02x%02x %d%%" % (c[0], c[1], c[2], k * 10))
print("GRADIENT:", ", ".join(stops))
for lbl, v in [("arm 372", 372.3), ("head 507", 507.4), ("base 898", 897.8), ("prior 1000", 1000.0)]:
    c = (ramp(v) * 255).round().astype(int)
    print("SWATCH %-12s #%02x%02x%02x" % (lbl, c[0], c[1], c[2]))
