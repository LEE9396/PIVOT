"""실패가 알고리즘 탓인가, 시험 물체의 모양 탓인가.

지금 nlink 는 링크가 150->60mm 로 짧아지고 관절축이 z/-y 로 번갈아 들어서
사슬이 되말린다. 그래서 말단 부위들이 손목에서 거의 같은 거리에 모인다.
되말리지 않는 사슬이면 어떨까.

  기준     : 지금 그대로
  등길이   : 모든 링크 100 mm (짧아지지 않음)
  좁은범위 : 관절 범위 0~90 도 (되말릴 수 없음)
  축통일   : 모든 관절이 z 축 (평면 사슬)
"""
import numpy as np, nlink, density_id_objects as obj, design_core as dc, density_id_drake as alg
from density_id_objects import Joint, ObjectSpec, Part, MEASURED_HINGE_KG
G = dc.CANONICAL_TRIAD
CROSS, GAP, OFF, FILL = 44.0, 4.0, 27.0, 0.98

def make(n, lengths=None, limits=(0.0, np.pi), same_axis=False):
    L = lengths or nlink.link_lengths(n)
    rho = nlink.densities(n); parts, joints = [], []
    center = (0.0, 0.0)
    for i, ln in enumerate(L):
        st = 0.0 if i == 0 else GAP/2
        cy, cz = center; c = (st+ln/2, cy, cz)
        parts.append(Part(name=f"link{i}"+("_base" if i==0 else ""),
            bbox_mm=(ln, CROSS, CROSS), volume_cm3=round(FILL*ln*CROSS*CROSS*1e-3,2),
            rho_gt=float(rho[i]), bbox_center_in_link_mm=c, shell_centroid_in_link_mm=c,
            color=(0.3,0.5,0.7,1.0), rho_empty=250.0, cavity_cm3=round(0.4*FILL*ln*CROSS*CROSS*1e-3,1)))
        if i+1 < len(L):
            xj = st+ln+GAP/2
            if same_axis or i % 2 == 0:
                origin, axis = (xj, cy+OFF, cz), (0.,0.,1.); center = (-OFF, 0.0)
            else:
                origin, axis = (xj, cy, cz+OFF), (0.,-1.,0.); center = (0.0, -OFF)
            joints.append(Joint(name=f"joint{i+1}", parent=parts[i].name, child=f"link{i+1}",
                origin_in_parent_link_mm=origin, axis=axis, limits_rad=tuple(limits),
                hinge_mass_kg=MEASURED_HINGE_KG))
    return ObjectSpec(key=f"{n}v", label=f"{n}", parts=parts, joints=joints,
                      base_bbox_center_in_sensor_mm=parts[0].bbox_center_in_link_mm, notes="")

def score(spec, R=15, seed=1):
    obj.set_measurement_averaging(); gt = obj.bind_object(spec)
    obj.apply_weight_prior(spec, obj.assembled_mass_kg(spec))
    npart = len(spec.parts); b = [j.limits_rad for j in spec.joints]
    Sigma = alg.SIGMA0.copy(); rh = alg.MU0.copy()
    for it in range(1, R+1):
        th,_ = dc.continuous_best(b, lambda t: dc.utility(t, rh, Sigma, G, "D", 0.05),
                                  n_starts=6, seed=seed*100+it)
        Sigma = dc.posterior(Sigma, dc.regressor(th, G), dc.effective_cov(th, rh, G, 0.05))
    h = 100*dc.stopping_width(dc.half_width(Sigma, gt), npart)
    th0 = np.array([0.5*(lo+hi) for lo,hi in b])
    d = 1000*np.linalg.norm(alg.part_centroids_in_S(th0)[:npart], axis=1)
    return h, float(np.std(d)), float(np.min(np.abs(np.diff(np.sort(d)))))

print("15라운드 후 최악 반폭 [%] — 물체 모양을 바꾸면?\n")
print(f"{'p':>2}{'모양':>12}{'반폭%':>10}{'거리 표준편차mm':>16}{'가장 가까운 두 부위mm':>22}")
print("-"*64)
for p in (5, 6, 8):
    for name, kw in (("기준", {}),
                     ("등길이100", dict(lengths=[100.0]*p)),
                     ("범위0~90", dict(limits=(0.0, np.pi/2))),
                     ("축통일", dict(same_axis=True))):
        h, sd, mn = score(make(p, **kw))
        print(f"{p:>2}{name:>12}{h:>10.2f}{sd:>16.1f}{mn:>22.1f}")
    print()
