"""배달물 URDF 의 링크 이름이 spec 과 달라도 물성이 들어가는지 확인한다."""

from pathlib import Path
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "my_work"))


class ExportUrdfChecks(unittest.TestCase):
    def test_lamp_links_are_mapped_by_final_part(self):
        import desk_lamp
        import export_urdf as eu
        import numpy as np

        spec = desk_lamp.build_spec()
        rho = np.array([p.rho_gt for p in spec.parts])
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "estimated.urdf"
            rows = eu.export(spec, rho, out, log=lambda *a: None)
            self.assertEqual({r["name"] for r in rows}, {"base", "support", "head"})
            self.assertEqual({r["part"] for r in rows}, {p.name for p in spec.parts})
            root = ET.parse(out).getroot()
            masses = {link.get("name"): float(link.find("inertial/mass").get("value"))
                      for link in root.iter("link") if link.find("inertial/mass") is not None}
            for r in rows:
                self.assertAlmostEqual(masses[r["name"]], r["mass"], places=9)
            # 스캔 부피 x GT 밀도 = GT 질량 (힌지 몫은 extras 로 더해질 수 있어 이상)
            for r in rows:
                part = next(p for p in spec.parts if p.name == r["part"])
                self.assertGreaterEqual(r["mass"] + 1e-9, part.volume_m3 * part.rho_gt)


if __name__ == "__main__":
    unittest.main()
