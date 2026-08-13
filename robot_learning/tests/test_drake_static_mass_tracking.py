#!/usr/bin/env python3

from pathlib import Path
import sys
import unittest

import numpy as np


sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "scripts"),
)

import run_drake_static_mass_tracking as tracking  # noqa: E402


class DrakeStaticMassTrackingTest(unittest.TestCase):
    def test_two_link_mass_tracking_closes_against_gt(self):
        result = tracking.run(
            joint_min_deg=20.0,
            joint_max_deg=120.0,
            steps=4,
            seed=20260728,
        )
        self.assertLess(max(result["final"]["mass_relative_error"]), 0.05)
        self.assertLess(
            max(
                row["drake_inverse_dynamics_max_error"]
                for row in result["trace"]
            ),
            1e-10,
        )
        self.assertTrue(
            np.all(np.asarray(result["final"]["part_mass_kg"]) > 0.0)
        )


if __name__ == "__main__":
    unittest.main()
