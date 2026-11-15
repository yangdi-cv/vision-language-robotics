"""Protect functional-part labels and runtime reset/angle contracts."""

from pathlib import Path
import sys, unittest
from types import SimpleNamespace
import numpy as np

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "scripts"))
from scene import MobileScene
from affordance_pilot import label


class Contracts(unittest.TestCase):
    def test_yaw_equivalence(self):
        s = MobileScene.__new__(MobileScene)
        s.joints = np.array([0, 1, 2])
        s.dock = np.array([-0.76, 0, 0])
        s.data = SimpleNamespace(qpos=np.array([0.0, 0.0, 4 * np.pi + 0.01]))
        self.assertAlmostEqual(s.pose()[2], 0.01)

    def test_point_hits_visible_part_not_empty_bbox_center(self):
        mask = np.zeros((1000, 1000), bool)
        mask[300:303, 300:700] = True
        mask[697:700, 300:700] = True
        mask[300:700, 300:303] = True
        mask[300:700, 697:700] = True
        r = label(mask)
        self.assertTrue(r["visible"])
        x, y = r["point"]
        self.assertTrue(mask[y, x])
        self.assertFalse(mask[500, 500])

    def test_invisible_part_is_not_given_fabricated_point(self):
        r = label(np.zeros((512, 512), bool))
        self.assertFalse(r["visible"])
        self.assertIsNone(r["point"])

    def test_runtime_reset_guard(self):
        s = MobileScene(initial=3, mobile_start=(0.0, 0.0, 0.0), stow=True)
        try:
            before = s.data.qpos.copy()
            with self.assertRaisesRegex(RuntimeError, "Runtime scene reset"):
                s.e.env.reset()
            with self.assertRaisesRegex(RuntimeError, "Runtime scene reset"):
                s.e.env.reset_from_xml_string(s.original_xml)
            np.testing.assert_array_equal(s.data.qpos, before)
        finally:
            s.close()


if __name__ == "__main__":
    unittest.main()
