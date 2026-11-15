# ruff: noqa: E402
from pathlib import Path
import json
import sys
import unittest

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "scripts"))
from checker_protocol import parse, metrics


class IterationTests(unittest.TestCase):
    def test_parser_requires_boolean(self):
        self.assertTrue(parse('{"done":true}'))
        for raw in ['{"done":"true"}', '{"done":1}', '{"done":true,"extra":0}']:
            with self.assertRaises(ValueError):
                parse(raw)

    def test_invalid_not_credited_as_true_negative(self):
        x = [
            {"expected": True, "predicted": True, "valid": True},
            {"expected": False, "predicted": None, "valid": False},
        ]
        m = metrics(x)
        self.assertEqual(m["balanced_accuracy"], 0.5)
        self.assertEqual(m["invalid"], 1)

    def test_episode_init_disjoint(self):
        rows = json.loads((R / "data/cases.json").read_text())
        for key in ["episode", "initial_state_sha256"]:
            sets = {
                s: {r[key] for r in rows if r["split"] == s}
                for s in ["train", "validation", "test"]
            }
            self.assertFalse(sets["train"] & sets["validation"])
            self.assertFalse(sets["train"] & sets["test"])
            self.assertFalse(sets["validation"] & sets["test"])

    def test_video_label_alignment(self):
        import numpy as np

        for row in json.loads((R / "data/cases.json").read_text()):
            t = np.load(R.parent / Path(row["source"]).parent / "trace.npz")
            self.assertEqual(bool(t["predicate"][row["step"] - 1]), row["done"])

    def test_no_early_frame_positive(self):
        rows = json.loads((R / "data/cases.json").read_text())
        self.assertTrue(all(not x["done"] for x in rows if x["stratum"] != "terminal"))


if __name__ == "__main__":
    unittest.main()
