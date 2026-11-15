# ruff: noqa: E402
from pathlib import Path
import json
import sys
import unittest

R = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(R / "scripts"), str(R.parent / "scripts")]
from protocol import evaluate, threshold_on_validation
from completion_checker_v2 import supported


class V2Tests(unittest.TestCase):
    def test_abstention_does_not_get_positive_credit(self):
        r = [
            {
                "done": True,
                "scores": {"yes": 0.1, "no": 0.2, "unknown": 0.7, "yes_log_odds": -2.2},
            },
            {
                "done": False,
                "scores": {"yes": 0.1, "no": 0.2, "unknown": 0.7, "yes_log_odds": -2.2},
            },
        ]
        m = evaluate(r, 0)
        self.assertEqual(m["recall"], 0)
        self.assertEqual(m["fpr"], 0)
        self.assertEqual(m["rejected"], 2)

    def test_margin_keeps_order_despite_probability_saturation(self):
        r = [
            {
                "done": False,
                "scores": {"yes": 1.0, "no": 0.0, "unknown": 0.0, "yes_log_odds": 20.0},
            },
            {
                "done": True,
                "scores": {"yes": 1.0, "no": 0.0, "unknown": 0.0, "yes_log_odds": 21.0},
            },
        ]
        t = threshold_on_validation(r)
        self.assertEqual(t, 21)
        self.assertEqual(evaluate(r, t)["recall"], 1)

    def test_group_holdout(self):
        r = json.loads((R / "data/cases.json").read_text())
        sets = {
            s: {x["group"] for x in r if x["split"] == s}
            for s in ["train", "validation", "test"]
        }
        self.assertFalse(sets["train"] & sets["validation"])
        self.assertFalse(sets["train"] & sets["test"])
        self.assertFalse(sets["validation"] & sets["test"])

    def test_physical_release_and_support_labels(self):
        for x in json.loads((R / "data/cases.json").read_text()):
            if x["source"] == "synthetic":
                self.assertEqual(
                    x["done"],
                    x["relation_predicate"] and x["support_contact"] and not x["grasp"],
                )
            if x["grasp"]:
                self.assertFalse(x["done"])
            if x["status"] == "yes":
                self.assertTrue(x["done"])

    def test_scope_fallback(self):
        for x in [
            "turn on stove",
            "open drawer",
            "pick up the book",
            "put the first moka pot on stove",
        ]:
            self.assertFalse(supported(x))
        self.assertTrue(
            supported("pick up the white mug and place it on the left plate")
        )


if __name__ == "__main__":
    unittest.main()
