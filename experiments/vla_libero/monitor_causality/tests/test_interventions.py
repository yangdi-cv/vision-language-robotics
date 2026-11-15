"""CPU-only contracts: checker image isolation, fixed controls and plan preservation."""

from pathlib import Path
import json
import tempfile
import unittest
import sys
from PIL import Image

S = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(S / "scripts"))
from interventions import Monitor


class Worker:
    def __init__(self):
        self.requests = []

    def request(self, payload, timeout=120):
        self.requests.append(payload)
        return dict(ok=True, done=False, checker="fake")


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.ref = self.root / "ref_step0000.png"
        self.current = self.root / "check_step0150.png"
        Image.new("RGB", (8, 8), "red").save(self.ref)
        Image.new("RGB", (8, 8), "blue").save(self.current)
        self.payload = dict(
            op="check",
            image=[str(self.ref), str(self.current)],
            subgoal="put object in basket",
        )
        self.worker = Worker()

    def tearDown(self):
        self.temp.cleanup()

    def monitor(self, mode):
        return Monitor(
            self.worker, ["first", "second"], mode, self.root / (mode + ".jsonl")
        )

    def test_live_input_and_plan_preserved(self):
        m = self.monitor("live_base")
        plan = m.request({"op": "plan"})
        self.assertEqual(
            [x["instruction"] for x in plan["subgoals"]], ["first", "second"]
        )
        m.request(self.payload)
        self.assertEqual(self.worker.requests[0]["image"], self.payload["image"])
        self.assertEqual(self.worker.requests[0]["checker_mode"], "base")

    def test_frozen_only_replaces_checker_current_image(self):
        payload = json.loads(json.dumps(self.payload))
        m = self.monitor("frozen_base")
        m.request(self.payload)
        self.assertEqual(
            self.worker.requests[0]["image"], [str(self.ref), str(self.ref)]
        )
        self.assertEqual(self.payload, payload)
        row = json.loads((self.root / "frozen_base.jsonl").read_text())
        self.assertEqual(len(set(row["actual_pixel_files_sha256"])), 1)
        self.assertEqual(len(set(row["original_pixel_files_sha256"])), 2)

    def test_timer_has_no_model_call_and_fixed_boundary(self):
        m = self.monitor("timer150")
        earlier = self.root / "check_step0149.png"
        Image.new("RGB", (8, 8)).save(earlier)
        self.assertFalse(
            m.request({**self.payload, "image": [str(self.ref), str(earlier)]})["done"]
        )
        self.assertTrue(m.request(self.payload)["done"])
        self.assertEqual(m.actual_calls, 0)
        self.assertEqual(self.worker.requests, [])

    def test_hold_first_never_votes_done(self):
        m = self.monitor("hold_first")
        self.assertFalse(m.request(self.payload)["done"])
        self.assertEqual(m.actual_calls, 0)

    def test_lora_mode_explicit(self):
        m = self.monitor("live_lora")
        m.request(self.payload)
        self.assertEqual(self.worker.requests[0]["checker_mode"], "v2")

    def test_unplanned_recovery_rejected(self):
        with self.assertRaises(ValueError):
            self.monitor("live_base").request({"op": "replan"})
        with self.assertRaises(ValueError):
            Monitor(self.worker, ["one"], "live_base", self.root / "none")


if __name__ == "__main__":
    unittest.main()
