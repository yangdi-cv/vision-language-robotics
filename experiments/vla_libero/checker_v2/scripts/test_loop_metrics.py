"""Check timestamp alignment and separation of forced versus confirmed switches."""

import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from summarize_loop import R, metrics, truth


class LoopMetricsTests(unittest.TestCase):
    def score(self, atoms, step, reason="done", task=0):
        (R / "results").mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=R / "results") as d:
            p = Path(d) / "result.json"
            r = {
                "task_id": task,
                "initial_state_id": 0,
                "success": False,
                "status": "timeout",
                "steps": len(atoms),
                "qwen_checks": 0,
                "events": [
                    {"kind": "subgoal_start", "index": 0, "step": 0},
                    {
                        "kind": "subgoal_end",
                        "index": 0,
                        "step": step,
                        "reason": reason,
                        "long_task_atoms": atoms[step - 1],
                    },
                ],
            }
            p.write_text(json.dumps(r))
            np.savez(p.with_name("trace.npz"), goal_atoms=atoms)
            return metrics(p)

    def test_state_at_event_is_previous_trace_entry(self):
        r = self.score(
            [
                [False, False],
                [False, False],
                [True, False],
                [True, False],
                [False, False],
            ],
            4,
        )
        self.assertEqual(r["correct_switches"], 1)
        self.assertEqual(r["delay_steps"], [1])

    def test_reached_then_lost_does_not_make_false_switch_correct(self):
        r = self.score(
            [[True, False], [True, False], [False, False], [False, False]], 4
        )
        self.assertEqual(r["false_switches"], 1)
        self.assertEqual(r["delay_steps"], [])

    def test_budget_is_not_visual_confirmation(self):
        r = self.score([[True, False]] * 4, 4, "budget")
        self.assertEqual(r["confirmed"], 0)
        self.assertEqual(r["true_before_budget"], 1)
        self.assertEqual(r["missed_confirmation"], 1)

    def test_identical_pots_and_unscorable_pickup(self):
        self.assertTrue(truth(8, [False, True, True]))
        self.assertFalse(truth(8, [False, False, True]))
        self.assertIsNone(truth(5, [False]))


if __name__ == "__main__":
    unittest.main()
