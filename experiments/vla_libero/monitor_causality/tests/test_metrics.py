"""Truth timing, held-object scope and paired-outcome checks without a GPU."""

from pathlib import Path
import tempfile
import json
import sys
import unittest
import numpy as np

S = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(S / "scripts"))
from analyze import truth, episode, paired


class MetricsTests(unittest.TestCase):
    def test_book_pickup_is_not_scored_by_final_goal(self):
        self.assertIsNone(truth(5, [False]))
        self.assertTrue(truth(8, [False, True]))
        self.assertFalse(truth(8, [False, False]))

    def score(self, atoms, event_step):
        with tempfile.TemporaryDirectory(dir=S) as tmp:
            root = Path(tmp)
            r = dict(
                task_id=0,
                initial_state_id=0,
                study_mode="live_base",
                success=False,
                status="timeout",
                steps=len(atoms),
                protocol_sha256="test",
                monitor_model_calls=0,
                monitor_opportunities=0,
                events=[
                    dict(
                        kind="subgoal_end",
                        step=event_step,
                        reason="done",
                        long_task_atoms=atoms[event_step - 1],
                    )
                ],
            )
            (root / "result.json").write_text(json.dumps(r))
            np.savez(
                root / "trace.npz", goal_atoms=atoms, applied=np.zeros((len(atoms), 7))
            )
            return episode(root)

    def test_pre_action_event_uses_previous_trace_state(self):
        r = self.score(
            [[False, False], [True, False], [True, False], [False, False]], 3
        )
        self.assertEqual(r["true_switches"], 1)
        self.assertEqual(r["delay_steps"], [1])

    def test_lost_completion_does_not_get_delay_credit(self):
        r = self.score(
            [[True, False], [True, False], [False, False], [False, False]], 4
        )
        self.assertEqual(r["false_switches"], 1)
        self.assertEqual(r["delay_steps"], [])

    def test_paired_gain_and_loss(self):
        rows = [
            dict(task=0, initial=0, mode="a", success=False, steps=5),
            dict(task=0, initial=0, mode="b", success=True, steps=4),
            dict(task=1, initial=0, mode="a", success=True, steps=5),
            dict(task=1, initial=0, mode="b", success=False, steps=4),
        ]
        r = paired(rows, "a", "b")
        self.assertEqual((r["gains"], r["losses"], r["nominal_mcnemar_p"]), (1, 1, 1.0))


if __name__ == "__main__":
    unittest.main()
