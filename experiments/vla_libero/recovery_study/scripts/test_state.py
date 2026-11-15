# ruff: noqa: E402
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from subgoal_state import SubgoalState, requires_release


class StateTests(unittest.TestCase):
    def test_false_and_invalid_break_streak(self):
        for valid, done in [(True, False), (False, True)]:
            s = SubgoalState(["a", "b"])
            self.assertFalse(s.observe(True, True))
            self.assertFalse(s.observe(valid, done))
            self.assertFalse(s.observe(True, True))

    def test_unconfirmed_cannot_advance(self):
        s = SubgoalState(["a", "b"])
        with self.assertRaises(ValueError):
            s.advance()
        self.assertEqual(s.confirmed, [])

    def test_confirmed_advance(self):
        s = SubgoalState(["a", "b"])
        s.observe(True, True)
        self.assertTrue(s.observe(True, True))
        s.advance()
        self.assertEqual((s.current, s.confirmed, s.streak), ("b", ["a"], 0))

    def test_replan_excludes_unfinished(self):
        s = SubgoalState(["a", "b"])
        s.observe(True, True)
        s.replace_pending(["fix a", "b"])
        self.assertEqual(s.confirmed, [])
        self.assertEqual(s.streak, 0)
        self.assertEqual(s.current, "fix a")

    def test_replan_preserves_confirmed_only(self):
        s = SubgoalState(["a", "b", "c"], confirm=1)
        s.observe(True, True)
        s.advance()
        s.replace_pending(["repair b"])
        self.assertEqual(s.instructions, ["a", "repair b"])
        self.assertEqual(s.confirmed, ["a"])

    def test_final_never_advanced_by_checker(self):
        s = SubgoalState(["a"], confirm=1)
        s.observe(True, True)
        with self.assertRaises(ValueError):
            s.advance()

    def test_bad_plan_atomic(self):
        s = SubgoalState(["a", "b"])
        with self.assertRaises(ValueError):
            s.replace_pending([])
        self.assertEqual(s.instructions, ["a", "b"])


class GateTests(unittest.TestCase):
    def test_fixture_goals_allow_closed_gripper(self):
        for text in [
            "turn on the stove",
            "turn off the stove",
            "open the drawer",
            "close the microwave door",
        ]:
            self.assertFalse(requires_release(text))

    def test_transport_requires_release(self):
        self.assertTrue(requires_release("put the mug on the plate"))
        self.assertTrue(requires_release("pick up mug and place it in microwave"))

    def test_unknown_remains_conservative(self):
        self.assertTrue(requires_release("do an unsupported action"))


if __name__ == "__main__":
    unittest.main()
