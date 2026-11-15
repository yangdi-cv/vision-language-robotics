# ruff: noqa: E402
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from qwen_planner_recovery import parse_repair


class ContractTests(unittest.TestCase):
    goals = ["turn on the stove", "put the pot on the stove"]

    def call(self, steps):
        return parse_repair(json.dumps({"subgoals": steps}), self.goals)

    def test_valid(self):
        self.assertEqual(
            len(
                self.call(
                    [
                        {"goal_id": "g0", "instruction": self.goals[0]},
                        {"goal_id": "g1", "instruction": self.goals[1]},
                    ]
                )["subgoals"]
            ),
            2,
        )

    def test_missing_goal(self):
        with self.assertRaises(ValueError):
            self.call([{"goal_id": "g1", "instruction": self.goals[1]}])

    def test_duplicate_goal(self):
        with self.assertRaises(ValueError):
            self.call(
                [
                    {"goal_id": "g0", "instruction": "turn on stove"},
                    {"goal_id": "g0", "instruction": "turn on stove"},
                ]
            )

    def test_pickup_only(self):
        with self.assertRaises(ValueError):
            self.call(
                [
                    {"goal_id": "g0", "instruction": "turn on stove"},
                    {"goal_id": "g1", "instruction": "pick up pot"},
                ]
            )

    def test_combined_pick_place(self):
        self.assertEqual(
            len(
                self.call(
                    [
                        {"goal_id": "g0", "instruction": "turn on stove"},
                        {
                            "goal_id": "g1",
                            "instruction": "pick up pot and place it on stove",
                        },
                    ]
                )["subgoals"]
            ),
            2,
        )

    def test_unknown_goal(self):
        with self.assertRaises(ValueError):
            self.call([{"goal_id": "g2", "instruction": "turn on stove"}])


if __name__ == "__main__":
    unittest.main()
