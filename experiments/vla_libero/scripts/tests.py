"""Protocol and observation-interface checks, independent of policy performance."""

import unittest, numpy as np
from evaluate import make_input, quat_axis, plans


class InterfaceTests(unittest.TestCase):
    def test_quaternion_identity(self):
        np.testing.assert_allclose(quat_axis([0, 0, 0, 1]), 0)

    def test_quaternion_halfturn(self):
        np.testing.assert_allclose(quat_axis([0, 0, 1, 0]), [0, 0, np.pi])

    def test_image_direction_and_state(self):
        im = np.zeros((224, 224, 3), np.uint8)
        im[0, 0] = [10, 20, 30]
        obs = {
            "agentview_image": im,
            "robot0_eye_in_hand_image": im,
            "robot0_eef_pos": np.array([0.1, 0.2, 0.3]),
            "robot0_eef_quat": np.array([0, 0, 0, 1]),
            "robot0_gripper_qpos": np.array([0.01, -0.01]),
        }
        out = make_input(obs, "pick target")
        np.testing.assert_equal(out["observation/image"][-1, -1], [10, 20, 30])
        np.testing.assert_allclose(
            out["observation/state"], [0.1, 0.2, 0.3, 0, 0, 0, 0.01, -0.01]
        )
        np.testing.assert_equal(obs["robot0_eef_quat"], [0, 0, 0, 1])

    def test_fixed_unique_trials(self):
        jobs = list(plans("all"))
        self.assertEqual(len(jobs), 246)
        self.assertEqual(len(set(jobs)), 246)
        self.assertEqual(sum(j[-1] == "baseline" for j in jobs), 120)

    def test_paired_initial_conditions(self):
        jobs = list(plans("all"))
        baseline = {j[:3] for j in jobs if j[-1] == "baseline"}
        for j in jobs:
            self.assertIn(j[:3], baseline)

    def test_no_truth_in_policy_input(self):
        im = np.zeros((224, 224, 3), np.uint8)
        obs = {
            "agentview_image": im,
            "robot0_eye_in_hand_image": im,
            "robot0_eef_pos": np.zeros(3),
            "robot0_eef_quat": np.array([0, 0, 0, 1]),
            "robot0_gripper_qpos": np.zeros(2),
            "object_position": np.ones(3),
            "depth": np.ones((224, 224)),
        }
        self.assertEqual(
            set(make_input(obs, "task")),
            {
                "observation/image",
                "observation/wrist_image",
                "observation/state",
                "prompt",
            },
        )


if __name__ == "__main__":
    unittest.main()
