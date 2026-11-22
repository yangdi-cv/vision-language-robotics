import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from vla_runtime.config import load_config
from vla_runtime.geometry import collision_matrix, fk_error, local_point, rotation
from vla_runtime.records import Journal, atomic_json, error_record, run_metadata
from vla_runtime.remote import SSH
from vla_runtime.schemas import ArmState, action_success, grounding, trajectory

IDENTITY = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


class Contracts(unittest.TestCase):
    def test_grounding_preserves_normalized_coordinates(self):
        self.assertEqual(
            grounding({"ok": True, "point_2d": [500, 750]})["point_2d"], [500.0, 750.0]
        )

    def test_grounding_rejects_untrusted_values(self):
        for value in (
            None,
            [],
            {"ok": "true", "point_2d": [1, 2]},
            {"ok": True, "point_2d": [True, 1]},
            {"ok": True, "point_2d": [1000, 1]},
            {"ok": True, "point_2d": [math.nan, 1]},
            {"ok": True, "point_2d": [1]},
            {"ok": True, "point_2d": ["1", 2]},
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                grounding(value)

    def test_arm_state_dimensions(self):
        state = {"joint_positions": [0] * 7, "finger_positions": [0] * 2, "tcp_p": [0] * 3}
        self.assertEqual(len(ArmState.parse(state).joint_positions), 7)
        for field in state:
            bad = dict(state, **{field: []})
            with self.assertRaises(ValueError):
                ArmState.parse(bad)

    def test_rotation_rejects_reflection_and_nonorthogonal(self):
        for value in (
            [[1, 0, 0], [0, 1, 0], [0, 0, -1]],
            [[2, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[math.inf, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[1, 0, 0]],
        ):
            with self.assertRaises(ValueError):
                rotation(value)

    def test_transformed_frame_not_just_translation(self):
        turn = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        self.assertEqual(local_point(turn, [1, 2, 0], [1, 3, 0]), [1.0, 0.0, 0.0])

    def test_fk_guard(self):
        self.assertAlmostEqual(fk_error([0, 0, 0], [0, 0, 1e-6]), 1e-6)
        for actual in ([0, 0, 0.01], [0, 0, math.nan]):
            with self.assertRaises(ValueError):
                fk_error([0, 0, 0], actual)

    def test_collision_default_denies_unlisted_contacts(self):
        names = ["link0", "finger", "table"]
        matrix = collision_matrix(names, {frozenset(("link0", "finger"))})
        self.assertTrue(matrix[0][1])
        self.assertEqual(matrix[0][1], matrix[1][0])
        self.assertFalse(matrix[1][2])
        with self.assertRaises(ValueError):
            collision_matrix(names, {frozenset(("finger", "missing"))})

    def test_action_requires_both_status_and_execution(self):
        action_success(1, 4, 1)
        for args in ((1, 6, 1), (-1, 4, 1), (1, 4, 0)):
            with self.assertRaises(ValueError):
                action_success(*args)

    def test_recorded_trajectory_contract(self):
        trajectory(
            ["a", "b"],
            [
                {"positions": [0, 1], "time_from_start": 0},
                {"positions": [1, 2], "time_from_start": 0.1},
            ],
        )
        for names, points in (
            (["a", "a"], [{"positions": [0, 1], "time_from_start": 0}]),
            (["a"], []),
            (["a"], [{"positions": [math.nan], "time_from_start": 0}]),
            (
                ["a"],
                [
                    {"positions": [0], "time_from_start": 1},
                    {"positions": [1], "time_from_start": 1},
                ],
            ),
        ):
            with self.assertRaises(ValueError):
                trajectory(names, points)


class Operations(unittest.TestCase):
    def test_atomic_json_failure_keeps_previous_result(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            atomic_json(path, {"old": 1})
            with self.assertRaises(ValueError):
                atomic_json(path, {"new": math.nan})
            self.assertEqual(json.loads(path.read_text()), {"old": 1})
            with patch("os.replace", side_effect=OSError("disk error")):
                with self.assertRaises(OSError):
                    atomic_json(path, {"new": 2})
            self.assertEqual(len(list(Path(directory).iterdir())), 1)

    def test_progress_durable_before_close(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            journal = Journal(path)
            try:
                journal.append("planning", attempt=0)
                self.assertEqual(json.loads(path.read_text())["phase"], "planning")
                with self.assertRaises(FileExistsError):
                    Journal(path)
            finally:
                journal.close()

    def test_full_exception_chain(self):
        try:
            try:
                raise ValueError("bad point")
            except ValueError as exc:
                raise RuntimeError("inference") from exc
        except RuntimeError as exc:
            record = error_record(exc)
        self.assertIn("bad point", record["traceback"])
        self.assertIn("RuntimeError: inference", record["traceback"])

    def test_config_override_paths_with_spaces_and_unknown_keys(self):
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(prefix="vla config ") as directory:
            path = Path(directory) / "local.json"
            path.write_text(
                json.dumps({"schema_version": 1, "paths": {"qwen_python": "my env/bin/python"}})
            )
            with patch.dict(os.environ, {"VLA_CONFIG": str(path), "VLA_ROOT": str(root)}):
                config = load_config()
                self.assertEqual(config.path("qwen_python"), root / "my env/bin/python")
                self.assertEqual(config.limit("visual_residual_m"), 0.025)
            path.write_text('{"schema_version":1,"paths":{"typo":"wrong"}}')
            with patch.dict(os.environ, {"VLA_CONFIG": str(path)}), self.assertRaises(ValueError):
                load_config(root)

    def test_metadata_captures_parameters_and_commit(self):
        value = run_metadata(Path(__file__).resolve().parents[2], parameters={"seed": 2})
        self.assertEqual(len(value["parameters_sha256"]), 64)
        self.assertIn("commit", value)

    def test_ssh_has_no_password_or_unquoted_remote_arguments(self):
        ssh = SSH("jetson-vla")
        command = ssh.command(["python3", "/path with space/a.py", "$(touch /tmp/no)"])
        self.assertIn("-oBatchMode=yes", command)
        self.assertIn("-oStrictHostKeyChecking=yes", command)
        self.assertIn("'/path with space/a.py'", command[-1])
        self.assertIn("'$(touch /tmp/no)'", command[-1])
        for host in ("-oProxyCommand=evil", "bad host", "x;rm", ""):
            with self.assertRaises(ValueError):
                SSH(host)
        with self.assertRaises(ValueError):
            ssh.tunnel(0, 18888)


class Artifacts(unittest.TestCase):
    def test_manifest_detects_modification_extra_and_symlink(self):
        from vla_runtime.artifacts import manifest, verify

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.txt").write_text("original")
            expected = manifest(root)
            verify(root, expected)
            (root / "a.txt").write_text("changed")
            with self.assertRaises(ValueError):
                verify(root, expected)
            (root / "a.txt").write_text("original")
            (root / "extra.txt").write_text("unexpected")
            with self.assertRaises(ValueError):
                verify(root, expected)
            (root / "extra.txt").unlink()
            (root / "link").symlink_to(root / "a.txt")
            with self.assertRaises(ValueError):
                manifest(root)
