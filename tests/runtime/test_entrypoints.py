import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from vla_runtime.config import RuntimeConfig
from vla_runtime.milestones import MilestoneMonitor
from vla_runtime.schemas import collision_geometry, target_pose

ROOT = Path(__file__).resolve().parents[2]


class Entrypoints(unittest.TestCase):
    def load_runner(self):
        path = ROOT / "experiments/mobile_qwen/closed_loop_20261115/scripts/run_cases.py"
        spec = importlib.util.spec_from_file_location("tested_cases", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_simulator_crash_preserves_progress_and_returns_failure(self):
        runner = self.load_runner()
        with tempfile.TemporaryDirectory(prefix="vla crash ") as directory:
            stage = Path(directory)
            (stage / "scripts").mkdir()
            (stage / "scripts/grounding_worker.py").write_text("# unused worker")
            (stage / "scripts/sim_precontact.py").write_text("raise SystemExit(7)")
            config = RuntimeConfig(
                ROOT,
                {
                    **runner.CONFIG.values,
                    "paths": {**runner.CONFIG.values["paths"], "policy_python": sys.executable},
                },
            )
            with (
                patch.object(runner, "D", stage),
                patch.object(runner, "CONFIG", config),
                patch.object(
                    sys, "argv", ["run_cases", "--prefix", "crash_test", "--modes", "oracle"]
                ),
            ):
                self.assertEqual(runner.main(), 1)
            logs = stage / "logs/crash_test_nominal_i00_oracle"
            result = json.loads((logs / "case_snapshot.json").read_text())
            self.assertFalse(result["success"])
            self.assertIn("Simulator exited", result["traceback"])
            phases = [
                json.loads(line)["phase"]
                for line in (logs / "progress.jsonl").read_text().splitlines()
            ]
            self.assertEqual(phases, ["starting", "failed", "finished"])
            self.assertEqual(result["process_exit_codes"], [7])

    def test_invalid_prefix_cannot_create_paths(self):
        runner = self.load_runner()
        with patch.object(sys, "argv", ["run_cases", "--prefix", "../escape"]):
            with self.assertRaises(SystemExit) as context:
                runner.main()
            self.assertEqual(context.exception.code, 2)

    def test_remote_dry_run_never_contacts_device(self):
        script = (
            ROOT / "experiments/mobile_qwen/jetson/stage3_20261115/scripts/run_joint_service.py"
        )
        result = subprocess.run(
            [sys.executable, str(script), "--tag", "dry_validation", "--dry-run"],
            capture_output=True,
            text=True,
            check=True,
        )
        value = json.loads(result.stdout)
        self.assertIn("-oBatchMode=yes", value["service"])
        self.assertIn("MemoryMax=6G", value["service"][-1])
        self.assertNotIn("sshpass", result.stdout)
        self.assertNotIn("/home/csip", result.stdout)

    def test_milestone_string_false_is_not_confirmation(self):
        monitor = MilestoneMonitor(
            {"prompt": "test", "milestones": [{"name": "done", "fact": "done", "confirm": 2}]}
        )
        monitor.update(10, {"ok": True, "satisfied": "false"})
        self.assertEqual(monitor.streak, 0)
        monitor.update(20, {"ok": True, "satisfied": True})
        self.assertFalse(monitor.done)
        monitor.update(30, {"ok": True, "satisfied": True})
        self.assertTrue(monitor.done)
        with self.assertRaises(ValueError):
            monitor.update(40, {"ok": True, "satisfied": True})

    def test_collision_geometry_mesh_and_dimensions(self):
        geometry = {
            "id": "table",
            "p": [0, 0, 0],
            "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "type": "box",
            "dimensions": [1, 1, 0.1],
        }
        collision_geometry(geometry)
        for dimensions in ([1, 1, -1], [1, 1, float("nan")], [1]):
            with self.assertRaises(ValueError):
                collision_geometry({**geometry, "dimensions": dimensions})
        with self.assertRaises(ValueError):
            collision_geometry(
                {**geometry, "type": "mesh", "vertices": [[0, 0, 0]], "triangles": [[0, 1, 2]]}
            )

    def test_target_quaternion_validation(self):
        target_pose({"target_position": [0, 0, 0], "target_orientation": [0, 0, 0, 1]})
        with self.assertRaises(ValueError):
            target_pose({"target_position": [0, 0, 0], "target_orientation": [0, 0, 0, 0]})

    def test_ros_launcher_retains_runtime_but_isolates_foreign_paths(self):
        import os

        with tempfile.TemporaryDirectory(prefix="ros setup ") as directory:
            prefix = Path(directory) / "ros"
            prefix.mkdir()
            (prefix / "setup.bash").write_text("export PYTHONPATH=/simulated/ros/packages\n")
            config = Path(directory) / "config.json"
            config.write_text(
                json.dumps({"schema_version": 1, "paths": {"ros_prefix": str(prefix)}})
            )
            command = [
                "bash",
                str(ROOT / "experiments/mobile_vla/scripts/ros-python"),
                "-c",
                'import json,os,vla_runtime; print(json.dumps({"runtime":vla_runtime.__file__,"paths":os.environ["PYTHONPATH"]}))',
            ]
            result = subprocess.run(
                command,
                env={
                    **os.environ,
                    "VLA_ROOT": str(ROOT),
                    "VLA_CONFIG": str(config),
                    "VLA_ROS_INTERPRETER": sys.executable,
                    "PYTHONPATH": "/foreign/model/env",
                },
                capture_output=True,
                text=True,
                check=True,
            )
            value = json.loads(result.stdout)
            self.assertIn(str(ROOT), value["runtime"])
            self.assertNotIn("/foreign/model/env", value["paths"])
            self.assertIn("/simulated/ros/packages", value["paths"])

    def test_vision_service_import_does_not_load_models_or_start_processes(self):
        path = ROOT / "experiments/mobile_qwen/jetson/stage3_20261115/scripts/vision_service.py"
        code = """import importlib.util,sys
from unittest.mock import patch
spec=importlib.util.spec_from_file_location("service",sys.argv[1])
module=importlib.util.module_from_spec(spec)
with patch("subprocess.Popen",side_effect=AssertionError("unexpected process")):
    spec.loader.exec_module(module)
assert "torch" not in sys.modules
"""
        subprocess.run(
            [sys.executable, "-c", code, str(path)], check=True, capture_output=True, text=True
        )
