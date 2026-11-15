# ruff: noqa: E402
"""CPU regression checks for input orientation, provenance and interrupted episodes."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image

R = Path(__file__).resolve().parents[1]
ROOT = R.parents[1]
sys.path[:0] = [str(R / "scripts"), str(ROOT / "perception/qwen_scene")]
from engine import SceneEngine, prepare_image
from qwen_planner import Planner
import run_hierarchical as runner


class DeliveryTests(unittest.TestCase):
    def test_clockwise_rotation_changes_pixels_and_shape(self):
        pixels = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
        for degrees in (0, 90, 180, 270):
            result, recorded = prepare_image(Image.fromarray(pixels), degrees)
            np.testing.assert_equal(np.array(result), np.rot90(pixels, -degrees // 90))
            self.assertEqual(recorded, degrees)

    def test_live_frame_not_rotated_twice(self):
        pixels = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
        result, recorded = prepare_image(Image.fromarray(pixels), 270, True)
        np.testing.assert_equal(np.array(result), pixels)
        self.assertEqual(recorded, 270)
        with self.assertRaises(ValueError):
            prepare_image(result, 13)

    def test_analyze_saves_pixels_passed_to_inference(self):
        e = SceneEngine.__new__(SceneEngine)
        e.load_seconds = 0
        seen = []

        def generate(image, prompt, max_new_tokens):
            seen.append(image.copy())
            return dict(raw="{}", seconds=0, tokens=1, inference={"prompt": prompt})

        e.generate = generate
        with tempfile.TemporaryDirectory() as tmp:
            result, directory = e.analyze(Image.new("RGB", (3, 2), "red"), tmp, 90)
            self.assertEqual(result["image_size"], [2, 3])
            with Image.open(directory / "input.png") as saved:
                self.assertEqual(saved.tobytes(), seen[0].tobytes())
            self.assertTrue(result["rotation_applied_here"])

    def test_planner_uses_shared_generation(self):
        p = Planner.__new__(Planner)
        calls = []
        p.engine = types.SimpleNamespace(
            generate=lambda images, prompt, max_new_tokens: calls.append(
                (images, prompt, max_new_tokens)
            )
            or dict(raw="{}", inference={})
        )
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "input.png"
            Image.new("RGB", (3, 2)).save(image)
            r = p.generate(str(image), "actual prompt", 17)
        self.assertEqual(calls[0][1:], ("actual prompt", 17))
        self.assertIn("planner_sha256", r["inference"])

    def test_shared_generation_records_actual_prompt_and_pixels(self):
        class Inputs(dict):
            def to(self, device):
                return self

        class Processor:
            def apply_chat_template(self, messages, **kw):
                return Inputs(input_ids=np.zeros((1, 2), dtype=int))

            def decode(self, *args, **kwargs):
                return "response"

        class Model:
            def generate(self, **kwargs):
                return np.zeros((1, 4), dtype=int)

            def parameters(self):
                return iter([types.SimpleNamespace(device="cpu-test")])

        import contextlib

        e = SceneEngine.__new__(SceneEngine)
        e.processor = Processor()
        e.model = Model()
        e.torch = types.SimpleNamespace(
            inference_mode=contextlib.nullcontext,
            cuda=types.SimpleNamespace(
                synchronize=lambda: None,
                memory_allocated=lambda: 0,
                memory_reserved=lambda: 0,
            ),
        )
        image = Image.new("RGB", (3, 2), "blue")
        with patch("engine.importlib.metadata.version", return_value="test"):
            a = e.generate(image, "first", 9)
            b = e.generate(image, "second", 9)
        meta = a["inference"]
        self.assertEqual(meta["prompt"], "first")
        self.assertNotEqual(meta["prompt_sha256"], b["inference"]["prompt_sha256"])
        self.assertEqual(
            meta["images"][0]["pixel_sha256"],
            hashlib.sha256(image.tobytes()).hexdigest(),
        )
        self.assertEqual(meta["generation"], dict(max_new_tokens=9, do_sample=False))
        self.assertTrue(meta["revision"])

    def test_sigkill_leaves_readable_partial_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            code = """import sys,time
from episode_journal import EpisodeJournal
j=EpisodeJournal(sys.argv[1],{'status':'running','events':[]})
j.step(2,{'applied':[1,2,3]})
print('saved',flush=True)
time.sleep(60)
"""
            env = dict(os.environ, PYTHONPATH=str(R / "scripts"))
            p = subprocess.Popen(
                [sys.executable, "-c", code, tmp],
                env=env,
                stdout=subprocess.PIPE,
                text=True,
            )
            try:
                self.assertEqual(p.stdout.readline().strip(), "saved")
                p.kill()
                p.wait(timeout=5)
                self.assertEqual(
                    json.loads((Path(tmp) / "result.json").read_text())["steps"], 2
                )
                self.assertEqual(
                    json.loads((Path(tmp) / "trace.jsonl").read_text())["applied"],
                    [1, 2, 3],
                )
            finally:
                if p.poll() is None:
                    p.kill()
                    p.wait()
                p.stdout.close()

    def test_runner_exception_keeps_completed_steps(self):
        self.check_runner_exception(runner)

    def test_strict_runner_exception_keeps_completed_steps(self):
        import run_hierarchical_recovery

        self.check_runner_exception(run_hierarchical_recovery)

    def check_runner_exception(self, runner):
        class Env:
            def __init__(self, **kw):
                self.n = 0
                self.env = types.SimpleNamespace(parsed_problem={"goal_state": []})

            def seed(self, x):
                pass

            def reset(self):
                pass

            def set_init_state(self, x):
                return self.obs()

            def obs(self):
                return {
                    "agentview_image": np.zeros((4, 4, 3), np.uint8),
                    "robot0_eef_pos": np.zeros(3),
                    "robot0_gripper_qpos": np.zeros(2),
                }

            def step(self, action):
                self.n += 1
                if self.n == 14:
                    raise RuntimeError("injected step failure")
                return self.obs(), 0, False, {}

            def check_success(self):
                return False

            def close(self):
                pass

        suite = types.SimpleNamespace(
            get_task=lambda _: types.SimpleNamespace(
                language="test", problem_folder="", bddl_file="test"
            )
        )
        ev = types.SimpleNamespace(
            initial_states=lambda *a: np.zeros((1, 2)),
            SUITES=["libero_10"],
            LIMITS={"libero_10": 20},
            DUMMY=np.zeros(7),
            make_input=lambda obs, instr: {},
        )
        libero = types.ModuleType("libero.libero")
        libero.benchmark = types.SimpleNamespace(
            get_benchmark_dict=lambda: {"libero_10": lambda: suite}
        )
        libero.get_libero_path = lambda _: "."
        envmodule = types.ModuleType("libero.libero.envs")
        envmodule.OffScreenRenderEnv = Env

        class Writer:
            def append_data(self, x):
                pass

            def close(self):
                raise RuntimeError("injected cleanup failure")

        args = types.SimpleNamespace(
            suite="libero_10",
            task_id=0,
            init=0,
            instruction=None,
            tag="test",
            max_steps=20,
            planner="none",
            monitor="qwen",
            subgoal_budget=300,
            monitor_every=1,
            monitor_res=4,
            gripper_gate=True,
            confirm=2,
            max_replans=0,
            replan_steps=5,
            recovery_mode="replan",
        )
        # render is independent of policy observations
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(runner, "R", Path(tmp)),
            patch.object(
                runner, "monitor_view", return_value=np.zeros((4, 4, 3), np.uint8)
            ),
            patch.dict(
                sys.modules,
                {
                    "evaluate": ev,
                    "libero": types.ModuleType("libero"),
                    "libero.libero": libero,
                    "libero.libero.envs": envmodule,
                },
            ),
            patch("imageio.v2.get_writer", return_value=Writer()),
        ):
            with self.assertRaisesRegex(RuntimeError, "cleanup failure"):
                runner.run(
                    args,
                    types.SimpleNamespace(
                        infer=lambda *a, **kw: {"actions": np.zeros((10, 7))}
                    ),
                    None,
                )
            out = Path(tmp) / "hierarchical/test/libero_10/task00_init00"
            result = json.loads((out / "result.json").read_text())
            self.assertEqual(result["status"], "error")
            self.assertEqual(result["steps"], 3)
            self.assertIn("injected step failure", result["error"])
            self.assertEqual(len((out / "trace.jsonl").read_text().splitlines()), 3)
            with np.load(out / "trace.npz") as trace:
                self.assertEqual(len(trace["applied"]), 3)


if __name__ == "__main__":
    unittest.main()
