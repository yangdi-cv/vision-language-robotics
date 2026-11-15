"""One co-residency/interface probe, no rollout and no motor commands."""

from pathlib import Path
import json, sys, subprocess, time, shutil, traceback
import numpy as np
from PIL import Image

R = Path(__file__).resolve().parents[1]
ROOT = R.parents[1]
OUT = R / "records/qwen_handoff"
OUT.mkdir(exist_ok=True, parents=True)
sys.path.insert(0, str(ROOT / "perception/qwen_scene"))
from worker_client import read_json_line, close_json_worker


def snapshot():
    return subprocess.check_output(
        [
            shutil.which("nvidia-smi"),
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader",
        ],
        text=True,
        close_fds=False,
    ).strip()


def main():
    p = R / "results/baseline/libero_10/task00_init00"
    x = np.load(p / "query_inputs.npz")
    c = json.loads((p / "result.json").read_text())
    image = OUT / "long0_initial.png"
    Image.fromarray(x["base_rgb"][0]).save(image)
    result = {
        "xla_preallocate": __import__("os").environ.get(
            "XLA_PYTHON_CLIENT_PREALLOCATE"
        ),
        "xla_fraction": __import__("os").environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION"),
        "instruction": c["original_prompt"],
        "image_source": str(p / "query_inputs.npz"),
        "same_process_tested": False,
        "reason": "Preserve separate Transformers4.53.2 (OpenPI) and4.57.6 (Qwen) environments",
        "before": snapshot(),
    }
    proc = None
    with (OUT / "qwen_worker.log").open("w") as log:
        try:
            # Spawn before importing JAX to avoid forking its threads.
            proc = subprocess.Popen(
                [
                    str(ROOT / ".venvs/qwen-scene/bin/python"),
                    str(R / "scripts/qwen_planner.py"),
                    "--worker",
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=log,
                text=True,
                start_new_session=True,
            )
            result["worker_pid"] = proc.pid
            result["ready"] = read_json_line(proc, timeout=180)
            result["qwen_loaded"] = snapshot()
            import jax
            from openpi.training import config
            from openpi.policies import policy_config

            policy = policy_config.create_trained_policy(
                config.get_config("pi05_libero"),
                R / "checkpoints/pi05_libero",
                sample_kwargs={"num_steps": 10},
            )
            obs = {
                "observation/image": x["base_rgb"][0],
                "observation/wrist_image": x["wrist_rgb"][0],
                "observation/state": x["state"][0],
                "prompt": c["original_prompt"],
            }
            noise = np.zeros((10, 32), np.float32)
            start = time.monotonic()
            a = policy.infer(obs, noise=noise)["actions"]
            result["pi_first_s"] = time.monotonic() - start
            result["pi_action_shape"] = list(a.shape)
            result["both_loaded"] = snapshot()
            proc.stdin.write(
                json.dumps({"image": str(image), "instruction": c["original_prompt"]})
                + "\n"
            )
            proc.stdin.flush()
            result["qwen_response"] = read_json_line(proc, timeout=120)
            result["after_qwen_inference"] = snapshot()
            start = time.monotonic()
            b = policy.infer(obs, noise=noise)["actions"]
            result["pi_second_s"] = time.monotonic() - start
            result["pi_outputs_finite"] = bool(
                np.isfinite(a).all() and np.isfinite(b).all()
            )
            result["both_after_second_pi"] = snapshot()
            result["co_resident_inference_ok"] = result["pi_outputs_finite"] and bool(
                result["qwen_response"].get("tokens", 0)
            )
            result["planner_schema_ok"] = bool(result["qwen_response"].get("ok"))
            result["jax_devices"] = str(jax.devices())
        except Exception:
            result["error"] = traceback.format_exc()
            result["co_resident_inference_ok"] = False
        finally:
            if proc is not None:
                close_json_worker(proc)
            (OUT / "residency.json").write_text(
                json.dumps(result, indent=2, ensure_ascii=False)
            )
            print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
