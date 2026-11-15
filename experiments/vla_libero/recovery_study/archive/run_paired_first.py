"""Three fixed replay states, three recovery modes, two paired action-noise seeds.
No scene mutation, oracle decisions or hardware access. Official goal is scoring only.
"""

from __future__ import annotations
import collections
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parents[1]
V = R.parent
sys.path.insert(0, str(V / "scripts"))
import evaluate as ev
from run_hierarchical import (
    QwenWorker,
    monitor_view,
    gripper_released,
    save_png,
    upright,
)
from subgoal_state import SubgoalState

MODES = ["continue", "full", "replan"]
SEEDS = [31001, 31002]
TASKS = [2, 4, 8]
PREFIX = 150
REMAINING = 370


def hash_array(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def prepare_manifest():
    cases = []
    for tid in TASKS:
        base = V / "hierarchical/zh_eval_hier_v2/libero_10" / f"task{tid:02d}_init02"
        r = json.loads((base / "result.json").read_text())
        cases.append(
            {
                "task": tid,
                "init": 2,
                "source": str(base.relative_to(V)),
                "source_result_sha256": hashlib.sha256(
                    (base / "result.json").read_bytes()
                ).hexdigest(),
                "prefix_actions_sha256": hash_array(
                    np.load(base / "trace.npz")["applied"][:PREFIX]
                ),
                "subgoals": r["subgoals"],
            }
        )
    m = {
        "suite": "libero_10",
        "cases": cases,
        "prefix_steps": PREFIX,
        "remaining_steps": REMAINING,
        "action_chunk_execute": 5,
        "modes": MODES,
        "noise_seeds": SEEDS,
        "n": 18,
        "monitor_confirm": 2,
        "monitor_every_queries": 4,
        "gripper_gate": True,
        "pure_replan_no_full_fallback": True,
        "selection": "Three previously identified v2 failures, fixed before new rollout; conditional recovery, not suite success rate. Two seeds are stochastic repeats, not independent initial states.",
    }
    p = R / "protocol.json"
    if p.exists():
        assert json.loads(p.read_text()) == m
    else:
        p.write_text(json.dumps(m, indent=2))
    return m


def one(case, mode, seed, policy, qwen, reference_hashes):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    import imageio.v2 as imageio

    out = R / "results" / f"task{case['task']:02d}_init02" / f"{mode}_seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "result.json").exists():
        raise FileExistsError(out)
    frames = out / "frames"
    frames.mkdir(exist_ok=True)
    r = {
        "task": case["task"],
        "init": 2,
        "mode": mode,
        "seed": seed,
        "prefix_steps": PREFIX,
        "remaining_budget": REMAINING,
        "success": False,
        "status": "running",
        "events": [],
    }
    env = writer = None
    traces = []
    start = time.monotonic()

    def event(kind, step, **kw):
        r["events"].append({"kind": kind, "step": step, **kw})

    try:
        suite = benchmark.get_benchmark_dict()["libero_10"]()
        task = suite.get_task(case["task"])
        env = OffScreenRenderEnv(
            bddl_file_name=str(
                Path(get_libero_path("bddl_files"))
                / task.problem_folder
                / task.bddl_file
            ),
            camera_heights=256,
            camera_widths=256,
        )
        env.seed(7)
        env.reset()
        obs = env.set_init_state(np.asarray(ev.initial_states(suite, case["task"])[2]))
        for _ in range(10):
            obs, _, _, _ = env.step(ev.DUMMY.tolist())
        initial = save_png(monitor_view(env, 512), frames / "initial.png")
        src = V / case["source"]
        old = np.load(src / "trace.npz")
        for action in old["applied"][:PREFIX]:
            obs, _, _, _ = env.step(action.tolist())
        state = np.asarray(env.sim.get_state().flatten()).copy()
        state_hash = hash_array(state)
        if case["task"] not in reference_hashes:
            reference_hashes[case["task"]] = state_hash
        assert state_hash == reference_hashes[case["task"]], "paired restore differs"
        r["prefix_state_sha256"] = state_hash
        r["prefix_eef_error"] = float(
            np.max(np.abs(obs["robot0_eef_pos"] - old["eef"][PREFIX - 1]))
        )
        assert r["prefix_eef_error"] < 1e-10
        r["prefix_goal_atoms"] = [
            bool(env.env._eval_predicate(a))
            for a in env.env.parsed_problem["goal_state"]
        ]
        assert not env.check_success(), "recovery state already successful"
        np.savez_compressed(out / "prefix_state.npz", state=state)
        current = save_png(monitor_view(env, 512), frames / "recovery_start.png")
        # The same prefix had no confirmed earlier subgoal. No ground truth is sent to Qwen.
        instructions = case["subgoals"]
        ref = initial
        if mode == "full":
            instructions = [task.language]
            ref = current
        elif mode == "replan":
            t = time.monotonic()
            reply = qwen.request(
                {
                    "op": "replan",
                    "image": current,
                    "instruction": task.language,
                    "completed": [],
                    "failed": instructions[0],
                    "observation": "The current subgoal has used its execution budget and has not been confirmed complete. Reinspect the current image; do not assume it succeeded.",
                }
            )
            event("replan", 0, seconds=time.monotonic() - t, response=reply)
            if not reply.get("ok"):
                raise RuntimeError("replan schema error: " + str(reply.get("error")))
            instructions = [x["instruction"] for x in reply["subgoals"]]
            ref = current
        plan = SubgoalState(instructions, confirm=2)
        r["instructions"] = list(instructions)
        rng = np.random.default_rng(seed)
        queue = collections.deque()
        queries = 0
        last = np.asarray(old["applied"][PREFIX - 1]).copy()
        writer = imageio.get_writer(
            out / "rollout.mp4", fps=20, codec="libx264", quality=7, macro_block_size=16
        )
        event("subgoal_start", 0, index=plan.index, instruction=plan.current)
        policy_seconds = []
        for step in range(REMAINING):
            writer.append_data(upright(obs))
            if not queue:
                if (
                    not plan.last
                    and queries > 0
                    and queries % 4 == 0
                    and gripper_released(obs, last)
                ):
                    img = save_png(
                        monitor_view(env, 512), frames / f"check_{step:04d}.png"
                    )
                    t = time.monotonic()
                    reply = qwen.request(
                        {"op": "check", "image": [ref, img], "subgoal": plan.current}
                    )
                    done = plan.observe(bool(reply.get("ok")), bool(reply.get("done")))
                    event(
                        "check",
                        step,
                        index=plan.index,
                        seconds=time.monotonic() - t,
                        response=reply,
                        confirmed=done,
                        shadow_atoms=[
                            bool(env.env._eval_predicate(a))
                            for a in env.env.parsed_problem["goal_state"]
                        ],
                    )
                    if done:
                        oldidx = plan.index
                        plan.advance()
                        ref = img
                        event("subgoal_end", step, index=oldidx, reason="confirmed")
                        event(
                            "subgoal_start",
                            step,
                            index=plan.index,
                            instruction=plan.current,
                        )
                t = time.monotonic()
                chunk = np.asarray(
                    policy.infer(
                        ev.make_input(obs, plan.current),
                        noise=rng.standard_normal((10, 32)).astype(np.float32),
                    )["actions"],
                    dtype=np.float64,
                )
                policy_seconds.append(time.monotonic() - t)
                assert chunk.shape == (10, 7) and np.isfinite(chunk).all()
                queue.extend(chunk[:5])
                queries += 1
            last = np.asarray(queue.popleft())
            obs, _, done, _ = env.step(last.tolist())
            traces.append(
                {
                    "applied": last,
                    "eef": obs["robot0_eef_pos"].copy(),
                    "qpos": env.sim.data.qpos.copy(),
                    "subgoal_index": plan.index,
                    "official_success": bool(env.check_success()),
                    "goal_atoms": np.array(
                        [
                            bool(env.env._eval_predicate(a))
                            for a in env.env.parsed_problem["goal_state"]
                        ]
                    ),
                }
            )
            if done:
                break
        writer.append_data(upright(obs))
        save_png(monitor_view(env, 512), frames / "final.png")
        r.update(
            success=bool(env.check_success()),
            steps=len(traces),
            total_steps=PREFIX + len(traces),
            queries=queries,
            confirmed_completed=plan.confirmed,
            final_goal_atoms=[
                bool(env.env._eval_predicate(a))
                for a in env.env.parsed_problem["goal_state"]
            ],
            policy_seconds=policy_seconds,
        )
        r["status"] = "success" if r["success"] else "timeout"
    except Exception:
        r.update(status="error", error=traceback.format_exc())
        print(r["error"], flush=True)
    finally:
        if writer:
            writer.close()
        if env:
            env.close()
    if traces:
        np.savez_compressed(
            out / "trace.npz",
            **{k: np.asarray([x[k] for x in traces]) for k in traces[0]},
        )
    r["wall_seconds"] = time.monotonic() - start
    (out / "result.json").write_text(json.dumps(r, indent=2))
    print(
        json.dumps(
            {
                k: r.get(k)
                for k in [
                    "task",
                    "mode",
                    "seed",
                    "status",
                    "success",
                    "steps",
                    "wall_seconds",
                ]
            }
        ),
        flush=True,
    )
    if r["status"] == "error":
        raise RuntimeError("episode failed; stop instead of silently continuing")
    return r


def main():
    m = prepare_manifest()
    qwen = QwenWorker(R / "logs/qwen_worker.log")
    try:
        from openpi.training import config
        from openpi.policies import policy_config

        policy = policy_config.create_trained_policy(
            config.get_config("pi05_libero"),
            V / "checkpoints/pi05_libero",
            sample_kwargs={"num_steps": 10},
        )
        hashes = {}
        results = []
        # Rotate order by task/seed to reduce systematic warmup/order confounds.
        for i, case in enumerate(m["cases"]):
            for j, seed in enumerate(SEEDS):
                order = MODES[(i + j) % 3 :] + MODES[: (i + j) % 3]
                for mode in order:
                    results.append(one(case, mode, seed, policy, qwen, hashes))
        (R / "paired_summary.json").write_text(
            json.dumps(
                {
                    mode: {
                        "n": sum(x["mode"] == mode for x in results),
                        "success": sum(
                            x["success"] for x in results if x["mode"] == mode
                        ),
                    }
                    for mode in MODES
                },
                indent=2,
            )
        )
    finally:
        qwen.close()


if __name__ == "__main__":
    main()
