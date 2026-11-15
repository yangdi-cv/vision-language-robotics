# ruff: noqa: E402
"""Predeclared basket rollback, drawer prerequisite and mug rollback. No hardware."""

from pathlib import Path
import collections
import json
import hashlib
import sys
import time
import traceback
import numpy as np

R = Path(__file__).resolve().parents[1]
V = R.parent
sys.path.insert(0, str(V / "scripts"))
from run_hierarchical import (
    QwenWorker,
    monitor_view,
    save_png,
    upright,
    gripper_released,
)
from subgoal_state import SubgoalState, requires_release
import evaluate as ev


def prepare():
    specs = []
    for tid in [0, 3, 4]:
        for init in [0, 1, 2]:
            path = (
                V
                / f"hierarchical/zh_eval_hier_v3/libero_10/task{tid:02d}_init{init:02d}"
            )
            src = json.loads((path / "result.json").read_text())
            end = next((e for e in src["events"] if e["kind"] == "subgoal_end"), None)
            prefix = 0 if tid == 3 else (end["step"] if end else None)
            if tid != 3 and end:
                assert end["long_task_atoms"][0]
            specs.append(
                {
                    "task": tid,
                    "init": init,
                    "source": str(path.relative_to(V)),
                    "prefix": prefix,
                    "mutation": {
                        0: "return alphabet soup to its original tabletop pose",
                        3: "close initially open bottom drawer before bowl placement",
                        4: "return white mug to its original tabletop pose",
                    }[tid],
                    "nominal_remaining": src["subgoals"]
                    if tid == 3
                    else src["subgoals"][1:],
                }
            )
    p = R / "injection_protocol.json"
    protocol = {
        "cases": specs,
        "modes": ["continue", "full", "qwen"],
        "recovery_budget": 420,
        "count": 27,
        "known_intervention_time": True,
        "scope": "Recovery conditional on an announced external disturbance; not autonomous fault detection. Harness uses simulator state to construct/verify faults, never as controller input. All branches share the same perturbed state and action-noise seed.",
    }
    if p.exists():
        assert json.loads(p.read_text()) == protocol
    else:
        p.write_text(json.dumps(protocol, indent=2))
    return protocol


def run_case(c, mode, policy, qwen, hashes):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    import imageio.v2 as imageio

    out = R / "results/injection" / f"task{c['task']:02d}_init{c['init']:02d}" / mode
    if (out / "result.json").exists():
        return json.loads((out / "result.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    record = {
        "task": c["task"],
        "init": c["init"],
        "mode": mode,
        "success": False,
        "status": "running",
        "events": [],
        "budget": 420,
    }

    def event(kind, step, **kw):
        record["events"].append({"kind": kind, "step": step, **kw})

    env = writer = None
    traces = []
    start = time.monotonic()
    try:
        suite = benchmark.get_benchmark_dict()["libero_10"]()
        task = suite.get_task(c["task"])
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
        obs = env.set_init_state(
            np.asarray(ev.initial_states(suite, c["task"])[c["init"]])
        )
        for _ in range(10):
            obs, _, _, _ = env.step(ev.DUMMY.tolist())
        name = {0: "alphabet_soup_1", 4: "porcelain_mug_1"}.get(c["task"])
        if name:
            joint = env.env.get_object(name).joints[0]
            original = np.array(env.sim.data.get_joint_qpos(joint)).copy()
        src = np.load(V / c["source"] / "trace.npz")
        prefix = c["prefix"]
        if prefix is None:
            for k, a in enumerate(src["applied"]):
                obs, _, _, _ = env.step(a.tolist())
                if env.env._eval_predicate(
                    env.env.parsed_problem["goal_state"][0]
                ) and gripper_released(obs, a):
                    prefix = k + 1
                    break
            if prefix is None:
                raise RuntimeError("no released completed-goal checkpoint")
        else:
            for a in src["applied"][:prefix]:
                obs, _, _, _ = env.step(a.tolist())
        source_result = json.loads((V / c["source"] / "result.json").read_text())
        active = int(src["subgoal_index"][min(prefix, len(src["subgoal_index"]) - 1)])
        nominal_remaining = source_result["subgoals"][active:]
        record.update(prefix_steps=prefix, nominal_remaining=nominal_remaining)
        before = [
            bool(env.env._eval_predicate(a))
            for a in env.env.parsed_problem["goal_state"]
        ]
        save_png(monitor_view(env, 512), out / "before_fault.png")
        if name:
            assert before[0], "rollback predicate not established"
            env.sim.data.set_joint_qpos(joint, original)
            env.sim.data.set_joint_qvel(joint, np.zeros(6))
        else:
            state = env.env.object_states_dict["white_cabinet_1_bottom_region"]
            assert state.is_open(), "drawer prerequisite not open initially"
            state.set_joint(qpos=0.004)
        env.sim.forward()
        obs = env.regenerate_obs_from_state(env.sim.get_state().flatten())
        for _ in range(5):
            obs, _, _, _ = env.step(ev.DUMMY.tolist())
        after = [
            bool(env.env._eval_predicate(a))
            for a in env.env.parsed_problem["goal_state"]
        ]
        if name:
            assert not after[0]
        else:
            assert env.env.object_states_dict[
                "white_cabinet_1_bottom_region"
            ].is_close()
        assert not env.check_success()
        state = np.asarray(env.sim.get_state().flatten())
        h = hashlib.sha256(state.tobytes()).hexdigest()
        key = (c["task"], c["init"])
        assert hashes.setdefault(key, h) == h
        record.update(
            before_fault_atoms=before, after_fault_atoms=after, perturbed_state_sha256=h
        )
        np.savez_compressed(out / "perturbed_state.npz", state=state)
        image = save_png(monitor_view(env, 512), out / "after_fault.png")
        if mode == "continue":
            instructions = nominal_remaining
        elif mode == "full":
            instructions = [task.language]
        else:
            reply = qwen.request(
                {
                    "op": "plan",
                    "image": image,
                    "instruction": task.language,
                    "context": "An external event may have invalidated prior progress. Reinspect ALL requested conditions in the current scene. Plan the remaining actions, including opening closed containers before placing items inside. Do not assume previously completed steps remain true.\n",
                }
            )
            event("replan", 0, response=reply)
            if not reply.get("ok"):
                raise RuntimeError("planner invalid: " + str(reply.get("error")))
            instructions = [x["instruction"] for x in reply["subgoals"]]
        record["instructions"] = instructions
        plan = SubgoalState(instructions, confirm=2)
        reference = image
        rng = np.random.default_rng(910000 + c["task"] * 100 + c["init"])
        queue = collections.deque()
        queries = 0
        last = ev.DUMMY.copy()
        writer = imageio.get_writer(
            out / "rollout.mp4", fps=20, codec="libx264", quality=7, macro_block_size=16
        )
        for step in range(420):
            writer.append_data(upright(obs))
            if not queue:
                if (
                    not plan.last
                    and queries > 0
                    and queries % 4 == 0
                    and (
                        not requires_release(plan.current)
                        or gripper_released(obs, last)
                    )
                ):
                    img = save_png(
                        monitor_view(env, 512), out / f"check_{step:04d}.png"
                    )
                    reply = qwen.request(
                        {
                            "op": "check",
                            "image": [reference, img],
                            "subgoal": plan.current,
                        }
                    )
                    completed = plan.observe(
                        reply.get("ok", False), reply.get("done", False)
                    )
                    event(
                        "check",
                        step,
                        index=plan.index,
                        confirmed=completed,
                        response=reply,
                    )
                    if completed:
                        plan.advance()
                        reference = img
                        event("switch", step, index=plan.index)
                chunk = np.asarray(
                    policy.infer(
                        ev.make_input(obs, plan.current),
                        noise=rng.standard_normal((10, 32)).astype(np.float32),
                    )["actions"],
                    dtype=np.float64,
                )
                assert chunk.shape == (10, 7) and np.isfinite(chunk).all()
                queue.extend(chunk[:5])
                queries += 1
            last = np.asarray(queue.popleft())
            obs, _, done, _ = env.step(last.tolist())
            traces.append(
                {
                    "applied": last,
                    "qpos": env.sim.data.qpos.copy(),
                    "eef": obs["robot0_eef_pos"].copy(),
                    "subgoal_index": plan.index,
                    "official_success": bool(env.check_success()),
                }
            )
            if done:
                break
        writer.append_data(upright(obs))
        save_png(monitor_view(env, 512), out / "final.png")
        record.update(
            success=bool(env.check_success()),
            steps=len(traces),
            queries=queries,
            final_atoms=[
                bool(env.env._eval_predicate(a))
                for a in env.env.parsed_problem["goal_state"]
            ],
        )
        record["status"] = "success" if record["success"] else "timeout"
    except Exception:
        record.update(status="error", error=traceback.format_exc())
        print(record["error"], flush=True)
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
    record["wall_seconds"] = time.monotonic() - start
    (out / "result.json").write_text(json.dumps(record, indent=2))
    print(
        json.dumps(
            {
                k: record.get(k)
                for k in ["task", "init", "mode", "status", "success", "steps"]
            }
        ),
        flush=True,
    )
    if record["status"] == "error":
        raise RuntimeError("setup or execution error: stop and inspect")
    return record


def main():
    p = prepare()
    qwen = QwenWorker(R / "logs/injection_qwen.log", profile="contract")
    try:
        from openpi.training import config
        from openpi.policies import policy_config

        policy = policy_config.create_trained_policy(
            config.get_config("pi05_libero"),
            V / "checkpoints/pi05_libero",
            sample_kwargs={"num_steps": 10},
        )
        hashes = {}
        rows = []
        # One init per task first; only after valid setup proceed to the remaining six cases.
        cases = sorted(p["cases"], key=lambda c: (c["init"], c["task"]))
        for c in cases:
            for mode in p["modes"]:
                rows.append(run_case(c, mode, policy, qwen, hashes))
            (R / "injection_results.json").write_text(json.dumps(rows, indent=2))
    finally:
        qwen.close()


if __name__ == "__main__":
    main()
