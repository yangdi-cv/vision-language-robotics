# ruff: noqa: E402
"""Confirmed-only subgoal execution. Historical runner remains in run_hierarchical.py.

Timeouts never advance a plan. Recovery full/replan are explicit separate modes.
"""

from __future__ import annotations
import collections
import json
import sys
import time
import traceback
from pathlib import Path
import numpy as np
from episode_journal import EpisodeJournal
from subgoal_state import SubgoalState, requires_release

R = Path(__file__).resolve().parents[1]
ROOT = R.parents[1]
sys.path.insert(0, str(R / "scripts"))
sys.path.insert(0, str(ROOT / "perception/qwen_scene"))


# Compatibility module: helpers and the CLI are shared with the canonical entry.
from run_hierarchical import upright, monitor_view, gripper_released, save_png
from run_hierarchical import QwenWorker as SharedQwenWorker


class QwenWorker(SharedQwenWorker):
    def __init__(self, log_path):
        super().__init__(log_path, profile="contract")


def run(args, policy, qwen):
    import imageio.v2 as imageio
    import evaluate as ev
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[args.suite]()
    task = suite.get_task(args.task_id)
    init = np.asarray(ev.initial_states(suite, args.task_id)[args.init])
    instruction = args.instruction or task.language
    out = (
        R
        / "hierarchical"
        / args.tag
        / args.suite
        / f"task{args.task_id:02d}_init{args.init:02d}"
    )
    if (out / "result.json").exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    out.mkdir(parents=True, exist_ok=True)
    frames_dir = out / "qwen_inputs"
    frames_dir.mkdir(exist_ok=True)
    seed = 700000 + ev.SUITES.index(args.suite) * 10000 + args.task_id * 100 + args.init
    rng = np.random.default_rng(seed)
    max_steps = args.max_steps or ev.LIMITS[args.suite]
    record = {
        "suite": args.suite,
        "task_id": args.task_id,
        "initial_state_id": args.init,
        "original_task_language": task.language,
        "input_instruction": instruction,
        "planner": args.planner,
        "monitor": args.monitor,
        "max_steps": max_steps,
        "subgoal_step_budget": args.subgoal_budget,
        "monitor_every_queries": args.monitor_every,
        "monitor_res": args.monitor_res,
        "gripper_gate": args.gripper_gate,
        "confirm": args.confirm,
        "max_replans": args.max_replans,
        "recovery_mode": args.recovery_mode,
        "transition_version": "confirmed_only_v1",
        "operation_checks": getattr(args, "operation_checks", False),
        "last_subgoal_runs_to_limit": True,
        "noise_seed": seed,
        "runner_sha256": __import__("hashlib")
        .sha256(Path(__file__).read_bytes())
        .hexdigest(),
        "events": [],
        "status": "running",
        "success": False,
    }

    journal = EpisodeJournal(out, record)

    def event(kind, step, **kw):
        e = {"kind": kind, "step": step, **kw}
        record["events"].append(e)
        journal.event(e)
        print(json.dumps(e, ensure_ascii=False), flush=True)

    traces = []
    env = writer = None
    try:
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
        obs = env.set_init_state(init)
        for _ in range(10):
            obs, _, _, _ = env.step(ev.DUMMY.tolist())
        if env.check_success():
            raise ValueError("Initial state already satisfies goal")

        # ---- high-level plan ----
        if args.planner == "none":
            subgoals = [instruction]
        elif args.planner == "fixed":
            subgoals = json.loads(args.subgoals)
        else:
            img = save_png(
                monitor_view(env, args.monitor_res), frames_dir / "plan_step0000.png"
            )
            t0 = time.monotonic()
            op = "translate" if args.planner == "translate" else "plan"
            reply = qwen.request({"op": op, "image": img, "instruction": instruction})
            event(
                "plan",
                0,
                ok=reply.get("ok"),
                seconds=time.monotonic() - t0,
                raw=reply.get("raw"),
                error=reply.get("error"),
            )
            if not reply.get("ok"):
                raise RuntimeError(f"Planner failed: {reply.get('error')}")
            subgoals = [s["instruction"] for s in reply["subgoals"]]
        record["subgoals"] = list(subgoals)
        state = SubgoalState(subgoals, confirm=args.confirm)
        budget_reported = False

        monitors = None
        if args.monitor == "oracle" and args.planner != "none":
            monitors = json.loads(args.subgoal_predicates)
            if len(monitors) != len(subgoals):
                raise ValueError("--subgoal-predicates must align with subgoals")
            for atoms in monitors:
                for atom in atoms:
                    env.env._eval_predicate(atom)  # fail fast on unknown names

        writer = imageio.get_writer(
            out / "rollout.mp4", fps=20, codec="libx264", quality=7, macro_block_size=16
        )
        idx, sub_start, queries, checks, replans, last_reason = 0, 0, 0, 0, 0, ""
        full_task_mode = False
        action_plan = collections.deque()
        traces = []
        sub_ref = save_png(
            monitor_view(env, args.monitor_res), frames_dir / "ref_step0000.png"
        )
        last_action = np.asarray(ev.DUMMY, dtype=np.float64)
        event("subgoal_start", 0, index=0, subgoal=subgoals[0])
        for step in range(max_steps):
            writer.append_data(upright(obs))
            if not action_plan:
                # --- completion monitor at re-query boundaries ---
                done_sub = False
                last_sub = idx == len(subgoals) - 1
                if args.planner != "none" and step > sub_start and not last_sub:
                    if args.monitor == "oracle":
                        done_sub = all(
                            bool(env.env._eval_predicate(a)) for a in monitors[idx]
                        )
                    elif queries % args.monitor_every == 0 and (
                        not args.gripper_gate
                        or (
                            getattr(args, "operation_checks", False)
                            and not requires_release(state.current)
                        )
                        or gripper_released(obs, last_action)
                    ):
                        img = save_png(
                            monitor_view(env, args.monitor_res),
                            frames_dir / f"check_step{step:04d}.png",
                        )
                        t0 = time.monotonic()
                        reply = qwen.request(
                            {
                                "op": "check",
                                "image": [sub_ref, img],
                                "subgoal": subgoals[idx],
                            }
                        )
                        checks += 1
                        done_sub = state.observe(
                            bool(reply.get("ok")), bool(reply.get("done"))
                        )
                        last_reason = reply.get("reason") or last_reason
                        event(
                            "check",
                            step,
                            index=idx,
                            done=bool(reply.get("done")),
                            confirmed=done_sub,
                            seconds=time.monotonic() - t0,
                            reason=reply.get("reason"),
                            error=reply.get("error"),
                            shadow_long_task_atoms=[
                                bool(env.env._eval_predicate(a))
                                for a in env.env.parsed_problem["goal_state"]
                            ],
                        )
                # Confirmations take priority over a simultaneous timeout.
                if done_sub:
                    if args.monitor == "oracle":
                        state.streak = state.confirm
                    state.advance()
                    event(
                        "subgoal_end",
                        step,
                        index=idx,
                        reason="confirmed",
                        long_task_atoms=[
                            bool(env.env._eval_predicate(a))
                            for a in env.env.parsed_problem["goal_state"]
                        ],
                    )
                    idx = state.index
                    sub_start, last_reason = step, ""
                    budget_reported = False
                    sub_ref = save_png(
                        monitor_view(env, args.monitor_res),
                        frames_dir / f"ref_step{step:04d}.png",
                    )
                    event("subgoal_start", step, index=idx, subgoal=state.current)
                elif (
                    step - sub_start >= args.subgoal_budget
                    and not budget_reported
                    and args.planner not in ("none", "translate")
                ):
                    event(
                        "subgoal_timeout",
                        step,
                        index=idx,
                        subgoal=state.current,
                        confirmed_completed=list(state.confirmed),
                    )
                    budget_reported = True
                    can_recover = (
                        replans < args.max_replans
                        and not full_task_mode
                        and args.recovery_mode != "continue"
                    )
                    if can_recover:
                        img = save_png(
                            monitor_view(env, args.monitor_res),
                            frames_dir / f"replan_step{step:04d}.png",
                        )
                        t0 = time.monotonic()
                        if args.recovery_mode == "full":
                            reply = qwen.request(
                                {
                                    "op": "translate",
                                    "image": img,
                                    "instruction": instruction,
                                }
                            )
                        else:
                            reply = qwen.request(
                                {
                                    "op": "replan",
                                    "image": img,
                                    "instruction": instruction,
                                    "completed": list(state.confirmed),
                                    "remaining": list(
                                        state.instructions[state.index :]
                                    ),
                                    "failed": state.current,
                                    "observation": last_reason,
                                }
                            )
                        replans += 1
                        event(
                            "recovery",
                            step,
                            index=idx,
                            mode=args.recovery_mode,
                            ok=reply.get("ok"),
                            seconds=time.monotonic() - t0,
                            raw=reply.get("raw"),
                            error=reply.get("error"),
                            confirmed_completed=list(state.confirmed),
                        )
                        if reply.get("ok"):
                            state.replace_pending(
                                [x["instruction"] for x in reply["subgoals"]]
                            )
                            subgoals = state.instructions
                            full_task_mode = args.recovery_mode == "full"
                            sub_start, last_reason = step, ""
                            budget_reported = False
                            sub_ref = save_png(
                                monitor_view(env, args.monitor_res),
                                frames_dir / f"ref_step{step:04d}.png",
                            )
                            event(
                                "subgoal_start", step, index=idx, subgoal=state.current
                            )
                subgoals = state.instructions
                element = ev.make_input(obs, subgoals[idx])
                noise = rng.standard_normal((10, 32)).astype(np.float32)
                chunk = np.asarray(
                    policy.infer(element, noise=noise)["actions"], dtype=np.float64
                )
                if chunk.shape != (10, 7) or not np.isfinite(chunk).all():
                    raise ValueError(f"Invalid policy chunk {chunk.shape}")
                queries += 1
                action_plan.extend(chunk[: args.replan_steps])
            applied = last_action = np.asarray(action_plan.popleft())
            obs, _, done, _ = env.step(applied.tolist())
            traces.append(
                {
                    "subgoal_index": idx,
                    "applied": applied,
                    "eef": obs["robot0_eef_pos"].copy(),
                    "gripper": obs["robot0_gripper_qpos"].copy(),
                    "success_predicate": bool(env.check_success()),
                }
            )
            record.update(
                policy_queries=queries,
                qwen_checks=checks,
                replans=replans,
                active_subgoal_index=idx,
                active_subgoal=subgoals[idx],
            )
            journal.step(len(traces), traces[-1])
            if done:
                break
        writer.append_data(upright(obs))
        record["success"] = bool(env.check_success())
        record["status"] = (
            "success"
            if record["success"]
            else ("plan_exhausted" if idx >= len(subgoals) else "timeout")
        )
        record.update(
            steps=len(traces),
            policy_queries=queries,
            qwen_checks=checks,
            replans=replans,
            full_task_fallback=full_task_mode,
            final_subgoals=subgoals,
            confirmed_completed=list(state.confirmed),
            final_atoms=[
                bool(env.env._eval_predicate(a))
                for a in env.env.parsed_problem["goal_state"]
            ],
        )
    except (Exception, KeyboardInterrupt) as exc:
        record["status"] = (
            "interrupted" if isinstance(exc, KeyboardInterrupt) else "error"
        )
        record["error"] = traceback.format_exc()
        print(record["error"], flush=True)
    finally:
        # Save before cleanup: renderer/video shutdown must not erase the outcome.
        record["steps"] = len(traces)
        journal.save()
        try:
            if traces:
                np.savez_compressed(
                    out / "trace.npz",
                    **{k: np.asarray([t[k] for t in traces]) for k in traces[0]},
                )
        finally:
            try:
                if writer:
                    writer.close()
            finally:
                if env is not None:
                    env.close()
    print(
        json.dumps(
            {
                k: record.get(k)
                for k in [
                    "status",
                    "success",
                    "steps",
                    "policy_queries",
                    "qwen_checks",
                    "subgoals",
                ]
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return record


def main():
    from run_hierarchical import main as unified_main

    unified_main(
        [
            "--execution-profile",
            "strict",
            "--planner-profile",
            "contract",
            *sys.argv[1:],
        ]
    )


if __name__ == "__main__":
    main()
