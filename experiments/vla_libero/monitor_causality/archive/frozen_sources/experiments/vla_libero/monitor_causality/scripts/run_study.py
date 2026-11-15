"""Paired mechanism study; model weights, plans and historical results are unchanged."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

S = Path(__file__).resolve().parents[1]
V = S.parent
ROOT = V.parents[1]
sys.path[:0] = [str(V / "scripts"), str(ROOT / "perception/qwen_scene")]
from episode_journal import atomic_json
import runner_snapshot as runner
from interventions import Monitor, MODES


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protocol():
    priors = {}
    for task in range(10):
        for initial in range(3):
            p = (
                V
                / f"hierarchical/zh_eval_hier_v3/libero_10/task{task:02d}_init{initial:02d}/result.json"
            )
            r = json.loads(p.read_text())
            assert len(r["subgoals"]) == 2
            priors[f"{task:02d}_{initial:02d}"] = dict(
                instruction=r["input_instruction"],
                plan=r["subgoals"],
                source_sha256=sha(p),
            )
    adapter = V / "checker_v2/results/model/best_adapter"
    selection = json.loads((V / "checker_v2/results/model/summary.json").read_text())
    sources = [
        S / "scripts/runner_snapshot.py",
        S / "scripts/run_study.py",
        S / "scripts/interventions.py",
        V / "scripts/run_hierarchical.py",
        V / "scripts/episode_journal.py",
        V / "scripts/qwen_planner.py",
        V / "scripts/completion_checker_v2.py",
        V / "checker_v2/scripts/protocol.py",
        ROOT / "perception/qwen_scene/engine.py",
        ROOT / "perception/qwen_scene/worker_client.py",
    ]
    return dict(
        scope="Mechanism diagnosis on previously studied Long initial states, not held-out policy improvement",
        modes=list(MODES),
        cases=[[t, i] for t in range(10) for i in range(3)],
        plans=priors,
        policy="official pi05_libero;10 denoising steps; execute first5 of10 actions; explicit paired noise",
        environment_seed=7,
        max_steps=520,
        stage_budget=300,
        confirm=2,
        gripper_gate=True,
        monitor_every_queries=4,
        timer_threshold=150,
        timer_note="Threshold declared before results, not fitted per task; switching waits for the common gripper gate and two votes",
        frozen_note="Only checker current image replaced by subgoal-start image; policy images/states unchanged",
        hold_first_note="Original plan retained but all transition confirmations and the 300-step stage budget disabled; official success or 520-step termination only",
        shared_worker="Same base+adapter resident for all arms; generation checker disables adapter; unsupported V2 operations fall back to base",
        adapter=str(adapter.relative_to(ROOT)),
        threshold=selection["thresholds"]["adapter"],
        offline_pass=selection["offline_pass"],
        adapter_sha256={p.name: sha(p) for p in adapter.iterdir() if p.is_file()},
        model_manifest_sha256=sha(V / "checkpoints/pi05_libero/download_verified.json"),
        sources={str(p.relative_to(ROOT)): sha(p) for p in sources},
        evaluation="Official LIBERO long-task check; goal predicates logged for offline diagnosis only; task5 first pickup unscorable by final task predicates",
        no_new_training=True,
        no_hardware=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--group", default="primary")
    args = parser.parse_args()
    p = protocol()
    protocol_path = S / "PROTOCOL.json"
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text()) == p, (
            "Protocol changed; use a new study directory"
        )
    else:
        atomic_json(protocol_path, p)
    cases = [[0, 0], [9, 0]] if args.pilot else p["cases"]
    # Worker must start before JAX imports and allocation.
    worker = runner.QwenWorker(
        S / f"logs/worker_{args.group}_{time.time_ns()}.log",
        checker_adapter=ROOT / p["adapter"],
        checker_threshold=p["threshold"],
    )
    atomic_json(
        S / "STATUS.json",
        dict(state="loading_policy", group=args.group, pilot=args.pilot),
    )
    try:
        import numpy as np
        from openpi.training import config
        from openpi.policies import policy_config

        started = time.monotonic()
        policy = policy_config.create_trained_policy(
            config.get_config("pi05_libero"),
            V / "checkpoints/pi05_libero",
            sample_kwargs={"num_steps": 10},
        )
        # Warm compilation only; no environment and no action execution.
        policy.infer(
            {
                "observation/image": np.zeros((224, 224, 3), np.uint8),
                "observation/wrist_image": np.zeros((224, 224, 3), np.uint8),
                "observation/state": np.zeros(8, np.float32),
                "prompt": "put the object in the basket",
            },
            noise=np.zeros((10, 32), np.float32),
        )
        atomic_json(
            S / "LOAD_STATUS.json",
            dict(seconds=time.monotonic() - started, worker_ready=worker.ready),
        )
        completed = []
        for task, initial in cases:
            prior = p["plans"][f"{task:02d}_{initial:02d}"]
            # Rotate order deterministically, not all base runs followed by all interventions.
            offset = (task * 3 + initial) % len(MODES)
            order = list(MODES[offset:] + MODES[:offset])
            for mode in order:
                out = (
                    S
                    / f"episodes/{args.group}/{mode}/libero_10/task{task:02d}_init{initial:02d}"
                )
                if (out / "result.json").exists():
                    result = json.loads((out / "result.json").read_text())
                    if result.get("protocol_sha256") != sha(protocol_path):
                        raise RuntimeError(f"Incomplete/different episode {out}")
                else:
                    a = argparse.Namespace(
                        suite="libero_10",
                        task_id=task,
                        init=initial,
                        instruction=prior["instruction"],
                        tag="monitor_causality",
                        study_mode=mode,
                        study_group=args.group,
                        planner="qwen",
                        monitor="qwen",
                        max_steps=520,
                        subgoal_budget=300,
                        monitor_every=4,
                        monitor_res=512,
                        gripper_gate=True,
                        confirm=2,
                        max_replans=0,
                        recovery_mode="continue",
                        operation_checks=False,
                        subgoal_predicates=None,
                        replan_steps=5,
                        subgoals=None,
                        execution_profile="legacy",
                        planner_profile="base",
                        checker_adapter=p["adapter"],
                    )
                    monitor = Monitor(
                        worker, prior["plan"], mode, out / "monitor_calls.jsonl"
                    )
                    t0 = time.monotonic()
                    result = runner.run(a, policy, monitor)
                    result.update(
                        protocol_sha256=sha(protocol_path),
                        monitor_model_calls=monitor.actual_calls,
                        monitor_opportunities=monitor.opportunities,
                        wall_seconds=time.monotonic() - t0,
                    )
                    atomic_json(out / "result.json", result)
                completed.append(
                    dict(
                        task=task,
                        initial=initial,
                        mode=mode,
                        success=result["success"],
                        status=result["status"],
                        steps=result.get("steps"),
                    )
                )
                atomic_json(
                    S / "STATUS.json",
                    dict(
                        state="running",
                        group=args.group,
                        pilot=args.pilot,
                        completed=len(completed),
                        expected=len(cases) * len(MODES),
                        latest=completed[-1],
                    ),
                )
                print(json.dumps(completed[-1]), flush=True)
                if result["status"] in ["error", "interrupted"]:
                    raise RuntimeError(result.get("error"))
        atomic_json(
            S / "STATUS.json",
            dict(
                state="complete",
                group=args.group,
                pilot=args.pilot,
                completed=len(completed),
                rows=completed,
            ),
        )
    finally:
        worker.close()


if __name__ == "__main__":
    main()
