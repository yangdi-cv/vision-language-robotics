# ruff: noqa: E402
"""Adaptive follow-up: six contract-validated repairs plus two normal-path smoke cases."""

from pathlib import Path
import argparse
import hashlib
import json
import sys

R = Path(__file__).resolve().parents[1]
V = R.parent
sys.path[:0] = [str(V / "scripts"), str(R / "scripts")]
from run_paired import one, SEEDS
from run_hierarchical_recovery import QwenWorker, run


def main():
    original = json.loads((R / "protocol.json").read_text())
    p = {
        "reason": "Original pilot exposed missing unconfirmed goals, standalone pickup steps incompatible with release gate, and schema errors. Test a constrained repair interface without changing model or replay states.",
        "cases": [2, 4, 8],
        "seeds": SEEDS,
        "n_recovery": 6,
        "n_normal_smoke": 2,
        "no_full_task_fallback": True,
        "sources": {
            str(p.relative_to(V)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                V / "scripts/qwen_planner_recovery.py",
                V / "scripts/run_hierarchical_recovery.py",
                V / "scripts/subgoal_state.py",
                R / "scripts/run_paired.py",
            ]
        },
    }
    target = R / "contract_protocol.json"
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(p, indent=2))
    qwen = QwenWorker(R / "logs/qwen_contract_worker.log")
    try:
        from openpi.training import config
        from openpi.policies import policy_config

        policy = policy_config.create_trained_policy(
            config.get_config("pi05_libero"),
            V / "checkpoints/pi05_libero",
            sample_kwargs={"num_steps": 10},
        )
        hashes = {
            c["task"]: json.loads(
                (
                    R
                    / "results"
                    / f"task{c['task']:02d}_init02/full_seed31001/result.json"
                ).read_text()
            )["prefix_state_sha256"]
            for c in original["cases"]
        }
        for case in original["cases"]:
            for seed in SEEDS:
                one(case, "contract", seed, policy, qwen, hashes)
        for tid in [0, 9]:
            a = argparse.Namespace(
                suite="libero_10",
                task_id=tid,
                init=0,
                instruction=None,
                tag="recovery_contract_smoke",
                planner="qwen",
                monitor="qwen",
                max_steps=520,
                subgoal_budget=150,
                monitor_every=4,
                monitor_res=512,
                gripper_gate=True,
                confirm=2,
                max_replans=1,
                recovery_mode="replan",
                subgoal_predicates=None,
                replan_steps=5,
                subgoals=None,
            )
            result = run(a, policy, qwen)
            if result["status"] == "error":
                raise RuntimeError(result.get("error"))
    finally:
        qwen.close()


if __name__ == "__main__":
    main()
