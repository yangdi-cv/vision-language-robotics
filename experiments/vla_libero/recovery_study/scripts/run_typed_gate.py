# ruff: noqa: E402
"""Two paired follow-up trials isolating the fixture-operation gripper gate."""

from pathlib import Path
import hashlib
import json
import sys

R = Path(__file__).resolve().parents[1]
V = R.parent
sys.path[:0] = [str(R / "scripts"), str(V / "scripts")]
from run_paired import one, SEEDS
from run_hierarchical_recovery import QwenWorker


def main():
    case = next(
        c
        for c in json.loads((R / "protocol.json").read_text())["cases"]
        if c["task"] == 2
    )
    p = R / "typed_gate_protocol.json"
    if p.exists():
        raise FileExistsError(p)
    p.write_text(
        json.dumps(
            {
                "task": 2,
                "seeds": SEEDS,
                "prefix_steps": 150,
                "remaining_steps": 370,
                "reason": "Contract trials had zero completion queries despite stove-on becoming true while the fingers stayed closed. Change only the release gate for supported state-change imperatives; transport retains its gate. The predicates were diagnostic and never used to switch at runtime.",
                "sources": {
                    str(p.relative_to(V)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in [
                        V / "scripts/subgoal_state.py",
                        V / "scripts/run_hierarchical_recovery.py",
                        R / "scripts/run_paired.py",
                    ]
                },
            },
            indent=2,
        )
    )
    qwen = QwenWorker(R / "logs/qwen_typed_worker.log")
    try:
        from openpi.training import config
        from openpi.policies import policy_config

        policy = policy_config.create_trained_policy(
            config.get_config("pi05_libero"),
            V / "checkpoints/pi05_libero",
            sample_kwargs={"num_steps": 10},
        )
        h = json.loads(
            (R / "results/task02_init02/full_seed31001/result.json").read_text()
        )["prefix_state_sha256"]
        for seed in SEEDS:
            one(case, "typed", seed, policy, qwen, {2: h})
    finally:
        qwen.close()


if __name__ == "__main__":
    main()
