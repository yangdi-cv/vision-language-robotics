# ruff: noqa: E402
"""Two exploratory, operator-specified centering instructions. Not autonomous replanning."""

from pathlib import Path
import json
import sys

R = Path(__file__).resolve().parents[1]
V = R.parent
sys.path[:0] = [str(R / "scripts"), str(V / "scripts")]
from run_paired import one, SEEDS


def main():
    case = next(
        c
        for c in json.loads((R / "protocol.json").read_text())["cases"]
        if c["task"] == 4
    )
    p = R / "guided_protocol.json"
    if p.exists():
        raise FileExistsError(p)
    p.write_text(
        json.dumps(
            {
                "task": 4,
                "seeds": SEEDS,
                "prefix_steps": 150,
                "remaining_steps": 370,
                "reason": "Contact audit found 38.5mm offset against 30mm threshold. Test whether an explicit language request to re-center permits the policy to correct it; no numeric location or predicate is sent to the policy. This instruction is chosen by the experimenter, not Qwen.",
                "instruction": "pick up the white mug again and place it at the center of the left plate, then put the yellow and white mug at the center of the right plate",
            },
            indent=2,
        )
    )
    from openpi.training import config
    from openpi.policies import policy_config

    policy = policy_config.create_trained_policy(
        config.get_config("pi05_libero"),
        V / "checkpoints/pi05_libero",
        sample_kwargs={"num_steps": 10},
    )
    h = json.loads(
        (R / "results/task04_init02/full_seed31001/result.json").read_text()
    )["prefix_state_sha256"]
    for seed in SEEDS:
        one(case, "centering", seed, policy, None, {4: h})


if __name__ == "__main__":
    main()
