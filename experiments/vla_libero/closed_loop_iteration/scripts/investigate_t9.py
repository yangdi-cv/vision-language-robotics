# ruff: noqa: E402
"""Four frozen-plan tests isolate instruction naming, transition rules and recovery."""

from pathlib import Path
import argparse
import json
import sys
import time

R = Path(__file__).resolve().parents[1]
V = R.parent
sys.path.insert(0, str(V / "scripts"))
import run_hierarchical as legacy
import run_hierarchical_recovery as strict


class FrozenPlan:
    def __init__(self, qwen, plan):
        self.qwen, self.plan = qwen, plan

    def request(self, payload, timeout=120):
        if payload.get("op") == "plan":
            return {
                "ok": True,
                "subgoals": [{"instruction": s} for s in self.plan],
                "raw": json.dumps(self.plan),
            }
        return self.qwen.request(payload, timeout)


def main():
    old = json.loads(
        (
            V / "hierarchical/zh_eval_hier_v3/libero_10/task09_init00/result.json"
        ).read_text()
    )["subgoals"]
    new = json.loads(
        (
            V
            / "hierarchical/recovery_contract_smoke/libero_10/task09_init00/result.json"
        ).read_text()
    )["subgoals"]
    cases = [
        ("legacy_old", legacy.run, old, 300, 0),
        ("strict_old", strict.run, old, 300, 0),
        ("strict_new", strict.run, new, 300, 0),
        ("strict_new_recovery", strict.run, new, 150, 1),
    ]
    protocol = {
        "started": time.time(),
        "max_hours": 2,
        "cases": [
            {"tag": x[0], "plan": x[2], "budget": x[3], "max_replans": x[4]}
            for x in cases
        ],
        "shared": "same task9/init0, original noise seed, monitor prompt, 512px images, confirm2, gripper gate, 520 steps. Initial plan frozen; no model call changes its wording.",
    }
    (R / "t9_protocol.json").write_text(json.dumps(protocol, indent=2))
    q = legacy.QwenWorker(R / "logs/t9_qwen.log", profile="contract")
    try:
        from openpi.training import config
        from openpi.policies import policy_config

        policy = policy_config.create_trained_policy(
            config.get_config("pi05_libero"),
            V / "checkpoints/pi05_libero",
            sample_kwargs={"num_steps": 10},
        )
        rows = []
        for name, run, plan, budget, replans in cases:
            a = argparse.Namespace(
                suite="libero_10",
                task_id=9,
                init=0,
                instruction="put the yellow and white mug in the microwave and close it",
                tag="t9_isolate_" + name,
                planner="qwen",
                monitor="qwen",
                max_steps=520,
                subgoal_budget=budget,
                monitor_every=4,
                monitor_res=512,
                gripper_gate=True,
                confirm=2,
                max_replans=replans,
                recovery_mode="replan",
                operation_checks=False,
                subgoal_predicates=None,
                replan_steps=5,
                subgoals=None,
                execution_profile="legacy" if run == legacy.run else "strict",
                planner_profile="contract",
            )
            r = run(a, policy, FrozenPlan(q, plan))
            rows.append(
                {
                    "case": name,
                    "success": r["success"],
                    "status": r["status"],
                    "steps": r.get("steps"),
                    "final_atoms": r.get("final_atoms"),
                }
            )
            (R / "t9_results.json").write_text(json.dumps(rows, indent=2))
            print(json.dumps(rows[-1]), flush=True)
            if r["status"] == "error":
                raise RuntimeError(r.get("error"))
            if time.time() - protocol["started"] > 7200:
                break
    finally:
        q.close()


if __name__ == "__main__":
    main()
