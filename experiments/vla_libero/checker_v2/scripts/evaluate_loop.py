# ruff: noqa: E402
"""Frozen paired simulation; exploratory mode retains the failed offline verdict."""

from pathlib import Path
import argparse
import json
import sys
import hashlib

R = Path(__file__).resolve().parents[1]
V = R.parent
sys.path.insert(0, str(V / "scripts"))
import run_hierarchical as runner


class Frozen:
    def __init__(self, worker, plan, mode):
        self.worker, self.plan, self.mode = worker, plan, mode

    def request(self, payload, timeout=120):
        if payload.get("op") == "plan":
            return {
                "ok": True,
                "subgoals": [{"instruction": x} for x in self.plan],
                "raw": json.dumps(self.plan),
            }
        if payload.get("op") == "check":
            payload = {**payload, "checker_mode": self.mode}
        return self.worker.request(payload, timeout)


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument(
        "--exploratory",
        action="store_true",
        help="Simulation-only paired evaluation despite offline failure; never promotes weights",
    )
    options = cli.parse_args()
    exploratory = options.exploratory
    selection = (
        json.loads((R / "accepted_model.json").read_text())
        if (R / "accepted_model.json").exists() and not exploratory
        else {"model_dir": "results/model", "summary": "summary.json"}
    )
    modeldir = R / selection["model_dir"]
    summary = json.loads((modeldir / selection["summary"]).read_text())
    if not summary["offline_pass"] and not exploratory:
        (R / "results/closed_loop_status.json").write_text(
            json.dumps(
                {"status": "not_started", "reason": "frozen offline acceptance failed"},
                indent=2,
            )
        )
        return
    adapter = modeldir / "best_adapter"
    threshold = summary["thresholds"]["adapter"]
    prefix = "exploratory" if exploratory else "closed_loop"
    p = R / (prefix.upper() + "_PROTOCOL.json")
    protocol = {
        "exploratory": exploratory,
        "offline_pass": summary["offline_pass"],
        "no_deployment_approval": True,
        "comparison": "existing base generation checker versus V2 prompt+LoRA+threshold package, not isolated LoRA ablation",
        "cases": [[t, i] for t in range(10) for i in range(3)],
        "modes": ["base", "v2"],
        "frozen_plans": "zh_eval_hier_v3 recorded English subgoals",
        "instruction": "same recorded Chinese task instruction",
        "transition": "legacy v3, budget300, confirm2, gate on, monitor every4 policy calls, 512px",
        "max_steps": 520,
        "threshold": threshold,
        "adapter_sha256": hashlib.sha256(
            (adapter / "adapter_model.safetensors").read_bytes()
        ).hexdigest(),
        "only_checker_changes": True,
    }
    if p.exists():
        assert json.loads(p.read_text()) == protocol
    else:
        p.write_text(json.dumps(protocol, indent=2))
    q = runner.QwenWorker(
        R / f"logs/{prefix}_worker.log",
        checker_adapter=adapter,
        checker_threshold=threshold,
    )
    try:
        from openpi.training import config
        from openpi.policies import policy_config

        policy = policy_config.create_trained_policy(
            config.get_config("pi05_libero"),
            V / "checkpoints/pi05_libero",
            sample_kwargs={"num_steps": 10},
        )
        rows = []
        for tid, init in protocol["cases"]:
            prior = json.loads(
                (
                    V
                    / f"hierarchical/zh_eval_hier_v3/libero_10/task{tid:02d}_init{init:02d}/result.json"
                ).read_text()
            )
            modes = ["base", "v2"] if (tid + init) % 2 == 0 else ["v2", "base"]
            for mode in modes:
                tag = (
                    "checker_v2_exploratory_" if exploratory else "checker_v2_paired_"
                ) + mode
                target = (
                    V
                    / f"hierarchical/{tag}/libero_10/task{tid:02d}_init{init:02d}/result.json"
                )
                if target.exists():
                    r = json.loads(target.read_text())
                else:
                    args = argparse.Namespace(
                        suite="libero_10",
                        task_id=tid,
                        init=init,
                        instruction=prior["input_instruction"],
                        tag=tag,
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
                        checker_adapter=str(adapter),
                    )
                    r = runner.run(args, policy, Frozen(q, prior["subgoals"], mode))
                rows.append(
                    {
                        "task": tid,
                        "init": init,
                        "mode": mode,
                        "success": r["success"],
                        "status": r["status"],
                        "steps": r.get("steps"),
                        "source": str(target.relative_to(V)),
                    }
                )
                (R / f"results/{prefix}_rows.json").write_text(
                    json.dumps(rows, indent=2)
                )
                print(json.dumps(rows[-1]), flush=True)
                if r["status"] == "error":
                    raise RuntimeError(r.get("error"))
        result = {
            mode: {
                "n": sum(x["mode"] == mode for x in rows),
                "success": sum(x["success"] for x in rows if x["mode"] == mode),
            }
            for mode in ["base", "v2"]
        }
        (R / f"results/{prefix}_summary.json").write_text(json.dumps(result, indent=2))
    finally:
        q.close()


if __name__ == "__main__":
    main()
