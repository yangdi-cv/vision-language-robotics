"""Read-only paired-loop metrics. Simulator predicates are evaluation labels only."""

from pathlib import Path
import argparse
import csv
import json
import math
import statistics
import numpy as np

R = Path(__file__).resolve().parents[1]
V = R.parent
# Nonfinal subgoal mapping, fixed from BDDL order and the recorded v3 plans.
# Book grasp is not a final-task predicate and is intentionally unscorable here.
ATOM = {0: [0], 1: [0], 2: [0], 3: [1], 4: [0], 6: [0], 7: [0], 8: [0, 1], 9: [0]}


def truth(task, atoms):
    if task not in ATOM:
        return None
    values = np.asarray(atoms, dtype=bool)[ATOM[task]]
    return bool(values.any()) if task == 8 else bool(values.all())


def metrics(path):
    r = json.loads(path.read_text())
    task = r["task_id"]
    tr = np.load(path.with_name("trace.npz"))
    atoms = tr["goal_atoms"]
    row = dict(
        task=task,
        init=r["initial_state_id"],
        success=r["success"],
        status=r["status"],
        steps=r["steps"],
        checks=r["qwen_checks"],
        confirmed=0,
        false_switches=0,
        correct_switches=0,
        budget_advances=0,
        unscored_switches=0,
        incorrect_budget_advances=0,
        missed_confirmation=0,
        unknown_checks=0,
        v2_checks=0,
        fallback_checks=0,
        true_before_budget=0,
        source=str(path.relative_to(V)),
    )
    delays = []
    inference = []
    transitions = []
    for e in r["events"]:
        if e["kind"] == "check":
            inference.append(e["seconds"])
            row["unknown_checks"] += e.get("status") == "unknown"
            row["v2_checks"] += e.get("checker") == "v2"
            row["fallback_checks"] += e.get("checker") == "base_unsupported"
        if e["kind"] != "subgoal_end":
            continue
        assert e["index"] == 0, (
            "Frozen two-subgoal plans only; extend mapping before using new plans"
        )
        actual = truth(task, e["long_task_atoms"])
        start = max(
            x["step"]
            for x in r["events"]
            if x["kind"] == "subgoal_start"
            and x["index"] == 0
            and x["step"] <= e["step"]
        )
        t = dict(step=e["step"], reason=e["reason"], correct=actual, delay_steps=None)
        if e["reason"] == "budget":
            row["budget_advances"] += 1
            row["incorrect_budget_advances"] += actual is False
            row["true_before_budget"] += actual is True
            row["missed_confirmation"] += actual is True
        else:
            row["confirmed"] += 1
            row["unscored_switches"] += actual is None
            row["false_switches"] += actual is False
            row["correct_switches"] += actual is True
            if actual:
                # Trace index k is state after action k, i.e. state at event step k+1.
                onset = e["step"]
                for state_step in range(e["step"], max(start, 1) - 1, -1):
                    if not truth(task, atoms[state_step - 1]):
                        break
                    onset = state_step
                delay = e["step"] - onset
                delays.append(delay)
                t["delay_steps"] = delay
        transitions.append(t)
    # Include reached-but-never-confirmed cases, rather than presenting only fast successes.
    if not transitions and task in ATOM:
        row["missed_confirmation"] += any(truth(task, a) for a in atoms)
    row["delay_steps"] = delays
    row["check_seconds"] = inference
    row["transitions"] = transitions
    return row


def aggregate(rows):
    if not rows:
        return {"n": 0}
    delays = [x for r in rows for x in r["delay_steps"]]
    checks = [x for r in rows for x in r["check_seconds"]]
    keys = [
        "success",
        "steps",
        "checks",
        "confirmed",
        "false_switches",
        "correct_switches",
        "budget_advances",
        "unscored_switches",
        "incorrect_budget_advances",
        "missed_confirmation",
        "unknown_checks",
        "v2_checks",
        "fallback_checks",
        "true_before_budget",
    ]
    return {
        "n": len(rows),
        **{k: sum(r[k] for r in rows) for k in keys},
        "timeouts": sum(r["status"] == "timeout" for r in rows),
        "success_without_transition": sum(
            r["success"] and not r["transitions"] for r in rows
        ),
        "mean_correct_switch_delay_steps": statistics.mean(delays) if delays else None,
        "median_correct_switch_delay_steps": statistics.median(delays)
        if delays
        else None,
        "delay_sample_count": len(delays),
        "mean_check_wall_seconds": statistics.mean(checks) if checks else None,
        "median_check_wall_seconds": statistics.median(checks) if checks else None,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prefix", default="checker_v2_exploratory")
    args = p.parse_args()
    modes = {
        m: [
            metrics(x)
            for x in sorted(
                (V / "hierarchical" / f"{args.prefix}_{m}" / "libero_10").glob(
                    "*/result.json"
                )
            )
        ]
        for m in ["base", "v2"]
    }
    summary = {m: aggregate(rows) for m, rows in modes.items()}
    maps = {m: {(r["task"], r["init"]): r for r in rows} for m, rows in modes.items()}
    paired = sorted(maps["base"].keys() & maps["v2"].keys())
    base_only = sum(
        maps["base"][k]["success"] and not maps["v2"][k]["success"] for k in paired
    )
    v2_only = sum(
        maps["v2"][k]["success"] and not maps["base"][k]["success"] for k in paired
    )
    n = base_only + v2_only
    exact = (
        min(
            1.0,
            2 * sum(math.comb(n, k) for k in range(min(base_only, v2_only) + 1)) / 2**n,
        )
        if n
        else 1.0
    )
    prefix_checks = []
    for key in paired:
        a, b = maps["base"][key], maps["v2"][key]
        ra = json.loads((V / a["source"]).read_text())
        rb = json.loads((V / b["source"]).read_text())
        first_switch = min(
            [
                e["step"]
                for r in [ra, rb]
                for e in r["events"]
                if e["kind"] == "subgoal_end"
            ]
            + [a["steps"], b["steps"]]
        )
        ta = np.load((V / a["source"]).with_name("trace.npz"))
        tb = np.load((V / b["source"]).with_name("trace.npz"))
        prefix_checks.append(
            {
                "task": key[0],
                "init": key[1],
                "steps": first_switch,
                "same_plan": ra["subgoals"] == rb["subgoals"],
                "same_seed": ra["noise_seed"] == rb["noise_seed"],
                "prefix_max_abs_action_error": float(
                    np.max(
                        np.abs(
                            ta["applied"][:first_switch] - tb["applied"][:first_switch]
                        )
                    )
                )
                if first_switch
                else 0.0,
                "full_action_equal": a["steps"] == b["steps"]
                and np.array_equal(ta["applied"], tb["applied"]),
            }
        )
    summary["pair_integrity"] = {
        "pairs": len(prefix_checks),
        "same_plan_and_seed": all(
            x["same_plan"] and x["same_seed"] for x in prefix_checks
        ),
        "max_prefix_action_error": max(
            [x["prefix_max_abs_action_error"] for x in prefix_checks], default=0.0
        ),
        "full_identical_action_pairs": sum(
            x["full_action_equal"] for x in prefix_checks
        ),
    }
    (R / "results/exploratory_pair_integrity.json").write_text(
        json.dumps(prefix_checks, indent=2)
    )
    summary["paired"] = {
        "n": len(paired),
        "base_only_success": base_only,
        "v2_only_success": v2_only,
        "mcnemar_exact_two_sided_p": exact,
    }
    delay_pairs = [
        (maps["base"][k]["delay_steps"][0], maps["v2"][k]["delay_steps"][0])
        for k in paired
        if maps["base"][k]["delay_steps"] and maps["v2"][k]["delay_steps"]
    ]
    summary["paired_correct_delays"] = {
        "n": len(delay_pairs),
        "base_mean_steps": statistics.mean(x[0] for x in delay_pairs)
        if delay_pairs
        else None,
        "v2_mean_steps": statistics.mean(x[1] for x in delay_pairs)
        if delay_pairs
        else None,
        "mean_v2_minus_base_steps": statistics.mean(x[1] - x[0] for x in delay_pairs)
        if delay_pairs
        else None,
        "v2_faster": sum(x[1] < x[0] for x in delay_pairs),
        "v2_slower": sum(x[1] > x[0] for x in delay_pairs),
    }
    summary["definitions"] = {
        "false_switch": "Qwen done-confirmed nonfinal transition while corresponding official task predicate is false; excludes budget",
        "delay": "Steps since latest continuously true predicate segment, only correctly confirmed nonfinal goals; 20 steps/s simulation, not wall time",
        "missed": "Official predicate reached but not confirmed, including timeout/budget; not equivalent to task failure",
        "unscored": "Book pick-up has no corresponding official final goal predicate; excluded from false-switch denominator",
        "scope": "Task8 either moka pot on stove; not necessarily internally numbered first pot; supported-only results additionally reported",
        "physical_release": "Official relation predicate does not independently prove release/support; metrics are benchmark-relation correctness",
    }
    summary["transport_scope"] = {
        m: aggregate([r for r in rows if r["task"] in [0, 1, 3, 4, 6, 7, 9]])
        for m, rows in modes.items()
    }
    (R / "results/exploratory_metrics.json").write_text(json.dumps(summary, indent=2))
    (R / "results/exploratory_episode_metrics.json").write_text(
        json.dumps(modes, indent=2)
    )
    with (R / "results/exploratory_metrics.csv").open("w") as f:
        fields = [
            "mode",
            "task",
            "init",
            "success",
            "status",
            "steps",
            "false_switches",
            "budget_advances",
            "missed_confirmation",
            "correct_switches",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for m, rows in modes.items():
            for r in rows:
                writer.writerow({"mode": m, **{k: r[k] for k in fields if k != "mode"}})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
