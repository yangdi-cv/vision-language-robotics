"""Paired task outcomes, decision errors, delays, and causal input-prefix checks."""

import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
from scipy.stats import binomtest

S = Path(__file__).resolve().parents[1]
V = S.parent
sys.path.insert(0, str(V / "scripts"))
from episode_journal import atomic_json
from interventions import MODES

# Final BDDL goal atom that corresponds to the first frozen subgoal. Task5 pickup is not one.
ATOM = {0: [0], 1: [0], 2: [0], 3: [1], 4: [0], 6: [0], 7: [0], 8: [0, 1], 9: [0]}


def truth(task, atoms):
    if task not in ATOM:
        return None
    x = np.asarray(atoms, dtype=bool)[ATOM[task]]
    return bool(x.any()) if task == 8 else bool(x.all())


def episode(path):
    r = json.loads((path / "result.json").read_text())
    if (
        r.get("status")
        not in ["success", "timeout", "plan_exhausted", "error", "interrupted"]
        or "protocol_sha256" not in r
    ):
        return None
    with np.load(path / "trace.npz") as f:
        atoms = f["goal_atoms"]
    switches = [e for e in r["events"] if e["kind"] == "subgoal_end"]
    checks = [e for e in r["events"] if e["kind"] == "check"]
    result = dict(
        task=r["task_id"],
        initial=r["initial_state_id"],
        mode=r["study_mode"],
        success=r["success"],
        status=r["status"],
        steps=r["steps"],
        actual_model_calls=r["monitor_model_calls"],
        opportunities=r["monitor_opportunities"],
        switches=len(switches),
        visual_switches=0,
        false_switches=0,
        true_switches=0,
        budget_switches=0,
        false_budget_switches=0,
        unscorable_switches=0,
        query_tp=0,
        query_fp=0,
        query_fn=0,
        query_tn=0,
        query_invalid=0,
        delay_steps=[],
        check_seconds=[],
        first_switch=switches[0]["step"] if switches else None,
        success_without_switch=bool(r["success"] and not switches),
        source=str(path.relative_to(V)),
    )
    mode = r["study_mode"]
    calls = (
        [json.loads(x) for x in (path / "monitor_calls.jsonl").read_text().splitlines()]
        if (path / "monitor_calls.jsonl").exists()
        else []
    )
    for call in calls:
        if call["actual_model_call"]:
            result["check_seconds"].append(call["seconds"])
        actual = truth(r["task_id"], atoms[call["step"] - 1])
        if actual is None or not call["actual_model_call"]:
            continue
        prediction = call["reply"].get("done")
        if not call["reply"].get("ok") or not isinstance(prediction, bool):
            result["query_invalid"] += 1
            continue
        result[
            "query_"
            + ("t" if prediction == actual else "f")
            + ("p" if prediction else "n")
        ] += 1
    for e in switches:
        actual = truth(r["task_id"], e["long_task_atoms"])
        if e["reason"] == "budget":
            result["budget_switches"] += 1
            result["false_budget_switches"] += actual is False
        else:
            result["visual_switches"] += mode in [
                "live_base",
                "frozen_base",
                "live_lora",
            ]
            result["false_switches"] += actual is False
            result["true_switches"] += actual is True
            result["unscorable_switches"] += actual is None
            if actual:
                onset = e["step"]
                for state in range(e["step"], 0, -1):
                    if not truth(r["task_id"], atoms[state - 1]):
                        break
                    onset = state
                result["delay_steps"].append(e["step"] - onset)
    result["reached_without_confirmed_switch"] = (
        bool(any(truth(r["task_id"], a) for a in atoms) and not result["true_switches"])
        if r["task_id"] in ATOM
        else None
    )
    return result


def paired(rows, left, right):
    a = {(r["task"], r["initial"]): r for r in rows if r["mode"] == left}
    b = {(r["task"], r["initial"]): r for r in rows if r["mode"] == right}
    keys = sorted(a.keys() & b.keys())
    gains = sum(not a[k]["success"] and b[k]["success"] for k in keys)
    losses = sum(a[k]["success"] and not b[k]["success"] for k in keys)
    jointly = [k for k in keys if a[k]["success"] and b[k]["success"]]
    return dict(
        left=left,
        right=right,
        n=len(keys),
        gains=gains,
        losses=losses,
        nominal_mcnemar_p=binomtest(gains, gains + losses, 0.5).pvalue
        if gains + losses
        else 1.0,
        jointly_successful=len(jointly),
        mean_step_delta_on_joint_success=float(
            np.mean([b[k]["steps"] - a[k]["steps"] for k in jointly])
        )
        if jointly
        else None,
        note="Descriptive paired analysis; initial states clustered within tasks; multiple comparisons not confirmatory significance",
    )


def prefix_checks(root, rows):
    results = []
    for task, initial in sorted({(r["task"], r["initial"]) for r in rows}):
        group = [r for r in rows if r["task"] == task and r["initial"] == initial]
        if len(group) < 2:
            continue
        bound = min(
            [r["first_switch"] for r in group if r["first_switch"] is not None]
            + [min(r["steps"] for r in group)]
        )
        anchor = root / group[0]["mode"] / f"libero_10/task{task:02d}_init{initial:02d}"
        ref = json.loads((anchor / "result.json").read_text())
        with np.load(anchor / "trace.npz") as f:
            base = f["applied"][:bound]
        basequeries = [
            json.loads(x)
            for x in (anchor / "policy_queries.jsonl").read_text().splitlines()
            if json.loads(x)["step"] < bound
        ]
        for row in group[1:]:
            path = root / row["mode"] / f"libero_10/task{task:02d}_init{initial:02d}"
            current = json.loads((path / "result.json").read_text())
            with np.load(path / "trace.npz") as f:
                difference = (
                    float(np.max(np.abs(base - f["applied"][:bound]))) if bound else 0
                )
            queries = [
                json.loads(x)
                for x in (path / "policy_queries.jsonl").read_text().splitlines()
                if json.loads(x)["step"] < bound
            ]
            inputequal = basequeries == queries
            results.append(
                dict(
                    task=task,
                    initial=initial,
                    left=group[0]["mode"],
                    right=row["mode"],
                    steps_before_any_switch=bound,
                    max_action_difference=difference,
                    policy_inputs_and_noise_exact=inputequal,
                    initial_state_exact=ref["initial_state_sha256"]
                    == current["initial_state_sha256"],
                    settled_state_exact=ref["settled_state_sha256"]
                    == current["settled_state_sha256"],
                )
            )
    return results


def full_action_comparisons(root, rows):
    """Identical nominal outcome is distinguished from identical model actions."""
    indexed = {(r["mode"], r["task"], r["initial"]): r for r in rows}
    comparisons = []
    for task, initial in sorted({(r["task"], r["initial"]) for r in rows}):
        anchor = indexed.get(("live_base", task, initial))
        if anchor is None:
            continue
        suffix = f"libero_10/task{task:02d}_init{initial:02d}"
        with np.load(root / "live_base" / suffix / "trace.npz") as f:
            reference = f["applied"]
        for mode in MODES[1:]:
            other = indexed.get((mode, task, initial))
            if other is None:
                continue
            with np.load(root / mode / suffix / "trace.npz") as f:
                current = f["applied"]
            n = min(len(reference), len(current))
            different = np.flatnonzero(np.any(reference[:n] != current[:n], axis=1))
            comparisons.append(dict(
                task=task, initial=initial, right=mode,
                common_steps=n,
                identical_full_actions=bool(np.array_equal(reference, current)),
                first_different_action_step=int(different[0]) if len(different) else None,
                left_steps=len(reference), right_steps=len(current),
                same_success=anchor["success"] == other["success"],
            ))
    return comparisons


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--group", default="primary")
    a = cli.parse_args()
    root = S / "episodes" / a.group
    rows = [
        value
        for p in sorted(root.glob("*/libero_10/task*"))
        if (value := episode(p)) is not None
    ]
    groups = {}
    counters = [
        "success",
        "steps",
        "actual_model_calls",
        "opportunities",
        "switches",
        "false_switches",
        "true_switches",
        "budget_switches",
        "false_budget_switches",
        "unscorable_switches",
        "success_without_switch",
        "query_tp",
        "query_fp",
        "query_fn",
        "query_tn",
        "query_invalid",
    ]
    for mode in MODES:
        selected = [r for r in rows if r["mode"] == mode]
        delay = [x for r in selected for x in r["delay_steps"]]
        seconds = [x for r in selected for x in r["check_seconds"]]
        groups[mode] = dict(
            n=len(selected),
            **{name: sum(r[name] for r in selected) for name in counters},
            reached_without_confirmed_switch=sum(
                r["reached_without_confirmed_switch"] is True for r in selected
            ),
            mean_correct_switch_delay_steps=float(np.mean(delay)) if delay else None,
            mean_model_check_seconds=float(np.mean(seconds)) if seconds else None,
        )
    comparisons = [
        paired(rows, "live_base", mode) for mode in MODES if mode != "live_base"
    ]
    prefixes = prefix_checks(root, rows)
    atomic_json(
        S / "SUMMARY.json",
        dict(
            episodes=len(rows),
            expected=150,
            groups=groups,
            paired=comparisons,
            prefix_checks=prefixes,
            full_action_comparisons=full_action_comparisons(root, rows),
            rows=rows,
        ),
    )
    if rows:
        with (S / "metrics.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(dict(episodes=len(rows), groups=groups), indent=2))


if __name__ == "__main__":
    main()
