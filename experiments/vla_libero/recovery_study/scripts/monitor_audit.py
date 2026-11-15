"""Restricted alignment: only stove-on and white-mug first goals are hand verified."""

import json
from pathlib import Path

R = Path(__file__).resolve().parents[1]
V = R.parent
summary = {}
records = []
for tag in ["zh_eval_hier_v2", "zh_eval_hier_v3"]:
    cells = {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "confirmed_false": 0, "n": 0}
    for p in sorted((V / "hierarchical" / tag / "libero_10").glob("*/result.json")):
        r = json.loads(p.read_text())
        if r["task_id"] not in [2, 4]:
            continue
        for e in r["events"]:
            if e["kind"] != "check" or e["index"] != 0:
                continue
            expected = bool(e["shadow_long_task_atoms"][0])
            got = bool(e["done"])
            key = ("tp" if expected else "fp") if got else ("fn" if expected else "tn")
            cells[key] += 1
            cells["n"] += 1
            cells["confirmed_false"] += int(e.get("confirmed", False) and not expected)
            records.append(
                {
                    "tag": tag,
                    "task": r["task_id"],
                    "init": r["initial_state_id"],
                    "step": e["step"],
                    "expected": expected,
                    "observed": got,
                    "confirmed": e.get("confirmed", False),
                    "reason": e.get("reason"),
                }
            )
    summary[tag] = cells
(R / "monitor_diagnostic.json").write_text(
    json.dumps(
        {
            "scope": "Only tasks2/4 first subgoal; corresponding final atom0 manually checked against BDDL and plan text. Queries are correlated, not independent samples.",
            "counts": summary,
            "records": records,
        },
        indent=2,
    )
)
print(json.dumps(summary, indent=2))
