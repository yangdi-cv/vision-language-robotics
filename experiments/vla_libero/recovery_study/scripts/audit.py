"""Index existing experiments without overwriting any source results."""

from pathlib import Path
import collections
import csv
import hashlib
import json
import statistics

R = Path(__file__).resolve().parents[1]
V = R.parent
TAGS = [
    "zh_eval_flat",
    "zh_eval_hier",
    "zh_eval_hier_v2",
    "zh_eval_hier_v3",
    "zh_eval_translate",
    "replan_probe2",
    "replan_probe3",
]
rows = []
summaries = {}
for tag in TAGS:
    group = []
    for path in sorted((V / "hierarchical" / tag).glob("*/task*_init*/result.json")):
        r = json.loads(path.read_text())
        events = r.get("events", [])
        ends = [e for e in events if e["kind"] == "subgoal_end"]
        checks = [e for e in events if e["kind"] == "check"]
        row = dict(
            tag=tag,
            task=r["task_id"],
            init=r["initial_state_id"],
            success=bool(r["success"]),
            status=r["status"],
            steps=r.get("steps"),
            checks=len(checks),
            confirmed=sum(bool(e.get("confirmed")) for e in checks),
            budget_switches=sum(e.get("reason") == "budget" for e in ends),
            replans=sum(e["kind"] == "replan" for e in events),
            full_fallback=sum(e["kind"] == "fallback_full_task" for e in events),
            qwen_seconds=sum(e.get("seconds", 0) for e in events),
            runner_sha256=r.get("runner_sha256"),
            source=str(path.relative_to(V)),
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        rows.append(row)
        group.append(row)
    summaries[tag] = dict(
        n=len(group),
        success=sum(x["success"] for x in group),
        errors=sum(x["status"] == "error" for x in group),
        budget_switches=sum(x["budget_switches"] for x in group),
        episodes_budget_switch=sum(bool(x["budget_switches"]) for x in group),
        checks=sum(x["checks"] for x in group),
        replans=sum(x["replans"] for x in group),
        full_fallback=sum(x["full_fallback"] for x in group),
        median_steps=statistics.median(
            x["steps"] for x in group if x["steps"] is not None
        ),
    )
with (R / "episode_index.csv").open("w") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
# Exact paired outcome counts; no significance claim from small, selected cohorts.
pairs = {}
for left, right in [
    ("zh_eval_flat", "zh_eval_translate"),
    ("zh_eval_translate", "zh_eval_hier_v3"),
]:
    a = {(r["task"], r["init"]): r for r in rows if r["tag"] == left}
    b = {(r["task"], r["init"]): r for r in rows if r["tag"] == right}
    c = collections.Counter(
        (a[k]["success"], b[k]["success"]) for k in a.keys() & b.keys()
    )
    pairs[left + "__" + right] = {f"{int(k[0])}->{int(k[1])}": v for k, v in c.items()}
(R / "audit.json").write_text(
    json.dumps(
        {"tags": summaries, "paired_outcomes": pairs, "n_indexed": len(rows)}, indent=2
    )
)
print(json.dumps(summaries, indent=2))
