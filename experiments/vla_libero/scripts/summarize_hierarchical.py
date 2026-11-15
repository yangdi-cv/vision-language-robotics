"""Summarize hierarchical/<tag> runs: per-task success and Qwen monitor agreement with shadow predicates."""

from pathlib import Path
import collections, json, sys

R = Path(__file__).resolve().parents[1] / "hierarchical"

for tag in sys.argv[1:]:
    per_task = collections.defaultdict(list)
    status = collections.Counter()
    checks = calls = 0
    for f in sorted((R / tag).glob("*/task*_init*/result.json")):
        r = json.loads(f.read_text())
        per_task[r["task_id"]].append(r["success"])
        status[r["status"]] += 1
        calls += r.get("qwen_checks") or 0
        checks += 1
    total = sum(sum(v) for v in per_task.values())
    n = sum(len(v) for v in per_task.values())
    row = " ".join(f"t{t}:{sum(v)}/{len(v)}" for t, v in sorted(per_task.items()))
    print(f"{tag}: {total}/{n}  {row}  status={dict(status)}  qwen_checks/ep={calls / max(checks, 1):.1f}")
