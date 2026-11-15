"""Offline comparison of completion-check forms on a finished case's saved frames.

Shadow labels come from the recorded official predicate; Qwen never sees them.
Usage (qwen-scene venv): probe_checks.py <case_dir>
"""

from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen_worker import MissionPlanner  # noqa: E402
from skills import SKILLS  # noqa: E402

case = Path(sys.argv[1])
r = json.loads((case / "qwen_result.json").read_text())
skill = SKILLS[r["mission"]["skill"]]
ref = str(case / "qwen_inputs/manip_start.png")
planner = MissionPlanner()
rows, correct = [], 0
for e in r["checks"]:
    img = str(case / f"qwen_inputs/check_{e['policy_step']:04d}.png")
    q = planner.question_check([ref, img], skill["completion_questions"])
    pred, truth = q.get("done"), e["shadow_official_success"]
    correct += pred == truth
    rows.append((e["policy_step"], truth, pred, q.get("answers"), q.get("error")))
print(f"question_check: {correct}/{len(rows)}")
for row in rows:
    print(*row)
