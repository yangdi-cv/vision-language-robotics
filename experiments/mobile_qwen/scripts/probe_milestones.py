"""Replay the online milestone monitor on a finished case's saved check frames.

Official predicate labels are used only for scoring.
Usage (qwen-scene venv): probe_milestones.py <out_json> <case_dir> [<case_dir> ...]
"""

from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from milestones import MilestoneMonitor  # noqa: E402
from qwen_worker import MissionPlanner  # noqa: E402
from skills import SKILLS  # noqa: E402


def main():
    out = Path(sys.argv[1])
    planner = MissionPlanner()
    results = []
    for case in map(Path, sys.argv[2:]):
        r = json.loads((case / "qwen_result.json").read_text())
        monitor = MilestoneMonitor(SKILLS[r["mission"]["skill"]])
        events, stop = [], None
        for e in r["checks"]:
            q = monitor.request(str(case / f"qwen_inputs/check_{e['policy_step']:04d}.png"))
            res = planner.milestone_check(q["image"], q["task"], q["memory"], q["milestone"])
            ev = monitor.update(e["policy_step"], res)
            ev["shadow_official_success"] = e["shadow_official_success"]
            events.append(ev)
            if monitor.done:
                stop = ev
                break
        first = next((e["policy_step"] for e in r["checks"] if e["shadow_official_success"]), None)
        bowl = next((x["policy_step"] for x in events if x.get("confirmed") == "bowl_in_drawer"), None)
        summary = dict(case=case.name, stop_step=stop and stop["policy_step"],
                       official_at_stop=stop and stop["shadow_official_success"],
                       first_official_check_step=first, bowl_confirmed_step=bowl,
                       bowl_confirmed_before_official=bowl is not None and first is not None and bowl < first)
        print(json.dumps(summary), flush=True)
        results.append(dict(summary, events=events))
        out.write_text(json.dumps(results, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
