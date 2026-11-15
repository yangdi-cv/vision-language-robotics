"""Per-frame accuracy of drawer-closed question forms on saved check frames.

Positives: frames after official success (drawer closed). Negatives: frames at steps <= 220,
before the drawer is pushed shut (drawer open in all saved cases). Labels are for scoring only.
Usage (qwen-scene venv): probe_drawer.py <out_json> <case_dir> [<case_dir> ...]
"""

from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen_worker import MissionPlanner, parse_json  # noqa: E402

GRIPPER = "The white object with two fingers near the cabinet handles is the robot gripper, not a drawer."
FORMS = {
    "neutral": f"""Look at the dark cabinet on the left side of the image. {GRIPPER}
Is any drawer of this cabinet pulled out so that you can see into the drawer (its inner floor or walls)?
Return only JSON: {{"drawer_open": true or false}}""",
    "describe": f"""Look at the dark cabinet on the left side of the image. {GRIPPER}
First describe the front of the cabinet in one sentence, then decide whether the bottom drawer is pulled out.
Return only JSON: {{"description": "...", "drawer_open": true or false}}""",
    "pair": f"""Image 1 is the scene when the task started; the bottom drawer of the dark cabinet on the left was pulled out. Image 2 is the scene now. {GRIPPER}
In image 2, is the bottom drawer still pulled out, as in image 1?
Return only JSON: {{"drawer_open": true or false}}""",
}


def main():
    out = Path(sys.argv[1])
    planner = MissionPlanner()
    rows = []
    for c in map(Path, sys.argv[2:]):
        r = json.loads((c / "qwen_result.json").read_text())
        start = str(c / "qwen_inputs/manip_start.png")
        for e in r["checks"]:
            if e["shadow_official_success"]:
                closed = True
            elif e["policy_step"] <= 220:
                closed = False
            else:
                continue
            img = str(c / f"qwen_inputs/check_{e['policy_step']:04d}.png")
            for name, prompt in FORMS.items():
                images = [start, img] if name == "pair" else [img]
                res = planner.generate(images, prompt, max_new_tokens=80)
                try:
                    pred = not parse_json(res["raw"])["drawer_open"]
                except (ValueError, TypeError, KeyError, AttributeError):
                    pred = None
                rows.append(dict(case=c.name, step=e["policy_step"], form=name, closed=closed,
                                 pred_closed=pred, raw=res["raw"]))
        out.write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    for name in FORMS:
        sub = [x for x in rows if x["form"] == name]
        tp = sum(x["closed"] and x["pred_closed"] is True for x in sub)
        tn = sum(not x["closed"] and x["pred_closed"] is False for x in sub)
        npos = sum(x["closed"] for x in sub)
        print(json.dumps(dict(form=name, closed_correct=f"{tp}/{npos}", open_correct=f"{tn}/{len(sub) - npos}")), flush=True)


if __name__ == "__main__":
    main()
