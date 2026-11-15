"""Offline comparison of completion-check prompts on saved frames with shadow ground truth."""

from pathlib import Path
import json, sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen_planner import Planner

ep = Path(sys.argv[1])
record = json.loads((ep / "result.json").read_text())
checks = [e for e in record["events"] if e["kind"] == "check" and e["index"] == 0]
frames = ep / "qwen_inputs"
ref = str(frames / "plan_step0000.png")
subgoal = record["subgoals"][0]

VARIANTS = {
    "single_where": (False, f"""You are monitoring a tabletop robot from a third-person camera.
Subgoal: "{subgoal}".
First describe where the target object is right now, then decide whether the subgoal is completed.
Return only JSON: {{"where": "short description", "done": true or false}}"""),
    "pair_change": (True, f"""Image 1 was taken when the robot started this subgoal; image 2 is the scene now.
Subgoal: "{subgoal}".
Compare the two images: which object has moved, and where is it now? Decide whether the subgoal is completed in image 2.
Return only JSON: {{"change": "short description", "done": true or false}}"""),
    "pair_dest": (True, f"""Image 1 was taken when the robot started this subgoal; image 2 is the scene now.
Subgoal: "{subgoal}".
Look at the destination named in the subgoal. Is there a new object at the destination in image 2 that was not there in image 1, and is it the target object?
Return only JSON: {{"change": "short description", "done": true or false}}"""),
}


def parse(raw):
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        return bool(json.loads(text)["done"])
    except Exception:
        return None


planner = Planner()
for name, (pair, prompt) in VARIANTS.items():
    correct = 0
    rows = []
    for e in checks:
        img = str(frames / f"check_step{e['step']:04d}.png")
        raw = planner.generate([ref, img] if pair else img, prompt, max_new_tokens=96)["raw"]
        pred, truth = parse(raw), e["shadow_long_task_atoms"][0]
        correct += pred == truth
        rows.append((e["step"], truth, pred, raw.replace("\n", " ")[:110]))
    print(f"== {name}: {correct}/{len(checks)}")
    for r in rows:
        print("  ", *r)
