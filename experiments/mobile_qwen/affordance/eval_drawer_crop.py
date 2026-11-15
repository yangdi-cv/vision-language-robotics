"""Drawer-state judgement: full image vs crop around the Qwen-predicted cabinet box.

Labels from the simulator drawer joint: closed if opening < 0.005 m, open if >= 0.02 m,
frames in between are reported separately as "nearly closed". The crop box comes from
Qwen's own cabinet bbox (eval_base_v2.json, same frame), expanded by 25% and upscaled to
512 px; no simulator geometry is used to place it.
Usage (qwen-scene venv): eval_drawer_crop.py <data_dir> <bbox_eval_json> <out_json> [--inits 0-12]
"""

from pathlib import Path
import argparse
import json
import sys

import numpy as np
from PIL import Image

Q = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Q / "scripts"))
from qwen_worker import MissionPlanner, parse_json  # noqa: E402
from skills import GRIPPER  # noqa: E402

W = 512
PROMPTS = {
    "full": f"Look at the small white cabinet with three drawers on the left side of the image. {GRIPPER} First describe the front of the cabinet in one sentence, then answer: is the lowest drawer pulled out?\nReturn only JSON: {{\"description\": \"...\", \"drawer_pulled_out\": true or false}}",
    "crop": f"This is a close-up of a small white cabinet with three drawers. {GRIPPER} First describe the front of the cabinet in one sentence, then answer: is the lowest drawer pulled out, even slightly?\nReturn only JSON: {{\"description\": \"...\", \"drawer_pulled_out\": true or false}}",
}


def crop_box(b, margin=0.25):
    x1, y1, x2, y2 = (np.array(b, float) / 1000 * W)
    w, h = x2 - x1, y2 - y1
    x1, x2 = max(0, x1 - margin * w), min(W, x2 + margin * w)
    y1, y2 = max(0, y1 - margin * h), min(W, y2 + margin * h)
    return [int(x1), int(y1), int(x2), int(y2)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data", type=Path)
    p.add_argument("bbox_json", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--inits", default="0-12")
    a = p.parse_args()
    lo, hi = map(int, a.inits.split("-"))
    boxes = {x["id"]: x["parsed"] for x in json.loads(a.bbox_json.read_text()) if x["query"] == "cabinet" and "parsed" in x}
    rows = [json.loads(x) for x in (a.data / "labels.jsonl").read_text().splitlines()]
    rows = [r for r in rows if lo <= r["initial"] <= hi and r["id"] in boxes]
    planner = MissionPlanner()
    tmp = a.out.parent / (a.out.stem + "_crops")
    tmp.mkdir(exist_ok=True)
    results = []
    for r in rows:
        o = r["drawer_opening_m"]
        label = "closed" if o < 0.005 else "open" if o >= 0.02 else "nearly_closed"
        img = Image.open(a.data / "rgb" / f"{r['id']}.png").convert("RGB")
        box = crop_box(boxes[r["id"]])
        if box[2] - box[0] < 16 or box[3] - box[1] < 16:
            continue
        c = img.crop(box)
        s = 512 / max(c.size)
        c = c.resize((int(c.width * s), int(c.height * s)), Image.BICUBIC)
        cpath = tmp / f"{r['id']}.png"
        c.save(cpath)
        item = {"id": r["id"], "source": r["source"], "opening_m": o, "label": label, "crop_box": box}
        for mode, path in (("full", a.data / "rgb" / f"{r['id']}.png"), ("crop", cpath)):
            res = planner.generate(str(path), PROMPTS[mode], max_new_tokens=100)
            try:
                item[mode] = bool(parse_json(res["raw"])["drawer_pulled_out"])
            except Exception:  # noqa: BLE001
                item[mode] = None
        results.append(item)
    a.out.write_text(json.dumps(results, indent=1))
    for mode in ("full", "crop"):
        line = {"mode": mode}
        for lab in ("closed", "nearly_closed", "open"):
            sub = [x for x in results if x["label"] == lab]
            want = lab != "closed"
            line[lab] = f"{sum(x[mode] is want for x in sub)}/{len(sub)}"
        print(json.dumps(line))


if __name__ == "__main__":
    main()
