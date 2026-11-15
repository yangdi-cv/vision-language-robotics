"""Visual-language goal selection for the sorting task; no actuator access."""

import json, sys, time, argparse
from pathlib import Path
from PIL import Image
from protocol import prompt as make_prompt
import torch

ROOT = Path(__file__).resolve().parents[3]
TASK = ROOT / "experiments/desktop_pickplace"
sys.path.insert(0, str(ROOT / "perception/qwen_scene"))
from engine import SceneEngine

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, default=TASK / "runs/qwen_goals")
parser.add_argument("--adapter", type=Path)
parser.add_argument("--heldout", action="store_true")
parser.add_argument("--cases", type=Path)
args = parser.parse_args()
out = args.out
out.mkdir(exist_ok=False)
data = TASK / "data/language"
cases = json.loads((args.cases if args.cases else data / "cases.json").read_text())
engine = SceneEngine()
if args.adapter:
    from peft import PeftModel

    engine.model = PeftModel.from_pretrained(engine.model, args.adapter).eval()
if args.heldout:
    cases = [c for c in cases if int(c["id"].split("_")[0]) >= 406]
torch.set_num_threads(8)
rows = []
with (out / "predictions.jsonl").open("w") as f:
    for case in cases:
        prompt = make_prompt(case)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": Image.open(data / case["image"]).convert("RGB"),
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        tick = time.perf_counter()
        inputs = engine.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to("cuda")
        with torch.inference_mode():
            answer = engine.model.generate(
                **inputs, max_new_tokens=100, do_sample=False
            )
        raw = engine.processor.decode(
            answer[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
        )
        row = {
            "case": case,
            "raw": raw,
            "seconds": time.perf_counter() - tick,
            "valid": False,
            "correct": False,
        }
        try:
            text = raw.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            goal = json.loads(text)
            assert type(goal["should_execute"]) is bool
            assert goal["target_color"] in ("red", "blue", "green", None) and goal[
                "destination"
            ] in ("left", "right", None)
            row.update(
                valid=True,
                goal=goal,
                correct=all(
                    goal[k] == case[k]
                    for k in ("should_execute", "target_color", "destination")
                ),
            )
        except (ValueError, KeyError, TypeError, AssertionError):
            pass
        rows.append(row)
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        if len(rows) % 20 == 0:
            print("Qwen", len(rows), flush=True)
pos = [r for r in rows if r["case"]["should_execute"]]
negative = [r for r in rows if not r["case"]["should_execute"]]
(out / "metrics.json").write_text(
    json.dumps(
        {
            "cases": len(rows),
            "correct": sum(x["correct"] for x in rows),
            "positive_cases": len(pos),
            "positive_correct": sum(x["correct"] for x in pos),
            "negative_cases": len(negative),
            "negative_correct": sum(x["correct"] for x in negative),
            "unsafe_negative_proposals": sum(
                x.get("goal", {}).get("should_execute", False) for x in negative
            ),
            "note": "Multiobject simulator images. Model proposals not sent to an actuator.",
        },
        indent=2,
    )
)
print("Qwen goals done")
