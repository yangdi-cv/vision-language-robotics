"""One-shot visual-language goal inference for the live simulation launcher."""

import argparse, json, sys
from pathlib import Path
from PIL import Image
import torch
from peft import PeftModel
from protocol import prompt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "perception/qwen_scene"))
from engine import SceneEngine

parser = argparse.ArgumentParser()
parser.add_argument("--image", type=Path, required=True)
parser.add_argument("--instruction", required=True)
parser.add_argument("--adapter", type=Path)
args = parser.parse_args()
engine = SceneEngine()
torch.set_num_threads(8)
if args.adapter:
    engine.model = PeftModel.from_pretrained(engine.model, args.adapter).eval()
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": Image.open(args.image).convert("RGB")},
            {"type": "text", "text": prompt({"instruction": args.instruction})},
        ],
    }
]
inputs = engine.processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
).to("cuda")
with torch.inference_mode():
    answer = engine.model.generate(**inputs, max_new_tokens=100, do_sample=False)
raw = engine.processor.decode(
    answer[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
)
text = raw.strip()
if text.startswith("```"):
    text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
try:
    goal = json.loads(text)
    assert type(goal["should_execute"]) is bool
    assert goal["target_color"] in ("red", "blue", "green", None) and goal[
        "destination"
    ] in ("left", "right", None)
    print(json.dumps({"valid": True, "goal": goal, "raw": raw}, ensure_ascii=False))
except (ValueError, AssertionError, KeyError, TypeError):
    print(json.dumps({"valid": False, "raw": raw}, ensure_ascii=False))
