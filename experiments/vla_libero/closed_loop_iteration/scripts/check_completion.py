# ruff: noqa: E402
"""Standalone two-frame completion checker. No robot or simulator control interface."""

from pathlib import Path
import argparse
import json
import sys
import time
from PIL import Image

R = Path(__file__).resolve().parents[1]
ROOT = R.parents[2]
sys.path[:0] = [str(ROOT / "perception/qwen_scene"), str(R / "scripts")]
from engine import SceneEngine
from checker_protocol import prompt, parse, boolean_probability


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--before", required=True)
    p.add_argument("--after", required=True)
    p.add_argument("--instruction", required=True)
    p.add_argument("--adapter", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--calibrated", action="store_true")
    p.add_argument("--threshold", type=float)
    a = p.parse_args()
    e = SceneEngine()
    if a.adapter:
        from peft import PeftModel

        e.model = PeftModel.from_pretrained(e.model, a.adapter, is_trainable=False)
    e.model.eval()
    images = [
        Image.open(x).convert("RGB").resize((256, 256)) for x in [a.before, a.after]
    ]
    if a.calibrated:
        threshold = a.threshold
        if threshold is None:
            if not a.adapter:
                raise ValueError("threshold or adapter required")
            threshold = json.loads(
                (a.adapter.parent / "calibrated_summary.json").read_text()
            )["adapter"]["threshold"]
        start = time.monotonic()
        score = boolean_probability(e, e.model, images, a.instruction)
        result = {
            "done": score >= threshold,
            "score": score,
            "threshold": threshold,
            "seconds": time.monotonic() - start,
            "mode": "validation-calibrated diagnostic; not accepted for robot switching",
        }
        if a.out:
            a.out.write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
        return
    msg = [
        {
            "role": "user",
            "content": [{"type": "image", "image": im} for im in images]
            + [{"type": "text", "text": prompt(a.instruction)}],
        }
    ]
    inputs = e.processor.apply_chat_template(
        msg,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to("cuda")
    start = time.monotonic()
    with e.torch.inference_mode():
        generated = e.model.generate(**inputs, max_new_tokens=24, do_sample=False)
    raw = e.processor.decode(
        generated[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
    )
    result = {
        "raw": raw,
        "done": parse(raw),
        "seconds": time.monotonic() - start,
        "adapter": str(a.adapter) if a.adapter else None,
    }
    if a.out:
        a.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
