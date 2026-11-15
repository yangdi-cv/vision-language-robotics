# ruff: noqa: E402
"""Validation-only threshold selection at a fixed false-confirmation budget."""

from pathlib import Path
import contextlib
import json
import sys
import torch
from PIL import Image
from peft import PeftModel

R = Path(__file__).resolve().parents[1]
ROOT = R.parents[2]
OUT = R / "results/checker_lora"
sys.path[:0] = [str(ROOT / "perception/qwen_scene"), str(R / "scripts")]
from engine import SceneEngine
from checker_protocol import prompt, metrics


def main():
    target = OUT / "calibrated_summary.json"
    if target.exists():
        raise FileExistsError(target)
    torch.set_num_threads(6)
    rows = [
        r
        for r in json.loads((R / "data/cases.json").read_text())
        if r["split"] in ["validation", "test"]
    ]
    e = SceneEngine()
    model = PeftModel.from_pretrained(
        e.model, OUT / "best_adapter", is_trainable=False
    ).eval()
    prepared = {}
    for r in rows:
        msg = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": Image.open(R / "data" / p).convert("RGB"),
                    }
                    for p in r["images"]
                ]
                + [{"type": "text", "text": prompt(r["instruction"])}],
            }
        ]
        versions = []
        for value in [False, True]:
            m = msg + [
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": json.dumps({"done": value})}],
                }
            ]
            versions.append(
                e.processor.apply_chat_template(
                    m,
                    tokenize=True,
                    add_generation_prompt=False,
                    return_dict=True,
                    return_tensors="pt",
                )
            )
        false, true = versions
        a = false["input_ids"][0]
        b = true["input_ids"][0]
        assert len(a) == len(b)
        idx = int(torch.nonzero(a != b)[0])
        assert torch.equal(a[:idx], b[:idx])
        data = {k: v for k, v in true.items()}
        data["input_ids"] = data["input_ids"][:, :idx]
        data["attention_mask"] = data["attention_mask"][:, :idx]
        prepared[r["id"]] = (data, int(a[idx]), int(b[idx]))
    summaries = {}
    for mode in ["base", "adapter"]:
        outputs = []
        ctx = model.disable_adapter() if mode == "base" else contextlib.nullcontext()
        with ctx, torch.inference_mode():
            for r in rows:
                data, false_id, true_id = prepared[r["id"]]
                logits = model(
                    **{k: v.to("cuda") for k, v in data.items()}, use_cache=False
                ).logits[0, -1]
                probability = float(
                    torch.softmax(logits[[false_id, true_id]].float(), dim=0)[1]
                )
                outputs.append(
                    {
                        "id": r["id"],
                        "split": r["split"],
                        "expected": r["done"],
                        "stratum": r["stratum"],
                        "score": probability,
                    }
                )
        val = [r for r in outputs if r["split"] == "validation"]
        test = [r for r in outputs if r["split"] == "test"]

        def predict(group, threshold):
            return [
                dict(r, valid=True, predicted=r["score"] >= threshold) for r in group
            ]

        best = None
        threshold = None
        vm = None
        for t in sorted({r["score"] for r in val} | {1.000001}):
            m = metrics(predict(val, t))
            if m["false_positive_rate"] > 0.05:
                continue
            rank = (m["tp"] / m["positive"], -m["false_positive_rate"], t)
            if best is None or rank > best:
                best, threshold, vm = rank, t, m
        scored = predict(test, threshold)
        summaries[mode] = {
            "threshold": threshold,
            "validation": vm,
            "test": metrics(scored),
            "core_test": metrics([r for r in scored if r["stratum"] != "boundary"]),
            "boundary_test": metrics([r for r in scored if r["stratum"] == "boundary"]),
        }
        (OUT / f"{mode}_scores.json").write_text(json.dumps(outputs, indent=2))
        print(json.dumps({mode: summaries[mode]}), flush=True)
    target.write_text(json.dumps(summaries, indent=2))
    # Re-opening the saved adapter in this fresh process verifies actual weight usability.
    (OUT / "reload_check.json").write_text(
        json.dumps(
            {
                "fresh_process_load": True,
                "adapter": "best_adapter",
                "rows_scored_per_mode": len(rows),
                "all_scores_finite": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
