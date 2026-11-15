# ruff: noqa: E402
"""One bounded V2 run; three labels, val-only checkpoint/threshold, held-out evaluation."""

from pathlib import Path
import json
import random
import sys
import time
from functools import lru_cache
import numpy as np
import torch
from PIL import Image
from peft import PeftModel, LoraConfig

R = Path(__file__).resolve().parents[1]
ROOT = R.parents[2]
sys.path[:0] = [str(ROOT / "perception/qwen_scene"), str(R / "scripts")]
from engine import SceneEngine
from protocol import (
    CLASSES,
    prepare,
    score_input,
    probabilities,
    evaluate,
    threshold_on_validation,
    macro_recall,
)


def main():
    out = R / "results/model_vision"
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    start = time.monotonic()
    torch.manual_seed(20261113)
    torch.set_num_threads(4)
    rng = random.Random(20261113)
    rows = json.loads((R / "data/cases.json").read_text())
    byid = {r["id"]: r for r in rows}
    splits = {
        s: [r for r in rows if r["split"] == s] for s in ["train", "validation", "test"]
    }
    count = min(
        60, *[sum(r["status"] == c for r in splits["validation"]) for c in CLASSES]
    )
    val = []
    for c in CLASSES:
        val.extend(
            rng.sample([r for r in splits["validation"] if r["status"] == c], count)
        )
    plan = {
        "updates": 360,
        "seed": 20261113,
        "r": 8,
        "alpha": 16,
        "lr": 5e-5,
        "accumulation": 3,
        "train_ids": [r["id"] for r in splits["train"]],
        "selection_ids": [r["id"] for r in splits["validation"]],
        "validation_ids": [r["id"] for r in splits["validation"]],
        "test_ids": [r["id"] for r in splits["test"]],
        "selection": "validation completion recall at FPR<=2%, then lower FPR and macro recall; no test selection",
        "objective": "cross entropy of the three deployed class logits; JSON syntax excluded",
        "warm_start": "dev_json/best_adapter selected at update360 using validation only",
        "data_sha256": __import__("hashlib")
        .sha256((R / "data/cases.json").read_bytes())
        .hexdigest(),
    }
    (out / "training_protocol.json").write_text(json.dumps(plan, indent=2))
    e = SceneEngine()
    model = e.model

    def images(r):
        return [Image.open(R / "data" / p).convert("RGB") for p in r["images"]]

    @lru_cache(maxsize=32)
    def batch(uid):
        r = byid[uid]
        return prepare(e.processor, images(r), r["instruction"], r["status"])

    @lru_cache(maxsize=32)
    def scoring(uid):
        r = byid[uid]
        return score_input(e.processor, images(r), r["instruction"])

    def get_scores(group):
        model.eval()
        model.gradient_checkpointing_disable()
        model.config.use_cache = False
        result = []
        for r in group:
            p = probabilities(model, scoring(r["id"]))
            result.append(
                {
                    k: r[k]
                    for k in [
                        "id",
                        "group",
                        "split",
                        "status",
                        "done",
                        "source",
                        "kind",
                    ]
                }
                | {"scores": p}
            )
        return result

    base_val = get_scores(splits["validation"])
    base_t = threshold_on_validation(base_val, 0.02)
    (out / "base_validation.json").write_text(json.dumps(base_val, indent=2))
    warm = R / "results/model/best_adapter"
    cfg = LoraConfig.from_pretrained(warm)
    cfg.target_modules = r".*(?:language_model\.layers\.\d+\.self_attn\.(?:q_proj|v_proj)|visual\.blocks\.\d+\.attn\.qkv)$"
    model = PeftModel.from_pretrained(model, warm, is_trainable=True, config=cfg)
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert any(p.requires_grad for n, p in model.named_parameters() if ".visual." in n)
    plan["vision_qkv_lora"] = True
    plan["warm_start"] = "classification model update360 selected by validation"
    plan["trainable_parameters"] = sum(p.numel() for p in trainable)
    (out / "training_protocol.json").write_text(json.dumps(plan, indent=2))
    model.enable_input_require_grads()
    opt = torch.optim.AdamW(trainable, lr=5e-5)
    classes = {c: [r for r in splits["train"] if r["status"] == c] for c in CLASSES}
    best = (-1.0, -1.0, -1.0)
    history = []
    beststep = None
    for step in range(1, 361):
        model.train()
        model.config.use_cache = False
        if not model.is_gradient_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        opt.zero_grad(set_to_none=True)
        losses = []
        for c in CLASSES:
            r = rng.choice(classes[c])
            prepared, ids = scoring(r["id"])
            data = {k: v.to("cuda") for k, v in prepared.items()}
            logits = (
                model(**data, use_cache=False, logits_to_keep=1)
                .logits[0, -1, ids]
                .float()
            )
            loss = torch.nn.functional.cross_entropy(
                logits[None, :], torch.tensor([CLASSES.index(c)], device="cuda")
            )
            assert torch.isfinite(loss)
            (loss / 3).backward()
            losses.append(float(loss.detach()))
        if step == 1:
            vision_gradient = sum(
                float(p.grad.detach().abs().sum())
                for n, p in model.named_parameters()
                if ".visual." in n and p.requires_grad and p.grad is not None
            )
            assert vision_gradient > 0
            plan["vision_gradient_first_update"] = vision_gradient
            (out / "training_protocol.json").write_text(json.dumps(plan, indent=2))
        torch.nn.utils.clip_grad_norm_(trainable, 1)
        opt.step()
        if step % 20 == 0:
            print(
                json.dumps(
                    {
                        "step": step,
                        "loss": float(np.mean(losses)),
                        "seconds": time.monotonic() - start,
                    }
                ),
                flush=True,
            )
        if step % 120 == 0:
            vr = get_scores(splits["validation"])
            vt = threshold_on_validation(vr, 0.02)
            vm = evaluate(vr, vt)
            macro = macro_recall(vr)
            score = (vm["recall"], -vm["fpr"], macro)
            saved = score > best
            if saved:
                best = score
                beststep = step
                model.save_pretrained(out / "best_adapter")
                e.processor.save_pretrained(out / "best_adapter")
            history.append(
                {
                    "step": step,
                    "macro_recall": macro,
                    "operating_point": vm,
                    "threshold": vt,
                    "saved": saved,
                }
            )
            (out / "history.json").write_text(json.dumps(history, indent=2))
            (out / f"validation_{step}.json").write_text(json.dumps(vr, indent=2))
            print(json.dumps(history[-1]), flush=True)
    model.load_adapter(out / "best_adapter", adapter_name="selected")
    model.set_adapter("selected")
    av = get_scores(splits["validation"])
    at = threshold_on_validation(av, 0.02)
    (out / "adapter_validation.json").write_text(json.dumps(av, indent=2))
    (out / "thresholds.json").write_text(
        json.dumps({"base": base_t, "adapter": at}, indent=2)
    )
    vm = evaluate(av, at)
    bvm = evaluate(base_val, base_t)
    development = {
        "validation_base": bvm,
        "validation_adapter": vm,
        "validation_gain": vm["recall"] - bvm["recall"],
        "thresholds": {"base": base_t, "adapter": at},
        "dev_pass": vm["recall"] >= 0.20 and vm["recall"] - bvm["recall"] >= 0.15,
        "best_update": beststep,
        "test_scored": False,
        "score_type": "log_odds",
        "calibration_fpr_limit": 0.02,
        "vision_adaptation": True,
    }
    (out / "development_summary.json").write_text(json.dumps(development, indent=2))
    print(json.dumps(development), flush=True)


if __name__ == "__main__":
    main()
