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
from peft import PeftModel

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
    out = R / "results/model"
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
        "selection": "validation completion recall at FPR<=5%, then lower FPR and macro recall; no test selection",
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
    base_t = threshold_on_validation(base_val)
    (out / "base_validation.json").write_text(json.dumps(base_val, indent=2))
    model = PeftModel.from_pretrained(
        model, R / "results/dev_json/best_adapter", is_trainable=True
    )
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert trainable and not any(
        p.requires_grad for n, p in model.named_parameters() if ".visual." in n
    )
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
            vt = threshold_on_validation(vr)
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
    at = threshold_on_validation(av)
    (out / "adapter_validation.json").write_text(json.dumps(av, indent=2))
    (out / "thresholds.json").write_text(
        json.dumps({"base": base_t, "adapter": at}, indent=2)
    )
    vm = evaluate(av, at)
    bvm = evaluate(base_val, base_t)
    if vm["recall"] < 0.20 or vm["recall"] - bvm["recall"] < 0.15:
        development = {
            "offline_pass": False,
            "test_not_scored": True,
            "reason": "validation operating point below recall/gain requirement; untouched test retained",
            "validation_base": bvm,
            "validation_adapter": vm,
            "best_update": beststep,
            "thresholds": {"base": base_t, "adapter": at},
        }
        (out / "summary.json").write_text(json.dumps(development, indent=2))
        print(json.dumps(development), flush=True)
        return
    # Test is first scored only after both weights and operating thresholds are frozen.
    with model.disable_adapter():
        bt = get_scores(splits["test"])
    adapted = get_scores(splits["test"])
    (out / "base_test.json").write_text(json.dumps(bt, indent=2))
    (out / "adapter_test.json").write_text(json.dumps(adapted, indent=2))
    bm = evaluate(bt, base_t)
    am = evaluate(adapted, at)
    accepted = (
        am["fpr"] <= 0.05
        and am["recall"] - bm["recall"] >= 0.15
        and am["recall"] >= 0.20
    )
    summary = {
        "base": bm,
        "adapter": am,
        "recall_gain": am["recall"] - bm["recall"],
        "offline_pass": accepted,
        "best_update": beststep,
        "validation_selection": best,
        "thresholds": {"base": base_t, "adapter": at},
        "validation_operating_points": {
            "base": evaluate(base_val, base_t),
            "adapter": evaluate(av, at),
        },
        "wall_seconds": time.monotonic() - start,
        "trainable_parameters": sum(p.numel() for p in trainable),
        "peak_mib": torch.cuda.max_memory_allocated() / 2**20,
        "closed_loop_batch_permitted": accepted,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
