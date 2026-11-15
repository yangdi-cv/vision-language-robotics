# ruff: noqa: E402
"""Bounded completion-checker LoRA. Validation selects; test is evaluated once afterward."""

from pathlib import Path
import json
import random
import sys
import time
import numpy as np
import torch
from PIL import Image
from peft import LoraConfig, get_peft_model

R = Path(__file__).resolve().parents[1]
ROOT = R.parents[2]
sys.path[:0] = [str(ROOT / "perception/qwen_scene"), str(R / "scripts")]
from engine import SceneEngine, MODEL_ID, REVISION
from checker_protocol import prompt, parse, metrics


def main():
    out = R / "results/checker_lora"
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    seed = 20261112
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(6)
    allrows = json.loads((R / "data/cases.json").read_text())
    split = {
        s: [r for r in allrows if r["split"] == s]
        for s in ["train", "validation", "test"]
    }
    # Sub-step contact thresholds can be visually ambiguous. Keep these cases as a
    # separate challenge set; do not train or select checkpoints on them.
    split["train"] = [r for r in split["train"] if r["stratum"] != "boundary"]
    split["validation"] = [r for r in split["validation"] if r["stratum"] != "boundary"]
    # A balanced fixed development subset selects the checkpoint; the full held-out test remains untouched.
    val = []
    rng = random.Random(seed)
    for label in [False, True]:
        choices = [r for r in split["validation"] if r["done"] == label]
        val.extend(rng.sample(choices, min(32, len(choices))))
    protocol = {
        "seed": seed,
        "model": MODEL_ID,
        "revision": REVISION,
        "steps": 180,
        "gradient_accumulation": 2,
        "learning_rate": 5e-5,
        "r": 8,
        "alpha": 16,
        "dropout": 0.05,
        "validation_every": 60,
        "validation_ids": [r["id"] for r in val],
        "test_ids": [r["id"] for r in split["test"]],
        "train_ids": [r["id"] for r in split["train"]],
        "selection": "highest validation balanced accuracy, then lower false-positive rate; no test-based checkpoint or hyperparameter selection",
        "precision": "BF16; frozen vision; language q/v LoRA",
        "images": "two 256px RGB frames, identical preprocessing for base/adapter",
        "limit_hours": 24,
        "boundary_policy": "Last-five-step and last-step negatives excluded from training/selection; all test cases retained and additionally reported by stratum",
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    e = SceneEngine()
    model = e.model

    def preprocess(row, answer):
        ims = [Image.open(R / "data" / p).convert("RGB") for p in row["images"]]
        msg = [
            {
                "role": "user",
                "content": [{"type": "image", "image": im} for im in ims]
                + [{"type": "text", "text": prompt(row["instruction"])}],
            }
        ]
        prefix = e.processor.apply_chat_template(
            msg,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        if not answer:
            return dict(prefix)
        msg.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": json.dumps({"done": row["done"]})}
                ],
            }
        )
        full = e.processor.apply_chat_template(
            msg,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
        )
        n = prefix["input_ids"].shape[-1]
        assert torch.equal(prefix["input_ids"], full["input_ids"][:, :n])
        labels = full["input_ids"].clone()
        labels[:, :n] = -100
        full["labels"] = labels
        return dict(full)

    train_batches = {r["id"]: preprocess(r, True) for r in split["train"]}
    prefixes = {r["id"]: preprocess(r, False) for r in val + split["test"]}

    def evaluate(rows, name):
        model.eval()
        model.gradient_checkpointing_disable()
        model.config.use_cache = True
        outputs = []
        with torch.inference_mode():
            for row in rows:
                inputs = {k: v.to("cuda") for k, v in prefixes[row["id"]].items()}
                t = time.monotonic()
                generated = model.generate(**inputs, max_new_tokens=24, do_sample=False)
                raw = e.processor.decode(
                    generated[0][inputs["input_ids"].shape[-1] :],
                    skip_special_tokens=True,
                )
                result = {
                    "id": row["id"],
                    "episode": row["episode"],
                    "suite": row["suite"],
                    "stratum": row["stratum"],
                    "expected": row["done"],
                    "valid": False,
                    "predicted": None,
                    "raw": raw,
                    "seconds": time.monotonic() - t,
                }
                try:
                    result.update(predicted=parse(raw), valid=True)
                except (ValueError, KeyError, TypeError):
                    pass
                outputs.append(result)
        (out / f"{name}.json").write_text(json.dumps(outputs, indent=2))
        m = metrics(outputs)
        print(json.dumps({"evaluation": name, **m}), flush=True)
        return m

    base = evaluate(split["test"], "base_test")
    model = get_peft_model(
        model,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=r".*language_model\.layers\.\d+\.self_attn\.(q_proj|v_proj)$",
        ),
    )
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert trainable and not any(
        p.requires_grad for n, p in model.named_parameters() if ".visual." in n
    )
    protocol["trainable_parameters"] = sum(p.numel() for p in trainable)
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    model.enable_input_require_grads()
    opt = torch.optim.AdamW(trainable, lr=5e-5)
    positives = [r for r in split["train"] if r["done"]]
    negatives = [r for r in split["train"] if not r["done"]]
    best = (-1.0, -1.0)
    best_step = None
    history = []
    started = time.monotonic()
    for step in range(1, 181):
        model.train()
        model.config.use_cache = False
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        opt.zero_grad(set_to_none=True)
        losses = []
        for group in [positives, negatives]:
            row = random.choice(group)
            batch = {k: v.to("cuda") for k, v in train_batches[row["id"]].items()}
            loss = model(**batch).loss
            assert torch.isfinite(loss)
            (loss / 2).backward()
            losses.append(float(loss.detach()))
        torch.nn.utils.clip_grad_norm_(trainable, 1)
        opt.step()
        if step % 10 == 0:
            print(
                json.dumps(
                    {
                        "step": step,
                        "loss": float(np.mean(losses)),
                        "seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )
        if step % 60 == 0:
            m = evaluate(val, f"validation_{step}")
            score = (m["balanced_accuracy"], -m["false_positive_rate"])
            saved = score > best
            if saved:
                best = score
                best_step = step
                model.save_pretrained(out / "best_adapter")
                e.processor.save_pretrained(out / "best_adapter")
            history.append(
                {
                    "step": step,
                    "loss": float(np.mean(losses)),
                    "validation": m,
                    "saved": saved,
                }
            )
            (out / "history.json").write_text(json.dumps(history, indent=2))
    model.save_pretrained(out / "last_adapter")
    model.load_adapter(out / "best_adapter", adapter_name="selected")
    model.set_adapter("selected")
    adapted = evaluate(split["test"], "adapter_test")
    summary = {
        "base_test": base,
        "adapter_test": adapted,
        "selected_step": best_step,
        "wall_train_seconds": time.monotonic() - started,
        "trainable_parameters": protocol["trainable_parameters"],
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "note": "Retrospective initial-state holdout, shared tasks. No claim of unseen-task or real-world generalization. Full-task labels; arbitrary intermediate-goal alignment not assumed.",
    }
    base_rows = json.loads((out / "base_test.json").read_text())
    adapter_rows = json.loads((out / "adapter_test.json").read_text())
    summary["strata"] = {
        name: {
            "base": metrics(
                [r for r in base_rows if (r["stratum"] == "boundary") == is_boundary]
            ),
            "adapter": metrics(
                [r for r in adapter_rows if (r["stratum"] == "boundary") == is_boundary]
            ),
        }
        for name, is_boundary in [("core", False), ("boundary", True)]
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
