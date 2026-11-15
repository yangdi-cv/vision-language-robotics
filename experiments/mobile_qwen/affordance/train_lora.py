"""LoRA fine-tune of Qwen3-VL-4B on simulator-labelled affordance grounding.

Train/validation data: affordance/data_train_v1 (initial states 13-49; validation = 13-16,
used only for checkpoint selection). The test set (initial states 0-12, data_v1) is never
read here; score it afterwards with eval_qwen.py --adapter.
Targets use Qwen3-VL's 0-1000 relative coordinates (confirmed on the base model).
Usage (qwen-scene venv): train_lora.py <data_dir> <out_dir> [--steps 600]
"""

from pathlib import Path
import argparse
import json
import random
import sys
import time

import numpy as np
import torch
from PIL import Image
from peft import LoraConfig, get_peft_model

Q = Path(__file__).resolve().parents[1]
ROOT = Q.parents[1]
sys.path[:0] = [str(ROOT / "perception/qwen_scene"), str(Q / "scripts"), str(Q / "affordance")]
from engine import SceneEngine  # noqa: E402
from qwen_worker import parse_json  # noqa: E402
from eval_qwen import QUERIES_V2, applicable  # noqa: E402

W = 512
WEIGHTS = {"handle": 0.5, "cabinet": 0.25, "place": 0.125, "bowl": 0.125}


def rel(v):
    return [int(round(x / W * 1000)) for x in v]


def target(name, r):
    if name == "handle":
        return {"point_2d": rel(r["handle_mask"]["centroid"])}
    if name == "bowl":
        return {"point_2d": rel(r["bowl_mask"]["centroid"])}
    if name == "place":
        return {"point_2d": rel(r["drawer_interior"]["uv"])}
    return {"bbox_2d": rel(r["cabinet_all_bbox"])}


def build(processor, image, prompt, answer=None):
    msg = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    prefix = processor.apply_chat_template(msg, tokenize=True, add_generation_prompt=True,
                                           return_dict=True, return_tensors="pt")
    if answer is None:
        return dict(prefix)
    msg.append({"role": "assistant", "content": [{"type": "text", "text": json.dumps(answer)}]})
    full = processor.apply_chat_template(msg, tokenize=True, add_generation_prompt=False,
                                         return_dict=True, return_tensors="pt")
    n = prefix["input_ids"].shape[-1]
    assert torch.equal(prefix["input_ids"], full["input_ids"][:, :n])
    labels = full["input_ids"].clone()
    labels[:, :n] = -100
    full["labels"] = labels
    return dict(full)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--val-inits", default="13-16")
    p.add_argument("--eval-every", type=int, default=150)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    a.out.mkdir(parents=True)
    seed = 20261114
    torch.manual_seed(seed)
    rng = random.Random(seed)
    rows = [json.loads(x) for x in (a.data / "labels.jsonl").read_text().splitlines()]
    vlo, vhi = map(int, a.val_inits.split("-"))
    val = [r for r in rows if vlo <= r["initial"] <= vhi and applicable("handle", r)]
    train = [r for r in rows if not vlo <= r["initial"] <= vhi]
    pools = {k: [r for r in train if applicable(k, r)] for k in WEIGHTS}
    val = rng.sample(val, min(48, len(val)))
    (a.out / "protocol.json").write_text(json.dumps({
        "seed": seed, "steps": a.steps, "accum": a.accum, "lr": a.lr, "r": 8, "alpha": 16,
        "target_modules": "language_model q_proj,v_proj", "weights": WEIGHTS,
        "train_frames": len(train), "pool_sizes": {k: len(v) for k, v in pools.items()},
        "val_ids": [r["id"] for r in val], "train_inits": sorted({r["initial"] for r in train}),
        "selection": "validation handle median pixel error (lower better); test never read",
        "prompts": QUERIES_V2,
    }, indent=2))
    e = SceneEngine()
    model, proc = e.model, e.processor

    def image(r):
        return Image.open(a.data / "rgb" / f"{r['id']}.png").convert("RGB")

    def validate():
        model.eval()
        model.gradient_checkpointing_disable()
        model.config.use_cache = True
        errs = []
        for r in val:
            inputs = {k: v.to("cuda") for k, v in build(proc, image(r), QUERIES_V2["handle"]).items()}
            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=40, do_sample=False)
            raw = proc.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            try:
                u, v = (np.array(parse_json(raw)["point_2d"], float) / 1000 * W)
                c = r["handle_mask"]["centroid"]
                errs.append(float(np.hypot(u - c[0], v - c[1])))
            except Exception:  # noqa: BLE001
                errs.append(999.0)
        return float(np.median(errs)), float(np.mean(np.array(errs) <= 15))

    history = [{"step": 0, "val": validate()}]
    print(json.dumps(history[-1]), flush=True)
    model = get_peft_model(model, LoraConfig(
        task_type="CAUSAL_LM", r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=r".*language_model\.layers\.\d+\.self_attn\.(q_proj|v_proj)$"))
    trainable = [q for q in model.parameters() if q.requires_grad]
    assert trainable and not any(q.requires_grad for n, q in model.named_parameters() if ".visual." in n)
    model.enable_input_require_grads()
    opt = torch.optim.AdamW(trainable, lr=a.lr)
    names, weights = list(WEIGHTS), list(WEIGHTS.values())
    best, start = None, time.monotonic()
    for step in range(1, a.steps + 1):
        model.train()
        model.config.use_cache = False
        if not model.is_gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        opt.zero_grad(set_to_none=True)
        losses = []
        for _ in range(a.accum):
            name = rng.choices(names, weights)[0]
            r = rng.choice(pools[name])
            data = {k: v.to("cuda") for k, v in build(proc, image(r), QUERIES_V2[name], target(name, r)).items()}
            loss = model(**data).loss
            assert torch.isfinite(loss)
            (loss / a.accum).backward()
            losses.append(float(loss.detach()))
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        if step % 25 == 0:
            print(json.dumps({"step": step, "loss": float(np.mean(losses)), "sec": round(time.monotonic() - start)}), flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            med, within = validate()
            saved = best is None or med < best[0]
            if saved:
                best = (med, within, step)
                model.save_pretrained(a.out / "best_adapter")
            history.append({"step": step, "val": [med, within], "saved": saved})
            print(json.dumps(history[-1]), flush=True)
            (a.out / "history.json").write_text(json.dumps(history, indent=2))
    (a.out / "summary.json").write_text(json.dumps({
        "best_step": best[2], "val_median_px": best[0], "val_within_15px": best[1],
        "base_val": history[0]["val"], "wall_seconds": time.monotonic() - start,
        "trainable_parameters": sum(q.numel() for q in trainable),
        "peak_mib": torch.cuda.max_memory_allocated() / 2**20}, indent=2))
    print("done", best, flush=True)


if __name__ == "__main__":
    main()
