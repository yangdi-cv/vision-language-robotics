"""Validation-controlled goal adapter; test labels never select a checkpoint."""

import sys, json, random, time, csv, argparse
from pathlib import Path
import torch, numpy as np
from PIL import Image
from peft import LoraConfig, get_peft_model, PeftModel

ROOT = Path(__file__).resolve().parents[3]
T = ROOT / "experiments/alibaba_pc_revision"
parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, default=T / "runs/lora_validation")
parser.add_argument(
    "--data", type=Path, default=ROOT / "experiments/desktop_pickplace/data/language"
)
parser.add_argument("--max-steps", type=int, default=240)
parser.add_argument("--validate-every", type=int, default=40)
parser.add_argument("--patience", type=int, default=3)
args = parser.parse_args()
DATA = args.data.resolve()
OUT = args.out.resolve()
OUT.mkdir(parents=True, exist_ok=False)
sys.path[:0] = [
    str(ROOT / "perception/qwen_scene"),
    str(ROOT / "experiments/desktop_pickplace/scripts"),
]
from engine import SceneEngine
from protocol import prompt

random.seed(20261020)
torch.manual_seed(20261020)
torch.set_num_threads(8)
cases = json.loads((DATA / "cases.json").read_text())
split = {
    "train": [c for c in cases if int(c["id"].split("_")[0]) < 406],
    "validation": [c for c in cases if 406 <= int(c["id"].split("_")[0]) <= 407],
    "test": [c for c in cases if int(c["id"].split("_")[0]) >= 408],
}
(OUT / "split.json").write_text(
    json.dumps({k: [c["id"] for c in v] for k, v in split.items()}, indent=2)
)
engine = SceneEngine()
model = get_peft_model(
    engine.model,
    LoraConfig(
        task_type="CAUSAL_LM",
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=r".*language_model\.layers\.\d+\.self_attn\.(q_proj|v_proj)$",
    ),
)


def prep(c, answer):
    msgs = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": Image.open(DATA / c["image"]).convert("RGB"),
                },
                {"type": "text", "text": prompt(c)},
            ],
        }
    ]
    prefix = engine.processor.apply_chat_template(
        msgs,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    if not answer:
        return prefix
    msgs.append(
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(
                        {
                            k: c[k]
                            for k in ["target_color", "destination", "should_execute"]
                        }
                    ),
                }
            ],
        }
    )
    full = engine.processor.apply_chat_template(
        msgs,
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
    return full


batches = {key: [prep(c, True) for c in cs] for key, cs in split.items()}
prefixes = {
    key: [prep(c, False) for c in cs] for key, cs in split.items() if key != "train"
}


def evaluate(key, predict=True):
    model.eval()
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    loss = []
    rows = []
    with torch.inference_mode():
        for c, b, p in zip(split[key], batches[key], prefixes[key]):
            loss.append(float(model(**b.to("cuda")).loss))
            p = p.to("cuda")
            out = model.generate(**p, max_new_tokens=96, do_sample=False)
            raw = engine.processor.decode(
                out[0][p["input_ids"].shape[-1] :], skip_special_tokens=True
            )
            try:
                goal = json.loads(raw)
                valid = (
                    set(goal) == {"target_color", "destination", "should_execute"}
                    and type(goal["should_execute"]) is bool
                )
                correct = valid and all(goal[k] == c[k] for k in goal)
            except Exception:
                valid = False
                correct = False
            rows.append({"id": c["id"], "raw": raw, "valid": valid, "correct": correct})
    return float(np.mean(loss)), rows


opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-5)
best_loss = float("inf")
best_score = -1
stale = 0
history = []
start = time.perf_counter()
best_step = 0
for step in range(1, args.max_steps + 1):
    model.train()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()
    opt.zero_grad(set_to_none=True)
    loss = model(**random.choice(batches["train"]).to("cuda")).loss
    assert torch.isfinite(loss)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
    opt.step()
    if step % args.validate_every == 0 or step == args.max_steps:
        vl, rows = evaluate("validation")
        score = sum(r["correct"] for r in rows)
        improved = score > best_score or (
            score == best_score and vl < best_loss - 0.001
        )
        history.append(
            {
                "step": step,
                "training_loss": float(loss),
                "validation_loss": vl,
                "validation_correct": score,
                "seconds": time.perf_counter() - start,
                "saved": improved,
            }
        )
        (OUT / f"validation_{step}.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2)
        )
        if improved:
            best_score = score
            best_loss = vl
            best_step = step
            stale = 0
            model.save_pretrained(OUT / "best_adapter")
            engine.processor.save_pretrained(OUT / "best_adapter")
        else:
            stale += 1
        print(history[-1], flush=True)
        if stale >= args.patience:
            break
model.save_pretrained(OUT / "last_adapter")
engine.processor.save_pretrained(OUT / "last_adapter")
last_loss, last_rows = evaluate("test")
model.load_adapter(OUT / "best_adapter", adapter_name="selected")
model.set_adapter("selected")
best_test_loss, best_rows = evaluate("test")
(OUT / "test_best.json").write_text(json.dumps(best_rows, ensure_ascii=False, indent=2))
(OUT / "test_last.json").write_text(json.dumps(last_rows, ensure_ascii=False, indent=2))
(OUT / "history.json").write_text(json.dumps(history, indent=2))
(OUT / "summary.json").write_text(
    json.dumps(
        {
            "best_step": best_step,
            "last_step": step,
            "best_validation_correct": best_score,
            "best_validation_loss": best_loss,
            "best_test_correct": sum(r["correct"] for r in best_rows),
            "last_test_correct": sum(r["correct"] for r in last_rows),
            "test_cases": len(best_rows),
            "patience": args.patience,
            "max_steps": args.max_steps,
            "selection": "validation exact match, then CE with min_delta .001",
            "test_used_for_selection": False,
            "seconds": time.perf_counter() - start,
        },
        indent=2,
    )
)
