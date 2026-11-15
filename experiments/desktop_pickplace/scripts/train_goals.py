import json, random, sys, time, csv
from pathlib import Path
import torch
from PIL import Image
from peft import LoraConfig, get_peft_model, TaskType
from protocol import prompt

ROOT = Path(__file__).resolve().parents[3]
TASK = ROOT / "experiments/desktop_pickplace"
sys.path.insert(0, str(ROOT / "perception/qwen_scene"))
from engine import SceneEngine

out = TASK / "runs/goal_lora"
out.mkdir(exist_ok=False)
cases = json.loads((TASK / "data/language/cases.json").read_text())
train = [c for c in cases if int(c["id"].split("_")[0]) < 406]
random.seed(20261015)
torch.manual_seed(20261015)
torch.set_num_threads(8)
engine = SceneEngine()
model = get_peft_model(
    engine.model,
    LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=r".*language_model\.layers\.\d+\.self_attn\.(q_proj|v_proj)$",
    ),
)
model.config.use_cache = False
model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)
model.enable_input_require_grads()
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-5)
start = time.perf_counter()
with (out / "training.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["step", "loss", "seconds"])
    for step in range(240):
        case = random.choice(train)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": Image.open(
                            TASK / "data/language" / case["image"]
                        ).convert("RGB"),
                    },
                    {"type": "text", "text": prompt(case)},
                ],
            }
        ]
        prefix = engine.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        n = prefix["input_ids"].shape[-1]
        messages.append(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                k: case[k]
                                for k in (
                                    "target_color",
                                    "destination",
                                    "should_execute",
                                )
                            }
                        ),
                    }
                ],
            }
        )
        batch = engine.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
        )
        assert torch.equal(batch["input_ids"][:, :n], prefix["input_ids"])
        labels = batch["input_ids"].clone()
        labels[:, :n] = -100
        batch["labels"] = labels
        opt.zero_grad(set_to_none=True)
        loss = model(**batch.to("cuda")).loss
        assert torch.isfinite(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % 20 == 0:
            w.writerow([step + 1, float(loss), time.perf_counter() - start])
            f.flush()
            print("step", step + 1, float(loss), flush=True)
model.save_pretrained(out / "adapter")
engine.processor.save_pretrained(out / "adapter")
(out / "config.json").write_text(
    json.dumps(
        {
            "steps": 240,
            "train_layout_seeds": list(range(400, 406)),
            "validation_seeds": [406, 407],
            "test_seeds": [408, 409],
            "train_cases": len(train),
            "trainable_parameters": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
            "r": 8,
            "alpha": 16,
            "learning_rate": 5e-5,
            "notes": "Same task templates on held-out layouts; not unseen instruction paraphrase generalization",
            "seconds": time.perf_counter() - start,
        },
        indent=2,
    )
)
