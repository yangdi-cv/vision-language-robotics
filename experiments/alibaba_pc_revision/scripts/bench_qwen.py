"""Interleaved task-specific token limits and finite-goal constrained decoding."""

from pathlib import Path
import sys, json, time, random, itertools
import numpy as np, torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
T = ROOT / "experiments/alibaba_pc_revision"
D = ROOT / "experiments/desktop_pickplace/data/language"
O = T / "runs/qwen_limits"
O.mkdir(exist_ok=False)
sys.path[:0] = [
    str(ROOT / "perception/qwen_scene"),
    str(ROOT / "experiments/desktop_pickplace/scripts"),
]
from engine import SceneEngine
from protocol import prompt

engine = SceneEngine()
torch.set_num_threads(8)
cases = [
    c
    for c in json.loads((D / "cases.json").read_text())
    if c["id"].startswith(("408_", "409_"))
]
cases = cases[:8] + cases[-4:]
allowed = [
    dict(target_color=c, destination=b, should_execute=True)
    for c, b in itertools.product(["red", "blue", "green"], ["left", "right"])
] + [dict(target_color=None, destination=None, should_execute=False)]
token_seqs = [
    engine.processor.tokenizer.encode(
        json.dumps(g, separators=(",", ":")), add_special_tokens=False
    )
    for g in allowed
]
eos = engine.model.generation_config.eos_token_id
eos = eos[0] if isinstance(eos, list) else eos
prefixes = []
for c in cases:
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": Image.open(D / c["image"]).convert("RGB")},
                {"type": "text", "text": prompt(c)},
            ],
        }
    ]
    prefixes.append(
        engine.processor.apply_chat_template(
            msgs,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to("cuda")
    )


def call(i, limit, grammar):
    inputs = prefixes[i]
    n = inputs["input_ids"].shape[-1]

    def constraint(batchid, ids):
        suffix = ids[n:].tolist()
        next_ids = {
            s[len(suffix)]
            for s in token_seqs
            if s[: len(suffix)] == suffix and len(s) > len(suffix)
        }
        if any(s == suffix for s in token_seqs):
            next_ids.add(eos)
        return sorted(next_ids) or [eos]

    torch.cuda.synchronize()
    tick = time.perf_counter()
    with torch.inference_mode():
        out = engine.model.generate(
            **inputs,
            max_new_tokens=limit,
            do_sample=False,
            **({"prefix_allowed_tokens_fn": constraint} if grammar else {})
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - tick
    new = out[0][n:]
    raw = engine.processor.decode(new, skip_special_tokens=True)
    try:
        g = json.loads(raw)
        valid = (
            set(g) == {"target_color", "destination", "should_execute"}
            and type(g["should_execute"]) is bool
        )
        correct = valid and all(g[k] == cases[i][k] for k in g)
    except Exception:
        valid = False
        correct = False
    return dict(
        case_id=cases[i]["id"],
        limit=limit,
        grammar=grammar,
        seconds=elapsed,
        tokens=len(new),
        truncated=int(new[-1]) != eos,
        valid=valid,
        correct=correct,
        raw=raw,
    )


for _ in range(3):
    call(0, 128, False)
jobs = list(
    itertools.product(
        range(len(cases)),
        range(2),
        [(64, False), (128, False), (256, False), (64, True)],
    )
)
random.Random(20261020).shuffle(jobs)
rows = []
for i, rep, (limit, grammar) in jobs:
    row = call(i, limit, grammar)
    row["repeat"] = rep
    rows.append(row)
    with (O / "predictions.jsonl").open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
summary = []
for limit, grammar in [(64, False), (128, False), (256, False), (64, True)]:
    r = [x for x in rows if x["limit"] == limit and x["grammar"] == grammar]
    summary.append(
        dict(
            limit=limit,
            grammar=grammar,
            cases=len(r),
            median_seconds=float(np.median([x["seconds"] for x in r])),
            p95_seconds=float(np.percentile([x["seconds"] for x in r], 95)),
            mean_tokens=float(np.mean([x["tokens"] for x in r])),
            schema_valid=sum(x["valid"] for x in r),
            correct=sum(x["correct"] for x in r),
            truncated=sum(x["truncated"] for x in r),
        )
    )
(O / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
