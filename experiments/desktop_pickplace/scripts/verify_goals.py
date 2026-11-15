from pathlib import Path
import json, re

ROOT = Path(__file__).resolve().parents[1]
rows = [
    json.loads(x)
    for x in (ROOT / "runs/lora_paraphrases/predictions.jsonl").read_text().splitlines()
]
wrong = [
    r
    for r in rows
    if r.get("goal", {}).get("should_execute") and not r["case"]["should_execute"]
]
blocked = []
for row in wrong:
    supported = (
        re.fullmatch(
            r"把(红色|蓝色|绿色)方块放进(左|右)边盒子。", row["case"]["instruction"]
        )
        is not None
    )
    assert not supported
    blocked.append(
        {
            "instruction": row["case"]["instruction"],
            "model_proposal": row["goal"],
            "dispatch": "blocked_unresolved_destination",
        }
    )
(ROOT / "runs/ambiguous_dispatch.json").write_text(
    json.dumps({"cases": len(blocked), "rows": blocked}, ensure_ascii=False, indent=2)
)
print("Ambiguous proposals blocked", len(blocked))
