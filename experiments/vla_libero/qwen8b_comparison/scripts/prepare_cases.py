"""Freeze eight diagnostic pairs before loading the 8B model."""

from pathlib import Path
import hashlib, json
from PIL import Image, ImageDraw, ImageFont

R = Path(__file__).resolve().parents[1]
ROOT = R.parents[2]
VLA = R.parent
specs = [
    ("wrong_plate", "zh_eval_hier_v2", 4, 2, 120),
    ("false_done_mug", "zh_eval_hier_v2", 4, 2, 160),
    ("plate_viewpoint", "zh_eval_hier_v3", 4, 2, 120),
    ("basket_occlusion", "zh_eval_hier_v3", 1, 0, 160),
    ("soup_done", "zh_eval_hier_v3", 0, 0, 140),
    ("soup_not_done", "zh_eval_hier_v3", 0, 0, 40),
    ("false_done_pot", "zh_eval_hier_v2", 8, 2, 100),
    ("microwave_done", "zh_eval_hier_v3", 9, 0, 180),
]
rows = []
for name, tag, tid, init, step in specs:
    p = VLA / "hierarchical" / tag / "libero_10" / f"task{tid:02d}_init{init:02d}"
    d = json.loads((p / "result.json").read_text())
    event = next(e for e in d["events"] if e["kind"] == "check" and e["step"] == step)
    assert event["index"] == 0
    sub = next(
        e["subgoal"]
        for e in d["events"]
        if e["kind"] == "subgoal_start" and e["index"] == 0
    )
    images = [
        p / "qwen_inputs/ref_step0000.png",
        p / "qwen_inputs" / f"check_step{step:04d}.png",
    ]
    assert all(Image.open(v).size == (512, 512) for v in images)
    rows.append(
        {
            "id": name,
            "source_tag": tag,
            "task_id": tid,
            "initial_state": init,
            "step": step,
            "subgoal": sub,
            "images": [str(x.relative_to(ROOT)) for x in images],
            "image_sha256": [
                hashlib.sha256(x.read_bytes()).hexdigest() for x in images
            ],
            "expected_done": bool(event["shadow_long_task_atoms"][0]),
            "label_source": "Logged first official goal atom; index0 aligned with first subgoal for these selected tasks only",
            "historical_4b_done": event["done"],
            "historical_reason": event["reason"],
        }
    )
(R / "cases.json").write_text(
    json.dumps(
        {
            "selection": "Eight targeted diagnostic pairs, 4 positive and 4 negative; not representative benchmark; same existing before/after prompts for both models",
            "cases": rows,
        },
        indent=2,
    )
)
font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 20)
canvas = Image.new("RGB", (1060, 8 * 315), "#edf2f7")
draw = ImageDraw.Draw(canvas)
for j, row in enumerate(rows):
    draw.text(
        (12, j * 315 + 6),
        f"{row['id']} | task {row['task_id']} step {row['step']} | predicate={row['expected_done']}",
        font=font,
        fill="#21384e",
    )
    for k, x in enumerate(row["images"]):
        canvas.paste(
            Image.open(ROOT / x).convert("RGB").resize((290, 290)),
            (k * 510 + 20, j * 315 + 25),
        )
canvas.save(R / "cases_contact_sheet.png")
print("Frozen", len(rows), "cases; positive", sum(r["expected_done"] for r in rows))
