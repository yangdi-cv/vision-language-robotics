from pathlib import Path
import json, numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
fig = ROOT / "figures"
fig.mkdir(exist_ok=True)


def read(name):
    return json.loads((ROOT / name).read_text())


results = read("runs/batch002/results.json")
rows = [r for r in results if r["group"] == "rgbd"]
stats = {
    "rgbd_mean_position_error_mm": float(
        np.mean([r["localization_error_m"] for r in rows]) * 1000
    ),
    "rgbd_max_position_error_mm": float(
        np.max([r["localization_error_m"] for r in rows]) * 1000
    ),
    "max_distractor_displacement_mm": float(
        max(max(r["distractor_displacement_m"].values()) for r in results) * 1000
    ),
    "demonstration_splits": {
        s: sum(
            i["split"] == s for i in read("data/demonstrations/manifest.json")["items"]
        )
        for s in ("train", "validation", "test")
    },
}
base = [
    json.loads(x)
    for x in (ROOT / "runs/qwen_goals/predictions.jsonl").read_text().splitlines()
]
adapt = [
    json.loads(x)
    for x in (ROOT / "runs/lora_heldout/predictions.jsonl").read_text().splitlines()
]
stats["goal_splits"] = {}
for name, lo, hi in [("validation", 406, 408), ("test", 408, 410)]:
    stats["goal_splits"][name] = {}
    for label, v in [("base", base), ("lora", adapt)]:
        selected = [x for x in v if lo <= int(x["case"]["id"].split("_")[0]) < hi]
        stats["goal_splits"][name][label] = {
            "cases": len(selected),
            "correct": sum(x["correct"] for x in selected),
        }
(ROOT / "runs/analysis.json").write_text(json.dumps(stats, indent=2))
f, ax = plt.subplots(figsize=(8, 4))
ax.bar(
    ["Slanted/oracle", "Slanted/RGB-D", "Vertical/oracle", "Vertical/RGB-D"],
    [24 / 30 * 100, 24 / 30 * 100, 100, 100],
    color=["#8095ac", "#8095ac", "#398d78", "#398d78"],
)
ax.set_ylim(0, 110)
ax.set_ylabel("Success %")
ax.set_title("Same30 tasks, five shuffled layouts")
ax.tick_params(axis="x", rotation=12)
f.tight_layout()
f.savefig(fig / "physical_baseline.png", dpi=160)
plt.close(f)
f, ax = plt.subplots(figsize=(8, 4))
labels = [
    "Original prompts/base",
    "Original prompts/LoRA",
    "New wording/base",
    "New wording/LoRA",
]
values = [35 / 40, 40 / 40, 32 / 44, 40 / 44]
ax.bar(
    labels, np.array(values) * 100, color=["#8095ac", "#398d78", "#8095ac", "#398d78"]
)
ax.set_ylabel("Exact goal match %")
ax.set_ylim(0, 110)
ax.tick_params(axis="x", rotation=12)
f.tight_layout()
f.savefig(fig / "goal_comparison.png", dpi=160)
plt.close(f)
folder = ROOT / "runs/integrated_lora_test/408_blue_left"
sheet = Image.new("RGB", (960, 480), "white")
for index, frame in enumerate(
    [
        "initial.png",
        "step_190.png",
        "step_280.png",
        "step_390.png",
        "step_530.png",
        "step_700.png",
    ]
):
    im = Image.open(folder / frame).resize((320, 240))
    ImageDraw.Draw(im).text((8, 8), frame, fill="yellow")
    sheet.paste(im, ((index % 3) * 320, (index // 3) * 240))
sheet.save(fig / "pickplace_sequence.jpg")
print(json.dumps(stats, indent=2))
