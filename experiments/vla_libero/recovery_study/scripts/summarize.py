"""Aggregate saved results and plot diagnostics; does not launch simulation or inference."""

from pathlib import Path
import hashlib
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from PIL import Image

R = Path(__file__).resolve().parents[1]
V = R.parent
font = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams["axes.unicode_minus"] = False
rows = []
for p in sorted((R / "results").glob("*/*/result.json")):
    r = json.loads(p.read_text())
    r["source"] = str(p.relative_to(R))
    rows.append(r)
assert len(rows) == 28, len(rows)
summary = {}
for mode in ["continue", "full", "replan", "contract", "centering", "typed"]:
    group = [r for r in rows if r["mode"] == mode]
    summary[mode] = {
        "attempts": len(group),
        "executed": sum("steps" in r for r in group),
        "success": sum(r["success"] for r in group),
        "errors": sum(r["status"] == "error" for r in group),
        "steps": sum(r.get("steps", 0) for r in group),
        "checks": sum(sum(e["kind"] == "check" for e in r["events"]) for r in group),
    }
for tid in [2, 4, 8]:
    group = [r for r in rows if r["task"] == tid]
    assert len({r["prefix_state_sha256"] for r in group}) == 1
    assert all(r["prefix_eef_error"] == 0 for r in group)
smokes = [
    json.loads(p.read_text())
    for p in sorted(
        (V / "hierarchical/recovery_contract_smoke/libero_10").glob("*/result.json")
    )
]
assert len(smokes) == 2
out = {
    "modes": summary,
    "smoke": [
        {
            k: r.get(k)
            for k in [
                "task_id",
                "success",
                "status",
                "steps",
                "final_atoms",
                "qwen_checks",
            ]
        }
        for r in smokes
    ],
    "paired_states_identical": True,
    "max_prefix_eef_error": max(r["prefix_eef_error"] for r in rows),
    "unique_failure_states": 3,
    "stochastic_repeats_per_state": 2,
    "recovery_attempts": 26,
    "guided_attempts": 2,
    "smoke_attempts": 2,
    "actual_simulated_steps": sum(r.get("steps", 0) for r in rows)
    + sum(r.get("steps", 0) for r in smokes),
}
(R / "summary.json").write_text(json.dumps(out, indent=2))
# Preserve both original source integrity and the exact archived experiment scripts.
old = json.loads((R / "source_hashes.json").read_text())
for n in [
    "scripts/run_hierarchical.py",
    "scripts/qwen_planner.py",
    "scripts/evaluate.py",
]:
    assert hashlib.sha256((V / n).read_bytes()).hexdigest() == old[n], n
# An already-contacting mug can still fail a stricter spatial predicate.
fig, axs = plt.subplots(
    1, 3, figsize=(13, 4), gridspec_kw={"width_ratios": [1, 1, 0.8]}
)
base = R / "results/task04_init02/full_seed31001/frames"
for ax, name, title in zip(
    axs[:2],
    ["recovery_start.png", "final.png"],
    ["恢复起点：白杯已接触目标盘", "整句重试结束：白杯位置基本未变"],
):
    ax.imshow(Image.open(base / name))
    ax.set_title(title, fontproperties=font, fontsize=11)
    ax.axis("off")
contact = json.loads((R / "plate_contact_diagnostic.json").read_text())
v = [
    contact[x]["predicate_details"][0]["xy_distance_m"] * 1000
    for x in ["prefix", "continue", "full", "replan"]
]
axs[2].bar(range(4), v, color=["#607d8b", "#769db0", "#4c7596", "#304b6b"])
axs[2].axhline(30, color="#cb4b46", ls="--", label="30 mm threshold")
axs[2].set_xticks(range(4), ["起点", "继续", "整句", "重规划"], fontproperties=font)
axs[2].set_ylabel("杯心与盘心水平距离 / mm", fontproperties=font)
axs[2].set_ylim(0, 47)
for i, x in enumerate(v):
    axs[2].text(i, x + 0.7, f"{x:.1f}", ha="center", fontsize=9)
axs[2].legend(fontsize=8)
fig.tight_layout()
fig.savefig(R / "figures/plate_boundary.png", dpi=180)
plt.close(fig)
# Show actual first-seed terminal states, with the common start state in the first column.
fig, axs = plt.subplots(3, 4, figsize=(12, 9))
for i, tid in enumerate([2, 4, 8]):
    for j, mode in enumerate(["start", "continue", "full", "contract"]):
        p = (
            R
            / "results"
            / f"task{tid:02d}_init02"
            / f"{'full' if mode == 'start' else mode}_seed31001"
            / "frames"
            / ("recovery_start.png" if mode == "start" else "final.png")
        )
        axs[i, j].imshow(Image.open(p))
        axs[i, j].axis("off")
        axs[i, j].set_title(
            f"t{tid} · "
            + {
                "start": "相同起点",
                "continue": "继续",
                "full": "整句重试",
                "contract": "约束重规划",
            }[mode],
            fontproperties=font,
            fontsize=10,
        )
fig.tight_layout()
fig.savefig(R / "figures/recovery_outcomes.png", dpi=150)
plt.close(fig)
print(json.dumps(out, indent=2))
