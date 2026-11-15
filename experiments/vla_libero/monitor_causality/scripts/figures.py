"""Plots and an actual-frame panel for the monitor intervention study."""

import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from interventions import MODES

S = Path(__file__).resolve().parents[1]
LABEL = {
    "live_base": "正常视觉",
    "frozen_base": "冻结旧图",
    "timer150": "预设时刻",
    "live_lora": "V2检查方案",
    "hold_first": "保持首条指令",
}
plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


def main():
    summary = json.loads((S / "SUMMARY.json").read_text())
    assert summary["episodes"] == 150
    dest = S / "figures"
    dest.mkdir(exist_ok=True)
    groups = summary["groups"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    x = np.arange(5)
    labels = [LABEL[m] for m in MODES]
    axes[0].bar(
        x,
        [groups[m]["success"] for m in MODES],
        color=["#397cb1", "#8e9bad", "#d79839", "#4c9c83", "#9271a7"],
    )
    axes[0].set_ylim(0, 32)
    axes[0].set_ylabel("官方成功回合 / 30")
    for i, m in enumerate(MODES):
        axes[0].text(
            i, groups[m]["success"] + 0.4, str(groups[m]["success"]), ha="center"
        )
    axes[1].bar(
        x, [groups[m]["success_without_switch"] for m in MODES], color="#4f7e9b"
    )
    axes[1].set_ylim(0, 32)
    axes[1].set_ylabel("成功且未切换子目标的回合")
    for ax in axes:
        ax.set_xticks(x, labels, rotation=15)
        ax.grid(axis="y", alpha=0.2)
    fig.savefig(dest / "outcomes.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    axes[0].bar(
        x, [groups[m]["false_switches"] for m in MODES], label="完成判断触发的错误切换"
    )
    axes[0].bar(
        x,
        [groups[m]["false_budget_switches"] for m in MODES],
        bottom=[groups[m]["false_switches"] for m in MODES],
        label="预算到期时目标仍未完成",
    )
    axes[0].set_ylabel("次数")
    axes[0].legend(fontsize=8)
    axes[1].bar(x, [groups[m]["actual_model_calls"] for m in MODES], color="#4c9c83")
    axes[1].set_ylabel("实际Qwen检查调用次数")
    for ax in axes:
        ax.set_xticks(x, labels, rotation=15)
        ax.grid(axis="y", alpha=0.2)
    fig.savefig(dest / "decisions_cost.png", dpi=160)
    plt.close(fig)
    # Same physical frame before the earliest switch; the frozen checker sees only the reference.
    path = S / "episodes/primary/frozen_base/libero_10/task00_init00/qwen_inputs"
    ref = path / "ref_step0000.png"
    current = path / "check_step0140.png"
    canvas = Image.new("RGB", (1440, 620), "#eef3f7")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(FONT, 25)
    for i, (image, title) in enumerate(
        [
            (ref, "子目标开始帧"),
            (current, "第140步真实当前帧"),
            (ref, "冻结组送给检查器的当前帧"),
        ]
    ):
        draw.text((15 + 480 * i, 15), title, font=font, fill="#17354c")
        with Image.open(image) as im:
            canvas.paste(im.convert("RGB").resize((460, 460)), (10 + 480 * i, 65))
    draw.text(
        (20, 545),
        "只替换检查输入；π0.5仍看真实当前画面。冻结组也可能在不切换指令时完成整项任务。",
        font=font,
        fill="#17354c",
    )
    canvas.save(dest / "image_intervention.png")


if __name__ == "__main__":
    main()
