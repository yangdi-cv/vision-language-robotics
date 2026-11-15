from pathlib import Path
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from PIL import Image


def main():
    R = Path(__file__).resolve().parents[1]
    f = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    s = json.loads((R / "results/checker_lora/summary.json").read_text())
    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    for ax, mode, title in zip(axs, ["base", "adapter"], ["Qwen基座", "检查器 LoRA"]):
        m = s["strata"]["core"][mode]
        a = [[m["tn"], m["fp"]], [m["fn"], m["tp"]]]
        ax.imshow(a, cmap="Blues", vmin=0, vmax=121)
        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    str(a[i][j]),
                    ha="center",
                    va="center",
                    fontsize=19,
                    color="white" if a[i][j] > 65 else "#233549",
                )
        ax.set_xticks([0, 1], ["预测未完成", "预测完成"], fontproperties=f)
        ax.set_yticks([0, 1], ["真值未完成", "真值完成"], fontproperties=f)
        ax.set_title(f"{title} · 准确率 {m['accuracy']:.1%}", fontproperties=f)
    fig.tight_layout()
    fig.savefig(R / "figures/checker_core.png", dpi=180)
    plt.close(fig)
    fig, axs = plt.subplots(1, 4, figsize=(12, 3))
    base = R / "results/injection/task00_init02"
    for ax, p, title in zip(
        axs,
        [
            base / "full/before_fault.png",
            base / "full/after_fault.png",
            base / "full/final.png",
            base / "qwen/final.png",
        ],
        ["扰动前", "将一件物体移回桌面", "整句重试：成功", "Qwen重规划：未完成"],
    ):
        ax.imshow(Image.open(p))
        ax.axis("off")
        ax.set_title(title, fontproperties=f, fontsize=10)
    fig.tight_layout()
    fig.savefig(R / "figures/recovery_pair.png", dpi=180)
    plt.close(fig)
    print("figures ready")


if __name__ == "__main__":
    main()
