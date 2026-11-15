"""Export the completed paired simulation without replacing earlier reports."""

from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import zipfile
from markdown_it import MarkdownIt
from pypdf import PdfReader
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[1]
V = R.parent
ROOT = V.parents[1]
OUT = ROOT / "deliverables/20261112_阿里_检查器闭环强化"
NAMES = [
    "双罐入篮",
    "芝士黄油",
    "开炉放壶",
    "碗入抽屉",
    "双杯放盘",
    "书入收纳",
    "杯与布丁",
    "汤罐芝士",
    "双壶上炉",
    "杯入微波炉",
]


def meanstr(x, scale=1):
    return "无可比样本" if x is None else f"{x * scale:.2f}"


def main():
    m = json.loads((R / "results/exploratory_metrics.json").read_text())
    assert m["base"]["n"] == m["v2"]["n"] == m["paired"]["n"] == 30, (
        "All 60 episodes required"
    )
    assert (
        m["pair_integrity"]["same_plan_and_seed"]
        and m["pair_integrity"]["max_prefix_action_error"] < 1e-6
    )
    rows = json.loads((R / "results/exploratory_episode_metrics.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    b, a = m["base"], m["v2"]
    sb, sa = m["transport_scope"]["base"], m["transport_scope"]["v2"]
    d = m["paired_correct_delays"]
    tasks = []
    for t in range(10):
        counts = [
            sum(x["success"] for x in rows[mode] if x["task"] == t)
            for mode in ["base", "v2"]
        ]
        tasks.append(f"| t{t} {NAMES[t]} | {counts[0]}/3 | {counts[1]}/3 |")
    plt.rcParams["font.family"] = "Noto Sans CJK JP"
    fig, ax = plt.subplots(figsize=(10, 3.6))
    x = list(range(10))
    width = 0.35
    for j, (mode, label, color) in enumerate(
        [("base", "基座检查器", "#577aaf"), ("v2", "V2 分类检查器", "#e19a4e")]
    ):
        ax.bar(
            [k + (j - 0.5) * width for k in x],
            [sum(r["success"] for r in rows[mode] if r["task"] == t) for t in x],
            width,
            label=label,
            color=color,
        )
    ax.set_xticks(x, [f"t{t}" for t in x])
    ax.set_ylim(0, 3.5)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_ylabel("成功回合数 / 3")
    ax.legend(ncol=2, loc="upper center")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(R / "figures/exploratory_success.png", dpi=160)
    plt.close(fig)
    status = (
        "保持了本批成功率，但未显示成功率提升"
        if a["success"] == b["success"]
        else (
            "本批成功数增加，仍需更多初态复核"
            if a["success"] > b["success"]
            else "本批成功数下降，暂不替换原检查器"
        )
    )
    report = f"""# Qwen 完成检查器：配对闭环强化报告

## 摘要

本轮在已有 V2 权重上完成 60 个 LIBERO-Long 仿真回合：同样的 10 个任务、每任务 3 个初态，分别使用基座检查器和 V2 分类检查器。没有重新训练，也没有调整阈值。基座成功 **{b["success"]}/30**，V2 成功 **{a["success"]}/30**；按官方任务关系核查的错误确认切换分别为 **{b["false_switches"]} / {a["false_switches"]} 次**（基座 / V2）。结论是：{status}。

V2 适用的 21 个搬运回合，两组均为 21/21、零误切换，原检查器已没有可供本轮降低的错误计数。全体唯一错误确认来自开炉步骤，仍由基座回退执行。V2 的达成未确认计数更多，没有显示检查召回在闭环中改善。

V2 适用的搬运任务中，平均单次检查耗时由 {meanstr(sb["mean_check_wall_seconds"])} 秒变为 {meanstr(sa["mean_check_wall_seconds"])} 秒。这里比较的是“原生成式检查”与“V2 提示词＋LoRA＋分类阈值”整个检查方案，速度变化不能全部归因于 LoRA。

## 1. 为什么继续跑闭环

此前独立离线测试中，V2 将完成召回从 2.9% 提高到 26.7%，误报由 0.6% 增至 6.7%，未达到预设 5% 上限。本轮保留这一结论，但将它与探索性仿真分开：实际闭环还有夹爪门控、连续两次确认和低层策略行为，离线误报不能直接换算为任务失败。

离线标签要求到位、支撑接触且已释放，闭环则保留 LIBERO 官方成功检查。两套标准不同。本轮不改离线标签、不重新选阈值，也不把原测试重新评分后当成新测试。

## 2. 固定的对照条件

- 低层均为同一个官方 π0.5 LIBERO 检查点，10 步去噪，每次执行动作块的前 5 步。
- 固定 v3 已记录的英文子目标及中文整句指令；每回合固定初态、环境种子和动作采样噪声，不重新生成计划。
- 第三人称 512 像素开始帧与当前帧；夹爪张开时检查，连续两次确认才切换。
- 每子目标预算 300 步，整回合最多 520 步；末子目标由官方成功或步数上限结束。超预算推进与视觉确认分别计数。
- V2 使用语言层已冻结适配器，log-odds 阈值 3.2480709186546264。门、炉开关和独立拿起等未覆盖步骤保留基座回退。
- 两组按任务与初态交替先后运行。仿真真值仅保存作评价，不参与检查或控制。

30 对计划和种子一致，首次切换前动作最大差为 {m["pair_integrity"]["max_prefix_action_error"]:.2g}；其中 {m["pair_integrity"]["full_identical_action_pairs"]} 对全程动作完全相同。本轮不是实体机械臂实验。

## 3. 闭环结果

| 指标 | 基座检查器 | V2 分类检查器 |
|---|---:|---:|
| 官方成功 | {b["success"]}/30 | {a["success"]}/30 |
| 超时回合 | {b["timeouts"]} | {a["timeouts"]} |
| 视觉确认推进 | {b["confirmed"]} | {a["confirmed"]} |
| 可核查的正确确认 | {b["correct_switches"]} | {a["correct_switches"]} |
| 错误确认切换 | {b["false_switches"]} | {a["false_switches"]} |
| 无对应最终谓词的确认 | {b["unscored_switches"]} | {a["unscored_switches"]} |
| 超预算推进 | {b["budget_advances"]} | {a["budget_advances"]} |
| 达成但未确认的回合/阶段 | {b["missed_confirmation"]} | {a["missed_confirmation"]} |
| 成功但未发生子目标切换 | {b["success_without_transition"]} | {a["success_without_transition"]} |
| 全部检查次数 | {b["checks"]} | {a["checks"]} |

达成但未确认包含最终任务已成功的情况，因此不等于任务失败。独立拿起书本没有对应的最终任务谓词，不能用“书已进收纳盒”替代拿起动作的真值；这类确认不进入错误切换计数。两只外观相同的壶，首阶段按任意一只已上炉评价。

仅统计正确确认的平均延迟分别为 {meanstr(b["mean_correct_switch_delay_steps"], 0.05)} / {meanstr(a["mean_correct_switch_delay_steps"], 0.05)} 秒仿真时间，样本数为 {b["delay_sample_count"]} / {a["delay_sample_count"]}。两组都正确确认的配对回合共 {d["n"]} 个，其平均延迟为 {meanstr(d["base_mean_steps"], 0.05)} / {meanstr(d["v2_mean_steps"], 0.05)} 秒，V2 减基座为 {meanstr(d["mean_v2_minus_base_steps"], 0.05)} 秒。未确认项单列，避免只统计快速成功的案例。

配对成功结果：仅基座成功 {m["paired"]["base_only_success"]} 回合，仅 V2 成功 {m["paired"]["v2_only_success"]} 回合；精确 McNemar 双侧 p={m["paired"]["mcnemar_exact_two_sided_p"]:.3g}。每任务仅 3 个初态，不据此作广泛泛化结论。

| 任务 | 基座 | V2 |
|---|---:|---:|
{chr(10).join(tasks)}

![各任务配对成功回合数](figures/exploratory_success.png)

## 4. 这轮验证了什么

第一，闭环可以在离线门槛未通过时作为探索继续运行，但两种结论应分别保留。本轮既没有证明单次误报会直接造成闭环失败，也没有证明离线召回提高就能改善任务成功。

第二，一部分回合中，π0.5 在首个子目标尚未被确认时就完成了整项任务。这说明当前基准上的低层行为可能覆盖比单条子目标更长的动作序列。单看任务成功率不足以证明高层完成检查有效，因此本轮补充了确认次数、未确认情况和全程动作相同比例。

第三，V2 的主要可测变化是检查方式和运行耗时；是否值得部署仍取决于对具体错误的改善，而不是权重已经训练出来。未覆盖操作的基座回退也必须单列，不能把它们当成 LoRA 已解决的类别。

## 5. 误报标签审核与后续取舍

原离线测试 12 个误报中，7 个是近接触、2 个明显偏移、2 个边缘外侧、1 个夹持未放；其中 9 个标签为 unknown，5 个仿真任务关系已成立但未满足更严格的支撑/释放要求。这是评分后的语义审核，不用于修改原阈值或擦除失败结果。

当前不再盲目扩大到相似的数千张静态图。若继续训练，应先针对确实影响闭环的错误补充自然过程和可辨识的困难对照；腕部相机先检查可见性再决定加入。完成检查器这一阶段按本轮结果收尾，后续优先补同一个物理场景中的导航与抓取，沿用已有检查与记录接口。

## 6. 代码与复核

```text
experiments/vla_libero/checker_v2/
  EXPLORATORY_PROTOCOL.json         模型、阈值与固定回合
  EXPLORATORY_AMENDMENT.json        探索性仿真与原验收分开
  EXPLORATORY_METRICS_PROTOCOL.json 指标定义与谓词映射
  scripts/evaluate_loop.py          配对执行入口
  scripts/summarize_loop.py         逐回合指标与动作一致性
  scripts/test_loop_metrics.py      时间对齐及切换统计测试
  results/exploratory_metrics.json  汇总
  results/exploratory_metrics.csv   逐回合表格
  results/exploratory_episode_metrics.json
  results/exploratory_pair_integrity.json
experiments/vla_libero/hierarchical/
  checker_v2_exploratory_base/libero_10/taskXX_initYY/
  checker_v2_exploratory_v2/libero_10/taskXX_initYY/
    result.json / trace.npz / rollout.mp4 / qwen_inputs/
```

相关执行代码在 scripts/evaluate_loop.py；相关指标代码在 scripts/summarize_loop.py；检查器运行封装在 experiments/vla_libero/scripts/completion_checker_v2.py。完整 60 个回合的结果、视频和模型输入保存在本地，交付目录附代表视频，源码包不包含基座和训练图像。

```bash
# 已存在结果会复用，用于中断续跑；不覆盖原回合。
experiments/vla_libero/scripts/python_shared experiments/vla_libero/checker_v2/scripts/evaluate_loop.py --exploratory
.venvs/openpi-libero/bin/python experiments/vla_libero/checker_v2/scripts/summarize_loop.py
.venvs/openpi-libero/bin/python -m unittest discover -s experiments/vla_libero/checker_v2/scripts -p 'test_*.py'
```
"""
    (R / "EXPLORATORY_REPORT.md").write_text(report)
    html = R / "EXPLORATORY_REPORT.html"
    css = """@page{size:A4;margin:16mm}body{font-family:"Noto Sans CJK SC",sans-serif;font-size:10pt;line-height:1.55;color:#243448}h1{font-size:20pt}h2{font-size:14pt;break-after:avoid}table{width:100%;border-collapse:collapse;font-size:9pt}td,th{border:1px solid #cbd5df;padding:4px}tr{break-inside:avoid}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f5f8;padding:8px;font-size:8pt}img{width:100%;max-height:85mm}p:has(img){break-inside:avoid}"""
    html.write_text(
        '<meta charset="utf-8"><style>'
        + css
        + "</style>"
        + MarkdownIt().enable("table").render(report)
    )
    pdf = OUT / "Qwen完成检查器_配对闭环强化报告.pdf"
    subprocess.run(
        [
            "/usr/bin/google-chrome",
            "--headless",
            "--no-sandbox",
            "--disable-gpu",
            "--no-pdf-header-footer",
            "--print-to-pdf=" + str(pdf),
            html.as_uri(),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    shutil.copy2(R / "EXPLORATORY_REPORT.md", OUT / "进度报告.md")
    files = (
        list((R / "scripts").glob("*.py"))
        + list(R.glob("*PROTOCOL.json"))
        + [
            R / "EXPLORATORY_AMENDMENT.json",
            R / "EXPLORATORY_REPORT.md",
            R / "EXPLORATORY_VERIFICATION.json",
            R / "README.md",
            R / "data/cases.json",
            R / "data/audit.json",
        ]
    )
    files += list((R / "results").glob("exploratory*")) + [
        R / "results/test_label_semantics_audit.json",
        R / "results/model/summary.json",
        R / "results/model/training_protocol.json",
    ]
    files += [
        V / "scripts" / n
        for n in [
            "run_hierarchical.py",
            "qwen_planner.py",
            "completion_checker_v2.py",
            "run_hierarchical_recovery.py",
            "qwen_planner_recovery.py",
            "subgoal_state.py",
            "evaluate.py",
            "python_shared",
        ]
    ]
    files += [
        ROOT / "perception/qwen_scene" / n for n in ["engine.py", "worker_client.py"]
    ]
    files += [
        V / "requirements.lock.txt",
        V / "environment.json",
        V / "configs/libero_long_tasks.json",
        V / "configs/libero/config.yaml",
        V / "README.md",
        V / "third_party/sources.json",
        V / "scripts/fetch_sources.py",
        V / "scripts/install.py",
        V / "scripts/install_simulation.py",
    ]
    files += sorted(
        (V / "hierarchical/zh_eval_hier_v3/libero_10").glob("*/result.json")
    )
    (OUT / "figures").mkdir(exist_ok=True)
    shutil.copy2(
        R / "figures/exploratory_success.png", OUT / "figures/exploratory_success.png"
    )
    with zipfile.ZipFile(
        OUT / "完成检查器_闭环源码.zip", "w", zipfile.ZIP_DEFLATED
    ) as z:
        for p in files:
            if p.is_file():
                z.write(p, p.relative_to(ROOT))
    # Include successful execution plus a counterpart with different switching behaviour.
    candidates = []
    for x in rows["v2"]:
        y = next(
            y for y in rows["base"] if (x["task"], x["init"]) == (y["task"], y["init"])
        )
        candidates.append(
            (
                x["success"] and y["success"] and x["confirmed"] != y["confirmed"],
                x["task"],
                x["init"],
            )
        )
    selected = max(candidates)
    for mode in ["base", "v2"]:
        source = (
            V
            / f"hierarchical/checker_v2_exploratory_{mode}/libero_10/task{selected[1]:02d}_init{selected[2]:02d}/rollout.mp4"
        )
        shutil.copy2(source, OUT / f"任务{selected[1]}_初态{selected[2]}_{mode}.mp4")
    (OUT / "视频说明.txt").write_text(
        f"两段来自相同任务{selected[1]}、初态{selected[2]}。base为原生成式检查器，v2为分类检查器。视频按20fps仿真时间播放，不反映模型推理的墙钟等待时间。切换记录在本地对应result.json中；画面本身不能代替官方成功谓词。\n"
    )
    manifest = {
        p.name: {
            "bytes": p.stat().st_size,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        }
        for p in OUT.iterdir()
        if p.is_file() and p.name != "manifest.json"
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)
    )
    print(OUT)
    print("pages", len(PdfReader(pdf).pages))


if __name__ == "__main__":
    main()
