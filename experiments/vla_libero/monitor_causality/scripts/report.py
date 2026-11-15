"""Chinese mechanism report and source-only archive; requires completed verification."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile
from markdown_it import MarkdownIt
from interventions import MODES
from figures import LABEL

S = Path(__file__).resolve().parents[1]
V = S.parent
ROOT = V.parents[1]


def main():
    summary = json.loads((S / "SUMMARY.json").read_text())
    verification = json.loads((S / "VERIFICATION.json").read_text())
    protocol = json.loads((S / "PROTOCOL.json").read_text())
    data = json.loads((S / "decision_data/summary.json").read_text())
    video = json.loads((S / "VIDEO.json").read_text())
    assert summary["episodes"] == 150 and verification["all_complete"]
    groups = summary["groups"]
    rows = summary["rows"]
    base = groups["live_base"]
    frozen = groups["frozen_base"]
    hold = groups["hold_first"]

    def table(header, rows):
        return "\n".join(
            [
                "| " + " | ".join(header) + " |",
                "|" + "|".join(["---"] * len(header)) + "|",
                *["| " + " | ".join(map(str, row)) + " |" for row in rows],
            ]
        )

    outcomes = [
        [
            LABEL[mode],
            f"{groups[mode]['success']}/30",
            groups[mode]["success_without_switch"],
            groups[mode]["actual_model_calls"],
            groups[mode]["switches"],
        ]
        for mode in MODES
    ]
    bytask = []
    for task in range(10):
        bytask.append(
            [
                task,
                *[
                    f"{sum(r['success'] for r in rows if r['task'] == task and r['mode'] == mode)}/3"
                    for mode in MODES
                ],
            ]
        )
    task_descriptions = [
        [task, protocol["plans"][f"{task:02d}_00"]["instruction"]]
        for task in range(10)
    ]
    decisions = []
    for mode in MODES:
        r = groups[mode]
        delay = r["mean_correct_switch_delay_steps"]
        seconds = r["mean_model_check_seconds"]
        decisions.append(
            [
                LABEL[mode],
                r["false_switches"],
                r["false_budget_switches"],
                r["unscorable_switches"],
                "-" if delay is None else f"{delay:.1f}",
                "-" if seconds is None else f"{seconds:.3f}",
            ]
        )
    paired = [
        [
            LABEL[r["right"]],
            r["n"],
            r["gains"],
            r["losses"],
            f"{r['nominal_mcnemar_p']:.3f}",
            "-" if r["mean_step_delta_on_joint_success"] is None else f"{r['mean_step_delta_on_joint_success']:+.1f}",
        ]
        for r in summary["paired"]
    ]
    action_comparisons = [
        [LABEL[mode], sum(r["identical_full_actions"] for r in summary["full_action_comparisons"] if r["right"] == mode), 30]
        for mode in MODES[1:]
    ]
    contrasting = []
    for task in range(10):
        for initial in range(3):
            selected = {r["mode"]: r for r in rows if r["task"] == task and r["initial"] == initial}
            if len({r["success"] for r in selected.values()}) > 1:
                contrasting.append([f"t{task} / 初态{initial}", *[f"{'成功' if selected[m]['success'] else '失败'} · {selected[m]['steps']}步" for m in MODES]])
    contrast_text = table(["条件", *[LABEL[m] for m in MODES]], contrasting) if contrasting else "本轮30个配对条件中，五种方法的成功/失败结局逐项一致。"
    text = f"""# 视觉阶段检查对机器人执行的实际作用

2026年11月13日 · π0.5 / Qwen / LIBERO-Long 仿真阶段报告

## 摘要

本轮检验的是固定分解之后的视觉完成检查，而不是重新训练模型。保持两条英文子目标、初态、动作噪声、低层输入和执行预算，比较正常视觉、冻结旧图像、预设时刻切换、已有V2检查方案，以及始终执行第一条子目标，共30个初态条件×5组=150个回合。终局评分使用原LIBERO官方长任务成功检查。

正常视觉成功{base["success"]}/30，冻结旧图像{frozen["success"]}/30，保持第一条指令{hold["success"]}/30。正常视觉有{base["success_without_switch"]}个成功回合没有阶段切换；保持首条指令组的{hold["success"]}个成功回合全部未切换。这些具体成功不能都归因于高层阶段推进，低层策略本身可能已经覆盖完整任务。

本轮没有训练新LoRA或更换模型，没有把旧检查器的离线未通过结论改成通过。额外整理{data["unique_pairs"]}组去重后的决策画面对，官方关系标签与物理支撑/人眼可判定性分开；只作为开发采集，不当成新独立测试集。

![图1 五组干预的官方成功及无切换成功回合数。](figures/outcomes.png)

## 目录

1. 问题与已有基础
2. 配对协议与五组干预
3. 官方结果与任务差异
4. 判断、切换与低层行为
5. 实际决策时刻的数据
6. 独立核验与限制
7. 结论与下一阶段
8. 代码目录与复现

## 1. 问题与已有基础

原π0.5在LIBERO完成117/120次闭环；已有中文Long对照中，Qwen分层28/30、仅翻译29/30。已有检查器V2在探索性配对闭环与基座均28/30，单次检查更快，但没有改善成功率。这些结果说明系统运行正常，尚不能证明视觉阶段检查带来了额外成功。

本轮的问题是：在不改动低层策略与计划的情况下，拿走当前视觉或取消阶段推进，任务是否仍能完成？若能，下一步需要设计真正需要高层介入的任务，而不是只在相同成功率上继续优化检查器。

既有数字只作背景，不与本轮回合混算。当前30个初态是已经研究过的Long任务0–9、初态0–2，属于机制诊断，不是新的泛化评测。

## 2. 配对协议与五组干预

| 组别 | 检查器可见内容 | 切换规则 |
|---|---|---|
| 正常视觉 | 子目标开始帧＋真实当前帧 | 基座Qwen判断、释放门控、连续两票 |
| 冻结旧图 | 开始帧＋同一开始帧 | 相同判断与门控；仅替换检查图像 |
| 预设时刻 | 不调用Qwen | 150步后给完成票，仍需相同门控和两票 |
| V2检查方案 | 开始帧＋真实当前帧 | 原LoRA＋提示词＋分类阈值；不支持的操作回退基座 |
| 保持首条指令 | 不调用Qwen | 不推进阶段，关闭300步阶段预算；直到官方成功或520步 |

前四组都保留300步阶段预算，超时推进与判断完成分别计数。预设150步是统一事前参数，不按任务调节；实际切换仍等待释放门控和两次查询，因此不一定在150步发生。保持首条指令是明确不同的机制控制，不能称为把完整长任务直接交给策略的flat基线。

所有方法都使用同一段冻结的历史Qwen计划，本轮不再调用规划器生成新计划。因此不能把结果解释为中文理解、翻译或分解质量对照。中文整句保留在记录中，低层实际接收的是相同英文子目标。

策略使用官方π0.5检查点，每次预测10步、执行前5步再观察；显式输入相同种子的噪声。Qwen3-VL-4B和V2适配器在全部方法运行期间共驻，基座生成时关闭适配器。V2对照是提示词、LoRA和评分阈值整个方案，不单独归因为LoRA。

![图2 冻结干预只改变检查输入。低层仍读取真实当前画面。](figures/image_intervention.png)

相关代码在 `monitor_causality/scripts/interventions.py`、`run_study.py` 和 `runner_snapshot.py`。旧入口和历史结果没有改动。

## 3. 官方结果与任务差异

{table(["方法", "官方成功", "成功且未切换", "实际Qwen检查调用", "阶段切换总数"], outcomes)}

{table(["任务", "正常视觉", "冻结旧图", "预设时刻", "V2检查", "保持首条"], bytask)}

任务编号对应如下，完整英文子目标逐初态保留在PROTOCOL.json：

{table(["任务", "长任务指令"], task_descriptions)}

同一初态中，以正常视觉为左侧参照，另一方法增加/丢失的成功如下：

{table(["对照方法", "配对数", "增加成功", "丢失成功", "名义精确McNemar p", "共同成功时平均步数差"], paired)}

p值仅描述配对不一致项；每任务只有3个初态，任务内相关，且同时比较多个方法，不能据此宣称一般等效或显著优越。步数差为对照方法减正常视觉，只比较双方都成功的回合，不是墙钟耗时。相同成功数也可能对应不同回合或不同动作轨迹。

不同结局的配对条件（步数只描述该回合，不用失败提前结束冒充速度提升）：

{contrast_text}

## 4. 判断、切换与低层行为

### 4.1 不把预算推进当作视觉确认

错误切换按切换当时的官方中间任务关系判断，预算到期的错误推进另计。任务5的第一步是独立拿书，最终任务谓词不能直接评估“已拿起”，因此标为不可评分，不强行算对或错。

{table(["方法", "完成票触发的错误切换", "预算推进时未完成", "不可评分切换", "正确切换平均延迟（步）", "平均模型检查耗时（s）"], decisions)}

正确切换延迟从切换前连续保持为真的最近起点计算。曾经达成又丢失的状态不会被算作正确切换；平均值只覆盖实际正确切换，未确认状态单独记录，不能用较少确认造成的低延迟“证明更好”。计时组的完成票是规则决定，不是视觉判断。

这里的两票是连续两次通过门控的检查，不是相邻视频帧。例如t2初态1在第40步被Qwen判断为已开炉，而官方关系仍为假；下一次符合门控的检查已到第120步，此时炉子确实开启，第二票才触发推进。中间80步没有新检查，旧规则也没有票据过期时间。本轮保留该规则，因此不能把两票解释为两份独立判断。

![图3 切换错误与实际Qwen调用成本。](figures/decisions_cost.png)

### 4.2 检查无贡献与策略不看视觉是两件事

冻结组只拿走Qwen的当前检查画面，π0.5仍以当前RGB、腕部图和机器人状态闭环预测动作。因而冻结组成功不能解释为低层不用视觉，也不表示VLA是动作回放。

保持首条指令组的{hold["success"]}/30成功说明：至少在这些初态中，推进到第二条指令不是达到官方长任务目标的必要条件。可能原因包括低层对已知场景的整任务先验、子目标指令本身覆盖更长行为等；本轮不能仅凭成功率确定模型内部原因。

相反，某些回合如需切换才能成功，也应逐项定位，不能被总体计数掩盖。全量逐回合结果、动作与检查事件保留，便于分析不同任务。

再按每个动作维度逐元素比较整段策略输出，下面统计与正常视觉组完全一致的回合：

{table(["对照", "整段执行动作完全相同", "配对总数"], action_comparisons)}

这把“官方结果一样”与“实际行为一样”分开。动作不同而同样成功，说明切换影响了行为，但尚未增加这些条件下的成功；整段动作相同，则本轮该配对没有产生执行层差异。

## 5. 实际决策时刻的数据

整理{data["calls"]}次实际检查机会，按参考RGB、当前RGB及指令去重后为{data["unique_pairs"]}对：官方关系为真{data["positive"]}、为假{data["negative"]}、不可评分{data["unscorable"]}，同像素标签冲突{data["label_conflicts"]}对。

这与原先从任意时刻或成功末帧抽样不同，覆盖了真实释放门控之后系统准备作判断的状态。但尚未完成可见性审核，不能自动把仿真关系为真解释成人眼可以看清，也不能把它当成严格支撑、无夹持的物理标签。

索引保留原始参考/当前图、实际送入检查器的图、方法、时间步、输出和标签来源。冻结或计时组没有看真实当前图，其预测不能直接当成正常视觉误判来挖训练样本。任务5保持不可评分；严格物理支撑标签为空，可判定性待审。

所有采集仍来自既有初态0–2，划为开发采集，未随机拆成新的“独立测试集”。本轮没有用它选择阈值、训练权重或报告新的离线模型提升。

相关代码在 `collect_queries.py`，数据索引在 `decision_data/cases.json`。

## 6. 独立核验与限制

{table(["核验项目", "结果"], [["完整回合", verification["episodes_verified"]], ["完整解码视频帧", verification["frames_decoded"]], ["策略输入/噪声/动作块查询", verification["queries_verified"]], ["检查机会", verification["decisions_verified"]], ["切换前跨方法配对", verification["prefix_pairs"]], ["切换前最大动作差", verification["max_prefix_action_difference"]], ["冻结组实际替换的不同当前画面", verification["frozen_distinct_current_frames_replaced"]], ["协议、共享源码和适配器", "哈希未变"], ["CPU单元检查", "10项通过"]])}

逐查询核对RGB、腕部图、状态和噪声。每个实际执行动作都与保存的模型动作块前5步一致；相同任务/初态在最早切换之前的输入和动作逐项相同。冻结组还核对模型实际处理的两张图像素哈希相同，计时/保持组实际模型调用为0。

仿真中间谓词仅用于日志与离线评估，不送给检查器或策略来决定阶段。最终成功停止沿用官方LIBERO规则，因此不笼统声称“任何终止决策都不用真值”。

视频为实际回合的20fps仿真回放，不包含模型计算时的墙钟等待。五组对照片在某组结束后保留其真实末帧并标注已停止，不继续生成动作。完整视频解码核对通过。

主要限制是样本小、任务与初态已被研究过、低层检查点熟悉该基准，以及释放门控会改变检查的状态分布。这里诊断的是现有固定分层方案的作用，不能推广为“VLM高层没有用”。

## 7. 结论与下一阶段

本轮提供了比单个28/30更具体的判断依据：正常视觉{base["success"]}/30、冻结旧图{frozen["success"]}/30、保持首条{hold["success"]}/30，同时区分了切换、预算和低层自行完成。是否要继续投入检查器训练，应结合这些机制结果，而不能只因为模型有可微调参数就继续扩大数据。

下一阶段优先设计低层先验无法自动满足、但单个技能已确认可执行的目标，例如只搬指定物体并禁止额外搬运，或在预先确认可恢复的状态下改变剩余目标。新任务必须单独定义评分，不与原Long官方成绩混算。先证明高层介入确实影响任务，再训练可判定的视觉完成检查。

若继续做LoRA，先审核真实决策时刻的可见性和标签差异，并使用新的场景/初态作为留出集。当前查询集合只用于开发与诊断，不拿已反复使用的初态证明泛化。

## 8. 代码目录与复现

```text
experiments/vla_libero/monitor_causality/
  PROTOCOL.json / STATUS.json
  scripts/run_study.py / runner_snapshot.py / interventions.py
  scripts/analyze.py / collect_queries.py / verify.py
  scripts/figures.py / video.py / report.py
  tests/                         10项自包含CPU检查
  archive/frozen_sources/        运行时共享源码快照
  decision_data/cases.json       自然决策帧索引，非训练图像包
  episodes/primary/<方法>/libero_10/taskXX_initYY/
    result.json / trace.npz / trace.jsonl
    rollout.mp4 / qwen_inputs/ / monitor_calls.jsonl
    policy_inputs/ / policy_queries.jsonl
```

相关共同代码在 `experiments/vla_libero/scripts/` 和 `perception/qwen_scene/`。源代码包保留必要共享代码、固定版本/计划、结果元信息和测试；不包含π0.5/Qwen基座、既有适配器、全部图像或原始大数组。完整仿真需已有固定环境和资源，CPU测试不需要这些模型或数据。

```bash
experiments/vla_libero/scripts/python_shared experiments/vla_libero/monitor_causality/scripts/run_study.py
.venvs/openpi-libero/bin/python experiments/vla_libero/monitor_causality/scripts/analyze.py
.venvs/openpi-libero/bin/python experiments/vla_libero/monitor_causality/scripts/collect_queries.py
.venvs/openpi-libero/bin/python experiments/vla_libero/monitor_causality/scripts/verify.py
```

OpenPI为Apache-2.0、LIBERO为MIT，沿用本地固定版本，不自行实现策略或基准。参考官方接口与SciPy精确二项检验说明： https://github.com/Physical-Intelligence/openpi/tree/main/examples/libero 、 https://github.com/Lifelong-Robot-Learning/LIBERO 、 https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html 。具体取舍见RESEARCH.md。
"""
    (S / "REPORT.md").write_text(text)
    rendered = (
        MarkdownIt()
        .enable("table")
        .render(text)
        .replace("<h2>目录</h2>", '<h2 class="toc">目录</h2>')
    )
    css = """@page{size:A4;margin:17mm 16mm 18mm}body{font:10pt/1.7 "Noto Sans CJK SC","Noto Sans CJK JP",sans-serif;color:#24374a}h1{font-size:24pt;color:#153b58}h2{font-size:16pt;color:#153b58;border-bottom:1px solid #ccd8e3;padding-top:12px;break-after:avoid}h3{font-size:12pt;break-after:avoid}.toc{break-before:page}.toc+ol{break-inside:avoid}table{width:100%;border-collapse:collapse;font-size:8.3pt;margin:12px 0;break-inside:avoid}td,th{border:1px solid #ccd8e3;padding:5px}th{background:#edf3f8}img{width:100%;height:auto}p:has(img){break-inside:avoid}pre{font:8pt/1.6 "DejaVu Sans Mono","Noto Sans CJK SC",monospace;white-space:pre-wrap;overflow-wrap:anywhere;padding:10px;background:#f1f5f8;break-inside:avoid}code,a{overflow-wrap:anywhere}"""
    (S / "REPORT.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>视觉阶段检查对机器人执行的实际作用</title><style>'
        + css
        + "</style>"
        + rendered
    )
    with tempfile.TemporaryDirectory(prefix="monitor_pdf_chrome_") as profile:
        subprocess.run(
            [
                "google-chrome",
                "--headless",
                "--no-sandbox",
                "--disable-gpu",
                f"--user-data-dir={profile}",
                "--no-pdf-header-footer",
                f"--print-to-pdf={S / 'REPORT.pdf'}",
                (S / "REPORT.html").as_uri(),
            ],
            check=True,
            timeout=90,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    assert (S / "REPORT.pdf").read_bytes().startswith(b"%PDF-")
    destination = ROOT / "deliverables/20261113_阿里_视觉检查闭环作用"
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(S / "REPORT.pdf", destination / "视觉阶段检查闭环作用报告.pdf")
    shutil.copyfile(
        S / "五组检查干预_实际回合对照.mp4",
        destination / "五组检查干预_实际回合对照.mp4",
    )
    files = (
        list((S / "scripts").glob("*.py"))
        + list((S / "tests").glob("*.py"))
        + list((S / "archive").rglob("*.py"))
        + list((S / "figures").glob("*.png"))
    )
    files += [
        S / name
        for name in [
            "README.md",
            "RESEARCH.md",
            "requirements-offline.txt",
            "PROTOCOL.json",
            "SUMMARY.json",
            "VERIFICATION.json",
            "VIDEO.json",
            "REPORT.md",
            "metrics.csv",
            "decision_data/cases.json",
            "decision_data/summary.json",
        ]
    ]
    files += [ROOT / name for name in protocol["sources"]]
    files += [
        V / "scripts/python_shared",
        V / "third_party/sources.json",
        V / "scripts/evaluate.py",
        V / "scripts/install.py",
        V / "scripts/install_simulation.py",
        V / "README.md",
        V / "environment.json",
        ROOT / "perception/qwen_scene/README.md",
        ROOT / "perception/qwen_scene/requirements-local.lock.txt",
    ]
    files += list((V / "configs").rglob("*.json")) + list(
        (V / "configs").rglob("*.yaml")
    )
    files += [V / "third_party/openpi/LICENSE", V / "third_party/libero/LICENSE"]
    files += [
        V / "checker_v2/results/model/summary.json",
        V / "checkpoints/pi05_libero/download_verified.json",
    ]
    files += list(
        (V / "hierarchical/zh_eval_hier_v3/libero_10").glob("task*_init*/result.json")
    )
    files += list((S / "episodes/primary").glob("*/libero_10/task*/result.json"))
    if (S / "CLEAN_ACCEPTANCE.json").exists():
        files.append(S / "CLEAN_ACCEPTANCE.json")
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required package files missing: " + ", ".join(missing))
    contents = {
        str(path.relative_to(ROOT)): path.read_bytes()
        for path in sorted(set(files))
    }
    contents["PACKAGE_SHA256.json"] = json.dumps(
        {name: hashlib.sha256(value).hexdigest() for name, value in contents.items()},
        indent=2,
    ).encode()
    with zipfile.ZipFile(
        destination / "视觉检查干预_源码.zip", "w", zipfile.ZIP_DEFLATED
    ) as z:
        for name, value in contents.items():
            z.writestr(name, value)
    (destination / "文件清单.json").write_text(
        json.dumps(
            {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in destination.iterdir()
                if p.is_file() and p.name != "文件清单.json"
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(destination)


if __name__ == "__main__":
    main()
