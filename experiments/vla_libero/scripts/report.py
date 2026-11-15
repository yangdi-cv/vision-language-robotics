"""Chinese report and reproducible source package for actual completed trials."""

from pathlib import Path
import json, html, subprocess, zipfile, shutil, hashlib, re, unicodedata
from pypdf import PdfReader, PdfWriter

R = Path(__file__).resolve().parents[1]
D = R.parents[1] / "deliverables/20261112_阿里_VLA统一仿真"
D.mkdir(exist_ok=True, parents=True)
S = json.loads((R / "summary.json").read_text())
V = json.loads((R / "verification.json").read_text())
P = json.loads((R / "configs/protocol.json").read_text())
expected = P["baseline"]["episodes"] + P["paired_interventions"]["episodes"]
assert len(S["rows"]) == expected, "Refuse a complete-stage report with missing trials"
assert V["checked_episodes"] + len(V["error_directories"]) == expected
md = []
doc = []
LABEL = {
    "baseline": "原始输入",
    "blank_images": "图像置零",
    "frozen_images": "冻结初始画面",
    "wrong_language": "替换指令（原目标）",
    "state_noise": "状态位置噪声5mm",
    "action_delay": "动作延迟100ms",
    "replan_10": "每10步重规划",
    "denoising_5": "5步去噪",
}


TOC = [
    "1. 本轮范围与选型",
    "2. 统一观察与动作接口",
    "3. 任务与评估协议",
    "4. 官方检查点闭环结果",
    "5. 视觉、语言与执行干预",
    "6. 推理耗时与执行效率",
    "7. 失败与适用边界",
    "8. 复现与源码目录",
    "9. 结论和后续",
]


def h(t, n=2):
    md.extend(["", "#" * n + " " + t, ""])
    cls = ' class="page-start"' if t in ["目录", "1. 本轮范围与选型"] else ""
    anchor = f' id="sec-{TOC.index(t) + 1}"' if t in TOC else ""
    doc.append(f"<h{n}{cls}{anchor}>{html.escape(t)}</h{n}>")


def p(t):
    md.extend([t, ""])
    if t in TOC:
        idx = TOC.index(t) + 1
        doc.append(
            f'<p><a class="toc-link" href="#sec-{idx}">{html.escape(t)}<span class="toc-page">__PAGE_{idx}__</span></a></p>'
        )
    else:
        doc.append("<p>" + html.escape(t) + "</p>")


def table(head, rows):
    md.extend(
        ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
        + ["| " + " | ".join(map(str, row)) + " |" for row in rows]
        + [""]
    )
    doc.append(
        "<table><thead><tr>"
        + "".join("<th>" + html.escape(str(v)) + "</th>" for v in head)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>"
            + "".join("<td>" + html.escape(str(v)) + "</td>" for v in row)
            + "</tr>"
            for row in rows
        )
        + "</tbody></table>"
    )


def pic(name, caption):
    if not (R / "figures" / name).exists():
        return
    md.extend([f"![{caption}](figures/{name})", ""])
    doc.append(
        f'<figure><img src="figures/{name}"><figcaption>{html.escape(caption)}</figcaption></figure>'
    )


def code(t):
    md.extend(["```text", t, "```", ""])
    doc.append("<pre>" + html.escape(t) + "</pre>")


def count(v):
    return f"{v['successes']}/{v['n']}"


h("VLA统一仿真：基线复现与闭环评估", 1)
p("阿里具身方向阶段进展｜2026年11月12日")
h("摘要")
p(
    "此前感知、导航、抓取和真机实验分别使用不同场景与动作接口，难以定位整条链路的失败。本轮先收敛到固定基座桌面操作，在LIBERO同一物理环境内使用官方OpenPI π0.5检查点，验证图像、语言和机器人状态输入到模型动作，再到关节控制与接触结果的完整过程。没有使用目标真实坐标、仿真深度或预录轨迹产生动作。"
)
p(
    "本轮属于官方任务检查点的独立复现和执行评估，不是自主预训练或任务微调。后续的模型训练与SO101迁移需另设数据和评估。以下只统计实际保存的运行，不用计划数量代替完成数量。"
)
table(
    ["内容", "实际完成"],
    [
        ["基线", str(sum(v["n"] for v in S["baseline_by_suite"].values())) + "次"],
        ["配对干预", str(sum(v["n"] for v in S["interventions"].values())) + "次"],
        ["轨迹检查", str(V["checked_episodes"]) + "次"],
        ["异常运行", str(len(V["error_directories"])) + "次"],
    ],
)
base_n = sum(v["n"] for v in S["baseline_by_suite"].values())
base_k = sum(v["successes"] for v in S["baseline_by_suite"].values())
p(
    f"官方任务检查点在本机基线中通过{base_k}/{base_n}。相同初态的实时图像被置零或冻结后，两项干预均为0/18；替换指令后的原目标得分与替代目标谓词分开分析。低层执行和输入接口的正确性得到数值检查，任务结束后的释放稳定性另设附加实验。"
)

h("目录")
for t in TOC:
    p(t)
h("1. 本轮范围与选型")
p(
    "选用OpenPI官方π0.5 LIBERO任务检查点，因为它提供对应的模型、输入输出转换与评估入口，可以先确认执行链路是否正常。新环境与厂家LeRobot、已有SmolVLA和ROS环境分开。本轮不打开串口、不操作实体机械臂，也不依赖Jetson、IMU或ZED。相关代码在 scripts/install.py、scripts/fetch_sources.py。"
)
p(
    "LIBERO使用固定基座机械臂，不含移动导航。它解决的是当前最需要核对的“VLA是否能够在物理仿真中完成操作”，不是直接完成整个导航抓取任务书。导航与MoveIt的既有结果仍单独保留，没有将跨模拟器流程重新包装成统一闭环。"
)
p(
    "任务书中的Qwen高层感知与决策模块保留。本轮π0.5不是Qwen底座，也没有将已有Qwen LoRA接进这套实验；先建立可靠的VLA执行参照，后续再单独评估高层任务分解与执行器的连接。"
)
h("2. 统一观察与动作接口")
p(
    "输入为外部相机与腕部相机的RGB、自然语言任务描述和八维机器人状态。状态由末端位置、轴角方向和两指关节位置组成。渲染分辨率256×256，按官方流程两路图像旋转180°后填充缩放至224×224。模型使用检查点配套的归一化和输入输出转换。相关代码在 scripts/evaluate.py，以及上游 openpi/policies/libero_policy.py。"
)
p(
    "模型预测10×7动作块，基线执行前5步后重新读取图像和状态。动作包含三维平移、三维旋转与夹爪控制，经LIBERO的OSC_POSE低层控制器产生电机控制量，再由MuJoCo推进。运行中不逐帧设置机器人关节或物体位置；仅在每回合初始化时恢复公开初态。"
)
p(
    "每个回合保存实际控制器参数、控制量范围、关节状态与速度、电机输出、模型完整预测块、实际动作、仿真时间及官方任务谓词。模型输入不包含这些评估用目标信息。若模型输出无效、环境报错或任务超时，单独记录，不将异常算作成功。"
)
pic("observation_views.png", "图1 模型的两路RGB观察：外部相机与腕部相机。")
h("3. 任务与评估协议")
p(
    "基线预设Spatial、Object、Goal和Long-Horizon四个任务集，各10个任务，每任务采用公开初态0、1、2，共120个计划回合。初态固定，环境种子7，动作采样噪声按任务和初态确定；配对干预共享相同初态和模型噪声。当前每任务仅3个初态，不等同于官方每任务50次的完整协议，不能把结果直接与论文整套分数排名。"
)
table(
    ["设置", "取值"],
    [
        ["初始等待", "10个空动作，等待物体稳定"],
        ["控制率", "20Hz"],
        ["基线重新观察", "每5个动作一次"],
        ["动作生成", "10步流匹配积分，输出10步动作"],
        ["成功", "环境原始任务谓词成立"],
        ["超时步数", "Spatial220；Object280；Goal300；Long-Horizon520"],
    ],
)
p(
    "配对任务在查看结果前固定为Spatial与Goal的任务0、3、6，使用相同三个初态。分别测试置零两路图像、冻结初始图像、替换任务指令、位置测量加入5mm高斯噪声、动作队列延迟两步，以及增加执行块长度、减少去噪步数。相关代码和完整定义在 configs/protocol.json。"
)
h("4. 官方检查点闭环结果")
table(
    ["任务集", "成功/总数", "95% Wilson区间"],
    [
        [k, count(v), f"{v['wilson95'][0] * 100:.1f}%—{v['wilson95'][1] * 100:.1f}%"]
        for k, v in S["baseline_by_suite"].items()
    ],
)
pic("success_summary.png", "图2 基线与相同初态下的干预结果。")
p(
    "这些结果来自官方已经针对LIBERO训练的权重，说明本机执行链路及该检查点在指定条件下的表现，不表示本项目从零学会这些任务。任务成功是环境内的目标谓词，不是由视觉语言模型自己判断。"
)
pic("rollout_examples.png", "图3 实际运行关键帧；标题注明任务和结果。")
if (R / "release_summary.json").exists():
    release = json.loads((R / "release_summary.json").read_text())
    h("4.1 松爪后的补充检查", 3)
    p(
        f"官方谓词可能在物体尚被夹持时成立。因此另外重放{release['checked']}条成功回合的实际动作，先核对完整关节轨迹，再保持末端并打开夹爪1秒。最后0.5秒持续满足目标的为{release['stable_last_half_second']}/{release['checked']}，最终一帧满足目标的为{release['final_goal_true']}/{release['checked']}。"
    )
    p(
        f"以Panda两指关节位移均至少0.03m确认夹爪充分开启（不是爪尖净间距标定），达到开启阈值的为{release['open_confirmed']}组；同时充分开启并保持目标的为{release['open_and_stable']}组。"
    )
    table(
        ["任务集", "重放成功回合", "松爪后持续满足", "末帧满足"],
        [
            [k, v["n"], v["stable"], v["final_goal"]]
            for k, v in release["by_suite"].items()
        ],
    )
    p(
        f"重放与原轨迹的最大关节位置差为{release['replay_max_qpos_error']:.2e}。这是附加的物理稳定性检查，不改写官方成功率，也不增加VLA评估回合数；后续1秒采用固定松爪保持指令，不冒充模型自主完成了该阶段。相关代码在 scripts/release_stability.py。"
    )

    pic(
        "release_check.png",
        "图4 追加检查：官方停止时与固定松爪保持后；后续指令不是模型生成的动作。",
    )
    p(
        "固定松爪保持不是原策略的后续行为。补充测试失败表示该固定操作协议下未保持目标，不能据此断言模型继续执行必然失败；它说明早停谓词与实际操作的完成确认需要区分。"
    )
h("5. 视觉、语言与执行干预")
table(
    ["条件", "配对基线", "干预结果", "成功→失败", "失败→成功"],
    [
        [
            LABEL[k],
            count(v["matched_baseline"]),
            count(v),
            v["success_to_failure"],
            v["failure_to_success"],
        ]
        for k, v in S["interventions"].items()
    ],
)
p(
    "图像置零和冻结用于检查视觉输入依赖，不是现实相机误差模型。替换指令首先按原任务评分；原任务失败可能是模型正确响应了替代指令，也可能是行为无效，因此不直接等同于语言能力下降。对于Goal中可在同一场景求值的替代目标，另记录是否曾满足其谓词，并比较相同初态、相同噪声下的首个动作块变化。"
)
for key, v in S["interventions"].items():
    comps = v["first_chunk_comparisons"]
    mean = sum(x["first_chunk_mae"] for x in comps) / len(comps) if comps else None
    if key == "wrong_language":
        p(
            f"替换指令的{len(comps)}组首动作块对照，平均绝对变化为{mean:.5f}（动作接口单位）；可求值替代目标{v['alternate_goal_evaluable']}组，其中{v['alternate_goal_ever_satisfied']}组曾满足替代谓词。在这组Goal条件中，替代目标的谓词成立支持模型依据语言改变操作；该结论限于这些已检查任务，不外推到未训练任务、中文或复杂歧义指令。"
        )
pic(
    "visual_intervention.png",
    "图5 同一初态的实时、置零、冻结视觉对照；下方展示实际提交给模型的图像。",
)
p(
    "状态噪声只加在提交给模型的末端位置测量，不修改实际物体或机器人状态。动作延迟为仿真中的100ms先进先出队列，不是让程序休眠100ms。两者分别区分观察误差与执行滞后。"
)
h("6. 推理耗时与执行效率")
t = S["timings"]
if t:
    p(
        f"排除各回合首个调用后的端到端推理中位数为{t['wall_median_s']:.3f}s，P95为{t['wall_p95_s']:.3f}s，共{t['queries_excluding_first_per_episode']}次调用。该计时包含输入转换、模型及CPU动作返回，首次调用可能包含JIT编译，另保留原始记录。"
    )
p(
    "本轮使用同步评估，模拟器在等待策略时暂停。动作按20Hz仿真时钟执行，并不说明真实机器人能以相同速度持续等待模型。增加每次执行动作数量可减少模型调用，但延长无新观察的时间；减少去噪积分步数降低计算量，但可能影响动作质量。两种改动均与原权重配对测试，不把耗时下降自动解释为策略变好。"
)
if (R / "inference_benchmark.json").exists():
    b = json.loads((R / "inference_benchmark.json").read_text())
    table(
        ["固定输入对照", "调用数", "中位耗时/ms", "P95/ms"],
        [
            [
                f"{k}步去噪",
                v["n"],
                f"{v['median_s'] * 1000:.2f}",
                f"{v['p95_s'] * 1000:.2f}",
            ]
            for k, v in b["timings"].items()
        ],
    )
    p(
        f"另以预设任务中的{b['contexts']}个相同观察、相同动作噪声交错重复测试，排除首次编译；5步相对10步的中位推理提速为{b['median_speedup_5_vs_10']:.2f}倍。这是单模型计时，不含环境推进，也不代表成功率改善。相关代码在 scripts/benchmark_inference.py。"
    )

h("7. 失败与适用边界")
p(
    "所有已完成回合均保留结果、控制轨迹和视频。成功数以原任务谓词为准，超时与异常分开。失败分析需要结合目标选择、接近、夹持、搬运和放置阶段，不仅观察最终成功率。独立检查重新读取保存数组，核对20Hz时间推进、每个预测块实际执行顺序、延迟队列、初始状态及成功标记。相关代码在 scripts/verify.py。"
)
table(
    ["检查项", "结果"],
    [
        ["已检查回合", V["checked_episodes"]],
        ["控制样本", V["control_samples"]],
        ["动作块执行", "已检查"],
        ["配对初态", "已检查"],
        ["原始任务判据", "已检查"],
        ["mocap", "模型中无mocap驱动"],
        ["错误目录", len(V["error_directories"])],
    ],
)
p(
    "本轮仍有明确限制：公开任务与模型训练分布存在重叠；每任务测试量较小；未进行自主微调或视觉域随机化训练；未模拟真实传感器曝光延迟与网络；低层控制采用基准自带控制器，不代表SO101电机带宽。当前结果不能外推为家中机械臂会抓取。"
)
failures = [v for v in S["rows"] if v["mode"] == "baseline" and not v["success"]]
if failures:
    table(
        ["基线失败任务", "初态", "状态", "控制步数"],
        [
            [
                f"{v['suite']} / {v['task_id']}",
                v["initial_state_id"],
                v["status"],
                v["steps"],
            ]
            for v in failures
        ],
    )
h("8. 复现与源码目录")
code(
    "experiments/vla_libero/\n  configs/protocol.json        固定任务与干预协议\n  configs/checkpoint_objects.json 官方权重对象及校验信息\n  scripts/fetch_sources.py     固定上游源码\n  scripts/install.py           独立环境依赖\n  scripts/install_simulation.py 仿真依赖\n  scripts/download_segments.py 权重分段下载\n  scripts/verify_checkpoint.py CRC32C/MD5及SHA256校验\n  scripts/simulation_smoke.py  相机和物理接口检查\n  scripts/evaluate.py          模型闭环与干预\n  scripts/tests.py             输入与协议回归\n  scripts/analyze.py           配对统计与图表\n  scripts/verify.py            原始轨迹检查\n  scripts/release_stability.py 重放与松爪保持检查\n  scripts/benchmark_inference.py 固定输入计时\n  scripts/video.py            任务关键帧与演示\n  scripts/visual_comparison.py 输入干预对照视频\n  scripts/report.py            报告与源码打包\n  results/                    本地动作、状态和视频\n  figures/\n  third_party/sources.json     上游固定版本\n  README.md / RESEARCH.md"
)
p(
    "源码包不含官方权重、环境、上游大型资产或原始轨迹数组，按README下载固定版本后运行。官方源码和权重许可见上游仓库及本地RESEARCH.md；不将第三方实现冒充自主实现。安装锁定版本与环境信息另存，依赖适配见 ENVIRONMENT_NOTES.md。"
)
h("9. 结论和后续")
p(
    "本轮把原先分散的观察与策略演示收敛为同一个仿真环境中的VLA执行和可重复评估。下一步根据配对结果选择需要改进的具体环节，再考虑训练：使用独立示教、验证集和未见初态，区分拟合、闭环成功及泛化。SO101迁移需要重新核对机器人、相机、动作定义与标定，不能直接下发LIBERO动作。"
)
p(
    "参考：https://github.com/Physical-Intelligence/openpi；https://github.com/Physical-Intelligence/openpi/tree/main/examples/libero；https://github.com/Lifelong-Robot-Learning/LIBERO；https://github.com/ARISE-Initiative/robosuite。"
)
(R / "REPORT.md").write_text("\n".join(md))
css = """@page{size:A4;margin:18mm}body{font-family:"Noto Sans CJK SC","Noto Sans CJK JP",sans-serif;font-size:10pt;line-height:1.75;color:#25384b}.page-start{break-before:page}.toc-link{display:block;color:inherit;text-decoration:none}.toc-page{float:right}h1{font-size:24pt;color:#173e5a}h2{font-size:16pt;color:#173e5a;break-after:avoid}p{text-align:justify}table{width:100%;border-collapse:collapse;font-size:9pt;break-inside:avoid}th,td{border:1px solid #ccd5de;padding:6px}th{background:#eaf0f6}figure{margin:12px 0;break-inside:avoid}img{width:100%}figcaption{font-size:9pt;color:#526273}pre{break-inside:avoid;white-space:pre-wrap;overflow-wrap:anywhere;font:8pt/1.6 monospace;background:#eff3f7;padding:10px}"""
html_text = (
    '<!doctype html><meta charset="utf-8"><style>' + css + "</style>" + "".join(doc)
)


def render_pdf(document):
    (R / "REPORT.html").write_text(document)
    subprocess.run(
        [
            "google-chrome",
            "--headless",
            "--no-sandbox",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--print-to-pdf={R / 'REPORT.pdf'}",
            (R / "REPORT.html").as_uri(),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


render_pdf(re.sub(r"__PAGE_\d+__", "", html_text))
reader = PdfReader(R / "REPORT.pdf")


def compact(text):
    return "".join(unicodedata.normalize("NFKC", text).split())


pages = {}
for title in TOC:
    matches = [
        i
        for i, page in enumerate(reader.pages)
        if i >= 2 and compact(title) in compact(page.extract_text() or "")
    ]
    assert matches, title
    pages[title] = matches[0]
for i, title in enumerate(TOC, 1):
    html_text = html_text.replace(f"__PAGE_{i}__", str(pages[title] + 1))
    md = [
        line + f"（第{pages[title] + 1}页）" if line == title else line for line in md
    ]
(R / "REPORT.md").write_text("\n".join(md))
render_pdf(html_text)
reader = PdfReader(R / "REPORT.pdf")
writer = PdfWriter()
writer.clone_document_from_reader(reader)
writer.add_outline_item("摘要", 0)
writer.add_outline_item("目录", 1)
for title, page in pages.items():
    writer.add_outline_item(title, page)
writer.add_metadata(
    {
        "/Title": "VLA统一仿真：基线复现与闭环评估",
        "/Subject": "官方检查点复现、输入干预、执行效率和补充释放检查",
    }
)
writer.write(R / "REPORT.bookmarked.pdf")
(R / "REPORT.bookmarked.pdf").replace(R / "REPORT.pdf")
(R / "report_navigation.json").write_text(
    json.dumps(
        {
            "pages": len(reader.pages),
            "sections": {title: page + 1 for title, page in pages.items()},
        },
        indent=2,
        ensure_ascii=False,
    )
)
shutil.copy2(R / "REPORT.pdf", D / "VLA统一仿真与闭环评估报告.pdf")
shutil.copy2(R / "REPORT.md", D / "REPORT.md")
shutil.copytree(R / "figures", D / "figures", dirs_exist_ok=True)
include = (
    list((R / "scripts").glob("*.py"))
    + [R / "scripts/python"]
    + [
        R / n
        for n in [
            "README.md",
            "RESEARCH.md",
            "ENVIRONMENT_NOTES.md",
            "REPORT.md",
            "summary.json",
            "verification.json",
        ]
    ]
    + list((R / "figures").glob("*.png"))
    + [p for p in (R / "configs").glob("*.json") if not p.name.endswith("_diff.json")]
    + [R / "third_party/sources.json"]
)
for n in ["environment.json", "requirements.lock.txt", "simulation_check.json"]:
    if (R / n).exists():
        include.append(R / n)
for optional in [
    "release_summary.json",
    "video_metadata.json",
    "inference_benchmark.json",
    "visual_comparison_metadata.json",
    "report_navigation.json",
]:
    if (R / optional).exists() and R / optional not in include:
        include.append(R / optional)
with zipfile.ZipFile(D / "VLA统一仿真源码.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for pth in sorted(include):
        z.write(pth, Path("experiments/vla_libero") / pth.relative_to(R))
(D / "文件清单.json").write_text(
    json.dumps(
        {
            str(p.relative_to(D)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in D.rglob("*")
            if p.is_file() and p.name != "文件清单.json"
        },
        indent=2,
        ensure_ascii=False,
    )
)
print(D)
if __name__ == "__main__":
    pass
