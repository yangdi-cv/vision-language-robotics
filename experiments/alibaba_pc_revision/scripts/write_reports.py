from pathlib import Path
import json, csv
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[3]
A = R / "experiments/alibaba_pc_revision"
B = R / "experiments/bytedance_week3"


def load(p):
    return json.loads(p.read_text())


def table(h, rows):
    return (
        "| "
        + " | ".join(h)
        + " |\n|"
        + "|".join(["---"] * len(h))
        + "|\n"
        + "\n".join("| " + " | ".join(map(str, row)) + " |" for row in rows)
        + "\n"
    )


lora = load(A / "runs/lora_validation/summary.json")
lh = load(A / "runs/lora_validation/history.json")
act = load(A / "runs/act_selection/summary.json")
ah = load(A / "runs/act_selection/history.json")
q = load(A / "runs/qwen_limits/summary.json")
vision = load(A / "runs/vision_precision/summary.json")
runtime = load(A / "runs/runtime/summary.json")
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
axs[0].semilogy([r["step"] for r in lh], [r["validation_loss"] for r in lh], "o-")
axs[0].set_ylabel("Validation cross-entropy")
axs[1].plot([r["step"] for r in lh], [r["validation_correct"] for r in lh], "o-")
axs[1].set_ylabel("Validation exact matches /20")
for ax in axs:
    ax.set_xlabel("Training update")
    ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(A / "figures/lora_validation.png", dpi=160)
plt.close(fig)
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
axs[0].plot([r["step"] for r in ah], [r["validation_l1"] for r in ah], "o-")
axs[0].set_ylabel("Validation normalized L1")
axs[1].plot([r["step"] for r in ah], [r["validation_success"] for r in ah], "o-")
axs[1].set_ylabel("Validation rollout successes /6")
for ax in axs:
    ax.set_xlabel("Training update")
    ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(A / "figures/act_selection.png", dpi=160)
plt.close(fig)
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
axs[0].bar(["PNG", "Shared memory"], [v["total_p50_ms"] for v in runtime["transport"]])
axs[0].set_ylabel("Producer -> two readers median [ms]")
axs[1].bar(["64", "128", "256", "64+grammar"], [v["median_seconds"] * 1000 for v in q])
axs[1].set_ylabel("Qwen goal inference median [ms]")
fig.tight_layout()
fig.savefig(A / "figures/pc_latency.png", dpi=160)
plt.close(fig)
text = f"""# PC端推理链路与训练评估改进

阿里方向阶段进展　2026年10月20日

## 摘要

本轮根据上一份报告的反馈，重点改进模型进程超时处理、图像传递、LoRA训练验证和ACT选模，并在4090工作站进行对照测试。新的代码与报告单独保存，原报告中的实验结果保持不变。

两读取进程的图像交付中位耗时由{runtime['transport'][0]['total_p50_ms']:.2f}ms降至{runtime['transport'][1]['total_p50_ms']:.2f}ms，像素与帧序号逐项核对一致。短目标JSON实际仅生成16个token，生成上限从256缩至64没有明显提速。LoRA加入周期验证和早停后，在第{lora['last_step']}步结束，选中第{lora['best_step']}步的适配器，测试集20/20完整字段匹配。

ACT同时比较离线L1和仿真成功率选模。前者选择第10000步模型，新24任务成功24次；后者选择第4000步模型，成功21次。这个结果表明，只有6个验证任务时，即使用成功率选模也可能过早选中模型，下一步需要扩大验证布局。

本轮仅为PC端。没有开展Jetson部署，也没有用办公室相机或机械臂进行新的联动测试。

## 目录

- 1. 改动范围与代码入口
- 2. 模型进程超时和退出处理
- 3. 图像内存传递与一致性
- 4. 输出长度与约束解码
- 5. 检测及深度推理对照
- 6. LoRA周期验证与早停
- 7. ACT数据划分及选模
- 8. 结论与后续

## 1. 改动范围与代码入口

| 内容 | 本轮实现 |
|---|---|
| worker阻塞读取 | 带截止时间的JSON行读取、异常退出和进程回收 |
| 图像交换 | 单份共享内存、锁保护元信息与像素、两个独立读取者 |
| 推理预算 | 短目标JSON的64/128/256上限对照，有限目标集合约束 |
| 检测/深度 | FP32/FP16及768/640深度输入对照，默认配置保留 |
| LoRA | 每40步验证、保存最佳适配器、早停 |
| ACT | manifest驱动划分、先抽episode再抽帧、定期仿真验证 |
| 重复代码 | 公开交接任务训练/评测共享实现；语言派发复用同一执行循环 |

主要实验在 `experiments/alibaba_pc_revision/`。生产代码改动位于 `perception/qwen_scene/` 和 `perception/web_workbench/server.py`。历史交接训练和评测入口保留原路径，通过兼容入口调用公共实现。

## 2. 模型进程超时和退出处理

原来的stdout.readline可能一直等不到一整行。新增读取器按截止时间等待，识别EOF、非法JSON和过大响应；失败时结束并回收模型进程。现有两个语言派发入口已接入，默认响应上限90秒。正常关闭先发停止请求，再按超时规则终止进程，避免主流程无限等待。

"""
text += table(
    ["软件测试", "结果", "含回收的总耗时 / s"],
    [
        [r["case"], r["status"], f"{r['total_seconds']:.3f}"]
        for r in runtime["worker_failures"]
    ],
)
text += """
测试包含静默、异常退出、非法JSON及正常回复，均使用本机模拟进程。读取超时与进程清理是两个阶段，表中的总时间包含清理等待。没有据此称为机器人硬件鲁棒性测试。

相关代码在 `perception/qwen_scene/worker_client.py`、`experiments/alibaba_baseline/scripts/integrated_demo.py`、`experiments/alibaba_extension/scripts/run_qwen_nav_act.py`。

## 3. 图像内存传递与一致性

原网页每帧保存PNG，两个推理进程分别解码。现在相机线程将RGB和帧号、接收时间、旋转信息写入一份共享内存，写入和读取时加锁。读取者复制完整一帧后立即释放锁，不在推理期间占锁，也不积压中间帧。已有PNG来源仍可作为兼容输入。

用12张已有腕部画面、每张重复三次，对比一次写入和两个独立进程读取。统计排除每组最初两次，包含消息通知和像素校验的开销；文件方案会受系统缓存影响，不是冷磁盘性能测试。

"""
text += table(
    ["方式", "有效测试帧数", "写入中位 / ms", "双读取完成中位 / ms", "双读取P95 / ms"],
    [
        [
            r["mode"],
            r["frames"],
            f"{r['write_p50_ms']:.3f}",
            f"{r['total_p50_ms']:.3f}",
            f"{r['total_p95_ms']:.3f}",
        ]
        for r in runtime["transport"]
    ],
)
text += """
共144次读取核对像素哈希与帧号，没有不一致；连续写入100帧后只读取到最新帧。提交给Qwen的分析图像与结果仍可归档，视觉结果也仍有保存开销；这次移除的是实时帧交换中的PNG往返，不是删除全部记录。

![图像交换与目标推理耗时](figures/pc_latency.png)

这个结果确认了图像交换有优化空间，但不能证明它是整个系统的最大瓶颈。目标推理仍约260ms，更长的场景描述耗时还会增加。网页若已在运行，需重启后使用新的传帧方式；本轮没有启动真实相机和主从控制。

相关代码在 `perception/qwen_scene/frame_transport.py`、`continuous_worker.py`、`vision_worker.py`、`perception/web_workbench/server.py`；测试在 `scripts/bench_runtime.py`。

## 4. 输出长度与约束解码

在同一Qwen基座、提示词和图像上，交错运行64、128、256生成上限及64加约束四种设置。12个目标用例各重复两次，预热后统计。

"""
text += table(
    [
        "上限 / 约束",
        "中位 / s",
        "P95 / s",
        "平均生成token",
        "格式有效",
        "目标完整匹配",
        "截断",
    ],
    [
        [
            str(v["limit"]) + (" / 有" if v["grammar"] else " / 无"),
            f"{v['median_seconds']:.3f}",
            f"{v['p95_seconds']:.3f}",
            f"{v['mean_tokens']:.1f}",
            f"{v['schema_valid']}/{v['cases']}",
            f"{v['correct']}/{v['cases']}",
            v["truncated"],
        ]
        for v in q
    ],
)
text += """
这组回答都在16个token正常结束，因此缩小上限没有明显加速；上限大不代表模型一定生成到上限。约束解码将输出限制为当前任务支持的7种JSON目标对象，能限定格式与字段组合，但不是通用JSON语法实现，也没有修复目标选择错误。

保留长场景描述的独立预算，短决策接口收紧至128并允许按任务配置；没有把整个观察窗口一律改为64。不能把语法有效等同于目标存在或指令明确。

相关代码在 `scripts/bench_qwen.py`、`perception/qwen_scene/engine.py`、`experiments/alibaba_baseline/scripts/qwen_goal_worker.py`。

## 5. 检测及深度推理对照

使用8个已有模拟RGB-D场景，各重复两次。检测输入固定640，深度比较768和640，记录串行模型调用耗时。它不包含完整网页渲染或相机采集。FP16映射到当前Ultralytics的quantize=16，额外检查实际网络参数类型为float16。

"""
text += table(
    ["精度 / 深度尺寸", "中位 / ms", "P95 / ms", "原尺度AbsRel", "逐图尺度对齐AbsRel"],
    [
        [
            ("FP16" if v["half"] else "FP32") + f" / {v['depth_size']}",
            f"{v['median_seconds']*1000:.2f}",
            f"{v['p95_seconds']*1000:.2f}",
            f"{v['mean_abs_rel']:.3f}",
            f"{v['mean_scale_aligned_abs_rel']:.3f}",
        ]
        for v in vision
    ],
)
text += """
本轮FP16收益较小；640略快，但尺度对齐后的相对误差由约0.089增加到0.102。原尺度误差很大，说明这些模拟场景中的预测距离不能直接当作可用于抓取的米制深度。

尺度对齐使用每张图的模拟真值估计比例，只用于诊断相对几何表现，部署时不能依赖这种真值。场景数量少、存在模拟域差异，模型设置按组顺序运行；不据此宣称真实相机精度或稳定生产加速。检测框数量变化仅作记录，没有目标检测人工真值，未报告检测准确率。

推理类已增加精度和深度尺寸配置，并移除重复的全设备显式同步；默认仍保留FP32/768。读取GPU输出到CPU时仍会发生必要同步，未宣称异步推理或双流已完成。

相关代码在 `scripts/bench_vision.py`、`perception/qwen_scene/vision_engine.py`。

## 6. LoRA周期验证与早停

沿用桌面目标选择的60条训练、20条验证、20条测试，按布局分开，不改标签。训练视觉模块冻结，仅对语言注意力q_proj/v_proj做r8、alpha16、学习率5e-5的LoRA。每40步计算验证损失和完整目标匹配，以匹配数优先、损失至少改善0.001作为同分选择依据；连续三次未改善则早停。

![LoRA验证过程](figures/lora_validation.png)

第40步验证12/20，第80步达到20/20；之后匹配数不再增加，损失改善也未达到所设阈值，在第200步结束。选中的第80步与最后第200步模型测试均20/20。

实际仍训练了200步，并承担周期推理验证的开销，不能把“选中80步”说成“只花80步训练成本”。这次验证的是选模与停止流程；沿用旧数据划分，也不能把20/20当作新的语言泛化突破。

相关代码在 `scripts/train_lora.py`；划分及逐次预测在 `runs/lora_validation/`。

## 7. ACT数据划分及选模

继续使用桌面任务66条脚本示教，54/6/6轨迹按布局划分，归一化只来自训练数据。训练从manifest读取episode列表与边界，先均匀抽episode，再在轨迹内部抽帧，动作块不跨轨迹。当前轨迹均为401帧，理论上两种采样的分布相同；这次重构主要提高可读性和变长数据兼容性，不把收益归因于改变采样。

ACT输入为RGB、关节/夹爪、任务符号及初始RGB-D坐标，训练10000步。每2000步计算验证L1，并在固定布局1100的六个任务运行物理仿真。分别保存最低L1模型和最高仿真成功数模型；成功数相同时保留较早模型。最终布局1200–1203的24个任务只在选模结束后评测。

![离线误差与仿真验证](figures/act_selection.png)

"""
text += table(
    ["选模依据", "选中更新数", "新任务成功", "离线测试归一化L1"],
    [
        [
            label,
            v["selected_step"],
            f"{v['test_success']}/{v['test_trials']}",
            f"{v['test_l1']:.4f}",
        ]
        for key, label in [("l1", "验证L1"), ("rollout", "验证仿真成功数")]
        for v in [act["selection"][key]]
    ],
)
text += """
验证L1从4000步到6000步下降，但仿真成功数由6降到4，说明两者确实不完全一致。不过仅凭一个布局六个任务选模，也会误选第4000步：它在新任务中只有21/24，而第10000步为24/24。当前结果不支持“换成成功率选模就一定更好”，下一轮应扩大验证布局、重复训练种子，并预先确定同分规则。

成功判据包含抬升、夹爪接触、指定盒内稳定、释放及盒底接触。执行仍使用原桌面任务的理想末端约束和模拟深度，未迁移SO101，也不是完整移动抓取机器人。只训练一个随机种子，测试仅四种新布局。

新的训练验证与最终评测复用 `scripts/act_runner.py`。旧公开交接任务的训练与评测兼容入口已改为公共实现，语言派发也复用同一交接执行/视频循环。公共训练入口做了一步训练、验证和保存检查，公共执行入口在已知布局1003完成一次交接；这些是兼容性检查，不是重新完成原30次成功率评测。

相关代码在 `scripts/train_act.py`、`act_runner.py`、`public_act_trainer.py`、`public_act_evaluate.py`、`aloha_runner.py`。训练数据目录、输出目录、训练步数、验证间隔和布局种子可通过CLI配置，仍需准备对应环境与数据，不称为跨机器零配置安装。

## 8. 结论与后续

本轮落实了反馈中的超时处理、帧交换、训练验证、仿真选模和主要重复代码整理。明显收益来自图像交换；短目标生成上限和FP16未显示同等幅度收益。选模对照还暴露了验证集规模不足的问题。

下一步先扩大独立验证布局及语言歧义用例，复测真实相机下的帧年龄与整体耗时。Jetson上的模型转换、算子兼容、显存和功耗另做，不用4090结果代替。

源码目录：

```text
experiments/alibaba_pc_revision/
├── scripts/                 训练、评测、传输测试与共享执行模块
├── runs/                    逐次预测、验证结果、轨迹和运行记录
├── figures/                 本报告图表
└── REPORT.md、README.md
perception/qwen_scene/       帧共享、worker客户端、推理类与读取进程
perception/web_workbench/    网页相机分发入口
experiments/alibaba_baseline/scripts/   历史兼容入口
experiments/alibaba_extension/scripts/  历史兼容入口
```

源码包只包含代码、必要配置和场景资源；权重、缓存、数据和运行结果保留于本地。运行方式与环境要求见README。
"""
(A / "REPORT.md").write_text(text)
# ByteDance contact results, using full1kHz logs for peak measurements.
rows = load(B / "runs/contact_suite_v3/summary.json")["results"]
core = [r for r in rows if r["group"] == "core"]
stress = [r for r in rows if r["group"] == "stress"]
approach = [r for r in rows if r["group"] == "approach"]
joint = load(B / "runs/joint_check_v3/summary.json")
a = np.load(B / "runs/contact_suite_v3/case_003.npz")["samples"]
b = np.load(B / "runs/contact_suite_v3/case_004.npz")["samples"]
c = np.load(B / "runs/contact_suite_v3/case_005.npz")["samples"]
fig, axs = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
for v, label in [(a, "Position"), (b, "Admittance K=0"), (c, "Admittance K=200")]:
    axs[0].plot(v[:, 0], v[:, 1], lw=0.8, label=label)
    axs[1].plot(v[:, 0], v[:, 5] * 1000, lw=0.8, label=label)
axs[0].plot(b[:, 0], b[:, 2], "k--", label="Target force")
axs[0].set_ylabel("Measured normal force [N]")
axs[1].set_ylabel("Admittance offset [mm]")
axs[1].set_xlabel("Time [s]")
axs[0].legend()
for ax in axs:
    ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(B / "figures/contact_force.png", dpi=160)
plt.close(fig)
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
for rate in [50, 100, 200]:
    vals = []
    for delay in [0, 20, 60]:
        rr = [
            r
            for r in stress
            if r["mode"] == "admittance"
            and r["rate"] == rate
            and r["delay_ms"] == delay
        ]
        vals.append(np.mean([v["steady_force_rmse_N"] for v in rr]))
    axs[0].plot([0, 20, 60], vals, "o-", label=f"{rate}Hz")
axs[0].set_xlabel("Added force-feedback delay [ms]")
axs[0].set_ylabel("Steady force RMSE [N]")
axs[0].legend()
axs[0].grid(alpha=0.2)
for mode in ["position", "admittance"]:
    rr = [v for v in approach if v["mode"] == mode]
    axs[1].plot(
        [v["approach_s"] for v in rr], [v["peak_force_N"] for v in rr], "o-", label=mode
    )
axs[1].set_xlabel("Approach duration [s]")
axs[1].set_ylabel("Peak normal force, 1kHz samples [N]")
axs[1].legend()
axs[1].grid(alpha=0.2)
fig.tight_layout()
fig.savefig(B / "figures/delay_approach.png", dpi=160)
plt.close(fig)
fig, ax = plt.subplots(figsize=(10, 4))
x = np.arange(7)
for off, mode in [(-0.18, "position"), (0.18, "admittance")]:
    v = next(r for r in joint if r["mode"] == mode and r["stiffness"] == 8000)
    ax.bar(x + off, v["joint_rmse_deg"], width=0.35, label=mode)
ax.set_xticks(x, labels=[f"J{i}" for i in range(1, 8)])
ax.set_ylabel("Joint reference tracking RMSE [deg]")
ax.legend()
fig.tight_layout()
fig.savefig(B / "figures/joint_tracking.png", dpi=160)
plt.close(fig)
rev = load(B / "assets/panda/SOURCE.json")["revision"]
text = f"""# Panda末端接触与导纳控制仿真实验

字节方向Week3阶段进展　2026年10月20日

## 摘要

本轮从自由空间轨迹跟踪推进到物理接触，使用七关节Panda动力学模型、受限关节力矩和末端探头，比较位置控制与法向导纳控制。主实验177组，包含三档接触参数、50/100/200Hz外环、人工测量噪声、反馈延迟及接近速度；另补六组关节参考跟踪记录与接触视频。

在中间接触参数档，固定位置控制的稳定段力误差约{core[3]['steady_force_rmse_N']:.2f}N，质量阻尼型导纳降到{core[4]['steady_force_rmse_N']:.3f}N。不过同样接近速度下，首次接触峰值仍约{core[4]['peak_force_N']:.1f}N，不能认为加导纳就消除了冲击。延长接近时间可进一步减小峰值。

实验在PC端MuJoCo中完成，与之前Isaac自由空间轨迹结果分开统计。本轮没有真机、力传感器或Jetson测量，也未完成六维柔顺操作和ROS调度下的接触闭环。

## 目录

- 1. 模型、接触与执行接口
- 2. 位置控制与导纳控制
- 3. 稳定接触和关节跟踪结果
- 4. 噪声、频率与延迟
- 5. 接近速度与首次冲击
- 6. 结论及后续

## 1. 模型、接触与执行接口

Panda来自Google DeepMind的MuJoCo Menagerie，版本固定为 `{rev[:12]}`，保留原模型惯量及关节结构，仅省去装饰网格。为了单独研究末端接触，在手部加入半径15mm球形探头与水平平面。其余机器人几何不参与碰撞，因此当前不研究自碰撞或机械臂其他部位撞桌。

采用独立MuJoCo2.3.7／NumPy1.26.4环境，未修改原ACT环境。物理步长1ms。末端位置与固定姿态通过阻尼雅可比逆解转换为七个关节参考，内环采用带重力/偏置补偿的关节PD力矩，前四关节限87Nm，后三关节限12Nm。除场景重置与逆解用的独立状态外，不直接改写动态机器人的关节位置，没有mocap焊接或把末端瞬移到目标。

接触力来自mj_contactForce的法向分量。这里只有水平平面的单一法向接触，不把它称为真实六维力传感器。接触求解器刚度参数设2000/8000/20000，不能直接当成已标定材料的N/m刚度。

相关代码在 `scripts/contact_task.py`，模型来源与许可在 `assets/panda/SOURCE.json` 和 `assets/panda/LICENSE`。

## 2. 位置控制与导纳控制

先保持0.5秒，再用五次曲线下探；位置控制继续保持固定参考。导纳控制在检测到接触后，以测量力与目标力之差调整法向位置，目标力在0.5秒内升至10N。

采用 `M·偏移加速度＋B·偏移速度＋K·偏移＝测量力−目标力`，M=2kg、B=100Ns/m。主方案K=0，为质量阻尼型力跟随；另用K=200N/m对照虚拟弹簧带来的影响。偏移限−25至50mm、偏移速度限40mm/s，限制条件均保留在配置中。

接触力及状态按1kHz记录，峰值从完整采样中统计，避免较低频率日志漏掉瞬时冲击。稳定段统一为第5至8秒。所有对照使用相同模型、接近参考和力矩限制。

![接触力和位置修正](figures/contact_force.png)

相关代码在 `scripts/contact_task.py`、`scripts/run_suite.py`，参数在 `config/suite.json`。

## 3. 稳定接触和关节跟踪结果

"""
text += table(
    [
        "接触参数档",
        "控制",
        "峰值力 / N",
        "稳定平均力 / N",
        "稳定力RMSE / N",
        "稳定力标准差 / N",
    ],
    [
        [
            v["stiffness"],
            {
                "position": "位置",
                "admittance": "导纳K0",
                "admittance_spring": "导纳K200",
            }[v["mode"]],
            f"{v['peak_force_N']:.2f}",
            f"{v['steady_mean_force_N']:.3f}",
            f"{v['steady_force_rmse_N']:.3f}",
            f"{v['steady_force_std_N']:.3f}",
        ]
        for v in core
    ],
)
text += """
K=0时可将稳定接触力保持在10N附近；加入虚拟弹簧后，保持偏移需要额外力，留下约1.7–1.9N的稳定段误差。这是当前控制形式的取舍，不把弹簧参数越大视为越好。

额外六次运行保存七关节参考及实际位置。下图为中间接触参数档的全程跟踪误差，包含外力作用，不能直接与之前无接触Isaac实验的误差比较。

![七关节参考跟踪](figures/joint_tracking.png)

关闭接触求解的对照得到零接触力；启用接触时机器人产生动态偏移，说明力并非由预设曲线或目标位置人工生成。

![Panda末端接触画面](figures/admittance_contact.png)

相关代码在 `scripts/joint_check.py`；参考与实际关节记录在 `runs/joint_check_v3/`。演示视频为 `runs/contact_suite_v3/position.mp4` 和 `admittance.mp4`。

## 4. 噪声、频率与延迟

对位置与导纳分别组合三档接触参数、50/100/200Hz外环、额外0/20/60ms反馈延迟。导纳测量叠加每次更新标准差0.5N的高斯噪声，三个随机种子，共162组。它是人为设定的扰动，不是自有传感器噪声。

下表为导纳在三档接触参数和三个种子上的平均稳定段力RMSE：

"""
text += table(
    ["外环 / Hz", "附加延迟0ms / N", "20ms / N", "60ms / N"],
    [
        [
            rate,
            *[
                f"{np.mean([v['steady_force_rmse_N'] for v in stress if v['mode']=='admittance' and v['rate']==rate and v['delay_ms']==delay]):.3f}"
                for delay in [0, 20, 60]
            ],
        ]
        for rate in [50, 100, 200]
    ],
)
text += """
附加延迟增大后误差增加，当前设置下更高更新率通常改善稳定段力跟随，但不保证首次碰撞峰值更低。各档每次更新注入相同标准差，连续时间噪声带宽并不严格相同，因此不能把所有差异只归因于频率。

177次运行共享一个姿态及平面接触任务，其中一些位置控制对照对随机种子不敏感；这些不是177个独立场景，也不代表覆盖全部稳定性边界。

## 5. 接近速度与首次冲击

将下探时间设为1、2、4秒，保持最终位置和控制参数不变：

"""
text += table(
    ["接近时间 / s", "控制", "接触峰值 / N", "稳定力RMSE / N"],
    [
        [
            v["approach_s"],
            "位置" if v["mode"] == "position" else "导纳",
            f"{v['peak_force_N']:.2f}",
            f"{v['steady_force_rmse_N']:.3f}",
        ]
        for v in approach
    ],
)
text += """
![反馈延迟与接近时间](figures/delay_approach.png)

导纳要在检测到力之后作出反应，不能撤销已经发生的首次碰撞。当前结果支持将接近速度、接触检测和接触后的柔顺调节一起设计，不能只给原位置控制外面加一层导纳就认为安全问题已经解决。

## 6. 结论及后续

本轮完成了受力接触、力矩驱动和单法向导纳对照，补充了七关节参考误差、噪声、时延和接近速度分析。主要收益是稳定接触力跟随；首次冲击和虚拟弹簧稳态偏差仍需单独处理。

下一步应扩大初始姿态与表面方向，增加缓慢接触切换和退出规则，再将控制节点接入ROS调度并测量实际周期。当前的简化探头和平面、短时单方向实验不等于完整六维柔顺操作，更不能直接用于SO101真机。Jetson及真实传感器测试后续单独开展。

代码目录：

```text
experiments/bytedance_week3/
├── scripts/
│   ├── fetch_model.py       固定来源与模型下载
│   ├── contact_task.py      动力学、力矩驱动、逆解与导纳
│   ├── run_suite.py         参数批量实验
│   ├── joint_check.py       七关节参考跟踪记录
│   └── pilot.py            初始单例检查
├── assets/panda/            模型、碰撞网格、来源与许可
├── config/suite.json        所有条件与控制限制
├── runs/contact_suite_v3/   177组记录及视频
├── runs/joint_check_v3/        七关节参考和实际状态
├── figures/
└── REPORT.md、README.md
```

模型参考：[MuJoCo Menagerie Panda](https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda)。力与运动学接口参考：[MuJoCo官方API](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html)。具体运行方式见README。源码包不含实验记录或视频，本地完整保存。
"""
(B / "REPORT.md").write_text(text)
print("two reports generated")
