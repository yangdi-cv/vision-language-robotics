# 运行记录

本阶段训练与PC完整评测均已结束。板端2B无swap的350帧/1014问答仍在运行，预计约80–100分钟（15W，单条5–7秒）。不要重复启动已有tag。

## 已完成

- 2B官方权重下载并校验SHA256；训练600步，同4B协议逐字段相等；best600，验证中位1.02554px，训练334.69s，1605632可训练参数。
- PC 2B BF16基座与LoRA各1014问答；2B/4B NF4各1014问答。所有*.resources.json完整。
- 2B BF16严格328/350、邻域345/350、3D1.5154cm；2B NF4严格319、邻域335、3D1.5506cm；4B NF4严格323、邻域347、3D1.3402cm。
- 基座2B长提示词在40token上限下843条JSON截断；CPU只读坐标诊断严格6/350、邻域22/350、3D16.11cm。正式输出未改，未重调模型。
- NF4导出最初内存膨胀，保留visual与lm_head的BF16、独立单元后通过。PC驱动收尾因运行时脚本编辑报错，所有评测已落盘且无重复，不重跑。

## 当前后台单元

PC：ali-jetson-2b-noswap-driver-20261114，等待板端完成、同步回文件并恢复swap。
板端：ali-jetson-2b-lora-nf4-512-noswap-20261114，MemoryHigh5G/Max6G。
PC：ali-stage2-finalize-20261114，等2B正式结果后尝试4B加载/单验证图资格检查（不做4B350帧），然后自动生成SUMMARY和REPORT.pdf。

主机连接优先USB192.168.55.1，局域网192.168.31.168备用；用户csip。源代码不存密码，自动化通过SSHPASS环境传入sshpass-e。

板端根目录：/home/csip/ali-jetson/project/experiments/mobile_qwen/jetson/stage2_20261114。
PC同目录：/home/yd/Documents/VLA/experiments/mobile_qwen/jetson/stage2_20261114。

只读进度：SSHPASS=<现有凭据> python3 scripts/progress.py（从项目根调用时带完整路径）。

## 后续收尾

确认无swap正式结果1014条、350帧、输入哈希一致；核对逐条输出率。write_report.py必须在SUMMARY齐备且accuracy_usable_without_swap=true才生成。

自动finalize完成后检查报告PDF排版、可选4B探针返回码与资源（不能把单验证图称全量精度）、恢复swap和电源模式状态；修正任何报告不实之处，补PROJECT_LOG。仅可供性，不进入延迟或里程碑。

384/256精度与P50/P95、MAXN SUPER放到下一阶段；当前固定512保持单因素。旧π0.5 LoRA闭环复测仍最后单列。

## 保留的无效尝试

早期jetson_2b_lora_nf4_512运行31问答，模型swap0但全局pswpin/out活动，已终止并标无效，不能计为正式350帧。正式新tag以_noswap结尾：加载后产生.load.json，外部控制器swapoff-a并触发.swap_ready，确认/proc/swaps只有表头才开始单遍测试。其后全局交换页应无增量。

板端kernel不提供cgroup memory.peak。资源报告采用VmHWM、CUDA峰值、稳定期cgroup采样最高及从加载前开始的tegrastats总RAM峰值；不可相加或冒称cgroup加载峰值。

## 完成更新（2026-11-15）

后台于00:43 AEDT自动完成并生成4页REPORT.pdf。正式2B结果350帧1014问答、无swap；原文和坐标均774/1014一致。4B单验证图加载探针通过（39.52s），未做板端350帧。额度恢复后已独立复核结果计数、协议和资源；板端推理进程退出，15W不变，swap恢复且使用0。本页前面的“仍在运行”是运行期间状态，以本节为准。
