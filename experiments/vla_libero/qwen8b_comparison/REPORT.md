# Qwen3-VL 4B 与 8B 离线检查对照

## 范围与方法

使用已有 LIBERO-Long 回合的 8 组 512×512 前后帧，两模型依次运行同一个完成检查提示词。样例在 8B 加载前固定，包含四个历史误判与四个对照，各四个正、负标签。这是难例诊断，不是随机测试集，也不是任务成功率评测。

两模型均为官方 Instruct 基座，BF16、SDPA、CUDA 单卡、greedy 解码、最多 96 个新 token，无 LoRA、无 CPU 卸载；先单次预热，再保留每例首次输出。8B 下载的四个权重分片已与官方 LFS SHA256 核对。旧引擎和分层执行程序没有改动。

## 实测结果

| 指标 | 4B | 8B |
|---|---:|---:|
| 实际参数量 | 4.438 B | 8.767 B |
| 与记录谓词一致 | 4/8 | 4/8 |
| 合法输出 | 8/8 | 8/8 |
| 加载耗时 | 6.15 s | 8.19 s |
| 单次检查耗时中位数，含预处理 | 0.527 s | 0.796 s |
| 检查阶段 PyTorch 分配峰值 | 8.44 GiB | 16.53 GiB |
| 检查阶段 PyTorch 保留峰值 | 8.48 GiB | 16.57 GiB |

显存统计不含桌面、CUDA 上下文等其他进程开销，也不是模型加载全程的整机峰值。耗时为当前提示词和短输出的单次实测。

| 样例 | 记录标签 | 4B | 8B |
|---|---|---|---|
| wrong_plate | False | False | False |
| false_done_mug | False | True | False |
| plate_viewpoint | True | False | True |
| basket_occlusion | True | False | False |
| soup_done | True | True | True |
| soup_not_done | False | False | True |
| false_done_pot | False | True | True |
| microwave_done | True | True | False |

## 解释与限制

本次 8B 修正了 false_done_mug 和 plate_viewpoint 两例，但在 soup_not_done 和 microwave_done 两例上退化，总一致数与 4B 相同；短输出检查耗时中位数增加约 51%。这组结果没有支持直接替换现有 4B，先保留原在线配置。更大参数量不保证这类完成判断更可靠。

8B BF16 在本机 RTX 4090 上已完成独立加载和双图推理。这次没有让它与 π0.5 同驻留；4B 与 π0.5 的既有显存记录不能直接替代 8B 共驻测试。若要替换在线系统，仍需另测量化或分时运行。

标签来自旧回合中对应首子目标的官方谓词，不放入模型提示词。左右参照、物体遮挡、是否仍被夹爪支撑、落在盘中的接触边界等，在单一视角下可能模糊；尤其 plate_viewpoint 与 wrong_plate 两例外观接近而标签不同。应结合原图与完整轨迹解释，不能把每次不一致都当作纯视觉识别错误。

未运行新仿真回合、重规划回归、微调或实体机械臂。8 个诊断样例不足以证明更大的模型能提高最终任务成功率；下一步应先审阅左右参照与遮挡样例，再决定是否做完整配对回归。

## 复现与文件

- 权重：`models/qwen3_vl_8b_instruct/`，revision 见 `model_source.json`。
- 固定输入：`cases.json`；原图位置和哈希随每例保存。
- 原始输出与统计：`results/4b/`、`results/8b/`。
- 相关代码在 `scripts/run_checks.py`、`scripts/prepare_cases.py`、`scripts/finish_comparison.py`、`scripts/summarize.py`。
- 在工程根目录运行 `.venvs/qwen-scene/bin/python experiments/vla_libero/qwen8b_comparison/scripts/run_checks.py --model 8b`。正式保留的首次结果应先备份，再复跑。
