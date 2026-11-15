# 分层闭环恢复与完成检查器

先读SUMMARY.md（一页摘要），详细过程见REPORT.md。附带MODEL_CARD.md说明权重范围。

- 统一入口：../scripts/run_hierarchical.py，execution-profile与planner-profile配置选择；旧recovery入口仅兼容包装。
- t9_protocol/results/conclusion：四组受控定位；未稳定复现旧超时，不指定原因。
- injection_protocol/results/summary：三类扰动27次；继续2/9、整句3/9、Qwen0/9。
- data/cases.json：720组帧与标签索引；audit.json记录划分。图像来自旧基线视频，可用prepare_checker_data.py提取。
- results/checker_lora/best_adapter：首轮LoRA；protocol/history/summary保存训练与测试，calibrated_summary为验证阈值分析。
- scripts/check_completion.py：只输出图像判断，不控制机器人；其余脚本分别负责数据、训练、校准和试验。
- archive：合并入口前版本，历史结果未覆盖。

独立测试改善了常规识别，但边界误报仍高。此次不替换闭环检查器，不继续扩大恢复实验。
