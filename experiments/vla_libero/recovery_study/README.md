# 分层执行与恢复分析

先读 REPORT.md。REPORT.pdf为同内容的便于阅读版本。

本目录记录三阶段工作：历史结果审计、配对恢复对照、目标约束与指定指令探测。所选故障状态未被救回，不能将接口修正描述成成功率提高。

- summary.json：新增实验汇总；26次恢复尝试含2次格式错误，另有2次指定指令、2次正常初态检查。
- protocol.json、contract_protocol.json、guided_protocol.json、typed_gate_protocol.json：各阶段设计；后两阶段为先导观察后的探索性工作。
- episode_index.csv：历史回合与源文件哈希，不修改历史结果。
- results/：每次首次输出、动作、起点、图片和视频。正常初态检查在上一级hierarchical/recovery_contract_smoke/。
- archive/：原入口和实验脚本版本快照。

新入口在 ../scripts/run_hierarchical_recovery.py，原入口仍保留。命令、环境、约束和文件层级见报告末节。
