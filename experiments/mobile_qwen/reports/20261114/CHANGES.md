# 本次改动与验证

日期：2026-11-14。未运行新模型、仿真回合、训练、硬件连接或旧π0.5 LoRA闭环复测。未改mobile_vla、vla_libero及历史results。

## 文档

- STAGE_REPORT.md/.html/.pdf：本阶段报告，包含摘要、目录、同场景链路、停止三版结果、定位指标、限制、代码目录。
- PAPER_v0.2.md/.html/.pdf：新的论文草稿，不覆盖十月旧稿。
- recompute_audit.py / AUDIT_METRICS.json：读取已有输出，按相同1014问答/350帧复核严格命中、邻域命中及训练中位固定点；不调用模型。
- README.md：当前默认改为里程碑；历史状态标明历史；明确训练17–49/验证13–16/测试0–12、动作模板共享、真值深度、控制步口径和日志位置。

## 运行代码

- scripts/launch.sh：保留systemd退出码，拒绝已有结果或已有journal的tag，确保logs目录存在。
- scripts/run_batch.sh、run_batch2.sh、run_batch3.sh：启用set -e；失败后不继续批处理或打印成功结尾。
- scripts/run_case_qwen.py：启动失败、导航进程非零退出、导航失败、缺少最终结果均返回非零；捕获后仍尽量清理自有进程并写summary；summary原子替换；启动记录包含所有运行参数；拒绝复用已有journal。
- scripts/durable_log.py：append-only JSONL逐条flush/fsync；同目录临时文件+原子替换进度快照。
- scripts/sim_server_qwen.py：Qwen请求先落盘、回复完整落盘（保留inference元数据）；记录每个已执行控制步的动作和关节状态、官方影子值；逐步progress；正常收尾与异常事件。journal在logs/<tag>_journal，避免提前创建out与父类的独占建目录冲突。
- 当前技能提示词、确认次数、动作生成及停止规则不变。对已有初态8–12的分数不重新命名为修改后代码结果。

## 验证

CPU测试：`.venvs/qwen-scene/bin/python -m unittest discover -s experiments/mobile_qwen/tests -v`，6项通过：异常进程退出后的已写日志保留、原子快照、完整回复及未完成请求、模拟systemd失败的返回码、编排启动失败返回非零、模拟控制步的动作日志与进度。

Python compileall与四个shell脚本bash -n通过；指标补算与报告一致。PDF使用本地Chrome和中文字体生成，检查页数、文本提取和首页排版。Markdown渲染依赖仅装入/tmp/mobile-qwen-doclibs，未安装到系统Python或模型环境。

没有执行完整在线回归；日志新增写盘开销未测量。完整视频/旧trace仍依赖finish，不支持恢复到崩溃处继续运行；最后一个控制步仍可能在仿真推进后、日志写入前遇到强杀。

## 最后单列的待办

旧π0.5 LoRA数值探针已完成，闭环复测未执行。后续仅做原生未合并适配器的小范围闭环核验，使用新tag；不覆盖历史结果，不作为当前mobile_qwen结果的必要前提。
