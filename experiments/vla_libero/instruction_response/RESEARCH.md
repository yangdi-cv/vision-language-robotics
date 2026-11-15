# 实现依据与范围

复用本地固定版本openpi（Apache-2.0）与LIBERO（MIT），提交号见PROTOCOL.json/third_party/sources.json。实现前核对官方LIBERO推理接口、动作块执行、初态加载，以及本地Policy.infer的输入变换和显式噪声接口。

- 官方策略示例：https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/main.py
- 官方环境说明：https://github.com/Physical-Intelligence/openpi/tree/main/examples/libero
- 策略实现（本地固定版本）：third_party/openpi/src/openpi/policies/policy.py
- 原始评测输入构造：../scripts/evaluate.py

本轮不另写策略或底层控制器。只新增受控输入和实际动作记录，区分“语言改变了动作数值”和“模型选择了要求的物体”。原LIBERO整任务done只记日志，不作为停止条件。

检查公开LIBERO-Object BDDL时发现：不同目标任务不仅指令不同，物体组合和摆放也不同。其高成功率不能直接证明同一Long场景下可以任意交换目标，不用更换场景的结果替代原场景修复。

初态0为多轮开发使用，初态11用于本次配对诊断；本研究是诊断，不是独立正式成功率评测。同状态反事实检查沿target0执行路径采样，因此后期状态也有anchor选择偏差。
