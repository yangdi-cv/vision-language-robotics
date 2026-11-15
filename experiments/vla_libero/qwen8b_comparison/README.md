# Qwen3-VL 4B与8B离线检查对照

本轮只检查模型容量和少量历史失败画面的判断，不运行新任务回合，不接实体设备。

- 原4B引擎和分层程序不修改。
- 8B使用官方Qwen3-VL-8B-Instruct BF16，revision在model_source.json。
- cases.json在加载8B前固定：8组已有512px前后帧（四正四负），专门包含4B历史误报/漏报及正常对照，不能作为总体准确率。
- 两模型复用当前Planner.check的同一提示词、SDPA、greedy解码和96新token上限，分别在独立进程运行，不与π0.5共驻。
- 标签为旧日志中与首子目标对应的官方目标原子谓词，仅用于离线评分，不进入提示词。遮挡、视角左右和接触阈值可能造成视觉歧义，需结合原图解释。
- 每例保留首次输出；不以重跑覆盖。相同输入不保证GPU内核逐位确定。
- 记录generate耗时、含预处理的wall时间、PyTorch分配/保留峰值；这些不是整机显存，也不是任务成功率。

运行：

```bash
.venvs/qwen-scene/bin/python experiments/vla_libero/qwen8b_comparison/scripts/prepare_cases.py
.venvs/qwen-scene/bin/python experiments/vla_libero/qwen8b_comparison/scripts/run_checks.py --model 4b
.venvs/qwen-scene/bin/python experiments/vla_libero/qwen8b_comparison/scripts/run_checks.py --model 8b
```

8B权重放在models/qwen3_vl_8b_instruct，下载后先按官方LFS SHA256核对，禁止用不完整权重推理。
