# 策略适配器

本阶段输出两个原生π0.5 LoRA增量：

- `lora_pilot/adapter_best.npz`：首轮200步，验证选中step200。
- `lora_paired/adapter_best.npz`：从首轮继续600步，阶段/目标分叉联合验证选中step400。

对应原基座：`gs://openpi-assets/checkpoints/pi05_libero/`，本地`experiments/vla_libero/checkpoints/pi05_libero/`。固定openpi提交与模型资源清单见`third_party/sources.json`和原检查点下载记录。适配器不包含原基座，不能单独推理；各自BEST.json记录SHA256、参数数和选模指标。

当前执行入口runtime.py使用bounded_restore.py，在原检查点的BF16存储块上合并适配增量，再用原模型结构推理。训练使用未合并LoRA结构；零增量与相同加载路径基座的408块哈希和动作全部一致。历史跨进程输出的微小差异保留在校验记录中。

代码ZIP不含大权重。交付目录内的“策略适配器”单独提供两个NPZ及对应说明。完整基础模型须按其原许可证获取，不随包重新分发。
