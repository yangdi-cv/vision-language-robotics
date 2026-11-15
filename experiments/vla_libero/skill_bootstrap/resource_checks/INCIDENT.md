# 2026-11-14 内存压力导致编辑器退出

systemd-oomd在14:37:17和14:41:39分别杀掉app-com.microsoft.VSCode-3080.scope、app-com.microsoft.VSCode-22943.scope。触发原因为user service内存压力持续20秒以上（85.06%、79.47%），不是这两个百分比的物理内存使用率。系统物理内存约30GiB，没有Swap。

恢复后双模型零增量校验进程实测RSS约21,350,664KiB、显存17,233MiB。校验及其launcher虽然start_new_session分离了Unix会话，仍在VSCode的systemd cgroup里，无法防止oomd将编辑器与实验一起终止。不能把这次故障归因于已证实的显卡或物理硬件故障。

停止双模型并驻。后续校验只串行加载一个模型，保存小的动作和参数哈希后释放模型再加载另一个；使用独立systemd用户服务，MemoryHigh14GiB/MemoryMax16GiB、OOMPolicy=stop，持续记录RSS和GPU使用量。未修改oomd策略，没有关闭系统保护，也没有擅自修改Swap或驱动。

原200步训练、适配器与数据完整；只有尚未完成的重载/闭环评测需要恢复。

已验证systemd用户服务支持内存限制；评测单元ali-vla-eval-safe.service设置MemoryHigh14GiB/MemoryMax16GiB、OOMPolicy=stop，不在VSCode cgroup内。校验进一步改为两个独立进程顺序执行，仅保存动作数组与参数哈希，不在同一进程保留两份模型。另有128MiB上限的小监控单元记录资源到safe_eval.jsonl。旧的同时载入两模型方案停用。

## 分块加载修复验证

采用TensorStore读取当前Orbax OCDBT检查点的原始存储块，限制共享缓存（16/32MiB）与文件读取线程，逐块转BF16并donate写入GPU参数缓冲；不复制完整主机权重，不同时载入两个模型。块内LoRA增量合并通过三种张量形状的数值等价和零增量测试。未改原检查点。

基座和零增量各在独立16GiB上限单元内成功执行。基座观测RSS峰约3.81GiB，cgroup含缓存峰约10.67GiB。51个参数数组、408个存储块逐块哈希一致，零增量与当前基座动作最大差0，通过原0.001阈值。历史进程输出仍有0.00146455差异，新基座与适配策略统一采用本次加载实现，不把历史微小差异算成训练效果。

闭环基座/LoRA评测由safe_eval.py顺序创建独立受限服务，每次只加载一个模型，监控先行，失败不自动重启。
