# 参考实现与选择

本轮没有重写VLA或机器人控制器，使用现有固定版本openpi与LIBERO。

| 参考 | 采用的部分 | 依赖和限制 |
|---|---|---|
| Physical-Intelligence/openpi（Apache-2.0） | 原π0.5配置、输入/动作变换、LoRA层和train_step | 使用本地固定提交；LoRA A/B均随机初始化，step0不能当成零增量；离线损失不能代替生成动作与闭环测试 |
| ARISE-Initiative/robosuite（本地1.4.1许可证见安装包LICENSE） | 原OSC与Panda关节/接触仿真 | 不升级控制器，不做mocap控制；物体相对抓取姿态仍需验证，候选中3条夹持失败剔除 |
| NVlabs/MimicGen | 参考物体相对姿态和示教转移思路 | 未引入或复制其代码、资产与依赖，不宣称实现完整MimicGen |
| google/orbax与google/tensorstore | 原OCDBT参数结构和公开TensorStore分块读取 | 固定已安装版本；整树并发恢复存在主机内存峰值。本轮共享小缓存、存储块循环及GPU donation，不更换检查点格式 |

来源：

- https://github.com/Physical-Intelligence/openpi
- https://github.com/ARISE-Initiative/robosuite
- https://github.com/NVlabs/mimicgen
- https://github.com/google/tensorstore
- https://orbax.readthedocs.io/en/stable/api_reference/checkpoint.handlers.html

已核对Orbax的restore_concurrent_gb配置及本地TensorStore序列化代码。此检查点包含约604MB的原始大存储块，单纯设置很低的并发预算不能保证所有数组可读。本轮按实际chunk_layout读取，每个块转换成约302MB或更小的BF16块后传入GPU，不同时保留完整主机权重。分块数学与零增量控制分别验证，完整原基座不随代码包分发。
