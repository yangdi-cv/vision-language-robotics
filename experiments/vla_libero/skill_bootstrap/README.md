# 同场景单物体技能适配

本阶段承接`../instruction_response/`的12回合诊断：语言改变了动作数值，却没有让Long场景中的执行顺序随目标改变。先生成经过物理验收的单物体示教，再对现有π0.5做200步原生LoRA小试；最后在与训练/验证分开的初态比较基座与适配策略。

## 环境

PC RTX4090，使用现有`.venvs/openpi-libero`，不需要Jetson，不连接实体机械臂。公开基座与LIBERO资产仍位于原目录，未修改。GPU复现需要这些已有资源；源码包不包含基座、全部图片/轨迹或第三方仓库。

```bash
# 原有PC环境，以下从VLA项目根目录运行
source .venvs/openpi-libero/bin/activate
# 已保存结果汇总，不加载模型
bash experiments/vla_libero/scripts/python_shared experiments/vla_libero/skill_bootstrap/scripts/summarize.py
# 核对仿真示教的物理标签；需要本地原始轨迹
bash experiments/vla_libero/scripts/python_shared experiments/vla_libero/skill_bootstrap/scripts/audit_data.py
```

## 数据和训练

`teacher.py`使用已验证的相对抓取姿态和原robosuite OSC接口，动作仍经过原有机器人动力学和接触模型。物体坐标只用于生成示教及离线评分；这不是策略评测。没有mocap或物体瞬移。训练初态20–25，验证26–27，策略测试28–29。失败示教完整保留，不训练。数据由`generate.py`按DATA_PROTOCOL.json生成，接受名单在MANIFEST.json。

训练调用固定版本openpi的`init_train_state`与`train_step`，只更新LoRA参数，冻结视觉编码器和全部基座参数。图像预处理、动作归一化、分词器和动作块格式沿用原模型。训练按任务/目标均衡采样，模型输入不包含物体位置或教师阶段标签。训练已完成，不要为了查看结果重复运行。

`lora_pilot/adapter_best.npz`是最佳验证检查点的约50M参数增量，不是完整基座。原生openpi的LoRA A/B初始化都非零，step0000不是“未加适配器基座”，验证损失中的baseline字段仅指训练前随机LoRA。真正零增量另存adapter_zero.npz。

## 受限运行与内存

曾因大模型加载/校验内存峰值触发systemd-oomd，导致VSCode进程组退出。不要在编辑器子进程内直接运行双模型，也不要同时启动base与lora。

现行`bounded_restore.py`复用TensorStore/Orbax OCDBT格式，按原始存储块恢复，逐块BF16转换和LoRA合并，GPU缓冲donation更新。共享缓存限制16/32MiB；基座与适配模型统一使用此加载方式。原基座文件不变。

`safe_eval.py`将每组放进独立systemd用户服务，MemoryHigh14GiB、MemoryMax16GiB、OOMPolicy=stop，顺序执行；`watch_resources.py`记录RSS与包含页缓存的cgroup内存，这两个指标不能混用。超限时实验单元可退出，不自动无限重试；不要自行关闭oomd。

```bash
# 完整运行前先看已有policy_eval/，程序拒绝覆盖，默认不重复评测。
# 新评测必须另建输出目录和实验版本，再按safe_eval.py的受限服务方式运行。
```

旧的`runtime_unmerged_reference.py`与`runtime_full_restore_reference.py`仅保留加载问题的开发参考，不是当前生产入口。当前仅用runtime.py + bounded_restore.py + chunk_math.py。

## 不需要GPU的数学验证

```bash
python3 -m venv /tmp/ali-skill-tests
/tmp/ali-skill-tests/bin/pip install -r experiments/vla_libero/skill_bootstrap/requirements-offline.txt
/tmp/ali-skill-tests/bin/python -m unittest discover -s experiments/vla_libero/skill_bootstrap/tests -v
```

创建隔离环境需要带venv/ensurepip的Python；系统Python缺少该组件时，可换用已安装的Python3.11。本次实际使用独立Python3.11环境验证。

四项检查使用合成小张量，验证注意力、KV、前馈三种布局下分块合并与整块乘法完全一致，并检查零增量不改变BF16比特。这不等于GPU评测或完整仿真复现。

## 代码位置

- `../instruction_response/scripts/run.py`：固定观察/噪声的目标交换与实际闭环，评测复用此执行引擎，只重定向输出目录。
- `scripts/teacher.py`、`generate.py`、`audit_data.py`：示教生成、划分、独立物理复算。
- `scripts/train_lora.py`：200步适配、16个固定验证样本、验证选模。
- `scripts/chunk_math.py`、`bounded_restore.py`、`runtime.py`：小块合并与实际模型重载。
- `scripts/evaluate_policy.py`、`safe_eval.py`、`summarize.py`：输入/动作检查、受限闭环和统计。
- `resource_checks/`：真实内存故障及修复记录；`lora_pilot/`：训练、验证、选模和权重。

本阶段只测试两个Long场景的指定单件搬运，不是完整LIBERO基准成绩。教师成功率与策略成功率分开报告；验证帧并不独立，单种子小试也不构成泛化结论。

## 目标分叉继续适配

首轮200步后，第二件先搬仍为0/4。PAIRED_BRANCH_AUDIT.json记录9个训练布局配对和4个验证布局配对，分叉前RGB/腕部图/状态完全相同。train_pair_lora.py用50%分叉配对、50%原阶段采样，从首轮权重继续600步batch2；仅使用原训练/验证初态，按阶段和分叉验证损失各半选中step400。不是只改采样的消融（更新数、优化器也不同）。新的独立策略测试初态为30/31，入口safe_eval.py --paired，结果SUMMARY_PAIRED.json。

本次额外适配完成后不继续扩大训练或沿用测试结果调参；先报告实际闭环结果。
