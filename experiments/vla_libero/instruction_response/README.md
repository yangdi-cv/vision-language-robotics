# 指令响应配对诊断

沿用原π0.5检查点和LIBERO两个Long场景，初态0/11，分别输入指定第一件、指定第二件、原始双物体指令，共12回合。沿第一件轨迹抽取24个同观察点，每点按相同噪声比较三种指令与重复第一件，共96次额外推理。

结果见SUMMARY.json：模型接收的目标词元不同，图像/本体状态/噪声相同，同条件重复动作差0；但4个场景条件的三种指令都先搬了原任务第一件。后续数据与适配在`../skill_bootstrap/`。这不能推断模型在所有场景都忽略语言。

```bash
bash experiments/vla_libero/scripts/python_shared experiments/vla_libero/instruction_response/scripts/analyze.py
```

分析需要本地保存的轨迹和输入；完整模型运行还需要原环境、检查点与LIBERO资产。源码包保留代码与统计，不包含全量大文件。当前run.py拒绝覆盖旧回合。
