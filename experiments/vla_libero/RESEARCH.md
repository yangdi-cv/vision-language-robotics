# 实现参考与选型

## 采用

- OpenPI：https://github.com/Physical-Intelligence/openpi 。Apache-2.0；上游另列Gemma相关许可。采用官方π0.5 LIBERO检查点与相应输入输出转换，不将官方训练成果算作本项目训练成果。
- 官方LIBERO评估入口：https://github.com/Physical-Intelligence/openpi/tree/main/examples/libero 。保留256渲染、224模型输入、双视角旋转180°、8维状态、7维动作、每5步重新观察与10步初始等待。
- LIBERO：https://github.com/Lifelong-Robot-Learning/LIBERO 。MIT；机器人与物体资产另按各自来源许可。采用原任务谓词和公开初始状态，不使用自定义成功判据替换官方任务。
- robosuite：https://github.com/ARISE-Initiative/robosuite 。MIT；LIBERO所用OSC_POSE低层控制经过关节力矩与MuJoCo接触，不直接改写末端或物体位置。

## 比较与取舍

- SmolVLA已有本机环境与SO101微调记录，但基础权重不能直接等同于LIBERO任务专家；若第一轮就重新训练和接环境，难以区分策略与接口错误。因此先复现有官方任务检查点的π0.5。
- 4090拥有24GB显存，官方列出的推理门槛可满足；不默认能够进行全参数训练。后续训练单独评估资源与任务数据。
- 保留固定基座操作范围。LIBERO成功不代表移动操作或SO101泛化，机器人动作表示和相机输入不同，不能直接把本轮权重发给实体机械臂。
- 官方基准通常每任务50次，本轮首先每任务3个公开初态，并报告小样本边界，不声称复现官方整套分数。
- 不把语言干预中的原任务失败直接解释为不服从指令。需要结合替代目标、实际运动及相同观察的动作变化分析。

源码固定版本记录在third_party/sources.json；安装时只修改新环境。
