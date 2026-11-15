# CAG免训练形式的本地复现

参考论文：https://arxiv.org/html/2602.17659v1 （式4，TF变体）。未核实可用官方代码，此处是根据公开公式实现的最小复现，不声称逐行复现作者实现。

同一个π0.5 LIBERO基座、同一图像/状态/采样噪声，顺序推理任务指令与空任务，最终动作 `a = a_cond + (w-1)*(a_cond-a_null)`。w=1严格返回原动作，w=1.5固定来自论文；不再搜索更多尺度。两个分支使用一份权重，不驻留两个模型，不训练VA先验。

空任务在原输入变换前设置prompt为空，原图像/状态输入均保留。实际pi05_libero配置discrete_state_input=False，状态不编码进任务文本；控制检查按实际配置验证，不能假定空任务包含State文本。当前混合7维反归一化后的控制量，包括夹爪；未添加裁剪，机器人原控制器裁剪行为保持不变。不是在流匹配的每步速度场内引导。论文官方实现未核实的噪声配对和动作空间细节，采用本地明确约定并记录。

运行环境和模型沿用openpi-libero。所有GPU任务应从systemd独立单元启动，MemoryHigh=14G、MemoryMax=16G、OOMPolicy=stop；启动器为`../skill_bootstrap/scripts/python_model.sh`。禁止在编辑器进程组直接加载大模型。本阶段不连接硬件。

`run.py`先验证真实模型尺度1恒等、同噪声重复恒等、空任务仍包含相同State，再运行16个开发回合：两场景×初态0/11×两目标×两方法。布局已参与旧诊断，不作为独立泛化测试。300策略步+25保持步，每次预测10步执行前5步，与旧引擎一致。Qwen、适配器、教师坐标均不参与控制。评分真值仅记录与计算结果。

脚本拒绝覆盖已有PROTOCOL和已完成回合；进程异常不自动追加重试。每次查询保存两个分支、组合动作、噪声与输入，最终结果由独立分析重新核对。

源码入口：
- scripts/mixing.py：公式与边界条件
- scripts/cag_policy.py：真实模型双分支与记录
- scripts/run.py：尺度1控制检查及开发回合
- scripts/summarize.py：输入、公式、实际执行、判分核验
- tests/test_mixing.py：5项合成数组CPU测试，不依赖GPU或模型

CPU测试：在装有numpy的Python中运行 `python -m unittest discover -s experiments/vla_libero/cag_reproduction/tests -v`。

完整模型测试需要本地固定openpi/LIBERO及基座资产；源码包不含它们。分析入口：`experiments/vla_libero/scripts/python_shared experiments/vla_libero/cag_reproduction/scripts/summarize.py`。

## 本轮结果

16个开发回合已完成。原策略/尺度1.5首次抬起正确目标均4/8，完整单件搬运0/8、1/8；所有回合先抬起的仍是目标0。唯一完整收益是汤罐原优先任务，没有修正目标1优先指令。查询中位时间约0.0818秒与0.1570秒。到此收尾，不搜索更多尺度，不启动训练。

真实模型尺度1与原输出差0，同噪声重复差0。首次预检的配置假设错误及当时源码保存在archive/attempt01_preflight，未生成回合。运行结果与冻结协议同时保留，重复运行应另建实验输出目录，不删除原记录或覆盖PROTOCOL。
