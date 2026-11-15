# 采用的接口与边界

先检索并核对官方实现，再复用当前固定版本，不新增策略或物理引擎依赖。

- openpi（Apache-2.0）：https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/main.py 。复用已有evaluate.make_input和pi05_libero检查点，显式提供动作噪声，每次执行动作块前5步。官方长任务评测会在完成时结束；本实验不能照搬这个结束条件，否则等于用真值替视觉系统停止。
- LIBERO（MIT）：https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/libero/libero/envs/env_wrapper.py ，以及官方collect_demonstration.py/create_dataset.py中的连续执行接口。使用ignore_done=True，并忽略step返回的原长任务done；目标关系只在写日志与离线评分中读取。
- robosuite：复用已安装版本的check_contact与对象contact_geoms，在离线评分中判断目标是否仍接触夹爪。没有把闭合空爪误作未释放，也没有用接触标签决定停止。

本地上游提交见vla_libero/third_party/sources.json，执行源码在PROTOCOL.json与archive中冻结。Qwen复用项目已有engine.generate与Planner.check/translate；新增部分仅为给定双物体词表中的目标选择、停止状态机和配对评测。

这里不是原LIBERO-Long成绩：同一世界中改变指令，新增“只搬指定一件”和保留物体中心位移约束。控制系统没有目标位置、接触真值或任务谓词输入。已知物体名称/ID词表由任务配置提供，不是开放词汇可供性定位。
