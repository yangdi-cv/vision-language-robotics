# VLA统一仿真验证

使用官方OpenPI π0.5 LIBERO检查点建立闭环基线，之后以固定评估协议检查动作接口、视觉与语言依赖、扰动和失败类型。

## 范围

- 独立环境 `.venvs/openpi-libero`，不修改厂家工程、已有SmolVLA或ROS环境。
- 只运行仿真，不打开实体串口、不启动机械臂。
- 首先复现官方任务检查点，不将官方训练结果当成本项目自主训练成果。
- LIBERO为固定基座操作；本轮不宣称已完成导航抓取统一系统。

## 计划

1. 固定源码版本、权重、依赖和任务初始状态。
2. 核对官方输入预处理、动作单位及低层控制接口。
3. 模型闭环执行并记录动作、实际状态、任务判据、耗时与视频。
4. 固定初始状态进行多任务评估；基线与干预采用相同初态。
5. 分别检查视觉、语言和执行方式的影响，保留失败与统计不确定性。
6. 结果整理为中文报告、源码与视频。

下载与安装日志见 logs/；上游源码版本见 third_party/sources.json。

## 从空目录准备（保持解压后的 experiments/vla_libero 层级）

```bash
python3 -m pip install --target experiments/vla_libero/tools uv
python3 experiments/vla_libero/scripts/fetch_sources.py
experiments/vla_libero/tools/bin/uv python install 3.11
experiments/vla_libero/tools/bin/uv venv --python 3.11 .venvs/openpi-libero
.venvs/openpi-libero/bin/python experiments/vla_libero/scripts/install.py
.venvs/openpi-libero/bin/python experiments/vla_libero/scripts/install_simulation.py
python3 experiments/vla_libero/scripts/download_segments.py
experiments/vla_libero/scripts/python experiments/vla_libero/scripts/tests.py
experiments/vla_libero/scripts/python experiments/vla_libero/scripts/simulation_smoke.py
experiments/vla_libero/scripts/python experiments/vla_libero/scripts/evaluate.py --phase smoke
experiments/vla_libero/scripts/python experiments/vla_libero/scripts/evaluate.py --phase all
experiments/vla_libero/scripts/python experiments/vla_libero/scripts/analyze.py
experiments/vla_libero/scripts/python experiments/vla_libero/scripts/verify.py
```

依赖下载与安装按顺序完成；不要对同一环境同时运行两个安装器。权重从官方公开GCS下载，按对象CRC32C、可用MD5和大小校验。分段下载支持未完成段重试。首次推理可能包含JIT编译；报告将其与后续调用区分。模拟器在同步推理期间暂停，不把离线闭环成功称为实时控制。

单独重跑一个已完成任务（不加入正式统计）：

```bash
experiments/vla_libero/scripts/python experiments/vla_libero/scripts/evaluate.py --phase baseline --suite libero_spatial --task-id 0 --limit 1 --repeat-tag demonstration_01
```

结果写入 `replays/demonstration_01/`，原始评估不覆盖。`pipeline_status.json`记录串行批次当前阶段；`logs/pipeline_*.log`记录每阶段输出。任何实体设备测试另行安排，本入口只调用LIBERO。

## 官方成功后的补充检查

`release_stability.py` 在单独进程中重放成功回合的实际动作，核对逐步关节状态，再以零笛卡尔增量并松爪保持1秒，统计最后0.5秒目标是否持续成立。它不修改官方分数，不增加VLA运行数；固定松爪指令也不算模型生成的动作。

```bash
experiments/vla_libero/scripts/python experiments/vla_libero/scripts/release_stability.py
```

下一轮微调的数据来源已核查：`physical-intelligence/libero`的对应黑碗任务有45条公开示教。候选按整条35/5/5划分，见 `configs/training_data_plan.json`；本轮没有启动新训练。公共基准初态不保证与示教初态分离，后续泛化需单独定义新布局测试。

完成所有运行后的报告与演示输出：

```bash
experiments/vla_libero/scripts/python experiments/vla_libero/scripts/release_stability.py
experiments/vla_libero/scripts/python experiments/vla_libero/scripts/benchmark_inference.py
experiments/vla_libero/scripts/python experiments/vla_libero/scripts/analyze.py
experiments/vla_libero/scripts/python experiments/vla_libero/scripts/verify.py
experiments/vla_libero/scripts/python experiments/vla_libero/scripts/video.py
experiments/vla_libero/scripts/python experiments/vla_libero/scripts/report.py
```

`release_checks/` 是对已保存动作的物理重放与松爪检查，不是增加的自主策略测试；`inference_benchmark.json`使用相同输入交错测量去噪步数，不与其他GPU任务同时运行。
