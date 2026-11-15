# 桌面指定物体抓放

输入“把蓝色方块放进左边盒子”，程序依次完成当前图像读取、Qwen目标解析、RGB-D位置估计、目标条件ACT执行和模拟器状态检查。场景包含三个颜色方块、两个固定盒子，使用ViperX双臂模型，右臂执行、左臂保持。

## 运行演示

在项目根目录 `/home/yd/Documents/VLA`：

```bash
experiments/desktop_pickplace/scripts/act-python experiments/desktop_pickplace/scripts/demo.py \
  --instruction '把蓝色方块放进左边盒子。' \
  --seed 910 --out experiments/desktop_pickplace/runs/demo_910
```

输出目录必须未存在。运行中保存当前图像、语言结果、策略动作、视频和最终检查；Ctrl+C停止。程序只访问模拟器，没有串口或真实机械臂控制接口。

当前执行指令模板为 `把{红色/蓝色/绿色}方块放进{左/右}边盒子。`。其他表述或不明确任务返回 `needs_clarification_or_unsupported_instruction`。例如“把蓝色方块放进盒子里”，模型可能选择左盒，但程序会拒绝该建议。完成判断来自模拟器位置与接触状态，视觉检查模块目前仅作实验。

## 代码结构

| 文件 | 功能 |
|---|---|
| `task.py` | 场景生成、随机布局、RGB-D位置估计、脚本轨迹 |
| `run_episode.py`、`batch.py` | 几何基线和可恢复批量测试 |
| `protocol.py`、`qwen_goals.py`、`train_goals.py` | 语言目标接口、评测、LoRA训练 |
| `goal_worker.py`、`demo.py` | 当前帧语言解析与运行入口 |
| `policy_runtime.py` | 8维末端参考位姿/夹爪动作执行、接触检查 |
| `export_demonstrations.py` | 将成功轨迹转换为训练样本 |
| `prepare_policy_data.py`、`prepare_position_features.py` | 解码缓存、划分及训练统计 |
| `train_policy.py`、`evaluate_policy.py` | 两种输入配置的ACT训练与评测 |
| `check_placement.py` | 全图/局部图放置检查对照 |
| `audit_policy_release.py` | 用保存动作复核放置与释放 |

`train_policy_rgbd.py` 和 `evaluate_policy_rgbd.py` 是参数兼容入口，实际实现共享。`task_v1.py` 保留初始斜向抓取方案。

## 环境

本机Python3.10、Torch2.7.1/CUDA12.6、MuJoCo2.3.7、dm-control1.0.14、gym-aloha0.1.1。ACT使用厂家衍生LeRobot0.3.4源码，位于 `SO101/code/lerobot/src`。Qwen使用固定revision的Qwen3-VL-4B-Instruct，Transformers4.57.6、PEFT0.18.1。本机OpenCV5.0.0来自既有环境；代码使用标准连通域接口。

现有入口复用 `.venvs/act-sim` 与 `.venvs/qwen-scene`。迁移时需要按 `requirements.txt` 和厂家框架依赖建立独立环境，并调整包装器解释器路径。当前工作站已验证运行，未验证另一台机器的一键安装。

需要的模型：

- `runs/goal_lora/adapter`：任务选择LoRA，须配合同一Qwen基座。
- `runs/goal_act_rgbd/best_checkpoint`：增强版目标条件ACT。
- Qwen基座固定revision：`ebb281ec70b05090aa6165b016eac8ec08e71b17`；从首轮项目缓存加载，缺失时可用 `experiments/alibaba_extension/scripts/download_qwen_base.py` 下载。

代码包不包含环境和基座大模型；模型和实验附件包另行提供。两个包解压到同一目录，保留 `experiments/desktop_pickplace`、`SO101`、`perception` 层级。附件含选中的ACT权重、LoRA、视频、动作CSV及示教清单；不含533MB的NPZ和约6GB缓存，可通过导出脚本从附带的batch002重建。

## 复现实验

```bash
# 可恢复物理基线；已完成的episode会跳过
/usr/bin/python3 experiments/desktop_pickplace/scripts/batch.py --out experiments/desktop_pickplace/runs/new_batch
# 首轮脚本默认读取batch002；换目录时同步调整数据入口
experiments/desktop_pickplace/scripts/act-python experiments/desktop_pickplace/scripts/export_demonstrations.py
experiments/desktop_pickplace/scripts/act-python experiments/desktop_pickplace/scripts/prepare_policy_data.py
experiments/desktop_pickplace/scripts/act-python experiments/desktop_pickplace/scripts/prepare_position_features.py
# 训练两个配置，均使用新输出目录
experiments/desktop_pickplace/scripts/act-python experiments/desktop_pickplace/scripts/train_policy.py \
  --steps 10000 --out experiments/desktop_pickplace/runs/new_rgb_policy
experiments/desktop_pickplace/scripts/act-python experiments/desktop_pickplace/scripts/train_policy.py \
  --target-position --steps 10000 --out experiments/desktop_pickplace/runs/new_rgbd_policy
# 指定新的测试布局
experiments/desktop_pickplace/scripts/act-python experiments/desktop_pickplace/scripts/evaluate_policy.py \
  --target-position --model experiments/desktop_pickplace/runs/new_rgbd_policy/best_checkpoint \
  --seed-start 1000 --seed-count 10 --out experiments/desktop_pickplace/runs/new_test
```

示教NPZ是本任务的自定义格式，包含320×240图像、关节/夹爪/任务符号状态及8维末端参考动作；不是旧ALOHA的14维关节动作。图像来自H264渲染视频，标注为下一25Hz采样点的控制目标；中间50Hz控制点未作为独立图像保存。54/6/6轨迹按布局seed划分，训练仍只有4种布局；缓存约6GB，原始压缩样本约533MB。

## 场景和依赖来源

机器人与桌面网格来自gym-aloha0.1.1（Apache-2.0），参照其末端约束控制方式增加多物体和盒子。ACT使用厂家衍生LeRobot0.3.4（Apache-2.0）。Qwen基座来源及版本沿用原项目。算法和场景接口在本目录修改，原工程未替换。
