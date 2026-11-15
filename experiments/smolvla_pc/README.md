# SmolVLA 工作站实验

直接使用 Hugging Face 的 SmolVLA 预训练视觉语言动作模型，测试 RGB、语言和六维关节状态到 50 步动作序列的推理。不是 Qwen 输出文本后调用预设轨迹，也不是 YOLO 底座追踪。

## 环境与来源

- 环境：项目 `.venvs/smolvla`，Python 3.10、Torch 2.7.1、Transformers 4.51.3、Accelerate 1.7.0。
- 复用厂家工程 `SO101/code/lerobot/src`，LeRobot 0.3.4（Apache-2.0）。没有升级厂家工程。
- 模型：<https://huggingface.co/lerobot/smolvla_base>，固定 revision `5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6`；本地 `models/smolvla_base/`。
- 独立环境通过 `.pth` 只读复用 act-sim 及原 hiwonder 环境已有基础依赖，新安装的 Transformers 等保存在新环境中。原环境没有被卸载或升级。当前不能将这个环境当作与原环境完全无依赖的可迁移环境。
- `requirements-local.lock.txt` 记录新增环境包。该实验不使用继承的 Diffusers；其版本与此环境 HF Hub 不兼容，不用于其他策略实验。

## 运行

在项目根目录执行：

```bash
.venvs/smolvla/bin/python experiments/smolvla_pc/scripts/probe_policy.py
```

脚本显式加载厂家源码路径，不连接串口、不启用电机。结果在 `runs/base_probe/`，包含原始动作 NPZ、输入图像、结果 JSON 及固定噪声的多 seed 条件对照。

## 实验条件与解释

使用 `experiments/so101_pickplace/recordings/grasp_rehearsal_20261025_03/` 第340帧图像及记录状态。该段示教存在人为松开与后段夹爪控制问题，不能作为独立成功测试集。本轮未训练模型。

只提供腕部相机，暂映射为模型 camera1；其他相机不复制、不伪造。状态与动作采用这一条本地记录的均值/标准差初始化归一化，未证实与预训练机器人坐标和视角兼容。基础模型附带的归一化文件含多种 SO100 的统计，不能任意选一组冒充本机校准。当前本地统计仅用于软件和条件敏感性测试。

每个 seed 内使用完全相同的流匹配初始噪声，对照抓取指令、远离目标指令和黑图。动作变化说明输入对输出有影响，不代表输出符合指令语义或能够成功抓取。不能将归一化后的接近值或动作变化当作真机成功率。

下一阶段先采用官方 SO101 数据集核对输入和动作定义，建立离线误差与基线，再在多条完整成功的本机示教上微调。基础权重不能直接启用到真实机械臂。

## PC 视觉工作台

```bash
VLA_CAMERA_ROTATION=0 \
VLA_DETECTOR_PATH=/home/yd/Documents/VLA/experiments/earbud_detector/runs/all100_v1/weights/best.pt \
perception/web_workbench/start
```

地址：<http://127.0.0.1:8080/>。本轮恢复 Qwen、YOLO 和深度观察，未将 SmolVLA 挂接到电机执行。相机旋转以当前硬件状态为准，环境变量不修改原有默认270°。

## 当前真机状态的影子测试

`live_shadow.py` 从白臂读取当前关节位置、扭矩和温度，从工作台共享内存取得新RGB，用SmolVLA生成动作并转换到原校准原始值，检查整段动作是否越界。脚本仅发串口READ，没有扭矩或目标位置写入。需要白臂供电、托稳且扭矩关闭。

```bash
.venvs/smolvla/bin/python experiments/smolvla_pc/scripts/live_shadow.py
```

输出在 `runs/live_时间戳/`。该工具不是运动入口，所有结果均 `execution_allowed:false`；原预训练模型与本机坐标/任务尚未验证，限幅不能替代这项验证。首次单步运动入口尚未启用。
