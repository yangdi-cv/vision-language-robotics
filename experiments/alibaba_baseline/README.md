# 离线具身 baseline：复现入口

先阅读 [阶段报告](REPORT.md) 的成果边界及 [执行记录](RUN_LEDGER.md)。所有本目录实验只处理图片/数据或 MuJoCo，不连接机械臂。

## 本机环境

项目根目录 `/home/yd/Documents/VLA`；原厂家衍生 LeRobot 0.3.4 在 `SO101/code/lerobot/src`。ACT Python 3.10 独立 venv `.venvs/act-sim`，Qwen `.venvs/qwen-scene`，继承原项目 Python 的基础包；没有修改原 Conda 或系统 Python。脚本 `act-python` 显式选择当前厂家源码并清除 ROS Python 污染，设置 EGL 无窗口渲染和系统 CA 证书。

主要版本：torch2.7.1、torchvision0.22.1、gym-aloha0.1.1、dm-control1.0.14、mujoco2.3.7；Qwen 使用 transformers4.57.6、peft0.18.1、accelerate1.15.0、huggingface-hub0.36.2。完整安装状态见 `setup_logs/*requirements.lock.txt`，两个环境 `pip check` 通过。锁文件含本机已有包与路径，不是跨机器直接安装的最小依赖清单。

## 在现有工作站重跑

```bash
cd /home/yd/Documents/VLA
experiments/alibaba_baseline/scripts/act-python experiments/alibaba_baseline/scripts/download_public.py
# 重建约18GB RGB缓存；不要与使用该缓存的训练同时执行
experiments/alibaba_baseline/scripts/act-python experiments/alibaba_baseline/scripts/prepare_act_data.py
experiments/alibaba_baseline/scripts/act-python experiments/alibaba_baseline/scripts/train_act.py --steps 5000 --out experiments/alibaba_baseline/runs/new_train
experiments/alibaba_baseline/scripts/act-python experiments/alibaba_baseline/scripts/evaluate_act.py --model experiments/alibaba_baseline/runs/new_train/final_checkpoint --out experiments/alibaba_baseline/runs/new_eval
experiments/alibaba_baseline/scripts/act-python experiments/alibaba_baseline/scripts/integrated_demo.py --guarded --out experiments/alibaba_baseline/runs/new_demo
```

输出目录要求未存在，保留历史。Ctrl+C 停止本轮仿真或训练；没有机械臂扭矩可关闭。

Qwen 入口使用 `env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH .venvs/qwen-scene/bin/python`，接 `scripts/qwen_benchmark.py`、`qwen_lora.py` 或 `verify_lora_reload.py`。首轮脚本目录名固定，重跑应复制脚本并修改输出目录及相应输入路径，避免覆盖；不要删除原始证据。基座由固定 revision 自动下载，约8.3GB，首次需要网络。LoRA 重加载必须使用相同训练提示词协议，缩写协议的失败记录见报告。

## 交付包与移机说明

包内保留报告、样本、全部结果/视频、日志、LoRA 适配器、自训练 ACT **最终**权重及本次使用的厂家框架源码。省略18GB解码缓存、公开 ACT 权重、公开50条示教原视频、重复的最佳 ACT 权重和所有环境。公开输入由固定 revision 下载，缓存由脚本重建。

包保留项目根目录结构，必须解压后在其根目录运行。移机需先建立 Python3.10 / CUDA匹配的两个独立环境，按上述版本及厂家 `pyproject.toml` 安装依赖，并调整 `act-python` 中本机旧 Conda PATH；没有验证另一台机器上的一键安装。训练参数和代码可复现，GPU/cuDNN 行为不保证逐位一致，重跑数值可能变化。

建议阅读顺序：报告 → 逐次预测/奖励 → `train_act.py` 数据边界与损失 → `evaluate_act.py` chunk缓存 → `qwen_lora.py` 标签监督 → `integrated_demo.py` 的模型建议与独立授权。真实失败记录也是交付的一部分。
