# 阿里方向追加实验：入口与复现

本目录追加原阶段实验，不替换 `experiments/alibaba_baseline`。先看 `REPORT.md` 和 `RESEARCH_DRAFT.md` 中的结果、数据来源及未完成边界。全部实验只调用本地模型或仿真，不打开真实机械臂/IMU串口。

## 环境与依赖

复用 `.venvs/act-sim`（厂家LeRobot0.3.4 + gym-aloha0.1.1/MuJoCo2.3.7）、`.venvs/qwen-scene`（Transformers4.57.6、PEFT0.18.1）以及系统ROS2Humble/Gazebo11/MoveIt2。新增bitsandbytes0.48.2只安装到Qwen独立venv，未升级原依赖；pip check通过。完整版本在 `setup_logs/qwen-requirements.lock.txt`。公开输入固定版本仍见原阶段 `PROVENANCE.json`。

执行位置 `/home/yd/Documents/VLA`，所有输出目录不得已存在。以下是当前工作站复现入口，移机需要建立相同独立环境和ROS包，不称未经测试的一键移机。

## ACT续训与调参

```bash
experiments/alibaba_baseline/scripts/act-python experiments/alibaba_extension/scripts/train_act_extended.py --steps 15000 --resume-model experiments/alibaba_baseline/runs/own_act_train_001/final_checkpoint --out experiments/alibaba_extension/runs/new_train
experiments/alibaba_baseline/scripts/act-python experiments/alibaba_extension/scripts/evaluate_act_extended.py --model experiments/alibaba_extension/runs/new_train/best_checkpoint --action-steps 100 --episodes 30 --seed-start 3000 --out experiments/alibaba_extension/runs/new_eval
```

数据使用原18GB缓存，40/5/5 episode拆分不变，优化器重新初始化，仅续接已训练权重。正式报告的动作间隔选择用2000–2004五个调参种子；测试3000–3029在选择后执行。不要把重复测试同种子作为增加独立样本数。

## Qwen RGB/深度、LoRA、INT8及KV Cache

```bash
# 生成独立于真实单条视频的模拟场景；原目录已存在，重建需更换脚本输出路径
experiments/alibaba_baseline/scripts/act-python experiments/alibaba_extension/scripts/build_rgbd_cases.py
# 新的独立输出目录
 env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH .venvs/qwen-scene/bin/python experiments/alibaba_extension/scripts/qwen_rgbd_eval.py --out experiments/alibaba_extension/runs/new_rgbd_eval
 env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH .venvs/qwen-scene/bin/python experiments/alibaba_extension/scripts/train_rgbd_lora.py --out experiments/alibaba_extension/runs/new_lora
 env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH .venvs/qwen-scene/bin/python experiments/alibaba_extension/scripts/qwen_rgbd_eval.py --adapter experiments/alibaba_extension/runs/new_lora/adapter --heldout-only --out experiments/alibaba_extension/runs/new_lora_eval
 env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH .venvs/qwen-scene/bin/python experiments/alibaba_extension/scripts/qwen_rgbd_eval.py --limit 40 --precision int8 --save-model --out experiments/alibaba_extension/runs/new_int8
```

量化只转换语言Linear，视觉和lm_head保留高精度；不是全部模型INT8。导出的实际量化checkpoint约5GB，位于 `runs/profile_int8_cache/quantized_checkpoint`。发布包省略该重复大模型权重，保留量化配置、指标、重新加载证明，可用以上命令从固定基座重建；新LoRA与选中ACT权重包含在包内。INT8本机更慢，因此不修改工作台默认BF16+KV Cache。

## ROS2导航、MoveIt与流程衔接

```bash
/usr/bin/python3 experiments/alibaba_extension/scripts/run_nav2.py --out experiments/alibaba_extension/runs/new_nav
/usr/bin/python3 experiments/alibaba_extension/scripts/run_moveit.py --out experiments/alibaba_extension/runs/new_moveit
/usr/bin/python3 experiments/alibaba_extension/scripts/run_qwen_nav_act.py --out experiments/alibaba_extension/runs/new_workflow
```

Nav2在domain62，Gazebo master11362，MoveIt在domain63/mock_components，隔离原工作空间。流程脚本采集MuJoCo当前初始帧，Qwen建议经完整指令授权后才派发Nav2到固定桌面站；公开ACT执行已知成功seed1003的方块交接，再Nav2返回，并记录真实仿真rosbag。Gazebo底盘与MuJoCo双臂**不是同一个物理场景**，无共同机器人模型或外参；这个结果只验证任务流程/接口，不算一台移动机械臂完成导航抓取。MoveIt三次关节目标的mock反馈也不能当成真实物理抓取。

Ctrl+C停止；编排脚本清理自己启动的进程。勿在相同Gazebo端口/domain同时运行两个本目录导航实验。

## 证据、失败与打包

`runs/` 保留每次模型预测、所有rollout动作/奖励/视频、导航里程计、MoveIt计划与模拟关节反馈、ROS2日志及仿真rosbag。首次Nav2动作过早发送、随后等待生命周期与初始定位互相阻塞的失败均保留；正确顺序为AMCL active→发布initialpose→bt_navigator active→发送action。

`CONTRACT_VERIFICATION.json` 为15项软件失败注入检查，覆盖不支持目标、否定、附加任务、过期结果、取消及阶段失败；不是15次实机鲁棒性实验。`VERIFICATION.json` 验证实际文件/视频/模型/消息和相对链接。量化运行在训练和其他本轮GPU实验结束后，40例三组顺序测试；首次生成排除在耗时分位数之外，仍非稳定硬实时或Jetson性能结论。

## 打包范围与前置材料

追加包保留选中的20k ACT最佳验证权重、新LoRA、RGB/深度/标签、全部预测/视频/ROS日志/仿真bag与主要依赖源码。省略20k最终权重副本、初次尺寸与颜色耦合的生成试验图像、5GB量化权重、环境及18GB缓存。首次生成试验未用于模型评测，正式场景将尺寸与颜色组合重新分配后才训练/评测。

包内补充厂家框架源码、基座engine与需要的首轮脚本/来源文档。原5k权重和首轮全部媒体在原交付包/现有项目内；重新进行权重续训需先放回原5k最终模型，或用首轮训练脚本重新训练5k，不将追加ZIP当成包含全部历史数据的单文件镜像。Qwen基座缓存缺失时执行 `scripts/download_qwen_base.py`，公开ACT/示教执行首轮 `download_public.py` 后重建缓存。复现本轮评测可直接用包内选中20k权重，不必重新训练。
