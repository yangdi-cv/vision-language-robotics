# 同场景导航与可供性预接触

此目录保留2026-11-15开始的PC仿真实验。使用原厨房世界，新增真实关节轨迹执行接口；不连接机械臂。成功的含义是到达把手前方的预接触位置，尚未抓把手或完成原LIBERO任务。

## 运行

从 `/home/yd/Documents/VLA` 启动。已有ROS2 Humble、MoveIt2、Panda资源、`.venvs/openpi-libero`与`.venvs/qwen-scene`；模型与LoRA沿用 `mobile_qwen/affordance`。不安装或改变旧环境。

```bash
systemd-run --user --collect --unit ali-precontact-example \
  -p MemoryHigh=16G -p MemoryMax=18G -p OOMPolicy=stop \
  -p WorkingDirectory=/home/yd/Documents/VLA \
  /usr/bin/python3 experiments/mobile_qwen/closed_loop_20261115/scripts/run_cases.py \
  --prefix example_unique --modes oracle,lora,base --initials 2,3 \
  --variants camera_right_3cm,camera_up_3cm --navigate
```

prefix必须新建，不允许覆盖。一次运行一个批次；ROS_DOMAIN_ID为118。`--navigate`执行Nav2停靠；省略则初始就停靠，只适合诊断。每回合独立视频、图像、控制反馈及结果在 `results/<tag>/`，子进程日志在 `logs/<tag>/`。中断可用 `systemctl --user stop <unit>`。

## 文件

- `scripts/sim_precontact.py`：单一MuJoCo世界、受限力矩PD、相机深度、碰撞和增量日志。
- `scripts/grounding_worker.py`：同一Qwen4B模型切换基座/LoRA，把手点预测及模型版本记录。
- `scripts/nav_only.py`：调用既有Nav2入口，停靠后交给同一仿真中的关节控制。
- `scripts/kinematics.py`：读取Panda URDF，统一工具坐标；不实现自定义逆解。
- `scripts/moveit.launch.py`、`plan_execute.py`：MoveIt KDL/OMPL与碰撞场景。
- `scripts/physics_bridge.py`：FollowJointTrajectory，反馈来自真实仿真关节状态。
- `scripts/run_cases.py`：配对运行、视觉重观测与最多一次修正。
- `PROTOCOL.md`：成功条件、冻结参数及数据边界。
- `research/SELECTION.md`：开源方案选择；`FK_CHECK.json`：静态工具坐标核对。

## 边界

深度和碰撞地图来自仿真真值；RGB把手坐标由模型给出。固定预接触偏移和姿态是人工技能先验，不能称端到端VLA抓取。oracle是真值定位上限，独立于基座和LoRA。所有初态已有使用历史，新相机扰动不能称跨场景。原任务成功另记，不能把预接触分数替代整任务分数。

MoveIt Humble退出时曾出现上游轨迹执行管理器析构崩溃；每回合单列cleanup_ok，保留日志，不因动作到位而忽略软件退出异常。

## 完整技能交接探针

`sim_combined.py`在实际预接触到位后取消关节PD模式，重置OSC目标（不改变qpos），再用已有bounded_restore加载官方π0.5。`combined_worker.py`在可供性查询时启用LoRA，在里程碑查询时禁用适配器，共用一个Qwen模型。Qwen每10个策略步检查原技能里程碑；官方谓词只记录，不控制停止。`run_combined.py`单列完整任务结果。

```bash
# 同样必须用systemd-run限制16/18GiB内存，且上一批模型进程应已退出。
/usr/bin/python3 experiments/mobile_qwen/closed_loop_20261115/scripts/run_combined.py \
  --prefix combined_unique --modes lora --initials 1,4,5 --variants nominal --navigate
```

源码包保留VLA层级，并附本地调用到的源码；不含模型、环境、第三方LIBERO/openpi、TF2二进制overlay。需要安装原README列出的环境与模型，不能脱离这些资源无GPU运行。REPORT_PRECONTACT.pdf保留仅预接触阶段版本，REPORT.pdf为本次阶段更新版，COMBINED_SUMMARY.json单列整任务统计。
