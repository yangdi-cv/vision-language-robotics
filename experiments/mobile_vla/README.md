# 同场景移动操作：当前开发阶段

本目录补“导航与操作不在同一个场景”的缺口，沿用现有LIBERO/Panda与π0.5，另建移动扩展和Nav2桥。没有修改原LIBERO包、系统ROS2或已有模型环境，不连接实体设备。

## 已实际完成

- 查询并比较robosuite、ManiSkill、RoboCasa、官方mujoco_ros2_control等实现、维护信息与许可证；记录在research/SELECTION.md。robosuite上游Omron XML及MIT许可原文以固定提交保存在third_party/，不是来源不明的复制。
- 给合适的厨房工作台加入移动底座，Panda、桌面、物体、相机、障碍共用同一MuJoCo实例。客厅场景的固定桌面安装方式会穿模，明确拒绝直接扩展。
- 底座使用三个平面关节和有限力速度执行器，参考robosuite官方设计；替换原Rethink固定支架，不将它一并移动。不包含差速轮、轮滑或悬架动力学。
- Nav2以已知地图、理想仿真位姿和实际MuJoCo射线工作；不是SLAM/AMCL。低位射线与高处点云分别覆盖地面障碍和桌面悬挑。
- 开发回合results/nav05完成Nav2停靠、展开手臂、π0.5抓碗放入底层抽屉并关闭。没有在导航后重置场景；导航误差约1.64cm，操作317步。

最新结果：局部加载官方tf2/tf2_ros 0.25.24库后，初态0/1/2均完成同场景导航和操作（3/3开发回归）；逐例实际图像、模型动作、连续时间、夹持与释放及视频核对通过。详见PROGRESS.md与results/regression_summary.json。此前超时、碰撞和时钟对照均保留，不把更换时钟单独当成修复。

## 还没有完成

Qwen可供性定位、指令分解LoRA和它们与本目录的正式集成尚未完成。当前同场景成功回合使用固定站点目标及英文操作指令，是执行基础验证，不是微调后Qwen驱动的完整产品。物体身份与真值仅用于记录和官方评分，π0.5输入仍是RGB、语言和机器人状态。

## 关键修正

1. 直接在原桌面安装点加底座会与家具穿模，不通过关闭碰撞解决。
2. 厨房Panda原模型包含约274kg的固定Rethink支架；移动扩展必须真正替换该支架并同步robosuite元信息。
3. 只有低位激光会漏掉桌面悬挑，必须把高处障碍也纳入局部点云和地图。
4. 导航初始TF需包含map→odom→base_footprint→base_link；每回合隔离ROS通信域和仿真时钟，避免跨回合混用。
5. 停车制动为有限摩擦力，运行时不改qpos或固定物体，不用轨迹重放冒充模型闭环。

## 入口

在项目根目录使用已有环境：
```bash
# 新标签必须不存在；每回合使用独立ROS domain。
experiments/vla_libero/scripts/python_shared experiments/mobile_vla/scripts/run_case.py --initial 1 --tag new_case --domain 85 --wall-clock --tf-overlay /home/yd/Documents/VLA/experiments/mobile_vla/runtime/geometry2_0_25_24/root
# 完整回合后核对模型输入、动作队列、视频和连续时钟。
experiments/vla_libero/scripts/python_shared experiments/mobile_vla/scripts/verify_case.py experiments/mobile_vla/results/new_case
```

run_case拥有并清理自己启动的进程；sim_server是唯一物理进程，ROS系统Python仅传消息，不重新实现物理。Nav2配置在configs/nav2.yaml（仿真时钟）与nav2_wall.yaml（墙钟对照）。ROS墙钟不改MuJoCo固定步长；两种时间记录分别保留，不声称模型推理时达到实体实时控制。

## 记录

每回合保留navigation.json、result.json、trace.json、protocol.json、rollout.mp4、导航路径、里程计和ROS日志。新版本另外保存逐次π0.5输入与完整预测块/实际执行动作，运行时禁止reset。首条开发成功nav05产生于这些额外记录加入之前，不能用新版本验证器要求替代它原来未保存的项目。
