# 同场景移动操作选型

核对日期：2026-11-12。GitHub提交、许可证原文与官方文档一起检查，不凭仓库名判断成熟度。

| 项目 | 维护与许可 | 可复用部分 | 本轮限制/决定 |
|---|---|---|---|
| ARISE robosuite | 2709 stars；API最后推送2026-07-11；LICENSE原文MIT，附Apache上游声明（GitHub自动识别NOASSERTION） | Panda、OSC控制、Omron平面速度执行器、接触物理 | 现有LIBERO依赖旧robosuite，不原地升级。固定上游提交，仅借鉴/保留底座XML与许可证，隔离扩展旧场景 |
| ManiSkill | 3509 stars；2026-08-04推送；Apache-2.0 | Fetch移动操作、丰富控制器、批量仿真 | 官方Fetch质量标B，部分参数不现实；现有π0.5 LIBERO检查点不对应Fetch状态/动作和相机，不强行套用 |
| RoboCasa | 1873 stars；2026-10-30推送；代码/资产需分别核验 | 移动厨房操作与PandaOmron任务设计 | 资产和依赖规模更大，已有LIBERO策略不能视为其预训练执行器，保留为扩展参考 |
| ros-controls/mujoco_ros2_control | 官方ros-controls维护；Apache-2.0；官方文档支持Humble | MJCF/URDF、ros2_control、相机/激光/传感器插件 | 与现有MuJoCo2.3/Python3.11策略环境不直接兼容；先保留单物理进程，通过只传状态和速度指令的ROS桥接Nav2，不改系统ROS包 |
| moveit/mujoco_ros2_control | 217 stars；2025-11-24推送；MIT | 早期MuJoCo/MoveIt集成 | 旧README仍列传感器为后续，不作为当前首选 |
| roqsim | Apache-2.0；0 stars；2026-11-12推送 | 丰富模型、来源与许可记录规范 | 活跃不等于成熟，尚无足够采用记录，本轮不作为核心依赖 |

## 本轮方案

在现有LIBERO/Panda场景内增加独立的移动底座关节与有限力速度执行器；手臂、相机、桌面和物体共享同一个MuJoCo状态。基础建模参考robosuite官方Omron，而不是重新实现物理引擎/规划器。Nav2继续使用现有ROS2 Humble官方组件；桥只负责传感器/TF/速度指令和时钟。先检验机械臂与底座是否可兼容，再接高层模型。

这是一种平面移动底座近似：不声称具备真实差速轮、轮滑或悬架动力学。所有运行时移动必须来自执行器，不能通过改qpos或重置场景冒充导航；初始化恢复公开状态和机器人起点单独记录。导航到达后也不重置物体或机器人来恢复旧策略条件。

关键验收：同一sim对象贯穿导航和操作；运动有速度/力约束；底座碰撞有效；导航后不重置场景；原有物体身份保持；实际夹持、释放和任务谓词分别记录。导航定位先使用仿真真值里程计作为明确的理想定位基线，不冒充AMCL/SLAM或真实定位精度。

## 已知问题和来源

- robosuite移动控制曾有XML/控制逻辑修复以及device demo问题，不能把最新API片段直接粘进旧LIBERO。
- 动作坐标可能是世界/机器人/末端系，必须用实际位姿变化测试，不按名称猜测。
- 保留LIBERO检查点不表示移动扩展后的画面与状态仍在其训练分布内；必须重新跑对照。

官方来源：
- https://github.com/ARISE-Initiative/robosuite
- https://robosuite.ai/docs/modules/robots.html
- https://github.com/ARISE-Initiative/robosuite/issues/704
- https://github.com/ARISE-Initiative/robosuite/releases
- https://github.com/mani-skill/ManiSkill
- https://maniskill.readthedocs.io/en/latest/robots/fetch/index.html
- https://github.com/robocasa/robocasa
- https://github.com/ros-controls/mujoco_ros2_control
- https://control.ros.org/humble/doc/mujoco_ros2_control/doc/index.html
- https://github.com/moveit/mujoco_ros2_control
- https://github.com/cps-test-lab/roqsim
- https://docs.nav2.org/configuration/packages/configuring-regulated-pp.html
