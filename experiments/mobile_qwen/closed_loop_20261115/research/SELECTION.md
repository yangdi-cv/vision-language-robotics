# 感知到MoveIt物理执行：实现选择

2026-11-15，优先复用本机官方ROS2 Humble组件，未升级系统ROS2或LIBERO。

- MoveIt2 / MoveItSimpleControllerManager（本机package.xml标注BSD）：使用标准MoveGroup与FollowJointTrajectory接口。官方文档明确可连接自行提供该ROS action的低层驱动，不要求mock_components。采用此接口作为已有单MuJoCo实例的物理驱动桥；不重新实现IK或OMPL。
  https://moveit.picknik.ai/humble/doc/examples/controller_configuration/controller_configuration_tutorial.html
- 官方ros-controls/mujoco_ros2_control（Apache-2.0）：长期完整硬件接口参考，Humble文档已支持；当前LIBERO绑定的Python MuJoCo实例与插件自持实例不同，直接另启插件会形成两个世界，所以本轮复用接口设计、保留唯一既有物理实例。
  https://control.ros.org/humble/doc/mujoco_ros2_control/doc/index.html
- MoveIt Panda资源（本机package.xml标注BSD）：URDF、SRDF、OMPL、KDL配置来自已安装moveit_resources，不复制未知来源模型。将固定手部偏移修正到当前robosuite Panda，另加TCP固定坐标；通过8个独立静态姿态与MuJoCo FK交叉核对。静态校验里的qpos赋值不作为控制或任务成功。
  https://github.com/moveit/moveit_resources
- MuJoCo轨迹执行沿用已有有限力矩PD+重力补偿控制模式，按轨迹插值发送关节目标、读取真实模拟关节反馈。只有初始化/独立静态校验可直接赋qpos；运行时不reset或改qpos，不用mock joint_states冒充物理执行。

已知边界：深度与碰撞地图先使用仿真已知量，Qwen只预测目标二维点；接近方向/姿态来自固定技能模板，不称自动6D抓取。导航继续理想仿真定位，执行器为移动底座有限力近似。预接触到位不是抓取、开抽屉或整项LIBERO成功。

## 本机MoveIt2退出问题

退出时的TrajectoryExecutionManager / CallbackGroup析构崩溃与上游moveit2#1597的栈相符；移除TF overlay未消除。保留日志，按运动任务结果与cleanup_ok分别记录，不把运行成功说成进程清理完美。本轮不升级系统ROS或重建整套MoveIt，以免影响已有环境。
https://github.com/moveit/moveit2/issues/1597

控制器开发：140/20的原展开PD在首次预接触轨迹末端未达到关节目标容差，记录为失败；在oracle开发回合上改为400/40后达到目标，无自碰撞/外部接触，随后冻结用于模型对照。没有为不同预测器设置不同增益。
