# Ubuntu 已验证运行记录

项目：/home/yd/Documents/VLA/SO101
代码：code/lerobot，LeRobot 0.3.4，保留 Windows 修改。
环境：hiwonder-so101，位于 /home/yd/Documents/VLA-old/.conda/envs/hiwonder-so101。
启动包装脚本使用新代码，隔离 ROS2 的 PYTHONPATH，并通过 dialout 组访问串口。

黑色主臂稳定串口：/dev/serial/by-id/usb-1a86_USB_Single_Serial_5CE5121669-if00
白色从臂稳定串口：/dev/serial/by-id/usb-1a86_USB_Single_Serial_5CE5122996-if00
此次连接主臂 /dev/ttyACM0，从臂 /dev/ttyACM1，编号可能变化。

校准：calibration/teleoperators/so_leader/my_awesome_leader_arm.json
校准：calibration/robots/so_follower/my_awesome_follower_arm.json
两臂电机校准值与上述文件只读核对一致；用户已确认 Ubuntu 遥操作正常。

无摄像头遥操作启动：
```bash
/home/yd/Documents/VLA/SO101/code/scripts/ubuntu/start-teleop
```
先确认固定、空间、断电位置，托稳白色臂。正常停止先支撑从臂再 Ctrl+C；异常运动断开 12V 电源。
脚本 10 Hz，关节目标限制在当前值附近 5 度；退出关闭扭矩，肩/肘 P 恢复 16。

摄像头：icSpring，/dev/video0 为此次图像设备，/dev/video1 是同一摄像头另一节点。
已验证可读取 640×480 图像。
预览命令：
```bash
/home/yd/Documents/VLA/SO101/code/scripts/ubuntu/project-python /home/yd/Documents/VLA/SO101/code/scripts/preview_camera.py --camera-index 0
```
预览窗口 Q、Esc 或关闭窗口退出；无需机械臂供电。
