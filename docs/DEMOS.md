# 视频演示

以下视频可在GitHub页面内点击播放，原始MP4也保留在仓库中。画面来自已有实验，不增加独立试验数量。

## 桌面ACT策略示例

仿真桌面取放，末端采用理想位置驱动；不能等同于真实电机控制。

https://github.com/user-attachments/assets/c7d395a4-e22e-47d6-8c61-0f5f704611de

[原始MP4](../reports/alibaba_engineering/videos/ch03/desktop_pickplace_example.mp4)

## Qwen分层与π0.5执行

LIBERO长任务示例：Qwen负责分解和中间完成检查，π0.5输出动作。正式30回合为28次成功，仅翻译对照为29次。

https://github.com/user-attachments/assets/8f9e9a59-083e-4486-a52a-0391b92c173a

[原始MP4](../reports/alibaba_engineering/videos/ch04/hierarchical_long_example.mp4)

## 带里程碑记忆的视觉完成检查

同场景抓碗入抽屉并关闭，Qwen依据画面和文字记忆判断阶段；这是单场景、单技能示例。

https://github.com/user-attachments/assets/3064e5c8-09fe-414a-b275-c156ba8bce83

[原始MP4](../reports/alibaba_engineering/videos/ch05/milestone_monitor_example.mp4)

## 可供性定位、预接触与策略交接

保存的一次成功示例。完整交接组为2/4，不把这段视频当作普遍稳定性。

https://github.com/user-attachments/assets/7f9c5dad-2e8d-4513-a863-6791e6cf6072

[原始MP4](../reports/alibaba_engineering/videos/ch06/policy_handover_success.mp4)

## Jetson视觉服务与PC预接触执行

板端2B NF4模型返回定位，PC仿真完成规划和预接触。模拟器同步等待，不代表实时达标。

https://github.com/user-attachments/assets/f3b872d7-9f98-4c89-a769-b0bc0f4f3e8b

[原始MP4](../reports/alibaba_engineering/videos/ch07/jetson_vision_pc_execution.mp4)

## 同场景导航与操作

Nav2停靠后，由π0.5抓碗、放入抽屉并关闭。视频为开发回合，未使用真实移动机器人。

https://github.com/user-attachments/assets/436bb141-7333-4f43-85c7-de7a269b3662

[原始MP4](../reports/alibaba_engineering/videos/ch05/mobile_navigation_operation.mp4)

## 官方π0.5在LIBERO的闭环复现

官方权重在长任务0的示例，使用模型动作和当前图像闭环，不是轨迹回放。

https://github.com/user-attachments/assets/20abbd93-ff6b-4cee-a306-2c3d4b9969d6

[原始MP4](../reports/alibaba_engineering/videos/ch03/pi05_official_long_task00.mp4)

## 相关资料

[返回首页](../README.md) · [章节与代码索引](CODE_MAP.md)
