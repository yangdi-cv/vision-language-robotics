# 具身视觉与策略算法阶段进展

阿里方向 · 2026年10月24日

## 摘要

本次把上一阶段PC端改进与本轮Jetson实测合在一起。10月20日的PC部分处理进程超时、图像传输、LoRA验证与ACT选模；10月24日的端侧部分接入真实腕部相机，完成检测和深度的TensorRT部署、Qwen连续观察以及4B目标指令适配器的端侧迁移，并扩大ACT验证布局与训练种子。

固定输入下，TensorRT检测＋深度处理的中位速度约为同一Jetson上的PyTorch FP32后端的1.76倍。这里比较的是Jetson上的两种后端，不是工作站与Jetson的速度。Qwen、目标指令和策略仿真的结果分别列出，真实场景判断仍有失败用例。

本次真机提供真实相机图像，没有启用机械臂电机。视觉部署、语言目标与ACT仿真是三个已经测试的模块，还没有组成真实机械臂自主抓取闭环。

## 目录

| 部分 | 内容 | 汇总PDF页序 |
|---|---|---|
| 阅读说明 | 摘要、目录及源码索引 | 1–2 |
| 第一部分 | PC端推理链路与训练评估改进 | 3–9 |
| 第二部分 | 端侧视觉部署与策略验证扩展 | 10–15 |

原阶段报告正文保留，页脚沿用原编号；上表按PDF阅读器显示的连续页序查阅。

<div style="break-before:page"></div>

## 源码目录与报告对应关系

```text
README.md
experiments/
  alibaba_pc_revision/
    scripts/                 LoRA训练验证、ACT选模及公共执行模块
    README.md
  alibaba_jetson_revision/
    scripts/
      jetson_vision.py        相机、视觉推理及引擎导出
      paired_vision.py        同图像同尺寸后端对照
      jetson_qwen.py          最新帧读取与连续视觉观察
      quantize_qwen.py        独立NF4模型转换
      jetson_adapter_eval.py  原4B适配器的端侧测试
      bridge_eval.py         端侧输出到同场景仿真策略
      bench_journal.py       日志编码开销对照
      train_act_expanded.py   扩大布局及重复训练
      run_act_expanded.sh
    README.md
    requirements-jetson-reference.txt
  desktop_pickplace/          训练评测使用的仿真与任务代码
  alibaba_baseline/           历史兼容入口
  alibaba_extension/          历史兼容入口
perception/
  qwen_scene/                worker客户端、共享帧与视觉模块
  web_workbench/             既有工作台
SO101/code/lerobot/           原版本依赖及许可
sensors/imu/                 既有传感器读取模块
```

相关代码在上述同名目录。报告正文各节另列具体入口；执行环境与模型路径见对应README。

源码包包含程序、配置、必要仿真资产及上游许可，不包含环境、权重、TensorRT引擎、训练数据、输入照片或逐步结果。PC、Jetson、ROS与仿真环境分开使用；现有Jetson的an-project环境未修改。模型权重和完整实验结果保留在本地，不把参考版本表称为跨平台一键部署。
