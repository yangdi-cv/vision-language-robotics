# 阿里方向PC端改进

本轮针对模型worker、相机帧交换、推理预算、LoRA验证和ACT选模做了实际对照。报告见REPORT.md。旧报告保留，Jetson与真实相机/机械臂联动复测尚未进行。

## 代码与默认目录

- scripts/train_lora.py：数据、输出、最大步数、验证间隔、早停patience均有CLI选项。默认使用desktop_pickplace/data/language中60/20/20划分。
- scripts/train_act.py：--cache、--out、--steps、--interval、--seed、--validation-seeds、--test-seeds可覆盖默认值。manifest决定轨迹分组，统计仅来自训练集。
- scripts/act_runner.py：桌面任务训练验证和最终测试共用。当前执行为原桌面理想末端约束，不是SO101驱动。
- scripts/public_act_trainer.py：旧公开交接训练的公共实现；--resume-model仅恢复权重，优化器重新创建。
- scripts/public_act_evaluate.py、aloha_runner.py：旧交接评测与语言派发共用执行和视频循环。
- scripts/bench_runtime.py：两个独立读取进程及worker异常检查，不打开相机或串口。
- scripts/bench_qwen.py、bench_vision.py：基座目标JSON、语法约束、检测和深度对照。固定输入文件来自已有实验。
- perception/qwen_scene/worker_client.py：超时、EOF、非法JSON及进程回收。
- perception/qwen_scene/frame_transport.py：单帧共享内存、元信息锁、PNG兼容读取。
- perception/qwen_scene/continuous_worker.py、vision_worker.py：读取最新帧；实际分析输入仍可保存。
- perception/web_workbench/server.py：相机仅分发一份内存帧，JPEG预览独立；服务重启后生效。

## 环境和运行

复用 `.venvs/qwen-scene` 与 `.venvs/act-sim`。Qwen固定revision ebb281ec70b05090aa6165b016eac8ec08e71b17，Transformers4.57.6、PEFT0.18.1；ACT使用厂家LeRobot0.3.4衍生源码，Torch2.7.1/CUDA12.6，MuJoCo2.3.7。Ultralytics8.4.175支持quantize=16，工作台默认仍为FP32和深度768。

在项目根目录、准备已有数据和权重后：

```bash
.venvs/qwen-scene/bin/python experiments/alibaba_pc_revision/scripts/train_lora.py --out 新的适配器目录
experiments/desktop_pickplace/scripts/act-python experiments/alibaba_pc_revision/scripts/train_act.py \
  --out 新的策略目录 --steps 10000 --interval 2000 --validation-seeds 1100 --test-seeds 1200 1201 1202 1203
```

新的正式实验应另选未参与开发的布局。复现本轮数值才使用报告中的布局；不要在看过测试结果后重选模型再将同一测试称为盲测。

纯源码包不含模型、数据、校准、缓存、照片、视频和运行记录。解压后保持项目根目录下experiments、perception、SO101和sensors的层级。训练前需准备已有desktop_pickplace数据缓存和Qwen/YOLO权重；网页硬件配置和校准仍按当前设备核对，不属于本轮PC验证。

共享内存仅用于同机多个进程，POSIX文件锁保护一致性。所有者负责unlink，读取者不主动删除共享段；启动的连续推理进程需使用对应更新后的代码，不能混用随意复制的旧入口。

## 结果范围

一次训练种子，ACT验证只有一个布局六个任务，因此仿真选模反而选中了泛化较差的早期权重。图像传输是两进程微基准，不能把22ms到1ms直接称为整个系统22倍加速。深度逐图尺度对齐用了模拟真值，不是可部署标定。约束解码为7个合法目标组合，不能保证语义正确。

代码来源：个人脚本复用厂家LeRobot（Apache2.0）、gym-aloha0.1.1（Apache2.0）、Qwen及Ultralytics模型。附带第三方源码或场景资源保留原许可，不将其实现计作个人原创。

源码包中将IMU解析依赖附在sensors/imu/vendor，并仅在包内将server.py的导入改为相对目录；本机原IMU服务路径没有改动。桌面训练数据和模型仍需单独准备。
