# Jetson 腕部视觉部署与实测

在现有 Jetson Orin Nano 上验证腕部相机、目标检测、单目深度与 Qwen 场景理解。程序不导入机械臂驱动，不发送电机命令。

现有 an-project 环境只读复用其 Python / PyTorch，额外 Ultralytics 包放入项目 deps/，不向原环境安装包。TensorRT 绑定读取系统已有安装；Qwen 读取设备上已存在的预量化 2B 模型与独立依赖目录，不加载其他任务适配器。

相关代码在 scripts/jetson_vision.py 与 scripts/jetson_qwen.py，具体运行参数使用 --help 查看。部署时需要指定自己的模型路径与依赖路径。相机为 V4L2 设备，当前画面逆时针旋转90°。

测得的单目深度没有物理尺度标定，不能直接当作抓取坐标。模型只生成观察结果；本实验不构成自主抓取验证。

## 本次测试顺序

- jetson_vision.py：相机采集、PyTorch精度冒烟与TensorRT引擎导出。初次矩形推理为冒烟数据，不直接与固定方形TensorRT输入作速度比。
- paired_vision.py：三种后端使用同一批保存帧、固定640×640填充，对比串行检测＋深度的耗时与数值差异。
- jetson_qwen.py：先运行Qwen单独实时取帧，再评测保存的目标可见性样例，最后同进程并行运行TensorRT检测/深度和Qwen。输入通过内存PIL对象交付，PNG仅作输入归档。两路各取当前最新帧，不累计待推理队列。
- pc_scene_eval.py：同一批保存图像上的PC端4B观察结果，不作2B量化前后的因果比较。
- train_act_expanded.py：扩大验证布局的ACT训练，复用alibaba_pc_revision/scripts/act_runner.py与原数据缓存。

## Jetson 启动示例

把Ultralytics同版本包放到项目deps/，Qwen所需Transformers/bitsandbytes等放到另外的独立依赖目录；不向an-project安装或升级包。先准备好两个TensorRT引擎，再启动完整视觉测试：

```bash
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DEACTIVATE_ASYNC_LOAD=1
export PYTORCH_CUDA_ALLOC_CONF=backend:native
export PYTHONPATH="/你的Qwen量化依赖目录:/你的Qwen通用依赖目录"
/home/jetson/miniforge3/envs/an-project/bin/python -s -u experiments/alibaba_jetson_revision/scripts/jetson_qwen.py --root experiments/alibaba_jetson_revision --model /你的Qwen3-VL-2B-NF4模型目录 --cycles 4 --tokens 64
```

运行前按脚本相对路径放好models/yolo26n.pt、yolo26n-depth.pt及对应.engine。engine必须在目标Jetson上生成；源码包不包含权重和引擎。若要复现实景可见性样例评测，还需自行准备scene/、scene_absent/、scene_shifted/、scene_partial/目录中的图像；未提供时跳过该组样例。原始相机记录保留于本地结果目录，不随纯源码包发送。

本机已有的Qwen NF4模型与依赖来自先前独立部署，本轮只读加载并重新测试，不将旧项目测量冒充新结果。复现需要相应预量化模型及能在ARM64/CUDA12.6运行的bitsandbytes构建；版本表不代表在其他机器上一条pip命令即可安装成功。an-project之外的yolo环境当前存在libcudss缺失，本轮未修改它。

连续联动测试可指定轮数，输出到单独目录：

```bash
python scripts/jetson_qwen.py --root . --model /模型目录 --cycles 0 --skip-saved --integrated-cycles 60 --output-name qwen_steady --tokens 64
```

这里相对于本实验目录运行。TensorRT需要在主线程先初始化并预热，随后才启动视觉线程和Qwen同时推理；初始化时CUDA图捕获与另一路生成重叠会失败，代码已按此顺序处理。

4B模型与适配器测试使用 `scripts/quantize_qwen.py --out 新模型目录` 在工作站另存NF4权重，再使用 `scripts/jetson_adapter_eval.py --model 新模型目录 --adapter 原best_adapter目录 --data 测试数据目录 --out 新结果目录`。数据目录需要cases.json及所引用图像；本次沿用原有20条测试用例，没有用这20条重新调适配器。设备上的大块CUDA预分配可能受页缓存和可用内存影响，加载脚本只对指定checkpoint建议释放干净页缓存，并使用有限预分配，不修改系统全局缓存策略。

本轮4B加载采用 `PYTORCH_CUDA_ALLOC_CONF=backend:native,max_split_size_mb:512`，先预留两个800MiB大块及较小缓存，避免加载词嵌入和输出头时大块被提前拆分。该设置只作用于当前进程，未修改系统或Conda配置。初次较大单块/小块组合及较小预分配均失败，日志保留，后续版本完成了实际加载与评测。

`bridge_eval.py` 可把保存的Jetson目标输出交给PC上的ACT，在对应布局执行仿真。它先逐像素检查初始化画面与语言输入一致，再执行模型提出的有效目标；评测标签只用于统计，不用于替换模型输出。该衔接是离线分阶段验证，不是实时网络控制，也不访问真机串口。

视觉线程的逐帧记录使用JSONL追加，在退出时再保存一次JSON汇总，避免每次都重新编码全部历史记录。每次运行须使用新的输出目录名；源码会拒绝覆盖已有Qwen输出目录。
