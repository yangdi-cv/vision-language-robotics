# Jetson视觉部署：软件栈阶段完成

2026-11-14。本阶段仅安装环境及小型兼容性测试；未加载Qwen2B/4B权重、未训练2B适配器、未跑350帧精度或联合闭环，不连接机械臂。

## 结论

JetPack6.2.1/CUDA12.6的Jetson专用PyTorch、torchvision与bitsandbytes已在ali-jetson安装，实际CUDA计算、NF4矩阵乘法、预量化存储跨机器加载、PEFT直接加载均通过。无须源码编译bitsandbytes。下一阶段先2B，4B保留在后。

| 组件 | 版本/来源 | 检查 |
|---|---|---|
| torch | 2.8.0，Jetson AI Lab jp6/cu126 | CUDA12.6，设备Orin，SM8.7，矩阵乘法通过 |
| torchvision | 0.23.0，同上 | CUDA NMS通过 |
| bitsandbytes | 0.48.0.dev0+ff389db，同上 | NF4双重量化、4bit Linear前向通过 |
| transformers | 4.57.6 | Qwen3-VL模型类和AutoProcessor可导入，未加载权重 |
| PEFT | 0.18.1 | 在预量化小模型上直接加载适配器通过，未合并 |
| accelerate | 1.12.0 | 依赖安装完成，pip check通过 |

torch/vision/bnb均来自https://pypi.jetson-ai-lab.io/jp6/cu126，而非通用aarch64 torch。安装报告记录下载URL与SHA256；后续HF依赖通过约束固定这三个包及NumPy1.26.4，避免解析器替换Jetson构建。系统Python未修改。

## 内存与运行状态

- 已从graphical.target切换为multi-user.target并设为默认；恢复桌面可用sudo systemctl set-default graphical.target、sudo systemctl isolate graphical.target。
- 板上无NVMe。用户随后明确允许SD卡swap，已在/swapfile-ali-jetson建立8GiB、权限600并写入fstab；现有zram约3.7GiB优先级5，新磁盘swap优先级-2。总swap约11.7GiB。原fstab备份为/etc/fstab.before-ali-jetson-20261114。
- 软件安装与板端小模型测试使用systemd-run --user，MemoryHigh=5G、MemoryMax=6G、OOMPolicy=stop。小测试不代表大模型的加载峰值。
- 电源保持nvpmodel模式0（15W）。jetson_clocks未启用；读取显示CPU为动态频率、GPU306–624.75MHz，EMC FreqOverride=0。未设置锁频或改变功耗模式。
- 切无桌面模式时USB网络服务停止，随后通过同一板子的局域网192.168.31.168恢复连接。板端用户csip，工作目录/home/csip/ali-jetson/deployment_20261114。

## 修复的兼容性问题

初次导入PEFT时，Transformers间接导入SciPy。Torch先加载了系统旧libstdc++，而Conda的SciPy需要GLIBCXX_3.4.32，发生导入异常。项目专用jetson_python.sh通过LD_PRELOAD显式使用ali-jetson/lib/libstdc++.so.6解决；只作用于该进程，没有替换系统库或全局修改shell配置。此脚本同时限制CPU线程。

## PC预量化与跨设备核对

测试使用随机初始化的小型一层Llama测试夹具（128隐藏维度），只用于存储/内核/API兼容性，不是Qwen能力测试。

1. PC在限内存systemd单元中，将FP16测试夹具量化为NF4（double_quant=true），save_pretrained保存。
2. 构造非零PEFT q_proj/v_proj适配器，独立保存，不合并。保存同一输入的基座/适配后logits。
3. PC重新加载量化目录验证；传输同一量化目录、适配器与参考到Orin，直接恢复预量化权重并加载PEFT。

| 指标 | PC重载 | Jetson重载 |
|---|---:|---:|
| 对PC保存前基座logits最大绝对差 | 0.0 | 0.0 |
| 对PC保存前适配logits最大绝对差 | 0.0 | 0.0 |
| 适配器引入的最大输出变化 | 0.17041015625 | 0.17041015625 |

PC为torch2.7.1+cu126/bnb0.48.2，Jetson为torch2.8.0/bnb0.48.0.dev0+ff389db，当前小夹具跨版本存储可恢复。零差仅对应这一固定输入的小模型，不能推断Qwen跨设备数值完全一致。实际Qwen加载、视觉前处理与图像输出精度仍需下一阶段验证。

## 本地文件

- install_stack.sh：固定版本安装与依赖约束。
- setup_headless_swap.sh：用户授权后的SD swap设置。
- jetson_python.sh：进程级C++运行库修复。
- stack_probe.py：CUDA/NF4导入和矩阵测试。
- smoke_nf4.py：PC prepare / Jetson verify小夹具测试。
- tiny_nf4_transfer/：PC夹具与PC重载结果。
- software_stack_20261114/deployment_20261114/：板端安装报告、pip前后清单、pip check、电源/swap状态、stack_result.json、nf4_transfer.log、板端verify_result.json。

## 下一阶段范围（尚未执行）

先Qwen3-VL-2B：PC预量化NF4保存→板端限6G加载；在PC同训练17–49、验证13–16、测试0–12训练独立2B可供性LoRA。先报告2B加载状况，再进入同350帧精度。测试512/384/256三档，分别报告准确率并记录实际视觉预处理尺寸，不能仅缩显示窗口；每档重新映射到原始512坐标评分。

然后测量P50/P95、实际输出token、描述/无描述准确率、tegrastats及swap。保持既定阈值：里程碑≤1秒首选、≤2秒放宽，额外判断0.5/1秒查询周期；可供性≤3秒。最后3–5个同步联合闭环，明确同步等待不能证明实时达标。旧π0.5 LoRA闭环复测仍单列最后。
