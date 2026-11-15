# 阿里项目 Jetson 独立环境

2026-11-14，在恢复后的自有Orin Nano 8GB上创建。系统Jetson Linux36.4.4/Ubuntu22.04，aarch64。

- Miniforge：`/home/csip/miniforge3`，官方ARM64安装包SHA256已校验，版本与来源见INSTALLER_SOURCE.json。
- 环境名：`ali-jetson`。
- Python：`/home/csip/miniforge3/envs/ali-jetson/bin/python`，3.10.22。
- 系统Python仍为`/usr/bin/python3`，3.10.12。
- 激活：`source ~/ali-jetson/activate.sh`；退出：`conda deactivate`。
- 保持base不自动激活；不修改系统Python，不沿用旧借用板的an-project环境。

## 当前可用范围

NumPy1.26.4、SciPy1.15.2、Pillow、OpenCV headless4.10.0、PySerial3.5、PyYAML、requests、psutil、tqdm、packaging、pip。用于图像/数据读取、远程接口与性能记录准备。OpenCV不含桌面imshow界面。

实际验证：环境路径和ARM64架构正确、依赖导入、OpenCV与Pillow无损PNG往返、pip check均通过。结果见SMOKE_RESULT.json。未连接相机或机械臂做实测。

尚未在此环境安装PyTorch、Transformers、TensorRT Python绑定、CUDA开发栈或模型权重，不将基础环境创建视为GPU推理部署成功。之后需匹配JetPack6.2.1/CUDA12.6与Python3.10，验证CUDA张量和实际模型后再测延迟。

## 文件

- environment.yml：可读的基础环境规格。
- environment-resolved.yml：本次实际安装版本。
- conda-explicit-linux-aarch64.txt：当前ARM64平台Conda包清单。
- pip-freeze.txt：包含pip安装组件的版本清单。
- smoke.py：基础环境检查，运行会在当前目录写SMOKE_RESULT.json。

这些ARM64显式包清单不用于x86工作站环境。基础组件由Conda管理，OpenCV和PySerial由此环境pip管理；安装时没有拉入通用torch轮子。
