# 环境适配说明

本轮使用独立 `.venvs/openpi-libero`，不修改已有机器人、ROS、SmolVLA或字节实验环境。

- 核心运行版本：Python3.11.18、PyTorch2.7.1+cu126、JAX0.5.3、NumPy1.26.4、MuJoCo3.2.3、robosuite1.4.1。
- OpenPI原锁文件的gym-aloha0.1.1要求MuJoCo低于3，LIBERO官方示例使用MuJoCo3.2.3。本轮LIBERO评估没有导入gym_aloha。交付前将未使用的gym-aloha更新至兼容MuJoCo3的0.1.5，核心模型、LIBERO、物理引擎及数值库版本不变。评估前后的版本分别记录在configs/evaluation_environment.json和environment.json；安装脚本会自动完成兼容组合。
- 本机CUDA发行库按相同版本复制到新环境；原环境只读，未共用可写安装目录。其他机器可由常规安装器下载相同依赖。
- LIBERO命名空间路径由scripts/python显式设置；本地配置位于configs/libero/，不修改用户原有的~/.libero配置。公开初态采用上游旧版torch序列化，仅对固定源码仓库内的初态文件显式使用旧格式加载。
- 官方检查点含GCS复合对象；16个对象全部核对CRC32C和大小，10个提供MD5的对象另核对MD5，并记录全部文件SHA256。权重内容未修改。
- 模拟器在策略推理时暂停；20Hz为仿真控制时钟。报告中的离线闭环和单模型计时不能代替真实网络或硬件实时性评估。
