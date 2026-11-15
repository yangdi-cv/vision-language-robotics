# 上游来源与许可

原项目代码、第三方模型与依赖分别保留，不统一改为新的开源许可证。本仓库为个人私有研究记录。

- ROS2、Isaac Sim、MuJoCo、NumPy、SciPy及优化库按原依赖说明安装，不包含软件发行环境。
- Panda官方USD不随源码复制；MuJoCo模型目录中的许可证与来源文件保留。
- G1原模型、网格与BSD-3-Clause许可在原third_party目录；厂家IMU解析库和LeRobot工程保留原来源。
- OpenPI、LIBERO、LeRobot上游固定提交见experiments/vla_libero/third_party/sources.json（适用时）。上游完整源码与权重不作为自主实现。
- 报告明确区分公开方案复现、模块适配、自编控制器和第三方求解器；没有将仿真结果表述为真实机器人部署。
