# 运行范围与环境

已验证的是本机现有环境：项目根目录`.venvs/openpi-libero/bin/python`，Ubuntu22.04/RTX4090。精确版本另见本次RUN_ENVIRONMENT.json。使用公开openpi和LIBERO固定源码、公开pi05_libero基座，引用版本在third_party/sources.json。

本机重新执行：`bash experiments/vla_libero/official_instance/start.sh --tag 新的英文tag --tasks 9 --initials 32`。这会启动新的策略推理和仿真，不是播放旧动作。tag仅用英文字母、数字、下划线或短横线。

源码ZIP不包含完整Python环境、官方仓库和大模型参数。迁移机器需要先准备Python3.11环境、NVIDIA/JAX运行依赖与LIBERO仿真依赖；参照requirements.lock.txt与ENVIRONMENT_NOTES.md核对版本。fetch_sources.py获取固定源码并生成当地路径配置，download_segments.py及configs/checkpoint_objects.json获取公开基座并验证。安装脚本随包提供供参考，但本次没有在第二台全新机器上重新验证自动安装，不承诺解压即运行GPU任务。

不需要教师数据即可重新运行官方原任务。Point接口的完整训练则依赖本机保留的原始示教数据；源码包没有这些全部轨迹，报告不宣称仅凭源码即可重训。单独提供的LoRA增量还需要原基座与相应目标图接口。

模型服务软14GiB/硬16GiB限额，只加载一份权重。网页服务仅绑定127.0.0.1:8793，停止命令为`systemctl --user stop ali-vla-instance-page`。没有实体设备访问。
