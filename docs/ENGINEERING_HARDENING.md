# 工程修订说明

本轮针对可迁移性、接口约束、失败排查和交付验证。模型权重、控制增益和历史实验结果没有重跑或覆盖。当前维护入口为 `closed_loop_20261115/scripts/run_cases.py`（预接触和策略交接共用）及 `jetson/stage3_20261115/scripts/run_joint_service.py`。

## 1. 安装与配置

在独立 Python 3.11 环境中执行：

```bash
python -m pip install -r requirements-offline.txt
python -m pip install --no-build-isolation -e .
python scripts/check_offline.py --out /tmp/offline_checks.json
```

`--no-build-isolation` 要求环境已安装 setuptools>=61 和 wheel。运行层本身只用标准库，兼容 Python 3.10 及以上。离线数据沿用包内的 cases.json 和轨迹，模型与第三方完整源码另按原说明准备。

复制 `configs/runtime.example.json` 到本机配置，例如 `~/.config/vla/runtime.json`，修改解释器、ROS 路径和远程部署目录。配置文件不放密码。优先级：显式环境变量 > 本机 JSON > 仓库示例。未知字段直接报错。

```bash
export VLA_ROOT=/path/to/vision-language-robotics
export VLA_CONFIG="$HOME/.config/vla/runtime.json"
vla-doctor
```

可覆盖 `VLA_QWEN_PYTHON`、`VLA_POLICY_PYTHON`、`VLA_ROS_PYTHON`、`VLA_ROS_PREFIX`、`VLA_REMOTE_HOST`、`VLA_REMOTE_STAGE_ROOT`、`VLA_REMOTE_PYTHON`、`VLA_REMOTE_LOCAL_PORT`、`VLA_REMOTE_REMOTE_PORT`。诊断命令只检查本机路径，不加载模型、不联网、不连接机械臂。

示例 JSON 中的 `jetson-vla` 是 SSH 别名；在自己的 `~/.ssh/config` 配置 HostName、User 和 IdentityFile，首次人工核对设备主机密钥后再使用。新入口强制 BatchMode 和主机密钥检查，不使用密码参数，不接管系统 swap 或电源模式。旧的日期化部署脚本仍保留在仓库中供历史阅读，不能视为已迁移的新入口；新源码包排除含密码登录逻辑的 11 个历史远程驱动，避免作为推荐运行入口交付。

## 2. 新回合的运行方式

先检查配置；下面是真正启动仿真的命令，离线验收不会执行它们。GPU 作业继续使用项目既有的 systemd 内存限制，PC Qwen 与 JAX 不放进同一进程。

```bash
python experiments/mobile_qwen/closed_loop_20261115/scripts/run_cases.py \
  --prefix local_new_trial --modes lora --initials 0 --navigate

python experiments/mobile_qwen/closed_loop_20261115/scripts/run_cases.py \
  --prefix combined_new_trial --task combined --modes lora --initials 0 --navigate
```

`run_combined.py` 是兼容入口，转入同一调度函数，不再维护第二份流程。模型 worker 原有训练与提示词实现保留，因此本轮不宣称解决全部历史路径耦合。

Jetson 端需先在其独立环境安装同版本运行层，并准备原有 NF4、适配器、input_manifest.json 和验证图；rsync 只同步服务脚本，不替你下载权重或传完整数据。管理员提前确认稳定测量阶段不使用 swap。新入口使用 5G/6G 内存上限，不自动执行 sudo。

```bash
python experiments/mobile_qwen/jetson/stage3_20261115/scripts/run_joint_service.py \
  --tag joint_new_trial --dry-run
```

检查输出后移除 `--dry-run` 才会连接设备和启动仿真。测试标签不可重复；远端输出在 `runs/<tag>/`，本机在对应 stage 的 `results/<tag>/`。

## 3. 接口与故障处理

- `vla_runtime/config.py`：路径和阈值读取。
- `schemas.py`：模型坐标、状态维度、网格索引、目标姿态和轨迹校验。字符串 `"false"` 不作布尔值接受。
- `geometry.py`：坐标转换、旋转矩阵、正运动学一致性和碰撞白名单校验。
- `transport.py`：统一 Unix socket、逐行 JSON 和子进程退出；限定消息长度和总读取期限。模型 stdout 仍只传协议，日志写 stderr。
- `records.py`：原子 JSON、逐条同步落盘、参数摘要、代码版本和异常堆栈。
- `milestones.py`：共用里程碑状态机，确认次数和任务文字不变，拒绝非布尔完成标记。
- `remote.py`：SSH 密钥与隧道参数；远程参数经过 shell 引号处理。
- `artifacts.py`：交付清单和文件校验。

新回合即使模拟器尚未创建结果目录，也会在 `logs/<tag>/` 写 `progress.jsonl`、`case_snapshot.json` 和 `run_metadata.json`。控制器每一步保存反馈，异常保留完整调用栈。JSON 快照写到同目录临时文件，同步后原子替换；强制断电无法承诺零丢失，SIGKILL 也不能执行 finally，但已同步的阶段记录仍保留。

MoveIt 成功要求错误码成功、动作状态成功且存在实际执行轨迹。发送规划请求前检查输入几何、姿态、碰撞白名单和 FK；仿真控制器拒绝非有限关节值、重复时间和维度错误。轨迹接口测试不等于已验证真实机械臂安全。

## 4. 依赖与验证边界

`requirements-offline.txt` 保持已验证 CPU 版本；`requirements-dev.txt` 追加固定版本的 Ruff 和 mypy；`requirements-dev-lock.txt` 固定 CPU 审查工具的传递依赖。Qwen、JAX、ROS2 和 Jetson CUDA 分属不同环境，不能合并成一个通用 lock。本轮只在独立 CPU 审查环境安装工具，没有修改模型环境或系统 Python。

新增测试包括协议分帧、超时与退出、错误 JSON、非有限数、坐标变换、FK、碰撞白名单、轨迹时间、SSH 参数和实际调度入口的启动失败。原有离线测试一并运行。ROS2/MoveIt 全仿真、GPU 推理和 Jetson 联调尚需在依赖齐全的环境用新标签回归；原有成功率不能视为修改后入口的新实验结果。

## 5. 保留的历史范围

历史报告、数据、权重说明和结果目录不改。早期训练/制图/部署脚本未全部迁移，原评审的全仓数量与本轮维护入口范围不同。不会把格式化、类型检查或 CPU 测试通过写成“企业级认证”。后续实际维护应从新入口接入，逐模块迁移历史工具，而不是再复制一套流程。

参考：[OpenSSH 配置](https://man.openbsd.org/ssh_config)、[Python 原子替换与文件同步](https://docs.python.org/3.11/library/os.html)、[ROS2 控制接口源码](https://github.com/ros-controls/control_msgs/tree/humble)。
