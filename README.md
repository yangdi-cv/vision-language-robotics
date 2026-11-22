# Vision-Language Robotics

[English engineering report (161 pages)](reports/alibaba_engineering/english/build/main_en.pdf) · [中文版报告](reports/alibaba_engineering/build/main_zh.pdf)

<!-- BEGIN INLINE DEMOS -->

### Qwen分层与π0.5执行

LIBERO长任务示例：Qwen负责分解和中间完成检查，π0.5输出动作。正式30回合为28次成功，仅翻译对照为29次。

https://github.com/user-attachments/assets/8f9e9a59-083e-4486-a52a-0391b92c173a

### 同场景导航与操作

Nav2停靠后，由π0.5抓碗、放入抽屉并关闭。视频为开发回合，未使用真实移动机器人。

https://github.com/user-attachments/assets/436bb141-7333-4f43-85c7-de7a269b3662

[查看全部可播放视频](docs/DEMOS.md)

<!-- END INLINE DEMOS -->

视觉语言引导的机器人操作：从桌面机械臂观察工具、公开视觉语言动作策略复现，到任务分层、视觉完成检查、可供性定位、同场景移动操作和Jetson推理服务。

这是个人研究与工程记录。代码保留各阶段的目录和实验条件，便于对照报告阅读。这里同时包含成功、未改善的对照和已发现的实现问题。

## 从哪里开始

- [技术工程报告（中文PDF）](reports/alibaba_engineering/build/main_zh.pdf)
- [按章节阅读代码](docs/CODE_MAP.md)：6个技术篇章、24个阶段。
- [原型与观察工具](perception/)：相机、场景理解与网页工作台。
- [LIBERO策略与分层实验](experiments/vla_libero/)
- [同场景导航与操作](experiments/mobile_vla/)
- [Qwen里程碑、可供性与端侧](experiments/mobile_qwen/)
- [预接触规划与策略交接](experiments/mobile_qwen/closed_loop_20261115/)

## 主要结果与适用范围

| 内容 | 保存结果 | 解释 |
|---|---|---|
| 官方π0.5在LIBERO闭环复现 | 117/120 | 四套任务，各30回合；属于公开权重复现 |
| 中文长任务：直接输入／Qwen分层／仅翻译 | 22/30、28/30、29/30 | 同30个任务初态；分层没有超过仅翻译 |
| 检查器LoRA闭环对照 | 基座与适配版均28/30 | 离线改善没有转化为整体成功率提升 |
| 可供性2B模型 | BF16严格把手命中93.7% | 350帧；初态隔离、动作模板共享，同场景泛化 |
| 可供性驱动预接触 | LoRA 4/4、基座0/4 | 使用仿真真值深度，属于到位测试 |
| 不重置交接到操作策略 | 2/4 | 到位不保证完整任务成功 |
| Jetson NF4视觉服务 | 三次联合预接触完成 | 仿真同步等待服务，不是实时达标 |

动作适配分支发现过低秩权重合并错误；修正后的完整闭环仍未复测。15瓦端侧延迟尚未达到3秒目标。真实SO101试验、解析轨迹回放和仿真模型控制在报告中分别呈现，不混作自主抓取成功。

## 先运行离线检查

使用Python3.11环境，安装离线依赖后，从仓库根目录运行：

```bash
python -m pip install -r requirements-offline.txt
python -m pip install --no-build-isolation -e .
python scripts/check_offline.py
```

检查脚本逐组启动独立进程，禁用GPU，不加载模型或连接机械臂。包含输入方向、状态机、数据划分与标签对齐、日志落盘、视觉干预和动作混合接口。测试所需的cases.json和低维轨迹已随包提供，详见[离线输入说明](repository/TEST_INPUTS.md)。

## 完整仿真与模型环境

不同阶段使用不同环境，不建议将全部依赖装进一个环境。先阅读对应实验README与锁定版本。主线使用`openpi-libero`、`qwen-scene`及系统ROS2 Humble；没有将ROS发行包写成通用pip依赖。

```bash
# 仅恢复固定版本的上游源代码，不下载模型或数据。
python scripts/fetch_upstream.py
```

完整LIBERO任务还需要按[运行说明](experiments/vla_libero/README.md)准备资产、权重及Python依赖。已有包装脚本支持`VLA_PYTHON`指定策略解释器；其他历史入口可能仍引用原工作站绝对路径，需要按本机目录配置。`mobile_vla`使用的局部tf2运行库未随源码上传，来源与版本保留在该目录的说明和研究记录中。

不要直接运行`evaluate.py --phase all`来检查安装，它会启动完整评测。先跑离线检查，再用单回合入口。新的试验使用新的输出标签；实体机械臂入口在SO101目录，与仿真入口分开。

## 仓库包含什么

保留源码、配置、小型核验输入、摘要记录、上游许可，以及完整工程报告的LaTeX、配图与代表视频。未上传模型权重、虚拟环境、原始训练图像或全部运行日志。报告中的原始文件引用有些只在研究工作站保留，不表示它们均随Git仓库提供。

`PROJECT_LOG.md`是整理前的共享技术日志，含另一控制方向的背景；本仓库的主题范围以本README、章节索引和报告为准。当前上传整理的文件校验清单见`repository/export_manifest.json`，离线检查输出见`repository/offline_checks.json`。

[第三方来源与许可](THIRD_PARTY.md)保留原项目约定。仓库私有，未为全部内容统一授予新的开源许可证。

## 工程修订（2026-11-22）

配置、SSH 密钥、共享接口、逐阶段日志和本轮验证见 [工程修订说明](docs/ENGINEERING_HARDENING.md)。新回合使用新标签；历史结果不重新解释为修改后的结果。
