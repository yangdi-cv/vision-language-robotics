# 选择性搬运：约束透传与多视角完成检查

本目录是选择性搬运实验的后续阶段。运行平台为 PC RTX 4090；本阶段没有实体机械臂操作，也没有训练新权重。

## 环境与运行

在项目根目录执行：

```bash
source .venvs/openpi-libero/bin/activate
# π0.5 与仿真入口；脚本内部调用独立 Qwen worker。
bash experiments/vla_libero/scripts/python_shared experiments/vla_libero/constraint_retention/scripts/run.py --stage pilot
# 汇总已保存结果，不启动推理。
bash experiments/vla_libero/scripts/python_shared experiments/vla_libero/constraint_retention/scripts/analyze.py
# 核对模型输入、动作和已冻结代码；需要原始轨迹和图像。
bash experiments/vla_libero/scripts/python_shared experiments/vla_libero/constraint_retention/scripts/audit.py
```

Qwen 环境为 `.venvs/qwen-scene/bin/python`，π0.5 为 `.venvs/openpi-libero/bin/python`。共享启动器设置 JAX 显存比例 0.40、关闭预分配，两个模型使用独立进程。模型仍为 Qwen3-VL-4B 基座与 pi05_libero。完整运行需要已有 LIBERO 资源、π0.5 检查点和 Qwen 模型缓存；源码压缩包不包含这些大文件。

## 不需要 GPU 的检查

```bash
python3 -m venv /tmp/constraint-tests
/tmp/constraint-tests/bin/pip install -r experiments/vla_libero/constraint_retention/requirements-offline.txt
/tmp/constraint-tests/bin/python -m unittest discover -s experiments/vla_libero/constraint_retention/tests -v
```

这八项测试验证约束模板、目标绑定、相机输入与完成状态解析，使用合成数据，不依赖未打包的轨迹、模型或 `cases.json`。它们不是仿真成功率测试。

## 文件对应

- `scripts/design.py`：目标、约束及停止方式组合；复用上一阶段的 `selective_execution/scripts/protocol.py` 和 `control.py`。
- `scripts/run.py`、`execution.py`：约束透传开发对照，`PROTOCOL.json` 固定实现与扩大实验条件。
- `scripts/token_preflight.py`：调用真实策略分词器检查截断。
- `scripts/collect_camera.py`、`prepare_views.py`：新初态轨迹与同一状态的第三人称/腕部图像。
- `scripts/view_checker.py`、`view_worker.py`、`view_run.py`：严格完成检查的离线对照，`VIEW_PROTOCOL.json` 固定提示词和源码。
- `scripts/episode.py`、`live.py`、`live_worker.py`：真正反馈至停止决策的在线试验，`LIVE_PROTOCOL.json` 固定实现。
- `scripts/minimal_view.py`：仅在开发集上检查保留原问法后的视角差异。
- `scripts/analyze.py`、`audit.py`：结果汇总与输入/动作检查。
- `REPORT.md`：本阶段报告；`*_SUMMARY.json`、`*_GATE.json`：统计及是否扩大实验的决定。

旧实验源码、结果和评测初态 3–7 均保持不变。多视角离线测试初态为 8–10；最小提示词复查只用开发初态 0。由于同一回合包含多帧，帧级样本不能当成独立回合进行显著性推断。

## 源码包范围

保留项目内相对路径，包含共享依赖、配置、协议、统计、单元测试和少量结果元数据。未包含模型权重、第三方仓库代码、全量图片、视频和轨迹。完整推理需要按项目环境说明安装依赖与资源；CPU 接口测试可在解压后直接安装上述最小依赖运行。图像和轨迹仍保存在工作站原实验目录。
