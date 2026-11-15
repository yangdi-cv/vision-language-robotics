# 视觉阶段检查是否改变执行

本目录仅做仿真机制对照，不训练、不改权重、不连接机器人。固定原Qwen分解的两条英文子目标，再比较：

- live_base：基座完成检查器读取参考帧与当前帧。
- frozen_base：检查器两张图都取参考帧；π0.5仍看当前画面。
- timer150：不调用Qwen，150步后在相同门控下给出完成票，仍需连续两票。
- live_lora：已有V2提示词+LoRA+分类阈值方案；不支持的任务回退基座。
- hold_first：保留计划记录，但关闭阶段推进和阶段预算，始终执行第一条指令。

前四组保持释放门控、两票规则和300步阶段预算。所有组最多520步，使用官方Long成功判定。固定时刻组并不保证正好150步切换，仍要等门控和下一次检查。hold_first是额外的机制控制，不能当作完整任务指令的flat基线。

## 工作区运行

```bash
experiments/vla_libero/scripts/python_shared experiments/vla_libero/monitor_causality/scripts/run_study.py
.venvs/openpi-libero/bin/python experiments/vla_libero/monitor_causality/scripts/analyze.py
.venvs/openpi-libero/bin/python experiments/vla_libero/monitor_causality/scripts/collect_queries.py
.venvs/openpi-libero/bin/python experiments/vla_libero/monitor_causality/scripts/verify.py
```

驱动入口先启动Qwen worker，再载入JAX π0.5；共驻显存设置沿用python_shared。已有完整回合会核对协议后复用，running/error记录不自动伪装成完成或覆盖。换控制方法/提示词需新建实验版本。

代码基于固定版本openpi与LIBERO，依赖现有openpi-libero和qwen-scene环境。Qwen为固定revision的4B基座；V2适配器沿用checker_v2/results/model/best_adapter，权重哈希及阈值在PROTOCOL.json。本轮未重新选择适配器，原离线未通过结论保留。

源码包不是环境或权重镜像：完整仿真需要公开LIBERO资产、π0.5检查点、Qwen缓存和此前V2适配器。新机器请按包内依赖/上游版本说明安装，不能将下面的CPU单元测试与完整模型仿真混为一谈。

## 无GPU单元测试

```bash
python3.11 -m venv .venv-offline
.venv-offline/bin/python -m pip install -r experiments/vla_libero/monitor_causality/requirements-offline.txt
.venv-offline/bin/python -m unittest discover -s experiments/vla_libero/monitor_causality/tests -p 'test_*.py'
```

10项检查自带临时图像和数组，不依赖实验图像、模型、轨迹或其他本地数据。源码ZIP须保持项目目录结构，包含共用episode_journal.py。

## 数据与核验

PROTOCOL.json在首次推理前冻结完整30×5条件、计划、源代码和适配器版本；源码快照在archive/frozen_sources。episodes/primary按方法/任务/初态保存视频、逐步轨迹、原子结果、检查输入、完整推理输出及每次策略查询的RGB/腕部图、状态、噪声和动作块。

verify.py核对切换前输入/噪声/动作是否逐项相同、执行动作是否来自模型、冻结干预是否仅改变检查输入、门控及两票条件、完整视频和官方末态。仅拿到了相同成功率，不代表方法等效或高层有因果收益。

真实查询索引在decision_data/cases.json。参考/当前帧使用真实原始场景，official_relation_label来自离线官方谓词；并不等于严格物理支撑或人眼可判定，physical_support_label为空，observability/review_status保留待审。冻结、计时与保持组记录的原始当前帧不是其实际模型输入，sent_images另记，不能直接拿其预测当正常视觉误判。

所有查询数据来自已经研究过的初态0–2，只作开发采集，不自动划成新的独立测试集，也未据此训练。本轮任务5的独立拿书没有可用最终目标原子，标为不可评分。数据索引不含原始大图；未重跑/没有原目录时不能用索引直接训练。
