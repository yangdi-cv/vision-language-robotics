# mobile_qwen：Qwen 驱动的同场景移动操作（2026-11-14）

本目录是独立新增目录，**没有修改** `experiments/mobile_vla`、`experiments/vla_libero` 中任何已有文件。通过继承 `mobile_vla/scripts/sim_server.Engine` 和原样调用 `mobile_vla/scripts/run_nav2.py` 复用已有的同场景导航与操作。

## 做了什么

在 mobile_vla 已跑通的"Nav2 停靠 → π0.5 抓碗入抽屉并关闭"（厨房任务3）之上，加了 Qwen 高层：

1. **任务规划 / 拒绝**：开始时渲染全景图（`mobile_overview`）和桌面特写（`agentview`），Qwen3-VL-4B 基座根据任意语言的用户指令，从站点表和"已验证技能库"中选择 `(station, skill)`，或拒绝（场景中没有该物体/电器、或没有技能能完成）。拒绝时不启动 Nav2、不加载 π0.5。
2. **执行**：可行时 π0.5 在导航前用 Codex 的低内存分块加载器（`vla_libero/skill_bootstrap/scripts/bounded_restore.py`）加载一次；Nav2 停靠后，π0.5 执行技能库中的英文提示词。
3. **视觉自主停止**：当前默认 `--monitor milestone --check-every 10`，第100步起只看当前图与文字记忆：碗入抽屉确认1次，再判断抽屉关闭并连续确认3次；否则到520步上限停止。历史 `--monitor task` 是开始帧与当前帧的整任务判断、连续2票确认。官方 LIBERO 谓词只记录为 `shadow_official_success` 和 `official_success_at_stop`，**不参与停止决策**。
4. **内存隔离**：每个回合用 `systemd-run --user`（MemoryHigh 16G / MemoryMax 18G）运行，不在编辑器 cgroup 内。Qwen worker 在导入 JAX 之前启动；π0.5 单进程分块加载。

## 文件

| 文件 | 作用 |
|---|---|
| `scripts/qwen_worker.py` | qwen-scene 环境 JSON-lines worker；复用 `vla_libero/scripts/qwen_planner.Planner`；op：`mission`、`check`（历史整任务判断）、`milestone`（当前默认）、`question_check`（技能自带是/否完成问题，离线试验） |
| `scripts/sim_server_qwen.py` | 继承 mobile_vla Engine：规划、预加载策略、视觉停止、`qwen_result.json` |
| `scripts/run_case_qwen.py` | 单回合编排（改写自 mobile_vla/run_case.py）；拒绝时不启动 Nav2 |
| `scripts/launch.sh` | 用受限 systemd 单元运行一个回合：`launch.sh <tag> <initial> <ros_domain> "<指令>"` |
| `scripts/run_batch.sh` | 本轮任务集 |
| `scripts/skills.py` | 站点与技能库：π0.5 提示词、里程碑定义（问题、答案键、期望值、确认次数）；sim_server_qwen 从这里读取 |
| `scripts/milestones.py` | 里程碑记忆监视器（在线与离线共用） |
| `scripts/probe_memory.py` / `probe_drawer.py` / `probe_milestones.py` | 离线问法与记忆对比（在已存帧上） |
| `scripts/run_batch2.sh` / `run_batch3.sh` | 里程碑监视器在线批处理（初态 3–7 / 8–12） |
| `affordance/gen_dataset.py` | 仿真自动标注数据：RGB/深度/分割 + 抽屉开度、把手/碗/抽屉内部 2D/3D 真值；回放、迁移回放、静态帧（用 `vla_libero/scripts/python_shared` 运行）；`add_camera.py` 补相机参数并检查反投影 |
| `affordance/eval_qwen.py` | 可供性指点/框评测 + 深度反投影 3D 误差，`--adapter` 测 LoRA，`--rescore` 重算 |
| `affordance/train_lora.py` | 可供性 LoRA（初态17–49训练，13–16验证选点） |
| `affordance/eval_drawer_crop.py` | 整图 vs 裁剪判断抽屉状态（负结果） |
| `affordance/data_v1`、`data_train_v1`、`lora_v1/best_adapter`、`eval_*.json` | 数据容器（含初态0–49，实际测试筛选0–12）、训练/验证数据（13–49）、LoRA 权重、评测原始输出 |
| `scripts/probe_checks.py` | 在已完成回合的保存帧上离线比较检查问法 |
| `results/<tag>/` | `mission.json`、`qwen_result.json`、`summary.json`、`qwen_inputs/`（Qwen 实际看到的图）、`rollout.mp4` 及 mobile_vla 原有记录 |
| `logs/` | server/client/launch 日志、启动时源码哈希 |

## 结果

见 `RESULTS.md`。简述：任务决策 7/7（5 可行、2 拒绝），可行回合导航+官方成功 5/5，旧整任务视觉停止 0/5。改为带文字记忆的里程碑监视器（现默认 `--monitor milestone --check-every 10`）后，在新初态 3–7 在线：Qwen 正确停止 3/5、误停 1/5、未停 1/5；修正碗问法（不列选项）并把抽屉确认提到 3 次后，新初态 8–12：停止决策 5/5 正确（4 次成功后 18–27 步停，1 次 π0.5 失败回合正确不停），误停 0。

## 可供性（简述，详见 RESULTS.md）

测试集（初态 0–12，训练未见）上把手指点命中 17.7% → 98.6%，深度反投影 3D 误差中位数 7.8 → 1.4 cm（≤5 cm 22% → 96%），柜子框 IoU 0.65 → 0.99。同场景、同相机内的泛化，不代表跨场景。

## 已知问题与下一步

- 历史整任务判断110次检查无完成票；记忆里程碑现已接入在线，最新初态8–12任务成功4/5、停止决策5/5正确。不同版本使用不同初态，不是配对消融；样本少，不能视为稳定可靠。
- 数据口径：实际训练17–49、验证13–16、测试0–12；初态隔离但动作模板共享。data_v1全容器802帧含0–49，评测必须筛选同一350帧；350帧不是350个独立回合。
- 把手命中17.7%→98.6%指4像素邻域；严格像素命中5.7%→92.3%。训练中位固定点对照分别32.9%与17.4%。3D来自仿真真值深度反投影，不是实测测距或抓取精度；停止延迟18–27为仿真控制步，不是毫秒。
- 50个官方初态均已在开发或数据中使用，后续不得随意称为全新未用初态。
- 技能库只有一个已验证技能（任务3）。mobile_vla 记录任务9重放失败，未扩展。
- LIBERO 的 "black bowl" 渲染为银灰色；带"黑色"的指令未被误拒（zh_black_i00 可行且成功）。
- 已修正碗里程碑照抄选项问题（新初态 8–12 验证）。i03 类"抽屉视觉已关但差几毫米"误停是纯视觉的硬限制，confirm=3 未在该情形下验证。
- 运行：`scripts/launch.sh <新tag> <初态> <未用ROS域> "<指令>"`；tag 不能重复（拒绝覆盖）。

## 新回合日志与退出状态（本次维护后）

`launch.sh`透传systemd退出码，单回合异常返回非零；批处理遇到失败即停，不再无条件报告BATCH_DONE。拒绝任务是正常结束，策略到步数上限也是正常评测终止，任务成功仍需查summary中的官方字段。

每个新tag的 `logs/<tag>_journal/events.jsonl` 逐条flush+fsync：Qwen完整请求、完整响应（含模型版本、prompt、图像像素哈希、生成配置）、视觉检查、每个已执行控制步的动作及关节状态。请求先落盘再推理，崩溃时允许只有请求没有回复。`results/<tag>/progress.json` 每控制步原子替换，正常finish后complete=true。历史回合不回填、不覆盖；不改当前dark cabinet问法。

日志支持故障定位，不提供断点续跑：完整视频和旧trace仍在finish时生成；进程在s.step与写日志之间被杀时可能缺失最后一个动作，文件系统断电持久性也不作绝对保证。每步fsync会增加墙钟开销，未重测延迟。原始推理日志只适用于新回合，历史输出没有的字段不能补造。

本阶段报告：`reports/20261114/STAGE_REPORT.md/.pdf`；论文v0.2：`reports/20261114/PAPER_v0.2.md/.pdf`。旧π0.5 LoRA闭环复测单列待办，本轮未执行。
