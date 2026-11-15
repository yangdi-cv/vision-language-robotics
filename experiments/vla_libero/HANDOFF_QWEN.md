> 最新迭代（2026-11-12）：执行入口已合并为run_hierarchical.py，详见closed_loop_iteration/README.md。检查器LoRA权重与独立测试已完成，边界误报仍高，未用于动作切换；受控恢复27次已收尾。本文更早的“尚未实现”仅属历史交接状态。

> 更新（2026-11-12）：本文主体是早期交接快照。分层执行、中文对照与后续恢复分析已经实现；最新状态见 `INTEGRATION_PLAN.md`、`recovery_study/REPORT.md` 和 PROJECT_LOG.md。现有强基线为仅翻译29/30，分层v3为28/30。恢复新版本尚未优于旧版；`--operation-checks` 为默认关闭的实验选项。不要依据本文早期“尚未实现”的描述重复搭建。

# Qwen-VL高层规划交接

2026-11-12。下一阶段优先接入Qwen高层，再考虑π0.5微调。本次不训练、不批量评估、不连接实体机械臂。

## 当前基线与目标

已完成官方π0.5-LIBERO任务检查点的246回合仿真：120基线117成功，126配对干预；详见REPORT.md、summary.json、verification.json。追加松爪检查是固定指令的物理重放，不算新VLA运行。当前固定基座Panda/单一LIBERO世界，不含导航。当前动作模型是PaliGemma系π0.5，不是Qwen。

下一步：Qwen读取当前RGB与长指令，输出英文子目标；π0.5执行当前子目标；独立完成监测后切换子目标，保留原始长任务的官方成功判据。先做一两个Long任务接口，勿直接批量扩展。不要修改现有正式结果目录。

## 1. 环境与路径

工作目录 `/home/yd/Documents/VLA`。

```bash
cd /home/yd/Documents/VLA
source .venvs/openpi-libero/bin/activate
# Python: /home/yd/Documents/VLA/.venvs/openpi-libero/bin/python
# 推荐用封装入口，它额外设置LIBERO路径、EGL和线程参数：
experiments/vla_libero/scripts/python experiments/vla_libero/scripts/tests.py

# Qwen独立环境（不要同时source两个环境；直接使用绝对解释器更清楚）
source .venvs/qwen-scene/bin/activate
# Python: /home/yd/Documents/VLA/.venvs/qwen-scene/bin/python
```

OpenPI：Python3.11.18、JAX0.5.3、PyTorch2.7.1、Transformers4.53.2；Qwen环境：Python3.10.18、PyTorch2.7.1+cu126、Transformers4.57.6。不要升级OpenPI的Transformers来迁就Qwen。各环境不共享可写包目录。版本适配详见ENVIRONMENT_NOTES.md。

封装脚本当前会强制 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.65`，直接在外面export较小值会被覆盖。共驻探测应直接使用OpenPI解释器并复制脚本其余环境变量，或后续将封装改为可覆盖默认值；本轮保持原封装与原评测代码不变。

## 2. 单回合运行

```bash
cd /home/yd/Documents/VLA
experiments/vla_libero/scripts/python experiments/vla_libero/scripts/evaluate.py \
  --phase baseline --suite libero_10 --task-id 0 --limit 1 --repeat-tag qwen_handoff_demo01
```

运行Long任务0、公开初态0。`--repeat-tag`必须用新名称，避免命中已保存结果直接跳过。不要运行`--phase all`或pipeline.py：它们会启动整批实验。

输出在 `experiments/vla_libero/replays/qwen_handoff_demo01/baseline/libero_10/task00_init00/`：`rollout.mp4`、`result.json`、`config.json`、`queries.json`、`trace.npz`、`query_inputs.npz`、`action_chunks.npz`。正式基线在`results/baseline/`，不要覆盖。首次调用含JIT，后续约70ms；同步推理时模拟器暂停，不代表真实系统实时性。

## 3. 策略调用与中途换指令

入口 `scripts/evaluate.py`：

- `make_input(obs, prompt) -> dict`：取`agentview_image`和`robot0_eye_in_hand_image`，两轴翻转即180°，填充缩放至224。状态拼接末端xyz、四元数转轴角、两指关节，共8维。返回键`observation/image`、`observation/wrist_image`、`observation/state`、`prompt`。
- `trial(policy, suite_name, task_id, init_id, mode, out, record_video=True)`：目前`prompt=task.language`仅在回合开始赋值。`wrong_language`模式仅用于旧配对实验，不是规划器。
- 约第217行，`if not action_plan:`中构造`element=make_input(obs,prompt)`；约第229行调用`policy.infer(element, noise=noise)`，返回`{'actions': ndarray(10,7), 'policy_timing': ...}`。真正签名见上游`openpi/policies/policy.py`：`infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict`。
- 约第252行将`chunk[:replan_steps]`放进deque，默认5步；每个20Hz仿真tick弹出一步，经`env.step(applied.tolist())`执行。动作是归一化末端增量/旋转/夹爪，不是关节角或电机力矩；OSC_POSE产生实际电机控制量。

中途换指令：建议新建`run_hierarchical.py`复用输入函数与环境初始化，不改历史evaluate.py。用`current_subgoal`代替固定prompt，在队列为空的重规划边界检查子目标完成，再更新prompt。必须清空旧`action_plan`，否则旧子目标动作还会执行；若以后启用延迟FIFO也需明确处理旧队列。可以按策略重规划边界切换，避免在动作块中途混用语义。

保留`original_long_instruction`与`current_subgoal`两个字段并逐次记录。最终成功仍用原长任务`env.check_success()`，不要用Qwen自述完成或单个子目标完成代替。不要把goal真值或对象真实坐标传给Qwen/π0.5；真值只用于仿真监测和评估。

## 4. Qwen基座、LoRA与加载

基座：`Qwen/Qwen3-VL-4B-Instruct`，固定revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`。
本地缓存：`/home/yd/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17/`。

加载入口：`perception/qwen_scene/engine.py:SceneEngine()`，BF16、SDPA、CUDA0、local_files_only=True。其`analyze()`默认提示词专门观察蓝色耳机壳，不能直接拿来规划LIBERO。复用`engine.model`、`engine.processor`，替换为新的子目标提示词。常驻JSON进程参考 `experiments/alibaba_baseline/scripts/qwen_goal_worker.py`；该旧worker只允许红方块交接技能，也不能直接复用其任务提示词。

| 适配器 | 本地路径 | 适用性 |
|---|---|---|
| 首轮真实帧LoRA | `experiments/alibaba_baseline/runs/qwen_lora_001/adapter/` | 单条耳机壳视频，测试未改善，不用于通用规划 |
| 模拟定位LoRA | `experiments/alibaba_extension/runs/rgbd_lora_001/adapter/` | 简单方块可见性、bbox和技能字段；不是长任务规划 |
| 周期验证最佳LoRA | `experiments/alibaba_pc_revision/runs/lora_validation/best_adapter/` | 第80步，有限颜色/盒子目标选择；适合复现旧结构化目标任务 |
| 同轮末版 | `experiments/alibaba_pc_revision/runs/lora_validation/last_adapter/` | 非最佳，不优先使用 |

**LIBERO子目标规划先用4B原始基座**：没有一版现有LoRA验证过长任务分解，窄域LoRA可能限制词汇和输出结构。不要因为best_adapter旧测试20/20就推断规划更好。后续如要比较LoRA，另设少量规划验证样例。

LoRA加载用 `peft.PeftModel.from_pretrained(engine.model, adapter_path).eval()`，参考 `experiments/alibaba_baseline/scripts/verify_lora_reload.py`、`experiments/alibaba_pc_revision/scripts/train_lora.py`。RGBD版训练入口`experiments/alibaba_extension/scripts/train_rgbd_lora.py`。原PC语言INT8在`experiments/alibaba_extension/runs/profile_int8_cache/quantized_checkpoint/`，不是普通LoRA；PC导出用于Jetson的4B NF4模型在`experiments/alibaba_jetson_revision/runs/qwen4b_nf4_model/`（加载/转换代码`experiments/alibaba_jetson_revision/scripts/quantize_qwen.py`），不是新LoRA；不要混到2B上。本轮共驻测试直接用BF16基座，无需量化。

## 5. LIBERO-Long任务与谓词

这里的Long即suite `libero_10`，以下ID来自固定源码`benchmark.get_benchmark_dict()['libero_10']()`，不是按文件名自行排序。机器可读表在`configs/libero_long_tasks.json`。

### 任务0

原始指令：`put both the alphabet soup and the tomato sauce in the basket`

BDDL：`experiments/vla_libero/third_party/libero/libero/libero/bddl_files/libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket.bddl`

最终目标的原子谓词：

```json
[
  [
    "in",
    "alphabet_soup_1",
    "basket_1_contain_region"
  ],
  [
    "in",
    "tomato_sauce_1",
    "basket_1_contain_region"
  ]
]
```

### 任务1

原始指令：`put both the cream cheese box and the butter in the basket`

BDDL：`experiments/vla_libero/third_party/libero/libero/libero/bddl_files/libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket.bddl`

最终目标的原子谓词：

```json
[
  [
    "in",
    "cream_cheese_1",
    "basket_1_contain_region"
  ],
  [
    "in",
    "butter_1",
    "basket_1_contain_region"
  ]
]
```

### 任务2

原始指令：`turn on the stove and put the moka pot on it`

BDDL：`experiments/vla_libero/third_party/libero/libero/libero/bddl_files/libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it.bddl`

最终目标的原子谓词：

```json
[
  [
    "turnon",
    "flat_stove_1"
  ],
  [
    "on",
    "moka_pot_1",
    "flat_stove_1_cook_region"
  ]
]
```

### 任务3

原始指令：`put the black bowl in the bottom drawer of the cabinet and close it`

BDDL：`experiments/vla_libero/third_party/libero/libero/libero/bddl_files/libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it.bddl`

最终目标的原子谓词：

```json
[
  [
    "close",
    "white_cabinet_1_bottom_region"
  ],
  [
    "in",
    "akita_black_bowl_1",
    "white_cabinet_1_bottom_region"
  ]
]
```

### 任务4

原始指令：`put the white mug on the left plate and put the yellow and white mug on the right plate`

BDDL：`experiments/vla_libero/third_party/libero/libero/libero/bddl_files/libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate.bddl`

最终目标的原子谓词：

```json
[
  [
    "on",
    "porcelain_mug_1",
    "plate_1"
  ],
  [
    "on",
    "white_yellow_mug_1",
    "plate_2"
  ]
]
```

### 任务5

原始指令：`pick up the book and place it in the back compartment of the caddy`

BDDL：`experiments/vla_libero/third_party/libero/libero/libero/bddl_files/libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy.bddl`

最终目标的原子谓词：

```json
[
  [
    "in",
    "black_book_1",
    "desk_caddy_1_back_contain_region"
  ]
]
```

### 任务6

原始指令：`put the white mug on the plate and put the chocolate pudding to the right of the plate`

BDDL：`experiments/vla_libero/third_party/libero/libero/libero/bddl_files/libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate.bddl`

最终目标的原子谓词：

```json
[
  [
    "on",
    "porcelain_mug_1",
    "plate_1"
  ],
  [
    "on",
    "chocolate_pudding_1",
    "living_room_table_plate_right_region"
  ]
]
```

### 任务7

原始指令：`put both the alphabet soup and the cream cheese box in the basket`

BDDL：`experiments/vla_libero/third_party/libero/libero/libero/bddl_files/libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket.bddl`

最终目标的原子谓词：

```json
[
  [
    "in",
    "alphabet_soup_1",
    "basket_1_contain_region"
  ],
  [
    "in",
    "cream_cheese_1",
    "basket_1_contain_region"
  ]
]
```

### 任务8

原始指令：`put both moka pots on the stove`

BDDL：`experiments/vla_libero/third_party/libero/libero/libero/bddl_files/libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove.bddl`

最终目标的原子谓词：

```json
[
  [
    "on",
    "moka_pot_1",
    "flat_stove_1_cook_region"
  ],
  [
    "on",
    "moka_pot_2",
    "flat_stove_1_cook_region"
  ],
  [
    "turnon",
    "flat_stove_1"
  ]
]
```

### 任务9

原始指令：`put the yellow and white mug in the microwave and close it`

BDDL：`experiments/vla_libero/third_party/libero/libero/libero/bddl_files/libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it.bddl`

最终目标的原子谓词：

```json
[
  [
    "in",
    "white_yellow_mug_1",
    "microwave_1_heating_region"
  ],
  [
    "close",
    "microwave_1"
  ]
]
```

### 每个任务的阶段监测候选

以下来自BDDL和谓词实现的静态核查，本轮没有逐任务额外运行验证。`goal_state`合取列表的顺序**不是**执行顺序。任务0/1/4/6/7/8的两个放置目标通常可选择顺序；任务3和9必须另外考虑开门阶段。

- 任务0：`[["in", "alphabet_soup_1", "basket_1_contain_region"], ["in", "tomato_sauce_1", "basket_1_contain_region"]]`
- 任务1：`[["in", "cream_cheese_1", "basket_1_contain_region"], ["in", "butter_1", "basket_1_contain_region"]]`
- 任务2：`[["turnon", "flat_stove_1"], ["on", "moka_pot_1", "flat_stove_1_cook_region"]]`
- 任务3：`[["open", "white_cabinet_1_bottom_region"], ["in", "akita_black_bowl_1", "white_cabinet_1_bottom_region"], ["close", "white_cabinet_1_bottom_region"]]`
- 任务4：`[["on", "porcelain_mug_1", "plate_1"], ["on", "white_yellow_mug_1", "plate_2"]]`
- 任务5：`[["in", "black_book_1", "desk_caddy_1_back_contain_region"]]`
- 任务6：`[["on", "porcelain_mug_1", "plate_1"], ["on", "chocolate_pudding_1", "living_room_table_plate_right_region"]]`
- 任务7：`[["in", "alphabet_soup_1", "basket_1_contain_region"], ["in", "cream_cheese_1", "basket_1_contain_region"]]`
- 任务8：`[["on", "moka_pot_1", "flat_stove_1_cook_region"], ["on", "moka_pot_2", "flat_stove_1_cook_region"], ["turnon", "flat_stove_1"]]`
- 任务9：`[["open", "microwave_1"], ["in", "white_yellow_mug_1", "microwave_1_heating_region"], ["close", "microwave_1"]]`

任务8的`turnon flat_stove_1`在BDDL初态已成立，应监测保持，未必需要再生成开炉动作。任务3底层目标是white_cabinet，不能照其他场景写成wooden_cabinet。任务9的open/close对象为microwave_1，而in目标为其heating_region。拿起阶段的抓持检查需要额外映射`_check_grasp`，不是虚构一个通用Grasped谓词。

### 中间状态怎么查

`env.env.parsed_problem['goal_state']`是原任务最终谓词列表；`env.env._eval_predicate(atom)`可单独求值；`env.check_success()`检查整体。可访问`env.env.object_states_dict`，二元/一元谓词实现见`third_party/libero/libero/libero/envs/problems/libero_tabletop_manipulation.py`与`envs/predicates/`。可以将最终合取谓词的每一项作为中间监测，例如两个对象放入篮子分别判断。

**LIBERO没有给每个长任务提供完整、带顺序的标准子目标轨迹。** “先打开抽屉→放入物体→关抽屉”等额外阶段需要依据该任务BDDL的`:init`、`:regions`、对象状态手工设计监测并核对可用对象名；不能把最终close谓词成立当成中途放入也完成。`_eval_predicate`是私有接口，先验证原子谓词可求值、状态名存在。机器人是否抓住物体可以参考robosuite `_check_grasp(gripper, object.contact_geoms)`，不是BDDL默认都提供的Grasped谓词。On/In等谓词可能在仍被夹持时成立，已知需额外完成确认。

示意（监测侧，不作为策略输入）：

```python
atoms = env.env.parsed_problem['goal_state']
flags = [bool(env.env._eval_predicate(atom)) for atom in atoms]
original_success = bool(env.check_success())
```

## 6. 显存与进程验证

**已实测：4090上双进程共驻与双向推理可行。** 不改两套依赖，Qwen使用`.venvs/qwen-scene`，π0.5使用`.venvs/openpi-libero`；通过现有`worker_client.read_json_line/close_json_worker`通信。

设置为`XLA_PYTHON_CLIENT_MEM_FRACTION=0.40`、`XLA_PYTHON_CLIENT_PREALLOCATE=false`，均在导入JAX前生效。不要通过旧封装再覆盖回0.65。本轮顺序：先启动Qwen worker并ready → 加载π0.5并推理一次 → Qwen对同一初态图规划一次 → Qwen仍驻留时π0.5再次推理 → 回收worker。

- Qwen BF16模型GPU加载约4.80秒；请求图224×224、预算256新token，实际输出153token，生成约2.72秒。
- π0.5第一次调用约9.34秒（含首次JIT），第二次约74.88ms，输出均为有限的10×7动作块。**探测没有执行动作。**
- 推理后进程快照：Qwen worker 9046MiB，π0.5主进程8614MiB，合计17660MiB≈17.25GiB。该数是快照，不是连续采样的峰值；不保证更大图像、长文本或并发请求同样占用。
- **同进程未测试**，不是实测同进程OOM；两套Transformers版本不一致，因此推荐双进程，不为验证同进程去改旧环境。
- 两套进程均已退出，未留下常驻服务。

原始结果：`records/qwen_handoff/residency.json`；worker日志`records/qwen_handoff/qwen_worker.log`；主日志`logs/qwen_residency_probe.log`。最小复现（只单次接口/显存验证，不跑回合）：

```bash
cd /home/yd/Documents/VLA
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
PYTHONPATH="$PWD/experiments/vla_libero/third_party/libero" \
LIBERO_CONFIG_PATH="$PWD/experiments/vla_libero/configs/libero" \
OPENPI_DATA_HOME="$PWD/experiments/vla_libero/cache" \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.40 XLA_PYTHON_CLIENT_PREALLOCATE=false \
.venvs/openpi-libero/bin/python experiments/vla_libero/scripts/probe_qwen_residency.py
```

### 最小规划骨架与接口结果（第3步）

`scripts/qwen_planner.py`包含`Planner.plan(image_path, instruction) -> dict`，CLI单次和`--worker`两种模式；复用SceneEngine模型/processor，使用新提示词，不调用蓝色耳机壳的analyze提示词。输入图像路径和长指令，输出`subgoals`列表，每项仅`instruction`、`completion_hint`，同时保留raw、耗时及显存。非法JSON返回ok=false，不静默补造计划。

已在Long任务0保存的初始观察（来自既有正式回合的首个输入，不重新跑评测）上验证：

1. Pick up the alphabet soup can from the table.
2. Place the alphabet soup can into the basket.
3. Pick up the tomato sauce bottle from the table.
4. Place the tomato sauce bottle into the basket.

本次输出将tomato sauce称为bottle，后续需按原指令和场景实体规范化名称，不能用自由文本直接匹配BDDL对象ID。

JSON字段检查通过，**只证明接口跑通，不证明规划正确率或分层执行成功**。completion_hint是自然语言，不能直接eval。现有π0.5是否能稳定执行这种“抓起/放下”粒度仍未测试；下一位可优先合并为“put alphabet soup in basket / put tomato sauce in basket”两个任务级子目标，并人工核对监测映射。

```bash
.venvs/qwen-scene/bin/python experiments/vla_libero/scripts/qwen_planner.py \
  --image experiments/vla_libero/records/qwen_handoff/long0_initial.png \
  --instruction "put both the alphabet soup and the tomato sauce in the basket" \
  --out experiments/vla_libero/records/qwen_handoff/standalone_plan.json
```

实际一次规划响应已存于`records/qwen_handoff/residency.json`的`qwen_response`，图为`long0_initial.png`。没有实现分层runner、子目标切换或执行评测；这些是后续接手工作。


## 7. 已知坑与接手顺序

1. Qwen与OpenPI的Transformers版本不同，优先独立worker。现有`perception/qwen_scene/worker_client.py:read_json_line(process, timeout=90)`支持ready握手、超时和异常回收；worker stdout只写一行一个JSON，日志写stderr。`Popen(..., start_new_session=True, stdin=PIPE, stdout=PIPE, text=True)`，关闭用`close_json_worker`；stdio必须保持独占，不能两个线程抢读。
2. JAX默认预分配会挤占PyTorch；共驻前先设环境变量，再导入JAX。PyTorch也会保留缓存；模型在GPU上的真实驻留须以进程显存与双向推理证明，不能只看CUDA可用。
3. 回合重置要清空动作缓存和规划状态；切子目标也要清缓存。暂时不用旧干预模式修改原始基线。
4. 模型图像180°是LIBERO训练约定，不是SO101腕部90°校准；两个项目不可混用。
5. benchmark公开初态可能与示教训练重叠；本轮117/120不是未见任务泛化，不应直接对比官方每任务50次分数。
6. `scripts/python`设置LIBERO命名空间与EGL，直接裸Python需要复制这些变量；不要改系统Python。ENVIRONMENT_NOTES.md记录MuJoCo与gym-aloha依赖适配。
7. 合法JSON不等于正确分解；先保存原始回答、校验字段、检查子目标与场景/原指令一致；不要输出虚构坐标。复杂语言可先人工审核后进入单回合。
8. 不要让Qwen逐控制tick重规划。先一次性生成短子目标列表，按完成监测切换；失败/超时再有限重规划。记录高层请求帧、原长指令、子目标、切换原因、时间和原长任务是否完成。
9. 若用仿真真值谓词切换阶段，这是额外监督信息；与原策略比较时必须单列，不称为纯视觉端到端提升。
10. 所有本轮脚本只运行仿真。LIBERO Panda的7维末端动作不能发给SO101。

建议接手后：先读此文与README → 单回合重跑 → 复核共驻结果 → 人工核查一个长任务计划 → 新建分层runner接入并保留原任务评分。π0.5微调优先级已后移。
