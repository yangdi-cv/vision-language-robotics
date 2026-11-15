# 选择性搬运与视觉停止

在原LIBERO的两个入篮场景里，只搬指定的一件，另一件保持原位。每个场景交换两件物体的角色。Qwen从给定物体词表选择目标，π0.5预测动作，Qwen可根据开始/当前图像决定停止。

这是自定义任务，不是原Long成功率。停止不读取任务真值，也不使用LIBERO返回的原长任务done。Qwen只接收512像素RGB、文字和物体名称词表；π0.5沿用224像素第三人称/腕部图与8维机器人状态。

## 对照

1. flat_zh：完整中文（含禁止搬另一件），执行到统一预算。
2. translated_en：Qwen翻译完整指令，执行到预算。
3. skill_continue：Qwen选出的目标，转换成固定英文单技能，执行到预算。
4. timer150：同一单技能，150步后在释放门控下两票停止。
5. timer_calibrated：同一单技能，按初态0的单技能小试标定每个目标的时刻，之后固定。
6. visual：同一单技能，Qwen实际图像检查，两票停止。
7. frozen：仅将检查器当前图换成开始图，其他与visual相同。

后五组使用完全相同的技能指令，切换前保存并核对策略图像/状态/动作噪声。全部方法共驻同一个Qwen基座，不加载LoRA，不变更环境。

## 阶段与评分

开发阶段：初态0的四个方向各跑一次参考技能（4回合），再跑七组小试（28回合）。目标选择错误不会自动改正。两个方向都保留，包括低层技能失败。

评测阶段：初态3–7，两场景×两方向×五初态×七方法=140回合。它们未用于本轮参数/定时器标定；这些仍是π0.5熟悉的公开场景，不能称为对全新场景泛化。参数不按评测结果调整。

最多执行300步，每次预测10步、执行前5步。每10步可检查一次，夹爪张开门控、连续两票；夹爪重新闭合或票间隔超过20步就清空旧票。停下后固定保持25步；保持上一条夹爪命令，不强制松爪。

自定义成功需同时满足：最后10个保持帧目标均满足原In篮子谓词、均不接触夹爪，指定保留物体在整个过程中中心位移不超过30毫米。这允许30毫米内的小位移，不等于完全无接触或无旋转。计分真值只在写日志与离线评估使用。第一次连续10帧完成且保留物体未超限的时刻，用于开发定时器标定。

flat策略没有独立结束token，预算结束不算自主停止；主比较仍是同一选择性成功条件，不能仅用“自主停止次数”宣称模型优劣。

## 运行

当前工作区已具有openpi-libero、qwen-scene环境以及官方基座/资产：

```bash
experiments/vla_libero/scripts/python_shared experiments/vla_libero/selective_execution/scripts/run.py --stage pilot
experiments/vla_libero/scripts/python_shared experiments/vla_libero/selective_execution/scripts/run.py --stage eval
.venvs/openpi-libero/bin/python experiments/vla_libero/selective_execution/scripts/analyze.py --phase eval
experiments/vla_libero/scripts/python_shared experiments/vla_libero/selective_execution/scripts/verify.py --phase eval --physics
```

完整运行需要公开LIBERO资产、pi05_libero检查点、指定Qwen模型缓存及两个现有环境，源码ZIP不包含它们。参考上游固定提交见RESOURCES.json。不要把CPU测试与完整模型评测混为一谈。

## 无GPU检查

```bash
python3.11 -m venv .venv-offline
.venv-offline/bin/python -m pip install -r experiments/vla_libero/selective_execution/requirements-offline.txt
.venv-offline/bin/python -m unittest discover -s experiments/vla_libero/selective_execution/tests -p 'test_*.py'
```

12项检查使用自生成数组及假模型回复，不需要本地图像、轨迹、模型或GPU。保持包内项目层级，包含共享planner和journal模块。源码包不包含全部原始轨迹和图像；完整物理重算需要原实验目录或重新运行。

每回合：result.json、events.jsonl、trace.jsonl、trace.npz、rollout.mp4、policy_inputs/、policy_queries.jsonl、images/。每步原子保存，异常保留已有记录；不自动覆盖失败或损坏回合。PROTOCOL.json冻结执行源码，EVALUATION_LOCK.json冻结评测配置/标定结果。development/open_gripper_score为开发修正前的旧评分小试，不计入任何正式统计。
