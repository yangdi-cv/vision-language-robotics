# 执行记录与复现入口

所有操作均在本地工作站完成，无新硬件操作，无真机运动，无外部消息发送。

执行顺序：
1. 建立独立act-sim环境，安装gym-aloha。dm-control依赖回溯后固定1.0.14与MuJoCo2.3.7；原安装日志保留。
2. 按固定revision下载官方ACT及50episode数据，保持v2.1格式；原有厂家代码版本不替换。
3. 从真实腕部示教抽24帧，独立视觉审阅暂定标签，生成72条指令用例；执行qwen_benchmark.py。
4. 公开ACT evaluate_act.py：seed1000–1009，10个实际MuJoCo rollouts。
5. prepare_act_data.py：解码与动作行严格对齐；40/5/5 episode拆分，训练统计仅来自前40episode。
6. train_act.py：独立ACT训练5000更新，保存训练曲线、验证、测试误差与权重；evaluate_act.py用相同10种子评测。
7. qwen_lora.py：48/12/12帧分离用例拆分，120更新，保存适配器；verify_lora_reload.py重新加载验证。
8. integrated_demo.py：基座Qwen选技能、公开ACT执行、MuJoCo反馈。测试5个正指令和2个负指令。
9. 发现负指令错选技能；添加独立有限指令语法授权，以--guarded复测，保留未加约束的原始记录。
10. 生成图表、失败分析、报告和交付包。

主要命令（在项目根目录）：

```bash
experiments/alibaba_baseline/scripts/act-python experiments/alibaba_baseline/scripts/prepare_act_data.py
experiments/alibaba_baseline/scripts/act-python experiments/alibaba_baseline/scripts/train_act.py --out experiments/alibaba_baseline/runs/new_train --steps 5000
experiments/alibaba_baseline/scripts/act-python experiments/alibaba_baseline/scripts/evaluate_act.py --model experiments/alibaba_baseline/runs/new_train/final_checkpoint --out experiments/alibaba_baseline/runs/new_eval
experiments/alibaba_baseline/scripts/act-python experiments/alibaba_baseline/scripts/integrated_demo.py --out experiments/alibaba_baseline/runs/new_demo --guarded
```

Qwen脚本用 `.venvs/qwen-scene/bin/python`，清除PYTHONPATH/PYTHONHOME。首轮脚本默认使用固定实验目录；重跑前修改输出目录，避免覆盖历史证据。数据与基座模型原始下载来源见PROVENANCE.json。
