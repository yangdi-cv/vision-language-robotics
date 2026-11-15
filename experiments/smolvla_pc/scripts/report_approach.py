"""Document short-horizon approach experiments, including failed validation."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3];BASE=ROOT/'experiments/smolvla_pc';OUT=BASE/'approach_report';OUT.mkdir(exist_ok=True)
r=json.loads((BASE/'runs/approach_refined_v1/summary.json').read_text());gate=json.loads((BASE/'runs/approach_refined_v1/offline_gate.json').read_text())
fig,ax=plt.subplots(figsize=(9,4))
for name in ['approach_holdout_v1','approach_all3_v1','approach_refined_v1']:
 p=BASE/'runs'/name;vals=[]
 for f in sorted(p.glob('validation_*.json'),key=lambda f:int(f.stem.split('_')[-1])):
  d=json.loads(f.read_text());vals.append((int(f.stem.split('_')[-1]),sum(d['horizon_joint_mae']['1'][:5])/5))
 ax.plot([x for x,y in vals],[y for x,y in vals],marker='o',label=name)
ax.set_xlabel('Optimizer steps in stage');ax.set_ylabel('First-step arm MAE (deg)');ax.legend();fig.tight_layout();fig.savefig(OUT/'first_step_mae.png',dpi=150);plt.close(fig)
text=f'''# SmolVLA 短窗口自主接近准备

目标是先完成竖直起点到接近耳机壳的可见自主动作，抓取前停止；不是持续追踪移动目标，也不是完整取放成功。

## 数据与训练

沿用三条已采集示教，将片段截止在夹爪首次明显闭合前16帧，首段保留竖直姿态。该截断由夹爪角度启发式生成，不是桌面空间高度标定。第一、二条232帧训练，第三条105帧留出；三条合训共337帧。图像为单腕RGB比例填充256×256，状态和目标均沿用本机校准坐标。腕部−90°已包含在记录动作中，不重复叠加。

预测窗口从50步改为10步（名义1秒），模型仍使用官方SmolVLA动作专家和流匹配，不用YOLO或预录轨迹生成动作。高学习率训练后留出误差反弹，已停止并保留最佳检查点；之后从三条合训的最佳权重以2e-5学习率细化3000步。

![首步误差变化](first_step_mae.png)

## 离线检查

细化阶段所选检查点为第{r['best_step']}步。三条参与训练的记录中均匀取36个观察点，前五关节首步MAE为{gate['first_five_joint_mae_deg']:.2f}°，绝对误差95分位{gate['first_five_joint_abs_error_p95_deg']:.2f}°，预测首步相对实际状态的变化量95分位{gate['proposed_first_step_delta_p95_deg']:.2f}°。

本轮预先设置的有限试动检查结果：{gate['bounded_trial_allowed']}。这是训练内的数值筛查，不是独立泛化、运动碰撞安全或真实任务成功验证。第三条留出表现仍不足，不宣称换位置后可以正确接近或追踪。

## 执行约束

实机程序 `scripts/run_approach_policy.py` 重新读取当前关节和相机，每轮重新推理，只取首步后限速；默认不写电机。执行时需显式参数和现场确认，初始姿态需接近记录起点。首轮最多8秒、起点相对角度上限约8°、每次4 tick，夹爪不使能；控制结束保持扭矩，托稳后OFF关闭。检查文件绑定所选权重SHA256，权重改变必须重做检查。

当前没有实体测试结果，两臂按用户说明保持断电。下一步仅安排操作人员在场的受限空载接近测试。模型即使输出有效数值，也不等于动作符合指令，更不等于完整取放已通过。

## 文件

- 权重与统计：`experiments/smolvla_pc/runs/approach_refined_v1/best_checkpoint/`、同级`normalization.json`。
- 检查结果：同级`offline_gate.json`、`selected_evaluation.json`。
- 训练：`scripts/train_approach.py`、`refine_approach.py`。
- 实机入口：`scripts/run_approach_policy.py`，当前未执行。
'''
(OUT/'REPORT.md').write_text(text);print(OUT/'REPORT.md')
