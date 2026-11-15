"""Summarize measured offline results without claiming robot task success."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3];BASE=ROOT/'experiments/smolvla_pc';OUT=BASE/'mousepad_report';OUT.mkdir(exist_ok=True)
h=json.loads((BASE/'runs/mousepad_holdout_1500/summary.json').read_text());f=json.loads((BASE/'runs/mousepad_all3_v1/summary.json').read_text())
labels=['Pan','Shoulder','Elbow','Wrist flex','Wrist roll','Gripper']
fig,ax=plt.subplots(figsize=(10,4.5));x=np.arange(6);w=.25
for shift,key,label in [(-w,'before','Base'),(0,'after','Fine-tuned'),(w,None,'Hold current')]:
 y=h[key]['joint_mae'] if key else h['after']['hold_current_joint_mae'];ax.bar(x+shift,y,w,label=label)
ax.set_xticks(x,labels);ax.set_ylabel('MAE (degrees; gripper in percentage points)');ax.set_title('Episode 3 held out: 50-step open-loop action error');ax.legend();fig.tight_layout();fig.savefig(OUT/'holdout_mae.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots(figsize=(9,3.5))
for name in ['mousepad_holdout_1500','mousepad_all3_v1']:
 r=[json.loads(x) for x in (BASE/'runs'/name/'training.jsonl').read_text().splitlines()];ax.plot([x['step'] for x in r],[x['loss'] for x in r],label=name)
ax.set_xlabel('Optimizer steps');ax.set_ylabel('Flow-matching training loss');ax.set_yscale('log');ax.legend();fig.tight_layout();fig.savefig(OUT/'training_loss.png',dpi=160);plt.close(fig)
rows=[]
for j,label in enumerate(['底座','肩关节','肘关节','腕俯仰','腕旋转','夹爪']):rows.append(f"| {label} | {h['before']['joint_mae'][j]:.2f} | {h['after']['joint_mae'][j]:.2f} | {h['after']['hold_current_joint_mae'][j]:.2f} |")
text='''# 耳机壳拾取与鼠标垫放置：SmolVLA 首轮微调

2026年10月25日

## 本轮完成

三条本机遥操作示教已用于真正的SmolVLA微调，不再只运行基础权重。原始1908帧保留，去掉前后等待和启动对齐后，训练索引保留900帧。每条保留竖直出发、接近、拾取、鼠标垫放置与回竖直，动作中的腕部−90°已生效，不重复叠加。

图像和关节读取使用主机时间戳。中位state到图像接收间隔约0.15毫秒，图像接收到发送动作约2.8～3.7毫秒；这些不是曝光同步精度。图像完整、六维状态和动作均有限。图像抽查可见操作阶段，不能证明每一帧无碰撞。

## 训练与留出检查

第一、二条589帧训练，第三条311帧整条留出。归一化仅使用前两条的逐关节统计。基于官方固定SmolVLA权重，冻结视觉语言主干，训练动作专家及配置允许的投影层，共约9988万个可训练参数。批量2、学习率1e-4、1500次优化更新；图像保持比例填充到256×256，输入单腕相机，预测未来50步六维绝对动作。

第三条均匀抽取12个观察位置，固定各位置的随机种子，对比基础权重与微调后输出。窗口止于本条结尾，不跨条取动作，尾部padding不参与评估。表中为未来最多50步的平均绝对误差，前五维单位度，夹爪为百分比点。

| 关节 | 微调前 | 微调后 | 保持当前位置基线 |
|---|---:|---:|---:|
'''+ '\n'.join(rows)+'''

![第三条留出动作误差](holdout_mae.png)

这些是录制轨迹上的开环误差，不是真机成功率。只留出一条、只比较一个任务，不能据此认定具有位置或语言泛化。静止、遮挡和视角接近时，仅凭单腕图像可能不能区分任务阶段；状态是模型的重要输入，真实执行时必须使用实时状态。

## 三条合训版本

完成留出检查后，从同一基础权重重新开始，用三条900帧合训。三条合训模型中的第三条也参与了训练，其误差只能称拟合检查。模型没有接入电机，没有执行任何自主抓取。

![训练损失](training_loss.png)

## 文件与下次测试

- 数据索引与审查图：`datasets/earbud_mousepad/prepared_v1/`。
- 训练代码：`experiments/smolvla_pc/scripts/train_mousepad.py`。
- 留出实验：`experiments/smolvla_pc/runs/mousepad_holdout_1500/`。
- 三条合训：`experiments/smolvla_pc/runs/mousepad_all3_v1/`。
- 模型文件：各运行目录内`checkpoint/`；配套统计在`normalization.json`，配置在`config.json`。

推理必须读取该次训练的checkpoint、归一化、256×256填充处理及任务指令，不能使用先前单条数据统计的旧网页worker直接替换权重。先做保存后重新加载一致性检查，再检查首步跳变量和物理限位；正式实机仍需操作人员在场、空载慢速试验与明确停止方式。当前两臂断电，训练不连接串口。
'''
(OUT/'REPORT.md').write_text(text)
print(OUT/'REPORT.md')
