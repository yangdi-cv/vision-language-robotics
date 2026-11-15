"""Generate the deployment report from completed measurement files."""
from pathlib import Path
import json, numpy as np, re
from datetime import datetime
from zoneinfo import ZoneInfo
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
T=Path(__file__).resolve().parents[1];J=T/'runs/jetson';F=T/'figures';F.mkdir(exist_ok=True)
def read(p):return json.loads(p.read_text())
paired=read(J/'paired_vision/summary.json');steady=read(J/'qwen_steady_jsonl/results.json');status=read(J/'qwen_steady_jsonl/status.json')
assert status['phase']=='complete' and status['completed']==60 and not status['camera_errors']
q=read(J/'qwen_live/results.json');adapter=read(J/'adapter_eval/results.json');astatus=read(J/'adapter_eval/status.json');assert astatus['phase']=='complete' and len(adapter)==20
pc=read(T/'runs/pc_nf4_adapter/results.json');original=read(T.parent/'alibaba_pc_revision/runs/lora_validation/summary.json');scene_pc=read(T/'runs/pc_scene/results.json')
seeds=[20261024,20261025];act=[read(T/f'runs/act_seed_{s}/summary.json') for s in seeds]
for s in seeds:
 exp=read(T/f'runs/act_seed_{s}/experiment.json');assert set(exp['rollout_validation_seeds']).isdisjoint(exp['final_test_seeds'])
valid=[x for x in steady if not x['warmup']];v=read(J/'qwen_steady_jsonl/vision_results.json');v=v[5:]
def stat(xs):return f"{np.median(xs):.2f} / {np.percentile(xs,95):.2f}"
base_rows=[x for x in q if x['source']=='saved_camera_case'];correct=sum(x['target_visibility_correct'] for x in base_rows)
fig,axs=plt.subplots(1,2,figsize=(11,3.8));axs[0].bar(['FP32','FP16','TRT FP16'],[x['p50_ms'] for x in paired['latency']]);axs[0].set_ylabel('Detection + depth median [ms]');axs[0].grid(axis='y',alpha=.2)
old_steady=read(J/'qwen_steady/results.json');axs[1].plot([x['cycle'] for x in old_steady if not x['warmup']],[x['total_inference_s'] for x in old_steady if not x['warmup']],label='Full-history JSON');axs[1].plot([x['cycle'] for x in valid],[x['total_inference_s'] for x in valid],label='JSONL');axs[1].legend();axs[1].set_xlabel('Cycle');axs[1].set_ylabel('Inference [s]');axs[1].grid(alpha=.2);fig.tight_layout();fig.savefig(F/'jetson_latency.png',dpi=150);plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(11,3.8))
for s in seeds:
 h=read(T/f'runs/act_seed_{s}/history.json');axs[0].plot([x['step'] for x in h],[x['validation_l1'] for x in h],'o-',label=str(s));axs[1].plot([x['step'] for x in h],[x['validation_success']/36 for x in h],'o-',label=str(s))
axs[0].set_ylabel('Validation normalized L1');axs[1].set_ylabel('Validation success / 36')
for ax in axs:ax.set_xlabel('Training update');ax.grid(alpha=.2);ax.legend()
fig.tight_layout();fig.savefig(F/'expanded_validation.png',dpi=150);plt.close(fig)
# Display four distinct scene conditions; adjacent frames do not count as independent scenes.
from PIL import Image,ImageDraw
canvas=Image.new('RGB',(960,700),'white');draw=ImageDraw.Draw(canvas)
for i,(folder,label) in enumerate([('scene','Target present'),('scene_absent','Target absent'),('scene_shifted','Target moved'),('scene_partial','Target partly visible')]):
 im=Image.open(J/folder/'frame_005.jpg').convert('RGB');im.thumbnail((450,310));x=(i%2)*480+10;y=(i//2)*350+30;canvas.paste(im,(x,y));draw.text((x,y-20),label,fill='black')
canvas.save(F/'scene_cases.jpg')
rows=['# 端侧视觉部署与策略验证扩展','', '阿里方向阶段进展　2026年10月24日','', '## 摘要','', '本轮继续处理上一阶段留下的端侧部署与验证集规模问题。在Jetson Orin Nano接入真实腕部相机，部署检测和单目深度的TensorRT引擎，并验证Qwen场景观察与视觉模型连续运行；同时把工作站上的4B模型另存为NF4权重，加载原LoRA，在Jetson复测原有目标指令测试集。ACT部分扩大验证布局，并用两个随机种子重复训练和独立测试。','', '本次机械臂只连接USB，相机提供真实图像；两臂12V断开，未发送运动指令。端侧视觉部署、目标指令测试与ACT仿真评测分别统计，不把它们写成真机自主抓取闭环。','', '## 1. 环境与数据流','', '| 项目 | 实际设置 |','|---|---|','| 硬件 | Jetson Orin Nano，约8GB共享内存 |','| 系统 | Ubuntu22.04.5，L4T R36.4.7，15W模式；未锁定频率 |','| 基础环境 | 现有an-project，Python3.10.19、PyTorch2.8.0、CUDA12.6；未修改安装包 |','| 视觉依赖 | 项目独立Ultralytics8.4.175覆盖目录，系统TensorRT10.7.0 |','| Qwen依赖 | 独立Transformers5.8.1、bitsandbytes0.50.2等加载目录 |','| 相机 | V4L2，640×480，逆时针旋转90°；设备支持最高30fps，实际吞吐另测 |','', '相机线程持续更新一个最新帧槽。Qwen在上一轮结束后取当前帧，TensorRT检测与深度使用另一路最新帧；同一次检测和深度共用图像。模型输入通过内存交付，保存PNG只用于输入归档，不承担模型间传图。两路可同时运行，但不是同一帧的严格时间同步融合。','', '相关代码在 `scripts/jetson_qwen.py`、`jetson_vision.py` 与 `paired_vision.py`。本报告中的scripts、runs均相对于 `experiments/alibaba_jetson_revision/`。','', '## 2. 检测与深度的后端对照','', '正式对照使用同一批10张真实腕部图像，各重复3次；三种设置都固定640×640填充，每种后端预热3次。计时包含串行检测、深度、预处理、后处理及CPU输出拷贝，不含相机采集和文件解码。FP16检查的是实际predictor后端参数，原始YOLO包装对象因深拷贝仍可保持FP32。','', '| 后端 | 样本数 | 中位耗时 ms | P95 ms |','|---|---|---|---|']
for x in paired['latency']:rows.append(f"| {x['backend']} | {x['n']} | {x['p50_ms']:.2f} | {x['p95_ms']:.2f} |")
ratio=paired['latency'][0]['p50_ms']/paired['latency'][2]['p50_ms'];rows+=['',f'本批输入下TensorRT的串行处理速度约为FP32的{ratio:.2f}倍。引擎在目标Jetson上编译；编译开销与稳态推理耗时分开。TensorRT10.7对检测头的导出兼容处理会改变后端实现，不能把结果简单归结为浮点位宽本身。','', '![端侧耗时](figures/jetson_latency.png)','', '深度模型没有实测距离真值，数值一致性不等于深度准确率；本轮未做米制标定或手眼标定。当前通用检测器在这批图像、0.35阈值下没有给出有效目标框，耳机壳的专用识别仍需任务数据或其他定位方法。不能用推理速度证明抓取检测已经有效。','', '相关代码在 `scripts/paired_vision.py`；逐帧结果在 `runs/jetson/paired_vision/`。','', '## 3. 连续运行与最新帧','', '轻量场景观察使用设备上已有Qwen3-VL-2B NF4基座，本轮重新测试，不加载其他项目适配器。它与工作站4B模型不同，不能据此得出4B量化前后提速结论。最初在另一个线程初始化TensorRT时，首次CUDA图捕获与Qwen生成发生冲突；改为在主线程先完成引擎初始化、预热与同步，再进入连续运行，复测通过。','', '第一轮60次联动测试发现，视觉线程每次重写全部历史JSON，序列越长，CPU序列化工作越多。改成每条结果追加JSONL，仅在结束时保存一次汇总，再按相同模型、输入模式和15W设置完整运行60轮。第一轮作为预热，以下Qwen统计为59轮；视觉线程前5次单列为初始化段。','', '| 指标 | 中位数 / P95 |','|---|---|',f"| Qwen完整推理 s | {stat([x['total_inference_s'] for x in valid])} |",f"| 选帧时主机帧年龄 ms | {stat([x['input_age_at_selection_ms'] for x in valid])} |",f"| 输出时对应画面年龄 s | {stat([x['result_frame_age_s'] for x in valid])} |",f"| 联动时检测＋深度 ms | {stat([x['seconds']*1000 for x in v])} |",f"| 联动视觉结果画面年龄 ms | {stat([x['result_age_ms'] for x in v])} |",'',f"60轮完成，视觉线程记录{len(v)+5}次结果，未记录相机或模型线程异常。Qwen CUDA分配峰值约{max(x['peak_cuda_allocated_bytes'] for x in steady)/2**30:.2f}GiB；进程RSS峰值约{max(x['process_peak_rss_kib'] for x in steady)/2**20:.2f}GiB。CUDA分配量不等于整机RAM占用。",'', '输入年龄从主机read返回后计时，不是曝光时间。结果显示几秒前的画面主要来自生成耗时；没有将旧帧排成待推理队列。静态场景下持续运行可验证稳定性与帧调度，不能代表动态目标跟踪能力。','', '相关代码在 `scripts/jetson_qwen.py`；结果在 `runs/jetson/qwen_steady_jsonl/`。','', '## 4. 真实场景观察的有效范围','', '实际场景包含深蓝色塑胶壳、黑色鼠标垫和夹爪。采集原位、完全移走、换位置、边缘部分可见四种状态，每种取两张邻近帧。最初一次“移走”仍在边缘露出目标，检查截图后改标为部分可见，并重新采集真正的无目标状态。','', '![场景样例](figures/scene_cases.jpg)','',f'2B基座的目标可见性与人工检查一致{correct}/{len(base_rows)}；PC端4B基座为{sum(x["target_visibility_correct"] for x in scene_pc)}/{len(scene_pc)}。相邻帧高度相关，这只是四种场景状态的小样例，不是独立大测试集。JSON可解析与目标判断正确分开统计，目标移走等失败用例保留。','', '这组观察结果说明，视觉语言输出仍需要定位校验和任务约束；当前不将单句“目标可见”直接转换为机械臂动作。','', '相关代码在 `scripts/pc_scene_eval.py` 和 `jetson_qwen.py`；输入在 `runs/jetson/scene*/`。','', '## 5. 自有4B模型与LoRA的Jetson迁移','', '在工作站读取原固定版本Qwen3-VL-4B-Instruct，另存NF4双重量化权重，计算使用BF16。语言与视觉线性层参与量化，输出头按配置排除；原BF16缓存与LoRA文件保持不变。随后把新权重和原第80步best_adapter传到Jetson，校验两端SHA256一致。适配器按原结构加载，没有合并，也没有把4B适配器套到2B模型上。','', '| 运行设置 | 原20个测试用例完整字段匹配 |','|---|---|',f'| 原PC BF16＋LoRA | {original["best_test_correct"]}/20 |',f'| 本轮PC NF4＋同一LoRA | {sum(x["correct"] for x in pc)}/20 |',f'| 本轮Jetson NF4＋同一LoRA | {sum(x["correct"] for x in adapter)}/20 |','',f'Jetson完整推理耗时中位数/P95为{stat([x["seconds"] for x in adapter if not x["warmup"]])}秒；每例输出token数保留在记录中。CUDA分配峰值约{max(x["cuda_peak_allocated_bytes"] for x in adapter)/2**30:.2f}GiB。','', '这些是原划分中的20条指令，只有两个测试布局，不能泛化为真实桌面任务成功率。没有使用测试集重新训练或选择适配器；它验证的是同一个适配器在NF4及Jetson运行条件下的迁移结果，不是导航或机械臂执行成功率。','', '相关代码在 `scripts/quantize_qwen.py` 与 `jetson_adapter_eval.py`；结果分别在 `runs/pc_nf4_adapter/` 和 `runs/jetson/adapter_eval/`。','', '## 6. ACT扩大验证与重复训练','', '原66条示教及54/6/6轨迹划分保持不变；本轮不是增加示教数据。训练随机种子20261024、20261025，各10000步，每2000步在1500–1505六个布局、每布局六任务上验证。选模规则事先固定为成功数优先，同分取验证L1较低，完全同分取较早更新；另保存最低验证L1模型。1600–1607八个布局、每布局六任务只在最后测试。','', '![扩大验证](figures/expanded_validation.png)','', '| 训练种子 | 选模方法 | 更新数 | 独立测试成功 | 测试归一化L1 |','|---|---|---|---|---|']
for seed,result in zip(seeds,act):
 for method,name in [('l1','验证L1'),('rollout','仿真成功数＋同分L1')]:
  x=result['selection'][method];rows.append(f"| {seed} | {name} | {x['selected_step']} | {x['test_success']}/{x['test_trials']} | {x['test_l1']:.4f} |")
rows+=['', '两种选模方法在同一训练种子内共用训练过程，避免把不同训练运行混为选模收益。验证布局和同分规则同时调整，本轮不能单独归因于其中一项。两种训练种子仍不足以覆盖训练方差；示教来自脚本，执行继续使用原仿真理想末端约束及模拟深度，没有证明SO101动力学或真实视觉泛化。','', '相关代码在 `scripts/train_act_expanded.py`，执行循环复用 `experiments/alibaba_pc_revision/scripts/act_runner.py`；全部历史与逐次测试在 `runs/act_seed_20261024/`、`runs/act_seed_20261025/`。','', '## 7. 结论与下一步','', '本轮完成真实相机到端侧模型的读取链路、TensorRT推理对照、连续联动及4B目标指令适配器迁移，并扩大ACT选模验证。实际收益与限制分别记录：TensorRT降低了这批输入的处理耗时；最新帧调度避免积累旧输入，但Qwen生成仍需数秒；语言观察仍有误判，通用检测器也没有解决耳机壳定位。','', '下一步围绕目标定位、负样例与距离标定继续做，而不是直接把未校验的观察结果变成真机运动。移动导航、MoveIt规划与当前桌面策略仍未合成同一真实场景的完整闭环，本次不声称已经完成。','', '## 代码与结果目录','', '```text','experiments/alibaba_jetson_revision/','  scripts/','    jetson_vision.py        相机、单模型推理与引擎导出','    paired_vision.py        同图像同尺寸后端对照','    jetson_qwen.py          最新帧、场景观察及连续联动','    quantize_qwen.py        另存4B NF4权重','    jetson_adapter_eval.py  4B适配器迁移评测','    bridge_eval.py          端侧输出到仿真策略','    bench_journal.py        日志编码成本对照','    pc_scene_eval.py        PC端场景样例','    train_act_expanded.py   扩大验证的ACT训练','  README.md','  requirements-jetson-reference.txt','  runs/jetson/             实体输入、逐次输出、运行记录','  runs/act_seed_*/         两次训练、验证和最终测试','  figures/','```','', '源码包只包含代码、配置、必要场景资产与许可。Python环境、权重、TensorRT引擎、输入照片、逐步结果和训练缓存保留在本地；运行条件与路径参数见README。现有工作站与Jetson依赖不同，不能将参考版本表当成跨平台一键安装保证。']
# The logging comparison is sequential, not a randomized full-system causal study.
journal=read(J/'runs/journal_benchmark.json')
pos=rows.index('相关代码在 `scripts/jetson_qwen.py`；结果在 `runs/jetson/qwen_steady_jsonl/`。')
log_rows=['日志成本单独在Jetson CPU测量，每档10次，仅计JSON编码，不含文件I/O或GPU。','', '| 历史条数 | 重写全部JSON中位 ms | 追加单条中位 ms |','|---|---|---|']
for n in [100,1000,2000,5000]:
 old=next(v for v in journal if v['history_length']==n and v['method']=='full_history');new=next(v for v in journal if v['history_length']==n and v['method']=='append_one_record')
 log_rows.append(f"| {n} | {old['median_serialization_ms']:.3f} | {new['median_serialization_ms']:.3f} |")
old_first=np.median([v['total_inference_s'] for v in old_steady[1:11]]);old_last=np.median([v['total_inference_s'] for v in old_steady[-10:]])
new_first=np.median([v['total_inference_s'] for v in steady[1:11]]);new_last=np.median([v['total_inference_s'] for v in steady[-10:]])
log_rows += ['', f'前后10轮中位推理耗时，旧日志方式为{old_first:.2f}→{old_last:.2f}秒，JSONL方式为{new_first:.2f}→{new_last:.2f}秒。逐条追加消除了随历史条数增长的编码量；两次整机运行先后进行、频率未锁定，不能把全部耗时差都归因于这一处修改。原始运行与失败日志保留。', '']
rows[pos:pos]=log_rows
# Board telemetry is scoped to the continuous integrated run, not model-only energy.
tele=[];start_wall=min(x['inference_start_wall_s'] for x in valid);end_wall=max(x['inference_end_wall_s'] for x in valid)
for line in (J/'runs/tegrastats_jsonl.log').read_text().splitlines():
 try:
  stamp=datetime.strptime(line[:19],'%m-%d-%Y %H:%M:%S').replace(tzinfo=ZoneInfo('Australia/Sydney')).timestamp()
  if not start_wall<=stamp<=end_wall:continue
  power=re.search(r'VDD_IN (\d+)mW/',line);temp=re.search(r'gpu@([0-9.]+)C',line);ram=re.search(r'RAM (\d+)/(\d+)MB',line)
  if power and temp and ram:tele.append((int(power[1])/1000,float(temp[1]),int(ram[1])))
 except ValueError:continue
if tele:
 pos=rows.index('输入年龄从主机read返回后计时，不是曝光时间。结果显示几秒前的画面主要来自生成耗时；没有将旧帧排成待推理队列。静态场景下持续运行可验证稳定性与帧调度，不能代表动态目标跟踪能力。')
 rows[pos:pos]=[f'连续运行期间板载采样共{len(tele)}条，VDD_IN平均{np.mean([v[0] for v in tele]):.2f}W，GPU温度最高{max(v[1] for v in tele):.1f}°C，整机已用RAM最高{max(v[2] for v in tele)}MB。该读数包括整机其他活动，不是外置功率计，也没有扣除空闲基线。','']
# A staged cross-device handoff uses actual Jetson outputs, with no labels in dispatch.
bridge=read(T/'runs/bridge_v2/summary.json')
position=rows.index('## 7. 结论与下一步')
rows[position]='## 8. 结论与下一步'
rows[position:position]=['## 7. 端侧目标到仿真策略的衔接','', '把Jetson上4B＋LoRA的20条实际输出交给PC，加载第一轮按仿真成功数选出的ACT模型。对可执行目标在对应的408、409布局重新建立仿真；执行前确认初始化RGB与语言模型所见图像一致。模型输出决定是否执行及目标颜色、盒子，评测标签只在事后统计使用。', '', f"12条可执行指令成功{bridge['positive_successes']}条；8条缺目标、歧义或否定指令均未执行。布局408的两条红色目标任务虽然完成抬升、夹持与释放，但没有稳定进入指定盒，保留为失败。目标字段20/20匹配不等于动作20/20成功。", '', '这是使用保存输出进行的离线分阶段衔接，未计网络往返时延，仿真也没有在Qwen等待期间推进，不能称为实时异构闭环。两种布局不替代前面的独立测试集。', '', '同配置完整重复一次仍为10/12成功、8条负向指令均未执行。相关代码在 `scripts/bridge_eval.py`；结果在 `runs/bridge_v2/` 与 `runs/bridge_repeat/`，首次一致性检查中断的部分运行保留在 `runs/bridge/`。', '']
(T/'REPORT.md').write_text('\n'.join(rows)+'\n')
(T/'RESULT_SUMMARY.json').write_text(json.dumps({'paired_vision':paired['latency'],'steady_cycles':len(steady),'adapter_correct':sum(x['correct'] for x in adapter),'adapter_trials':len(adapter),'act':dict(zip(map(str,seeds),act)),'bridge':bridge},ensure_ascii=False,indent=2))
print('REPORT GENERATED')
