"""Summarize paired physical precontact trials; incomplete batches cannot be final."""
from pathlib import Path
import argparse,json,sys,subprocess,hashlib,zipfile
import numpy as np
D=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--prefix',default='paired_v1');p.add_argument('--expected',type=int,default=12);a=p.parse_args()
rows=json.loads((D/'results'/f'{a.prefix}_summary.json').read_text())
assert len(rows)==a.expected, f'Batch incomplete: {len(rows)}/{a.expected}'
assert len({r['tag'] for r in rows})==len(rows)
lines=[];details=[];metrics={}
for mode in ['oracle','base','lora']:
 rr=[r for r in rows if r['mode']==mode];errors=[r.get('final_audit',r['attempts'][-1]['audit'] if r['attempts'] else {}).get('shadow_true_precontact_error_m') for r in rr];errors=[v for v in errors if v is not None]
 metrics[mode]={'n':len(rr),'success':sum(r['precontact_success'] for r in rr),'navigation_success':sum(r.get('navigation',{}).get('navigation_success',False) for r in rr),'median_error_mm':float(np.median(errors)*1000) if errors else None,'cleanup_failures':sum(not r['cleanup_ok'] for r in rr)}
 m=metrics[mode];err_text='—' if m['median_error_mm'] is None else f"{m['median_error_mm']:.1f}";lines.append(f"| {mode} | {m['navigation_success']}/{m['n']} | {m['success']}/{m['n']} | {err_text} | {m['cleanup_failures']} |")
for r in rows:
 audit=r.get('final_audit',r['attempts'][-1]['audit'] if r['attempts'] else {});e=audit.get('shadow_true_precontact_error_m');er='—' if e is None else f'{e*1000:.1f}'
 details.append(f"| {r['initial']} | {r['variant']} | {r['mode']} | {'通过' if r['precontact_success'] else '未通过'} | {er} | {len(r['attempts'])} | {r.get('error','')} |")
(D/'SUMMARY.json').write_text(json.dumps({'prefix':a.prefix,'metrics':metrics,'rows':rows},indent=2))
s=f'''# 可供性定位接入同场景导航与机械臂执行

2026年11月15日 · PC / RTX 4090 仿真阶段报告

## 摘要

本阶段将原来离线评测的Qwen可供性定位接入动作执行。Nav2先停靠，再用RGB预测把手点、仿真深度反投影到三维，交由MoveIt的KDL/OMPL规划；轨迹通过FollowJointTrajectory驱动同一MuJoCo世界中的真实模拟关节。到位后重新观察，必要时修正一次。

这是预接触验证：目标是把手斜前上方约18.5cm处，而非抓取、开抽屉或完成入抽屉任务。表中成功率不等于LIBERO整任务成功率。

## 目录

1. 对照结果
2. 实现与验收
3. 开发中遇到的问题
4. 局限与后续
5. 逐回合结果与文件

## 1. 对照结果

两个已有初态2、3，两个未用于本轮训练的相机平移条件（相机横轴/竖轴各3cm），每种条件配对oracle、base、lora，共12回合。oracle是真值目标上限；base与lora使用同一个4B BF16模型、相同提示词和控制参数，仅切换适配器。初态都存在历史使用，不能称独立新初态。

| 定位方式 | 导航到达 | 预接触通过 | 到位误差中位数 mm | 清理异常回合 |
|---|---:|---:|---:|---:|
'''+ '\n'.join(lines)+'''

样本很小，仅用于验证“定位输出能否改变实际到位效果”，不能外推跨场景能力。失败与规划拒绝均保留，不重跑覆盖。

## 2. 实现与验收

导航、感知与关节执行共享唯一物理实例。碰撞几何来自当前仿真场景，MoveIt执行接口的反馈读取实际qpos/qvel；控制为限幅力矩PD与重力补偿，原模型惯性与物理子步保留。没有运行时直接赋qpos或用mock关节反馈替代执行。

预接触目标为预测把手点加世界偏移(-0.12,-0.10,+0.10)m，姿态沿用初始TCP。这是固定技能模板，不是学习出的6D抓取。关节PD为400/40，轨迹20Hz，速度/加速度缩放0.12，关节到位容差0.015rad，保持1.5秒。所有定位模式使用同一组参数。

模型到位后再次定位，以TCP与新预测目标间距≤25mm作为视觉确认，最多修正一次。真值坐标不参与模型模式的修正与停止，只用于事后误差评分；预接触通过还要求真值误差≤30mm、无环境或自碰撞、物理实例未变。oracle单独记录，没有视觉确认。原LIBERO任务状态另存libero_original_task_success。

## 3. 开发中遇到的问题

首版PD140/20未达到关节容差，在真值开发回合上改为400/40并冻结。静态URDF工具坐标用8个姿态与MuJoCo核对；真实导航后仍出现约0.206mm的FK差异，严格检查阻止了执行。原因是积分后的qpos与派生位姿来自不同时刻，读取状态前调用forward同步后，导航开发回合误差降至约5×10⁻¹²m并成功执行。未放宽坐标检查阈值。

开发nominal初态0的LoRA预接触误差约9.2mm、视觉重观测残差5.6mm；基座约97.6mm，未通过。导航复测navigation_dev2也已到位。这些开发结果不混入上表。

MoveIt Humble在部分退出过程中发生轨迹执行管理器析构崩溃，与上游已知问题栈相符。动作结果和cleanup_ok分别记录；未升级系统ROS或声称已解决该退出问题。控制日志保留真实关节、力矩和接触信息。

## 4. 局限与后续

深度和碰撞地图来自仿真真值；导航使用理想里程计及已知地图，没有SLAM。只测单厨房、单把手、固定预接触姿态；相机扰动不是跨场景。没有把当前到位结果称为自主抓取，也没有替代此前π0.5整任务闭环结论。

后续应独立验证预接触到真实把手接触、夹持及拉动的安全轨迹，再接完整任务；先解决物理动作接口与任务验收，再比较端侧延迟。当前阶段没有连接实体机械臂、没有运行Jetson性能测试、没有新微调权重。

## 5. 逐回合结果与文件

| 初态 | 视角 | 定位 | 预接触 | 误差 mm | 规划次数 | 错误 |
|---:|---|---|---|---:|---:|---|
'''+ '\n'.join(details)+'''

相关代码在scripts/。PROTOCOL.md记录冻结设置，research/记录接口选型与坐标核对。结果在results/<tag>/case_result.json，视频rollout.mp4，逐步日志events.jsonl，控制器反馈controller_*.json，规划结果plans/attempt*/moveit_result.json。SUMMARY.json可逐条追溯本表。README.md提供命令、环境与限制。
'''
# Full-skill continuation is scored separately from the paired precontact comparison.
combined=[]
for name in ['combined_probe1','combined_validation1']:
 file=D/'results'/f'{name}_summary.json'
 if file.exists():combined.extend(json.loads(file.read_text()))
if combined:
 success=sum(bool(v.get('combined_official_success') and v.get('visual_milestones_complete')) for v in combined)
 for v in combined:
  events=D/'results'/v['tag']/'events.jsonl';truth=[]
  if events.exists():
   for line in events.read_text().splitlines():
    event=json.loads(line)
    if event.get('kind')=='policy_step' and event.get('shadow_official_success'):truth.append(event['policy_step'])
  v['shadow_first_success_step']=min(truth) if truth else None
  v['visual_delay_after_first_success_steps']=v.get('policy_steps',0)-min(truth) if truth and v.get('visual_milestones_complete') else None
  folder=D/'results'/v['tag'];checkfile=folder/'milestone_checks.json';v['intermediate_region_disagreements']=[]
  if checkfile.exists():
   checks=json.loads(checkfile.read_text());labels=sorted(folder.glob('shadow_labels_*.json'));prefix_count=len(labels)-len(checks)
   for j,event in enumerate(checks):
    if event.get('confirmed')=='bowl_in_drawer':
     shadow=json.loads(labels[prefix_count+j].read_text())
     if not shadow['bowl_in_drawer_region']:v['intermediate_region_disagreements'].append({'step':event['policy_step'],'label':labels[prefix_count+j].name})
 metrics_combined={'n':len(combined),'official_and_visual_success':success,'false_visual_stops':sum(v.get('visual_milestones_complete',False) and not v.get('combined_official_success',False) for v in combined),'rows':combined}
 (D/'COMBINED_SUMMARY.json').write_text(json.dumps(metrics_combined,indent=2))
 paragraph=f'另补做完整任务交接：已完成{len(combined)}回合，其中{success}回合同时通过Qwen视觉里程碑与官方整任务检查。完整任务与预接触结果分开统计，详见第6节。'
 s=s.replace('## 目录',paragraph+'\n\n## 目录',1)
 s=s.replace('5. 逐回合结果与文件','5. 逐回合结果与文件\n6. 无重置交接完整任务',1)
 s=s.replace('后续应独立验证预接触到真实把手接触、夹持及拉动的安全轨迹，再接完整任务；先解决物理动作接口与任务验收，再比较端侧延迟。','MoveIt直接完成把手接触、夹持及拉动仍未验证。本阶段另做预接触后交接π0.5完成原任务的功能测试，见第6节；不能把π0.5的操作归为MoveIt抓取。')
 table=[]
 for v in combined:
  f=D/'results'/v['tag']/'policy_handover.json'
  no_reset=json.loads(f.read_text()).get('no_state_reset',False) if f.exists() else False
  table.append(f"| {v['initial']} | {v.get('precontact_success',False)} | {no_reset} | {v.get('policy_steps','—')} | {v.get('visual_milestones_complete',False)} | {v.get('combined_official_success',False)} | {v.get('visual_delay_after_first_success_steps','—')} | {v.get('cleanup_ok',False)} |")
 s+='\n\n## 6. 无重置交接完整任务\n\n'
 s+=f'首个开发探针初态0通过后，增加已有初态1、4、5进行重复性检查。本节当前共{len(combined)}回合，完整任务通过{success}/{len(combined)}。这些初态均有历史使用，不是跨场景测试；没有直接比较“接入预接触”与“原生π0.5”成功率，不能宣称提升整任务表现。\n\n'
 s+='执行顺序为Nav2停靠、LoRA定位、MoveIt预接触、停止MoveIt与关节桥、保留当前qpos交给π0.5 OSC执行放碗并关抽屉。交接记录逐项断言前后关节状态完全相同，没有先展开或重置到熟悉姿态。π0.5为官方权重，没有在本轮微调。\n\n'
 s+='Qwen以原有文字记忆里程碑顺序检查碗入抽屉、抽屉关闭，每10个策略步查询一次；到达视觉确认才停止，520步上限兜底。官方任务谓词仅作影子记录及最终评分，不驱动策略停止。确认延迟按Qwen停止步减去首次官方成功步计算，未成功则不计；这只是事后诊断。策略时间仍是仿真步数，不能当端侧实时性能。此前提示词中的dark cabinet写法保持不变。\n\n'
 s+='| 初态 | 预接触通过 | 无重置交接 | 策略步数 | Qwen完成 | 官方整任务成功 | 确认延迟步 | 清理正常 |\n|---:|---|---|---:|---|---|---:|---|\n'+'\n'.join(table)+'\n\n'
 disagree=[v for v in combined if v['intermediate_region_disagreements']]
 if disagree:
  s+='\n中间里程碑仍有局限：'+ '；'.join(f"初态{v['initial']}在第{e['step']}步确认碗入抽屉时，与影子区域标注不一致" for v in disagree for e in v['intermediate_region_disagreements'])+'。区域检查为抽屉site边界加2cm，属于几何诊断而非官方整任务谓词，不能直接当作严格误报率。最终未误停也不能说明中间视觉判断可靠；本轮没有中途修改提示词或确认规则。\n\n'
 s+='相关代码在scripts/sim_combined.py、combined_worker.py、run_combined.py。COMBINED_SUMMARY.json及各回合policy_handover.json、milestone_checks.json、policy_inputs/、rollout.mp4保存交接与执行结果。该入口是本阶段探针，尚未整合为通用技能调度器；完整操作仍由π0.5完成，可供性只决定前置预接触位。\n'

native_file=D/'results/native_probe1_summary.json'
if native_file.exists():
 native=json.loads(native_file.read_text());(D/'NATIVE_CONTROL_SUMMARY.json').write_text(json.dumps(native,indent=2))
 s+='\n\n## 7. 失败初态的原生技能诊断\n\n'
 s+='在完整交接初态4失败后，仅选择该初态做一次不执行预接触的π0.5诊断，使用相同Nav2、原始技能和Qwen监视器。选择发生在观察失败之后，不是预先冻结的效益试验；Nav2墙钟执行也不保证逐帧状态相同，因此不能作为严格因果或总体成功率比较。\n\n'
 for v in native:
  s+=f"原生初态{v['initial']}：官方成功{v.get('combined_official_success',False)}，Qwen完成{v.get('visual_milestones_complete',False)}，策略步{v.get('policy_steps')}，错误{v.get('error','无')}。\n\n"
 s+='本次原生对照在320步成功，而前置预接触回合在520步失败，提示交接姿态可能影响策略表现；单个事后选取的回合尚不能证明因果。\n\n把手预接触用于验证定位到动作接口，不能默认前置到所有π0.5技能。当前整任务先搬碗、后关抽屉，应由阶段条件决定何时使用把手可供性；额外前置姿态可能造成策略输入分布偏移。该轮仅证明物理接口贯通，不将完整交接版本替换此前原生闭环。\n'

# Plot directly from saved predictions and shadow labels, never model-generated visuals.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
fig,axs=plt.subplots(1,3,figsize=(10,3.6))
for ax,mode in zip(axs,['oracle','base','lora']):
 r=next(v for v in rows if v['mode']==mode);folder=D/'results'/r['tag']
 ax.imshow(Image.open(folder/'grounding_000.png'))
 shadow=json.loads((folder/'shadow_labels_000.json').read_text());uv=shadow['handle']['uv']
 ax.scatter(*uv,c='lime',marker='+',s=90,label='Reference handle')
 if mode!='oracle':
  pred=json.loads((folder/'prediction_00.json').read_text());uv=np.asarray(pred['point_2d'])*512/1000
  ax.scatter(*uv,facecolors='none',edgecolors='red',s=100,label='Qwen prediction')
 ax.set_title(mode);ax.axis('off')
axs[1].legend(loc='upper right',fontsize=6)
fig.tight_layout();fig.savefig(D/'grounding_comparison.png',dpi=160);plt.close(fig)
s=s.replace('## 2. 实现与验收','![同配对条件初次观察，绿色为参考把手点，红色为模型预测](grounding_comparison.png)\n\n## 2. 实现与验收',1)
(D/'REPORT.md').write_text(s)
sys.path.insert(0,'/tmp/mobile-qwen-doclibs');import markdown
css='@page{size:A4;margin:16mm}body{font-family:"Noto Sans CJK SC",sans-serif;font-size:9.5pt;line-height:1.65;color:#263448}h1{font-size:23pt}h2{font-size:15pt;break-after:avoid}table{border-collapse:collapse;width:100%;font-size:8pt}td,th{border:1px solid #bcc8d2;padding:5px;overflow-wrap:anywhere}th{background:#eaf0f5}img{max-width:100%;max-height:85mm}code{overflow-wrap:anywhere}'
h=D/'REPORT.html';h.write_text('<meta charset="utf-8"><style>'+css+'</style>'+markdown.markdown(s,extensions=['tables','fenced_code']))
subprocess.run(['/usr/bin/google-chrome','--headless','--no-sandbox','--disable-gpu','--no-pdf-header-footer','--print-to-pdf='+str(D/'REPORT.pdf'),h.as_uri()],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
ROOT=D.parents[2]
with zipfile.ZipFile(D/'SOURCE.zip','w',zipfile.ZIP_DEFLATED) as z:
 files=[f for f in D.rglob('*') if f.is_file() and ('scripts' in f.relative_to(D).parts or f.name in ['README.md','PROTOCOL.md']) and '__pycache__' not in f.parts]
 for directory in ['experiments/mobile_vla/scripts','experiments/mobile_vla/configs','experiments/mobile_qwen/scripts','experiments/mobile_qwen/affordance','perception/qwen_scene','experiments/vla_libero/skill_bootstrap/scripts']:
  folder=ROOT/directory
  files.extend(f for f in folder.glob('*') if f.is_file() and (f.suffix in ['.py','.yaml','.yml'] or f.name=='ros-python'))
 for rel in ['experiments/vla_libero/scripts/evaluate.py','experiments/vla_libero/scripts/python_shared','experiments/vla_libero/scripts/qwen_planner.py']:
  files.append(ROOT/rel)
 for f in sorted(set(files)):z.write(f,str(f.relative_to(ROOT)))
 z.writestr('BUNDLE_README.md','源码保留VLA目录层级，新增入口见experiments/mobile_qwen/closed_loop_20261115/README.md。包含调用到的本地源码，不含模型权重、训练集、虚拟环境、LIBERO/openpi第三方库和TF2二进制overlay。运行需要原文列出的ROS2与Python环境及这些外部依赖。本包不承诺脱离模型和仿真资源即可运行，也不提供无GPU推理。')
manifest={str(f.relative_to(D)):hashlib.sha256(f.read_bytes()).hexdigest() for f in D.glob('scripts/*') if f.is_file()};(D/'SOURCE_SHA256.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(metrics,indent=2));print(D/'REPORT.pdf')
