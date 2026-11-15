"""Deliver a runnable original-task instance and a separate visual-goal pilot report."""
import hashlib,json,shutil,subprocess,tempfile,zipfile
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from markdown_it import MarkdownIt
import imageio_ffmpeg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from visual_prompt import mark
S=Path(__file__).resolve().parents[1];V=S.parent;ROOT=V.parents[1];I=V/'official_instance'
D=ROOT/'deliverables/20261114_阿里_自主实例与显式目标接口'
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
def table(head,rows):return '\n'.join(['| '+' | '.join(head)+' |','|'+'|'.join(['---']*len(head))+'|',*['| '+' | '.join(map(str,r))+' |' for r in rows]])
def frames(path):
 reader=imageio_ffmpeg.read_frames(str(path));m=next(reader)
 return [np.frombuffer(f,np.uint8).reshape(m['size'][1],m['size'][0],3).copy() for f in reader]
def figures(summary):
 F=S/'figures';F.mkdir(exist_ok=True);font=ImageFont.truetype(FONT,23);small=ImageFont.truetype(FONT,18)
 im=Image.new('RGB',(1000,1050),'#edf2f7');d=ImageDraw.Draw(im);d.text((20,12),'同一布局，两种目标；交换位置后仍分别指定',font=font,fill='#173b55')
 for row,l in enumerate(['normal','swap']):
  with np.load(S/f'preflight/task0_{l}.npz') as f:image=f['image'];boxes=f['boxes']
  for target in [0,1]:
   x=20+target*490;y=65+row*485;d.text((x,y),('原位置' if row==0 else '交换位置')+(' · 汤罐' if target==0 else ' · 番茄酱'),font=font,fill='#173b55');im.paste(Image.fromarray(mark(image,boxes[target])).resize((440,440)),(x,y+38))
 d.text((20,1020),'图示目标框来自仿真分割，是接口上限测试，不是模型预测框。',font=small,fill='#173b55');im.save(F/'goal_interface.png')
 plt.rcParams['font.family']='Noto Sans CJK JP';plt.rcParams['axes.unicode_minus']=False
 history=[json.loads(x) for x in (S/'training/validation.jsonl').read_text().splitlines()]
 fig,ax=plt.subplots(figsize=(8,3),constrained_layout=True)
 for k,label in [('text_loss','文字模式'),('grounded_loss','目标图模式')]:ax.plot([r['step'] for r in history],[r[k] for r in history],marker='o',label=label)
 ax.set_xlabel('更新步数');ax.set_ylabel('固定验证样本流匹配损失');ax.legend();ax.grid(alpha=.2);fig.savefig(F/'validation.png',dpi=170);plt.close(fig)
 fig,ax=plt.subplots(figsize=(8,3.4),constrained_layout=True);x=np.arange(3)
 for j,(k,label) in enumerate([('first_lift_correct','首次选对'),('success','完整单件搬运')]):
  yy=[summary['groups'][v][k] for v in ['base','text','goal']];ax.bar(x+(j-.5)*.3,yy,width=.3,label=label)
  for xx,y in zip(x+(j-.5)*.3,yy):ax.text(xx,y+.1,str(y),ha='center')
 ax.set_xticks(x,['原权重','适配·文字','适配·目标图']);ax.set_ylim(0,9);ax.set_ylabel('回合数 / 8');ax.legend();ax.grid(axis='y',alpha=.2);fig.savefig(F/'results.png',dpi=170);plt.close(fig)
 native=Image.new('RGB',(1170,480),'#edf2f7');d=ImageDraw.Draw(native)
 for j,(t,label) in enumerate([(0,'两罐入篮'),(1,'两盒入篮'),(9,'杯子入微波炉、关门')]):
  clip=frames(I/f'runs/fresh_original/task{t:02d}_init32/rollout.mp4');frame=clip[min(140,len(clip)-1)];x=j*390+10;d.text((x,10),label,font=font,fill='#173b55');native.paste(Image.fromarray(frame).resize((370,370)),(x,55));d.text((x,437),'本次原任务回合 · 完成',font=small,fill='#173b55')
 native.save(F/'official_instances.png')

def video(summary):
 pick=json.loads((S/'VIDEO_SELECTION.json').read_text());rows=[next(r for r in summary['episodes'] if r['variant']==v and all(r[k]==pick[k] for k in ['task','layout','target','initial'])) for v in pick['variants']];clips=[frames((ROOT/r['source']).parent/'rollout.mp4') for r in rows];assert all(len(c)==326 for c in clips)
 out=S/'文字与显式目标图_黄油对照.mp4';w,h=1180,760;proc=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}','-r','20','-i','-','-an','-vcodec','libx264','-crf','20','-pix_fmt','yuv420p',str(out)],stdin=subprocess.PIPE,stderr=subprocess.DEVNULL)
 font=ImageFont.truetype(FONT,23);small=ImageFont.truetype(FONT,19)
 for i in range(326):
  c=Image.new('RGB',(w,h),'#edf2f7');d=ImageDraw.Draw(c);d.text((20,12),'指定黄油 · 同一适配权重、相同初态41与采样噪声',font=font,fill='#173b55')
  for j,label in enumerate(['文字指定','准确目标框参考图指定']):
   x=20+j*580;d.text((x,57),label,font=font,fill='#173b55');c.paste(Image.fromarray(clips[j][i]).resize((550,550)),(x,97));d.text((x,662),'完整单件搬运：'+('成功' if rows[j]['success'] else '未完成'),font=small,fill='#173b55')
  d.text((20,711),f'第{i}步 / 325 · 按20Hz仿真时间播放，不包含推理等待。',font=small,fill='#173b55');proc.stdin.write(c.tobytes())
 proc.stdin.close();assert proc.wait()==0
 reader=imageio_ffmpeg.read_frames(str(out));next(reader);assert sum(1 for _ in reader)==326
 return out

def main():
 official=json.loads((I/'SUMMARY.json').read_text());assert json.loads((I/'CHECK.json').read_text())['actions_match_model'] and official['successes']==6
 a=json.loads((S/'SUMMARY.json').read_text());physics=json.loads((S/'PHYSICS_AUDIT.json').read_text());dataaudit=json.loads((S/'DATA_PHYSICS_AUDIT.json').read_text());assert physics['final_disagreements']==dataaudit['final_disagreements']==0
 a['audit'].update(physical_rescore=True,physical_audit_sha256=hashlib.sha256((S/'PHYSICS_AUDIT.json').read_bytes()).hexdigest());(S/'SUMMARY.json').write_text(json.dumps(a,ensure_ascii=False,indent=2))
 g=a['groups'];best=json.loads((S/'training/BEST.json').read_text());pairs=json.loads((S/'PAIRS.json').read_text());figures(a);movie=video(a);D.mkdir(parents=True,exist_ok=True)
 native_table=table(['原任务','初态','结果','控制步数'],[[{0:'汤罐与番茄酱入篮',1:'奶酪与黄油入篮',9:'杯子入微波炉并关门'}[r['task_id']],r['initial_state_id'],'成功',r['steps']] for r in official['rows']])
 result_table=table(['策略/输入','首次选对','合法完成窗口','完整单件搬运'],[[label,*[f"{g[v][k]}/8" for k in ['first_lift_correct','valid_window','success']]] for v,label in [('base','原权重·文字'),('text','适配权重·文字'),('goal','适配权重·目标图')]])
 layout_table=table(['输入模式','原位置：选对/成功','交换位置：选对/成功'],[[v,*[f"{g[v]['by_layout'][l]['first_lift_correct']}/4 · {g[v]['by_layout'][l]['success']}/4" for l in ['normal','swap']]] for v in ['base','text','goal']])
 improved=g['goal']['first_lift_correct']>g['text']['first_lift_correct'];conclusion=('显式目标图在本轮8个条件中的首次选择优于同权重文字模式，值得在下一批布局验证；样本仍小，不作统计显著或可靠部署的结论。' if improved else ('增加数据与联合适配后，至少一种输入模式的首次选择优于原权重，但目标图没有显示出超越同权重文字模式的额外收益。不能将整体收益单独归于红框。' if max(g['text']['first_lift_correct'],g['goal']['first_lift_correct'])>g['base']['first_lift_correct'] else '目标图模式在本次保留初态上退化，首次选对仅1/8、完整成功0/8，低于同权重文字模式。该适配尚不可用，本轮不追加训练或扩初态寻找正结果；这也不能据此否定原论文方法。官方原任务实例仍单独跑通。'))
 names={0:['汤罐','番茄酱'],1:['奶酪','黄油']};detail_rows=[]
 for t in [0,1]:
  for layout in ['normal','swap']:
   for target in [0,1]:
    rr=[next(r for r in a['episodes'] if (r['variant'],r['task'],r['layout'],r['target'])==(v,t,layout,target)) for v in ['base','text','goal']]
    detail_rows.append([names[t][target]+' / '+('原位' if layout=='normal' else '交换'),*[names[t][r['first_lift']] if isinstance(r['first_lift'],int) else '未明确抬起' for r in rr],'/'.join('成功' if r['success'] else '失败' for r in rr)])
 details=table(['指定目标/布局','原权重先拿','文字先拿','目标图先拿','完整成功：原/文/图'],detail_rows)
 probes=float(np.median([r['action_rms'] for r in a['box_only_probes']]))
 text=f'''# 自主任务实例与显式目标接口

阿里具身项目 · 本次阶段报告  
2026年11月14日 · RTX4090 / π0.5 / LIBERO

## 摘要

本阶段将“先运行一个现成VLA实例”与“扩展指定目标能力”分开。第一条线使用官方π0.5 LIBERO公开权重及原任务指令，重新执行6个完整回合，6/6成功，包含两物体入篮和杯子放入微波炉后关门。输入图像、预测动作块与实际施加动作均保存，并提供本地观看页面与单任务启动命令。这些是仿真结果，不是SO101真机演示，也不是本项目训练出的模型成绩。

第二条线借鉴Point-VLA的固定目标参考图与文字/视觉指定共训练，加入同布局双目标和重置时交换位置的数据设计。成对验收后44条训练轨迹、14条验证轨迹，1000步LoRA完成。新初态41的8个条件分别比较三种模式，首次选对为原权重{g['base']['first_lift_correct']}/8、适配文字{g['text']['first_lift_correct']}/8、适配目标图{g['goal']['first_lift_correct']}/8；完整单件搬运分别为{g['base']['success']}/8、{g['text']['success']}/8、{g['goal']['success']}/8。{conclusion}

## 目录

1. 官方开源方案的完整运行实例
2. 显式目标接口与数据设计
3. 有界联合适配与保留初态测试
4. 核对、边界与后续安排
5. 文件目录和启动方式

## 1. 官方开源方案的完整运行实例

使用Physical Intelligence公开的openpi框架、pi05_libero权重及配套LIBERO环境。不是只运行模型导入或离线预测，而是“新相机图像和原任务指令→动作块→Panda的OSC控制器→新图像”的连续闭环。每次预测10步、执行前5步，再取新图。没有Qwen、脚本教师动作、示教轨迹回放或物体瞬移替代策略。官方成功信号只用于结束和判分。

{native_table}

![本次三类自主任务](figures/official_instances.png)

这6个新回合独立保存，不混入此前117/120成绩。原入篮任务要求两件都搬入，下一节的新增任务要求只搬指定一件、另一件不动；两组任务和初态不同，6/6与后面的2/8不能直接作性能对比。其作用是提供可以观看、重新启动和核查的完整实例，不能据此声称任意改写任务均能完成。模型仍使用官方权重，复现结果不作为自主训练成果。相关代码在`official_instance/scripts/run.py`和`start.sh`。

## 2. 显式目标接口与数据设计

参考Point-VLA的思路：保持当前agentview与腕部图像，额外提供一张标出目标的初始参考图。把原先空置的第三图像槽位用于该图，并显式打开mask；文字模式保持原mask关闭。视觉模式使用不含物体名称的固定指令，目标由红框指定；文字模式使用物体名称。参考图在整个回合固定，不在中途从真值更新位置。

本轮目标框来自仿真分割，是准确定位条件下的接口上限测试。它不等于实现了Qwen目标定位，也不等于完整复现官方Point-VLA代码或权重。当前图像和参考图均为224像素输入，基座与视觉编码器冻结，只适配语言和动作专家的LoRA。

![同场景不同目标及位置交换](figures/goal_interface.png)

为减少“背景/位置决定目标”的关联，加入同一布局的两种目标，以及初始化时交换两件物体XY位置。交换只在策略开始前发生，控制过程中不改物体位置。新训练初态36–39，验证40；旧训练20–25、验证26–27保留。候选中任一目标未通过稳定放置与另一件全程位移不超过30毫米的要求，整对不进入训练。最终22个训练布局对、7个验证布局对，分别44条和14条轨迹。

脚本教师使用物体真值生成动作，仅用于造数据，不能计入VLA成功率。测试初态41未参与本轮数据生成或选模，42仍保留。公开基座是否接触过相似训练布局不是本试验能够排除的，因此这里只称本轮微调的保留初态。

相关代码在`point_target/scripts/layout.py`、`prepare.py`、`goal_inputs.py`。Point-VLA方法来源：https://arxiv.org/abs/2512.18933 ；数据设计参考Shortcut Learning研究：https://proceedings.mlr.press/v305/xing25a.html 。

## 3. 有界联合适配与保留初态测试

1000步、batch2，从原基座和零B的LoRA开始，不沿用上一轮训练后的适配器。仅49,987,584个LoRA参数可训练，主参数为FP32。文字/目标图batch精确1:1；任务与布局均匀采样，每batch成对采两种目标，40%取共享观察的目标分叉点，其余按动作阶段采样。新增平移将参考图和框一起移动；官方训练原有的当前相机裁剪、旋转和颜色增强仍保留。

固定验证样本的平均流匹配损失从{best['initial']['mean_loss']:.6f}降至{best['mean_loss']:.6f}，选定step{best['step']}。没有依据测试结果换权重或追加训练。

![固定验证损失](figures/validation.png)

{result_table}

![闭环结果](figures/results.png)

{layout_table}

{details}

首次选对要求夹爪接触、闭合命令且物体升高20毫米持续3步；合法窗口表示过程中曾达到要求，不保证最终保持。完整成功要求末10步目标稳定在篮中并与夹爪分离，另一件全程位移不超过30毫米。每回合300策略步、25保持步，不使用真值提前停止。三种模式每个条件的初态和采样噪声一致。

适配文字与目标图使用同一权重，因此它们是输入方式对照，不能单独分离增加数据、位置交换、联合训练各自的贡献。只换初始参考图红框的同观察探针，动作差RMS中位数为{probes:.6f}，说明输入会影响预测；是否选对必须看上面的闭环结果，不能由动作数值变化替代。

{conclusion}

相关代码在`point_target/scripts/train.py`、`policy_runtime.py`、`evaluate.py`和`summarize.py`。

## 4. 核对、边界与后续安排

文字分支变换与原方案逐项相同，目标图只新增第三图像及mask。原始相机、状态和同红框对照的任务词元已核验。目标框通过分割与RGB方向预检并保存图示；缓存图像与即时渲染存在微小时间差，方向检查与目标可见性分别验证，没有用翻转错误的框训练。

40条新教师候选（含拒绝项）共{dataaudit['states']}状态复算，最终标签全部一致；24条策略测试共{physics['states']}状态复算，最终判分全部一致。实际1440次策略查询与8次额外换框探针保留输入、预测动作和实际施加动作。官方6回合另核对实际动作与模型动作块一致、末步官方成功谓词为真。

全部模型任务在独立用户服务中单模型运行，软14GiB、硬16GiB限额，沿用分块加载；CPU数据和复算为4/6GiB限额。没有修改系统Python、Swap、ROS2或连接实体设备。

后续若显式接口有稳定收益，再接视觉预测框，并分别报告准确框与预测框表现；若无收益，先定位图像条件与动作对齐，不继续凭验证损失下降宣称技能提升。SO101真机任务需要对应机器人和相机的数据适配，本阶段不将Panda仿真权重直接用于真机。

## 5. 文件目录和启动方式

```text
official_instance/
  index.html、start.sh、python_model.sh
  scripts/run.py、build_page.py
  runs/fresh_original/             6条原任务输入、动作与录像
point_target/
  scripts/layout.py、prepare.py、goal_inputs.py
  scripts/train.py、policy_runtime.py、evaluate.py
  training/                       增量权重、训练和验证记录
  evaluation/base|text|goal/       24条策略测试
  SUMMARY.json、PHYSICS_AUDIT.json
```

本地实例页面：http://127.0.0.1:8793 。单任务重新运行（使用新tag）：

```bash
bash experiments/vla_libero/official_instance/start.sh --tag my_demo --tasks 9 --initials 32
```

交付中的“自主任务演示/index.html”可直接打开观看本次6条录像。源码ZIP不含基座、完整环境、训练轨迹及全部帧，完整GPU运行需要README列出的公开资产；增量适配器单独提供，不能独立作为完整模型运行。CPU图像测试使用合成输入，不依赖隐藏案例文件。相关代码目录保留原层级以便检查。
'''
 (S/'REPORT.md').write_text(text);css='@page{size:A4;margin:17mm}body{font:10pt/1.65 "Noto Sans CJK SC","Noto Sans CJK JP",sans-serif;color:#24374a}h1{font-size:23pt;color:#153b58}h2{font-size:15pt;break-after:avoid;border-bottom:1px solid #ccd8e3}table{width:100%;border-collapse:collapse;font-size:8pt;break-inside:avoid}td,th{padding:5px;border:1px solid #ccd8e3}th{background:#edf3f8}img{width:100%}pre{font-size:8pt;white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f5f8;padding:10px}code{overflow-wrap:anywhere}p:has(img){break-inside:avoid}'
 (S/'REPORT.html').write_text('<!doctype html><meta charset="utf-8"><style>'+css+'</style>'+MarkdownIt().enable('table').render(text))
 with tempfile.TemporaryDirectory() as profile:subprocess.run(['google-chrome','--headless','--no-sandbox','--disable-gpu',f'--user-data-dir={profile}','--no-pdf-header-footer',f'--print-to-pdf={S/"REPORT.pdf"}',(S/'REPORT.html').as_uri()],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=90)
 shutil.copyfile(S/'REPORT.pdf',D/'自主实例与显式目标接口_阶段报告.pdf');shutil.copyfile(movie,D/movie.name)
 demo=D/'自主任务演示';demo.mkdir(exist_ok=True);shutil.copyfile(I/'index.html',demo/'index.html')
 for r in official['rows']:
  dest=demo/r['folder'];dest.mkdir(parents=True,exist_ok=True)
  for name in ['rollout.mp4','result.json','queries.json']:shutil.copyfile(I/r['folder']/name,dest/name)
 weights=D/'策略适配器';weights.mkdir(exist_ok=True)
 for name in ['adapter_best.npz','BEST.json','STATUS.json','PROTOCOL.json']:shutil.copyfile(S/'training'/name,weights/name)
 (weights/'README.md').write_text('仅LoRA增量，基座为官方gs://openpi-assets/checkpoints/pi05_libero。放到experiments/vla_libero/point_target/training/adapter_best.npz后，按README准备公开基座和LIBERO资产。不是可独立推理的完整模型。')
 files=set()
 for stage in [I,S]:
  files.update(stage.glob('*.md'));files.update(stage.glob('*.json'));files.update(stage.glob('*.sh'));files.update(stage.glob('*.html'));files.update((stage/'scripts').glob('*.py'));files.update((stage/'tests').glob('*.py'));files.update((stage/'tests').glob('*.txt'))
 files.discard(S/'REPORT.html')
 for folder in [S/'training']:
  files.update(folder.glob('*.json'));files.update(folder.glob('*.jsonl'))
 files.update((I/'runs').glob('*/*.json'));files.update((I/'runs').glob('*/*/*.json'))
 files.update((S/'evaluation').glob('*/*.json'));files.update((S/'evaluation').glob('*/*/episodes/*/*/result.json'))
 dependencies=['scripts/evaluate.py','scripts/episode_journal.py','scripts/run_hierarchical.py','scripts/python_shared','scripts/fetch_sources.py','scripts/install.py','scripts/install_simulation.py','scripts/download_segments.py','scripts/environment.py','scripts/pin_local_libero.py','selective_execution/scripts/protocol.py','instruction_response/scripts/run.py','instruction_response/scripts/analyze.py','skill_bootstrap/scripts/runtime.py','skill_bootstrap/scripts/bounded_restore.py','skill_bootstrap/scripts/chunk_math.py','skill_bootstrap/scripts/teacher.py','skill_bootstrap/scripts/train_lora.py','skill_bootstrap/scripts/watch_resources.py','skill_bootstrap/scripts/python_model.sh','third_party/sources.json','third_party/openpi/LICENSE','third_party/libero/LICENSE','requirements.lock.txt','environment.json','checkpoints/pi05_libero/download_verified.json','checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json']
 files.update(V/p for p in dependencies);files.update([V/'ENVIRONMENT_NOTES.md',V/'scripts/verify_checkpoint.py',V/'configs/checkpoint_objects.json'])
 root_readme='本包为本次自主实例和显式目标接口源码。先阅读experiments/vla_libero/official_instance/README.md与point_target/README.md。CPU测试只需numpy/Pillow；GPU运行需固定版本openpi、LIBERO与公开pi05_libero基座。新训练原始轨迹不在包内，不能据此承诺从零复现适配训练；适配权重单独提供。原任务可用start.sh新tag运行。'
 manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)};manifest['README.md']=hashlib.sha256(root_readme.encode()).hexdigest()
 with zipfile.ZipFile(D/'自主实例与显式目标接口_源码.zip','w',zipfile.ZIP_DEFLATED) as z:
  for p in sorted(files):z.write(p,str(p.relative_to(ROOT)))
  z.writestr('PACKAGE_SHA256.json',json.dumps(manifest,indent=2));z.writestr('README.md',root_readme)
 (D/'文件清单.json').write_text(json.dumps({str(p.relative_to(D)):hashlib.sha256(p.read_bytes()).hexdigest() for p in D.rglob('*') if p.is_file() and p.name!='文件清单.json'},ensure_ascii=False,indent=2));print(D)
if __name__=='__main__':main()
