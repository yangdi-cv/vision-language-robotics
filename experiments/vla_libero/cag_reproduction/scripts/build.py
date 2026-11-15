"""Summarize the bounded reproduction without altering previous deliveries."""
import hashlib,json,shutil,subprocess,tempfile,zipfile
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from markdown_it import MarkdownIt
import imageio_ffmpeg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
S=Path(__file__).resolve().parents[1];V=S.parent;ROOT=V.parents[1]
D=ROOT/'deliverables/20261114_阿里_CAG免训练复现'
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
def table(head,rows):return '\n'.join(['| '+' | '.join(head)+' |','|'+'|'.join(['---']*len(head))+'|',*['| '+' | '.join(map(str,r))+' |' for r in rows]])
def video(summary):
 selection=json.loads((S/'VIDEO_SELECTION.json').read_text());rows=[next(r for r in summary['episodes'] if all(r[k]==selection[k] for k in ['task','initial','target']) and r['variant']==v) for v in selection['variants']]
 clips=[]
 for r in rows:
  reader=imageio_ffmpeg.read_frames(str((ROOT/r['source']).parent/'rollout.mp4'));m=next(reader);frames=[np.frombuffer(f,np.uint8).reshape(m['size'][1],m['size'][0],3).copy() for f in reader];assert len(frames)==326;clips.append(frames)
 out=S/'基座与CAG_指定黄油对照.mp4';w,h=1180,760
 p=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}','-r','20','-i','-','-an','-vcodec','libx264','-crf','20','-pix_fmt','yuv420p',str(out)],stdin=subprocess.PIPE,stderr=subprocess.DEVNULL)
 font=ImageFont.truetype(FONT,23);small=ImageFont.truetype(FONT,19)
 for i in range(326):
  c=Image.new('RGB',(w,h),'#edf2f6');d=ImageDraw.Draw(c);d.text((20,12),'指定黄油，奶酪保持原位 · 相同开发初态0',font=font,fill='#183b55')
  for j,label in enumerate(['原π0.5 · 尺度1','免训练CAG · 尺度1.5']):
   x=20+j*580;d.text((x,57),label,font=font,fill='#183b55');c.paste(Image.fromarray(clips[j][i]).resize((550,550)),(x,97));d.text((x,662),'完整搬运：'+('成功' if rows[j]['success'] else '未完成'),font=small,fill='#183b55')
  d.text((20,711),f'第{i}步 / 325 · 按仿真20Hz播放，不包含推理等待。',font=small,fill='#183b55');p.stdin.write(c.tobytes())
  if i==120:c.save(S/'figures/video_frame.png')
 p.stdin.close();assert p.wait()==0
 reader=imageio_ffmpeg.read_frames(str(out));next(reader);assert sum(1 for _ in reader)==326
 return out

def main():
 a=json.loads((S/'SUMMARY.json').read_text());physics=json.loads((S/'PHYSICS_AUDIT.json').read_text());assert physics['final_disagreements']==0
 a['audit'].update(physical_state_rescore=True,physical_audit_sha256=hashlib.sha256((S/'PHYSICS_AUDIT.json').read_bytes()).hexdigest(),physical_states=physics['states'],final_score_disagreements=physics['final_disagreements']);(S/'SUMMARY.json').write_text(json.dumps(a,ensure_ascii=False,indent=2))
 g=a['groups'];c=a['control'];D.mkdir(parents=True,exist_ok=True);(S/'figures').mkdir(exist_ok=True)
 plt.rcParams['font.family']='Noto Sans CJK JP';plt.rcParams['axes.unicode_minus']=False
 fig,ax=plt.subplots(figsize=(8,3.5),constrained_layout=True);x=np.arange(3)
 for i,(v,label) in enumerate([('base','原策略'),('cag15','CAG 1.5')]):
  y=[g[v][k] for k in ['target_contact','first_lift_correct','success']];ax.bar(x+(i-.5)*.32,y,width=.32,label=label)
  for xx,yy in zip(x+(i-.5)*.32,y):ax.text(xx,yy+.1,str(yy),ha='center')
 ax.set_xticks(x,['曾接触指定物体','首次抬起选对','完整单件搬运']);ax.set_ylim(0,9);ax.set_ylabel('回合数 / 8');ax.legend();ax.grid(axis='y',alpha=.2);fig.savefig(S/'figures/results.png',dpi=170);plt.close(fig)
 movie=video(a)
 outcomes=table(['方法','接触指定物体','首次抬起选对','合法完成窗口','完整成功'],[[label,*[f"{g[v][k]}/8" for k in ['target_contact','first_lift_correct','valid_window','success']]] for v,label in [('base','原π0.5'),('cag15','CAG 1.5')]])
 details=table(['场景 / 初态 / 指定目标','原策略先抬起','CAG先抬起','原策略完整成功','CAG完整成功'],[[f"{r['task']} / {r['initial']} / {r['target']}",r['base_first'],r['cag_first'],'是' if r['base_success'] else '否','是' if r['cag_success'] else '否'] for r in a['pairs']])
 gains=sum(r['cag_success'] and not r['base_success'] for r in a['pairs']);losses=sum(r['base_success'] and not r['cag_success'] for r in a['pairs'])
 selection_delta=g['cag15']['first_lift_correct']-g['base']['first_lift_correct'];success_delta=g['cag15']['success']-g['base']['success']
 conclusion=('本轮首次选择有所改善，但只有8对开发条件，尚未验证独立布局泛化。' if selection_delta>0 else ('目标选择没有改善；唯一新增完整成功来自场景0、初态0、指定原优先汤罐的条件，两方法都先选汤罐。收益表现为该条件下避免后续搬动另一件物体，不能解释为听懂了反向目标。本轮到此收尾，不扩大尺度搜索。' if success_delta>0 else '本轮没有显示出目标选择或完整成功的改善，不继续扩大尺度搜索，也不据此追加训练。'))
 text=f'''# CAG免训练引导：指定目标执行复现

阿里具身项目 · 阶段补充报告  
2026年11月14日 · RTX4090 / LIBERO / π0.5

## 摘要

针对“换了目标指令，策略仍搬熟悉物体”的问题，本阶段参照公开论文的Counterfactual Action Guidance，完成免训练形式的本地复现。沿用原π0.5 LIBERO权重，一个模型顺序计算有任务与空任务两路动作，不训练额外模型、不使用Qwen或教师坐标控制。

16个开发回合中，原策略首次选对{g['base']['first_lift_correct']}/8，CAG为{g['cag15']['first_lift_correct']}/8；完整单件搬运分别为{g['base']['success']}/8、{g['cag15']['success']}/8。配对完整成功新增{gains}例、损失{losses}例。{conclusion}

## 目录

1. 参考方法与复现边界
2. 输入及公式控制检查
3. 配对闭环结果
4. 资源、文件与后续判断

## 1. 参考方法与复现边界

参考论文《When Vision Overrides Language: Evaluating and Mitigating Counterfactual Failures in VLAs》（2026预印本）：https://arxiv.org/html/2602.17659v1 。采用式4的training-free变体，增强同一观察下由语言引起的动作差异：

`a = a_cond + (w - 1) × (a_cond - a_null)`

引导尺度固定为原策略1与论文使用的1.5，不搜索更多尺度。两个分支共享同一权重、图像和显式采样噪声；空任务在原输入变换前设置。当前在7维反归一化动作上混合，包括夹爪；不是在流匹配内部速度场上引导。模型动作反归一化是仿射变换，组合系数之和为1，因此数学上与其前的线性组合相容。保留原环境控制器的范围处理，未新加动作放大或裁剪。

作者项目链接本次无法访问，未核实官方代码。此处是按论文公式与明确本地约定实现，不声称与作者采样细节逐项一致。也没有复现另一个需要训练独立视觉动作先验的版本。

场景0的目标0/1分别为汤罐/番茄酱，场景1分别为奶酪/黄油；各初态0、11，分别指定两件物体，共8个条件，每条件比较两方法。初态已参与此前诊断，是开发对照，不是独立测试。每回合300策略步、25保持步；每次预测10步执行前5步，原Long成功只记录，不用于提前停止。

## 2. 输入及公式控制检查

5项CPU测试覆盖尺度1/0、公式、相同分支与非法参数。真实模型检查：尺度1相对原策略最大动作差{c['scale1_max_abs']}，同噪声重复差{c['repeat_max_abs']}，有/空指令动作RMS差{c['language_delta_rms']:.6f}。

预检曾发现具体配置与通用实现的区别：虽然π0.5通用tokenizer支持将状态编码为文本，本地固定pi05_libero显式关闭discrete_state_input。最初预设空任务仍有State文本的检查因此报错，尚未执行模型推理和回合；修正后按真实配置核验。实际空任务解码为换行；原图像和state字典字段均保留。没有修改模型配置来绕过检查。

全部960次策略查询核对保存输入、种子噪声、组合公式与实际执行动作，公式最大差{a['audit']['max_formula_error']}。两方法每条件的初始与稳定后状态哈希一致。另恢复{physics['states']}个物理状态复算，16条最终判分均一致。

## 3. 配对闭环结果

{outcomes}

![回合结果](figures/results.png)

“接触指定物体”只表示该回合中夹爪曾与目标接触，不等于成功选取。首次抬起要求目标升高超过20毫米、夹爪接触且闭合命令持续3步。完整成功要求最后10步目标在篮中且与夹爪分离，另一件全程位移不超过30毫米。三个指标不能互换，也不能把本地完整约束成绩当成原LIBERO排行榜成绩。

{details}

![固定条件回放截图](figures/video_frame.png)

视频条件在看到该条件结果前固定为场景1、初态0、指定黄油。保留完整326帧，按20Hz仿真时间播放，不含GPU推理等待，没有依据成功筛选片段。

## 4. 资源、文件与后续判断

原策略每次一次模型调用，CAG每次两次；实际中位查询时间分别{g['base']['seconds_per_query_median']:.3f}秒、{g['cag15']['seconds_per_query_median']:.3f}秒（不含仿真推进；包含本地输入检查与记录）。共1440次闭环分支推理，另4次模型预检。模型通过原分块加载，独立systemd单元MemoryHigh14GiB、MemoryMax16GiB，没有双模型驻留，没有改Swap或关闭内存保护。模型进程采样观测RSS峰约5.78GiB、进程组约5.21GiB。独立物理复算最初软上限2GiB过低导致回收节流，调整为软4GiB、硬6GiB后继续同一次复算；没有OOM或重跑策略回合。

{conclusion} 若后续验证视觉目标接口，需要区分正确目标框的上限实验与真实视觉预测框，不能默认未训练的π0.5能读懂新增标记。所有结果保留，不覆盖上一阶段交付。

```text
cag_reproduction/
  scripts/mixing.py、cag_policy.py  双分支与公式
  scripts/run.py                   模型控制检查、16开发回合
  scripts/summarize.py              输入与动作核验
  tests/test_mixing.py              5项CPU测试
  PROTOCOL.json、CONTROL_CHECK.json
  SUMMARY.json、PHYSICS_AUDIT.json
  development/base/、cag15/         原始回合、动作、输入和视频
```

相关代码在上述scripts目录。源码包包含本轮源码、统计和调用依赖源码；不包含基座、全部轨迹或图像，完整GPU复现仍需原环境与LIBERO资产。源码解压后的CPU测试只用合成数组。权重未改变，因此本轮没有新增适配器文件。
'''
 (S/'REPORT.md').write_text(text)
 css='@page{size:A4;margin:17mm}body{font:10pt/1.65 "Noto Sans CJK SC","Noto Sans CJK JP",sans-serif;color:#24374a}h1{font-size:23pt;color:#153b58}h2{font-size:15pt;break-after:avoid;border-bottom:1px solid #ccd8e3}table{width:100%;border-collapse:collapse;font-size:8pt;break-inside:avoid}td,th{padding:5px;border:1px solid #ccd8e3}th{background:#edf3f8}img{width:100%}pre{font-size:8pt;white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f5f8;padding:10px}code{overflow-wrap:anywhere}p:has(img){break-inside:avoid}'
 (S/'REPORT.html').write_text('<!doctype html><meta charset="utf-8"><style>'+css+'</style>'+MarkdownIt().enable('table').render(text))
 with tempfile.TemporaryDirectory() as profile:subprocess.run(['google-chrome','--headless','--no-sandbox','--disable-gpu',f'--user-data-dir={profile}','--no-pdf-header-footer',f'--print-to-pdf={S/"REPORT.pdf"}',(S/'REPORT.html').as_uri()],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=90)
 shutil.copyfile(S/'REPORT.pdf',D/'CAG免训练复现_阶段报告.pdf');shutil.copyfile(movie,D/movie.name)
 old=ROOT/'deliverables/20261114_阿里_目标条件化执行与策略适配/目标条件化执行_源码.zip'
 with zipfile.ZipFile(old) as z:files={n:z.read(n) for n in z.namelist() if n!='PACKAGE_SHA256.json'}
 candidates=[*S.glob('*.md'),*S.glob('*.json'),*(S/'scripts').glob('*.py'),*(S/'tests').glob('*.py'),*(S/'figures').glob('*.png'),*(S/'development').glob('*/episodes/*/*/result.json'),V/'research/TARGET_GROUNDING_REVIEW_20261114.md']
 for p in candidates:files[str(p.relative_to(ROOT))]=p.read_bytes()
 files['PACKAGE_README.md']='本包增加cag_reproduction的免训练复现代码。含上一阶段依赖源码与结果，旧结果不是本轮对照。完整模型/资产不随包提供；本轮没有训练权重。运行前阅读cag_reproduction/README.md。'.encode()
 manifest={n:hashlib.sha256(b).hexdigest() for n,b in files.items()}
 with zipfile.ZipFile(D/'CAG免训练复现_源码.zip','w',zipfile.ZIP_DEFLATED) as z:
  for n,b in files.items():z.writestr(n,b)
  z.writestr('PACKAGE_SHA256.json',json.dumps(manifest,indent=2))
 (D/'文件清单.json').write_text(json.dumps({str(p.relative_to(D)):hashlib.sha256(p.read_bytes()).hexdigest() for p in D.iterdir() if p.is_file() and p.name!='文件清单.json'},ensure_ascii=False,indent=2))
 print(D)
if __name__=='__main__':main()
