"""Chinese stage report, matched actual rollout, and source-only delivery."""
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
D=ROOT/'deliverables/20261114_阿里_目标条件化执行与策略适配'
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'

def table(head,rows):return '\n'.join(['| '+' | '.join(head)+' |','|'+'|'.join(['---']*len(head))+'|',*['| '+' | '.join(map(str,r))+' |' for r in rows]])

def figures(v1,v2):
    (S/'figures').mkdir(exist_ok=True)
    plt.rcParams['font.family']='Noto Sans CJK JP';plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),constrained_layout=True)
    for ax,summary,title,variant in [(axes[0],v1,'首轮 · 初态28/29','lora'),(axes[1],v2,'配对适配 · 初态30/31','paired')]:
        groups=summary['groups'];x=np.arange(2)
        for j,(field,label) in enumerate([('first_lift_correct','先抬起指定物体'),('success','完整单件搬运')]):
            values=[groups[m][field] for m in ['base',variant]];ax.bar(x+(j-.5)*.3,values,width=.3,label=label)
            for xx,y in zip(x+(j-.5)*.3,values):ax.text(xx,y+.1,str(y),ha='center')
        ax.set_xticks(x,['基座','适配']);ax.set_ylim(0,9);ax.set_title(title);ax.set_ylabel('成功回合数 / 8');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    fig.savefig(S/'figures/results.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.7),constrained_layout=True)
    a=[json.loads(x) for x in (S/'lora_pilot/validation.jsonl').read_text().splitlines()];axes[0].plot([r['step'] for r in a],[r['loss'] for r in a],marker='o');axes[0].set_title('首轮：固定16帧验证');axes[0].set_ylabel('流匹配损失')
    a=[json.loads(x) for x in (S/'lora_paired/validation.jsonl').read_text().splitlines()]
    for key,label in [('phase_loss','阶段样本'),('branch_loss','目标分叉'),('combined','选模指标')]:axes[1].plot([r['step'] for r in a],[r[key] for r in a],marker='o',label=label)
    axes[1].set_title('第二轮：阶段 + 分叉验证');axes[1].legend(fontsize=8)
    for ax in axes:ax.set_xlabel('本轮更新步数');ax.grid(alpha=.2)
    fig.savefig(S/'figures/loss.png',dpi=170);plt.close(fig)
    manifest=json.loads((S/'MANIFEST.json').read_text());rows=[r for r in manifest if r['task']==0 and r['initial']==23];rows.sort(key=lambda r:r['target'])
    canvas=Image.new('RGB',(1050,650),'#eef3f7');draw=ImageDraw.Draw(canvas);fonts={k:ImageFont.truetype(FONT,k) for k in [18,22,27]}
    draw.text((20,12),'同一观察状态，目标不同，示教动作不同',font=fonts[27],fill='#17354c')
    previous=None
    for j,r in enumerate(rows):
        with np.load(ROOT/r['trajectory']) as f:image=f['image'][9];action=f['action'][9]
        if previous is not None:assert np.array_equal(previous,image)
        previous=image.copy();ox=20+j*520
        draw.text((ox,65),'只搬汤罐头' if j==0 else '只搬番茄酱',font=fonts[22],fill='#17354c');canvas.paste(Image.fromarray(image).resize((440,440)),(ox,105))
        draw.text((ox,557),'动作 Δx/Δy/Δz：'+', '.join(f'{x:+.3f}' for x in action[:3]),font=fonts[18],fill='#17354c')
    draw.text((20,607),'训练初态23，第9帧；示教动作是OSC控制量，不是米或角度。',font=fonts[18],fill='#17354c');canvas.save(S/'figures/paired_observation.png')

def video(summary):
    # Fixed before results: task0/init30, request target1. Never select by success.
    rows=[next(r for r in summary['episodes'] if r['variant']==v and r['task']==0 and r['initial']==30 and r['target']==1) for v in ['base','paired']]
    clips=[]
    for r in rows:
        reader=imageio_ffmpeg.read_frames(str((ROOT/r['source']).parent/'rollout.mp4'));meta=next(reader);frames=[np.frombuffer(f,np.uint8).reshape(meta['size'][1],meta['size'][0],3).copy() for f in reader];assert len(frames)==326;clips.append(frames)
    out=S/'基座与目标配对适配_实际回合.mp4';w,h=1180,760
    proc=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}','-r','20','-i','-','-an','-vcodec','libx264','-crf','20','-pix_fmt','yuv420p',str(out)],stdin=subprocess.PIPE,stderr=subprocess.DEVNULL)
    font=ImageFont.truetype(FONT,23);small=ImageFont.truetype(FONT,19)
    for i in range(326):
        canvas=Image.new('RGB',(w,h),'#eef3f7');draw=ImageDraw.Draw(canvas);draw.text((20,12),'只搬番茄酱，汤罐头保持原位 · 相同初态30',font=font,fill='#17354c')
        for j,label in enumerate(['π0.5 基座','π0.5 配对适配']):
            ox=20+j*580;draw.text((ox,57),label,font=font,fill='#17354c');canvas.paste(Image.fromarray(clips[j][i]).resize((550,550)),(ox,97));draw.text((ox,662),'完整单件搬运：'+('成功' if rows[j]['success'] else '未完成'),font=small,fill='#17354c')
        draw.text((20,711),f'第{i}步 / 325 · 20Hz仿真时间；不含推理等待，不改动实际动作。',font=small,fill='#17354c');proc.stdin.write(canvas.tobytes())
        if i==120:canvas.save(S/'figures/video_frame.png')
    proc.stdin.close();assert proc.wait()==0
    reader=imageio_ffmpeg.read_frames(str(out));next(reader);assert sum(1 for _ in reader)==326
    (S/'VIDEO.json').write_text(json.dumps(dict(task=0,initial=30,target=1,frames=326,fps=20,decoded=True,selection='fixed task0/init30/target1, independent of result'),indent=2))
    return out

def main():
    D.mkdir(parents=True,exist_ok=True)
    a=json.loads((V/'instruction_response/SUMMARY.json').read_text());b=json.loads((S/'SUMMARY.json').read_text());c=json.loads((S/'SUMMARY_PAIRED.json').read_text());physics=json.loads((S/'POLICY_PHYSICS_AUDIT.json').read_text());p2=json.loads((S/'POLICY_PHYSICS_AUDIT_PAIRED.json').read_text())
    assert b['audit']['episodes']==c['audit']['episodes']==16
    figures(b,c);movie=video(c)
    g1=b['groups'];g2=c['groups'];best1=json.loads((S/'lora_pilot/BEST.json').read_text());best2=json.loads((S/'lora_paired/BEST.json').read_text());zero=json.loads((S/'lora_pilot/ZERO_ADAPTER_CHECK.json').read_text())
    result_table=table(['阶段 / 测试布局','策略','先抬起指定物体','有合法完成窗口','完整单件搬运'],[[stage,label,f"{g['first_lift_correct']}/8",f"{g['valid_window']}/8",f"{g['success']}/8"] for stage,items in [('首轮 / 28–29',[('基座',g1['base']),('LoRA200步',g1['lora'])]),('配对适配 / 30–31',[('基座',g2['base']),('配对适配',g2['paired'])])] for label,g in items])
    detail=table(['任务与目标','第二轮基座：先选对 / 完整成功','第二轮适配：先选对 / 完整成功'],[[f'场景{t} / 目标{g}',f"{g2['base']['by_task_target'][f'task{t}_target{g}']['first_lift_correct']}/2 · {g2['base']['by_task_target'][f'task{t}_target{g}']['success']}/2",f"{g2['paired']['by_task_target'][f'task{t}_target{g}']['first_lift_correct']}/2 · {g2['paired']['by_task_target'][f'task{t}_target{g}']['success']}/2"] for t in [0,1] for g in [0,1]])
    text=f'''# 目标条件化执行与单物体策略适配

阿里具身项目 · 阶段报告  
2026年11月14日 · RTX4090 / LIBERO / π0.5

## 摘要

此前的选择性搬运实验要求“只搬指定一件，另一件保持原位”。追加文字约束和腕部视角均没有解决主要问题。本阶段先检查低层策略是否收到并响应目标指令，再补充同场景单件搬运数据，完成轻量策略适配与独立布局闭环验证。

12条配对诊断中，目标词元和动作数值均会变化，但所有回合仍先抬起原任务的第一件物体。随后使用原机器人动力学和OSC控制器生成32条候选示教，29条通过物理验收，其中21条训练、8条验证。第一轮200步LoRA在初态28–29的完整单件搬运为基座0/8、适配1/8，尚未改变第二件优先的行为。

进一步检查发现，决定目标选择的共享观察分叉点在普通阶段采样中覆盖不足。第二轮将半数训练batch改为“相同观察、不同目标、不同正确动作”的配对，从第一轮权重继续600步，并改用初态30–31测试。第二轮完整单件搬运为基座{g2['base']['success']}/8、配对适配{g2['paired']['success']}/8；首次抬起指定物体为{g2['base']['first_lift_correct']}/8、{g2['paired']['first_lift_correct']}/8。完整逐项结果在第5节。

两轮使用不同测试布局，不将第二轮与第一轮的差值解释为单一采样因素的收益。此处是两个场景、一个训练种子的小规模适配实验，不是原LIBERO整套基准成绩。示教生成使用仿真真值，策略测试没有使用物体坐标或真值停止。

## 目录

1. 指令响应诊断
2. 单物体示教与物理验收
3. 第一轮策略适配
4. 目标分叉配对适配
5. 新布局闭环结果
6. 内存故障与加载修复
7. 代码、权重与后续工作

## 1. 指令响应诊断

选择Long中的汤罐头/番茄酱与奶酪/黄油两个入篮场景，初态0与11。每个条件分别输入“只搬第一件”“只搬第二件”的正向单技能指令，以及原双物体任务指令，共12条回合。模型、观察、初态和动作噪声保持一致，每次输出10步、执行前5步。

沿第一件轨迹的第0、50、100、150、200、250步采样，24个相同观察点各调用三种指令及一次同指令重复，共96次额外推理。图像、本体状态与噪声哈希一致，实际目标词元不同；同条件重复动作差0。四个初始观察点的目标交换动作RMS约0.016–0.030（模型控制量），说明语言进入了推理。

但是四个场景条件的三种指令都先抬起第一件；要求第二件优先的4条均未做到。这里的“抬起”按物体高于初始位置20毫米、夹爪接触且关闭连续3步记录，是行为代理指标，与最终任务得分分开。

因此，本轮没有修改分词或输入接口，而是补充目标条件化技能。相关代码在`instruction_response/scripts/run.py`、`analyze.py`，逐项记录在该目录SUMMARY.json。

## 2. 单物体示教与物理验收

参考物体相对抓取姿态的思路，复用原robosuite OSC控制接口，在同一Long场景中执行清空、靠近、夹持、抬起、移至篮子、释放和退回。抓取姿态取自已成功的第一件轨迹，再根据目标物体位置生成示教。执行仍经过原关节控制、动力学和接触模型，不用mocap，也不直接改动物体位置。

这种生成器具有仿真真值，是训练数据来源，不是VLA算法。两个开发反向小试成功只用于确认数据生成的物理可行性。正式候选按初态预先划分：训练20–25，验证26–27；第一轮策略测试28–29，第二轮测试30–31。

| 数据部分 | 候选轨迹 | 接受 | 未接受 |
|---|---:|---:|---:|
| 训练 | 24 | 21 | 3 |
| 验证 | 8 | 8 | 0 |

接受标准为目标在篮内稳定放置并脱离夹爪、指定保留物体的全程最大位移不超过30毫米。3条夹持失败轨迹完整保留，不进入训练。训练输入只有图像、本体状态、指令和实际动作，不包含教师使用的物体坐标或阶段标签；阶段标签只用于采样。

独立恢复10595个示教仿真状态，32条最终接受标签全部一致。恢复后与运行缓存的物体位置最大差约2.587毫米。16个布局的两目标示教在初始图像和状态上相同。

相关代码在`skill_bootstrap/scripts/teacher.py`、`generate.py`、`audit_data.py`；划分和筛选结果在DATA_PROTOCOL.json、MANIFEST.json、DATA_AUDIT.json。

## 3. 第一轮策略适配

使用原π0.5检查点，增加原生LoRA，仅训练49,987,584个参数，约占3.40B模型参数的1.47%。全部基座和视觉编码器保持冻结，图像处理、分词、动作归一化和动作块格式沿用原实现。调用固定版本openpi的train_step，没有替换底层策略。

首轮200步、batch1，按任务/目标与示教阶段采样。验证固定为初态26–27的16帧，检查点按验证流匹配损失选择。训练前随机LoRA损失为0.02769，最低为{best1['validation_loss']:.5f}，选中step{best1['step']}。该“训练前”不是未加适配器的原基座：此版本官方LoRA的A、B均随机初始化。

在独立初态28–29，完整单件搬运从0/8变为1/8，先抬起指定物体从3/8变为4/8。要求第二件优先仍为0/4。离线损失下降没有补齐目标选择能力。

相关代码在`scripts/train_lora.py`；训练、验证和选模在`lora_pilot/`。适配增量在adapter_best.npz，不覆盖原基座。

## 4. 目标分叉配对适配

两条成功示教可能有大量不同的中间画面，却只有很短一段在完全相同的状态下要求不同动作。检查接受轨迹后发现：9个训练布局配对、4个验证布局配对在分叉帧的RGB、腕部图像和本体状态均逐元素一致；动作最大差约0.59–0.90。汤罐头场景分叉在第9帧，奶酪场景在第20帧。

![相同观察与不同目标动作](figures/paired_observation.png)

第二轮从首轮权重继续600步、batch2。一半batch来自共享观察下的两个相反目标，另一半保持原阶段采样。配对时只给模型不同的自然语言目标和正确动作标签，避免只凭画面判断应搬哪一件。没有增加失败轨迹，也没有使用测试图片训练。

验证使用原16帧阶段样本，加8个验证布局的目标分叉样本，以两项平均损失各占一半选模。最终选中step{best2['step']}。本轮同时改变了采样、更新次数并重置优化器，因此不作为“仅改变采样”的消融结论。

![两轮验证损失](figures/loss.png)

相关代码在`scripts/train_pair_lora.py`；分叉核对在PAIRED_BRANCH_AUDIT.json，设置与选模在`lora_paired/`。

## 5. 新布局闭环结果

每轮各有8个条件：两个场景×两个目标×两个新初态。基座与适配策略使用相同初态、噪声、正向技能指令和动作接口，每条固定执行300步后保持25步。原Long成功信号只记录，不控制停止；没有接Qwen检查器，先隔离低层执行能力。

“完整单件搬运”要求目标末10步在篮内且不接触夹爪，指定另一件全程位移不超过30毫米。“合法窗口”只说明执行中曾满足条件，不代表策略会自主停住。

{result_table}

![两轮闭环结果](figures/results.png)

第二轮按场景/目标分列：

{detail}

第二轮首次选对物体仍为5/8对5/8，没有提高。唯一完整收益来自场景0、初态30、指定番茄酱：基座同样先抬起了这件物体，但没有完成放置；适配改善了这一条件的后续完成。奶酪/黄油场景的反向目标仍失败。因此，这轮尚未补齐可靠的目标选择能力，不能把单例完成当成普遍的语言条件化提升。

两轮测试布局不同，各自与自己的基座配对。第二轮配对新增完整成功{c['paired_success_gains']}条、损失{c['paired_success_losses']}条。只有一个训练种子、每种目标两个测试初态，不能据此推断广泛泛化或统计显著性。

第一轮恢复5200个仿真状态，16条最终判分一致；第二轮恢复{p2['states']}个状态，最终判分差异{p2['final_disagreements']}条。动作输入和保存轨迹分别核对960次策略查询，两轮共1920次；另有48×2个同状态反事实点。首轮图像与动作生成不能因后续结果被覆盖。

附带视频固定展示第二轮场景0、初态30、指定番茄酱的两组真实回合，不按成功与否挑选，完整326帧。视频使用20Hz仿真时间，不包含推理等待。

相关代码在`scripts/evaluate_policy.py`、`evaluate_paired.py`、`safe_eval.py`、`summarize.py`；结果在SUMMARY.json、SUMMARY_PAIRED.json。

## 6. 内存故障与加载修复

重载校验时同时驻留两个模型，进程RSS一度约20.4GiB。工作站只有约30GiB系统内存且无Swap，systemd-oomd两次终止了包含实验的VSCode进程组。分离Unix会话没有分离systemd资源组，因此原先的后台启动方式不足以保护编辑器。日志没有据此证明显卡或其他硬件损坏。

修复分两步：实验放入独立systemd用户服务，设置14GiB软上限、16GiB硬上限，超限退出实验；模型按TensorStore/Orbax原存储块读取、转换和传入GPU，缓存限制16/32MiB，不再将完整权重多次复制到主机内存。LoRA增量逐块合并回原模型结构，基座与适配推理统一采用相同加载路径。

分块加载的基座进程观测RSS峰约3.81GiB，包含页缓存的进程组约10.67GiB。51数组/408块的基座与真正零增量适配器哈希全部一致，动作最大差0，通过原0.001阈值。历史进程输出仍有0.00146差异，保留记录，不将其算成训练收益。当前数学合并通过四项自包含CPU测试，模拟注意力、KV、前馈布局及零增量比特保持。

没有关闭系统内存保护，没有修改Swap、显卡驱动或升级硬件。相关代码在`scripts/bounded_restore.py`、`chunk_math.py`、`watch_resources.py`，故障与实际资源日志在`resource_checks/`。

## 7. 代码、权重与后续工作

```text
instruction_response/
  scripts/run.py、analyze.py       配对输入与执行顺序诊断
  PROTOCOL.json、SUMMARY.json
skill_bootstrap/
  scripts/teacher.py、generate.py  仿真示教
  scripts/train_lora.py            首轮适配
  scripts/train_pair_lora.py       目标分叉配对适配
  scripts/runtime.py、bounded_restore.py、chunk_math.py
  scripts/evaluate_policy.py、evaluate_paired.py、safe_eval.py
  scripts/summarize.py、audit_policy.py
  lora_pilot/、lora_paired/         权重与训练记录
  policy_eval/、policy_eval_paired/ 实际回合及输入/动作
  resource_checks/、tests/
```

代码包只包含源码、规格、统计、示教接受名单和少量图示，不包含原基座、全部帧和轨迹。权重增量单独保存，并附SHA256与固定基座来源；不是可独立运行的完整模型。完整GPU复现需要README列出的模型和LIBERO资产。解压代码包后的四项CPU测试使用合成张量，不依赖隐藏数据文件。

本阶段新增的是π0.5执行策略适配，不是Qwen高层微调。下一阶段应依据各目标的闭环表现决定是否继续补执行数据；只有能够按目标执行的技能，才接回Qwen检查器比较停止收益。Jetson端部署、统一导航与抓取场景、实体测试均未在本阶段开展。
'''
    (S/'REPORT.md').write_text(text)
    html=MarkdownIt().enable('table').render(text);css='@page{size:A4;margin:17mm}body{font:10pt/1.65 "Noto Sans CJK SC","Noto Sans CJK JP",sans-serif;color:#24374a}h1{font-size:23pt;color:#153b58}h2{font-size:15pt;break-after:avoid;border-bottom:1px solid #ccd8e3}table{width:100%;border-collapse:collapse;font-size:8pt;break-inside:avoid}td,th{padding:5px;border:1px solid #ccd8e3}th{background:#edf3f8}img{width:100%}pre{font-size:8pt;white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f5f8;padding:10px}code{overflow-wrap:anywhere}p:has(img){break-inside:avoid}'
    (S/'REPORT.html').write_text('<!doctype html><meta charset="utf-8"><style>'+css+'</style>'+html)
    with tempfile.TemporaryDirectory() as profile:subprocess.run(['google-chrome','--headless','--no-sandbox','--disable-gpu',f'--user-data-dir={profile}','--no-pdf-header-footer',f'--print-to-pdf={S/"REPORT.pdf"}',(S/'REPORT.html').as_uri()],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=90)
    shutil.copyfile(S/'REPORT.pdf',D/'目标条件化执行与策略适配_阶段报告.pdf');shutil.copyfile(movie,D/movie.name)
    files=set()
    for stage in [S,V/'instruction_response']:
        files.update(stage.glob('*.md'));files.update(stage.glob('*.json'));files.update(stage.glob('*.txt'));files.update((stage/'scripts').glob('*.py'));files.update((stage/'scripts').glob('*.sh'));files.update((stage/'tests').glob('*.py'));files.update((stage/'figures').glob('*.png'))
        for archive in stage.glob('archive*'):files.update(archive.rglob('*.py'))
        files.update((stage/'episodes').glob('*/*/result.json'));files.update((stage/'episodes').glob('*/*/*/result.json'))
    for folder in [S/'lora_pilot',S/'lora_paired']:
        files.update(folder.glob('*.json'));files.update(folder.glob('*.jsonl'));files.update(folder.glob('*.md'))
    for folder in [S/'policy_eval',S/'policy_eval_paired']:
        files.update(folder.glob('*/*.json'));files.update(folder.glob('*/episodes/*/*/result.json'))
    files.update((S/'resource_checks').glob('*.md'));files.update((S/'resource_checks').glob('*.log'));files.update((S/'resource_checks').glob('*.jsonl'))
    for evaluation in ['policy_eval','policy_eval_paired']:
        for p in (S/evaluation).glob('*/PROTOCOL.json'):files.update(ROOT/n for n in json.loads(p.read_text())['source_sha256'])
    for p in [V/'instruction_response/PROTOCOL.json',S/'DATA_PROTOCOL.json']:
        value=json.loads(p.read_text());files.update(ROOT/n for n in value.get('sources',{}))
    for folder in [S/'lora_pilot',S/'lora_paired']:
        config=json.loads((folder/'PROTOCOL.json').read_text());files.update(ROOT/n for n in config.get('source_hashes',config.get('source_sha256',{})))
    files.update([V/'requirements.lock.txt',V/'checkpoints/pi05_libero/download_verified.json',V/'scripts/python_shared',V/'third_party/sources.json',V/'third_party/openpi/LICENSE',V/'third_party/libero/LICENSE',V/'environment.json',V/'README.md'])
    manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    with zipfile.ZipFile(D/'目标条件化执行_源码.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(files):z.write(p,str(p.relative_to(ROOT)))
        z.writestr('PACKAGE_SHA256.json',json.dumps(manifest,indent=2))
    weights=D/'策略适配器';weights.mkdir(exist_ok=True)
    for name,folder in [('首轮',S/'lora_pilot'),('目标配对',S/'lora_paired')]:
        shutil.copyfile(folder/'adapter_best.npz',weights/(name+'_adapter.npz'));shutil.copyfile(folder/'BEST.json',weights/(name+'_说明.json'))
    (D/'文件清单.json').write_text(json.dumps({str(p.relative_to(D)):hashlib.sha256(p.read_bytes()).hexdigest() for p in D.rglob('*') if p.is_file() and p.name!='文件清单.json'},ensure_ascii=False,indent=2));print(D)

if __name__=='__main__':main()
