"""Build the Chinese stage report, actual-rollout comparison, and source archive."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from markdown_it import MarkdownIt
import imageio_ffmpeg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from protocol import MODES, OBJECTS
from analyze import read_rows
from examples import main as make_examples

S=Path(__file__).resolve().parents[1];V=S.parent;ROOT=V.parents[1]
LABEL=dict(flat_zh='完整中文',translated_en='仅翻译',skill_continue='单技能持续',timer150='固定150步',timer_calibrated='开发定时',visual='视觉停止',frozen='冻结检查图')
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
D=ROOT/'deliverables/20261113_阿里_选择性搬运与视觉停止'


def table(head,rows):
    return '\n'.join(['| '+' | '.join(head)+' |','|'+'|'.join(['---']*len(head))+'|',*['| '+' | '.join(map(str,row))+' |' for row in rows]])


def figures(summary):
    plt.rcParams['font.family']='Noto Sans CJK JP';plt.rcParams['axes.unicode_minus']=False
    labels=[LABEL[m] for m in MODES];x=np.arange(len(MODES));groups=summary['groups']
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for ax,field,title in [(axes[0],'success','选择性成功 / 20'),(axes[1],'violations','保留物体位移超限 / 20')]:
        values=[groups[m][field] for m in MODES]
        ax.bar(x,values,color=['#8e9bad','#8e9bad','#8e9bad','#d79839','#d79839','#397cb1','#9271a7'])
        ax.set_ylabel(title);ax.set_ylim(0,22);ax.set_xticks(x,labels,rotation=25);ax.grid(axis='y',alpha=.2)
        for i,value in enumerate(values):ax.text(i,value+.3,str(value),ha='center')
    fig.savefig(S/'figures/outcomes.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,4),constrained_layout=True)
    for i,(name,stratum) in enumerate(summary['development_strata'].items()):
        values=[stratum[m]['success'] for m in MODES]
        ax.bar(x+(i-.5)*.36,values,width=.36,label='开发时已有可行窗口' if i==0 else '开发时未证实可行的反向目标')
    ax.set_xticks(x,labels,rotation=20);ax.set_ylim(0,11);ax.set_ylabel('选择性成功 / 各10条件');ax.legend(fontsize=9);ax.grid(axis='y',alpha=.2)
    fig.savefig(S/'figures/directions.png',dpi=160);plt.close(fig)


def video(rows):
    bycase={r['case']['key']:{q['mode']:q for q in rows if q['case']['key']==r['case']['key']} for r in rows}
    keys=sorted(bycase)
    eligible=[k for k in keys if bycase[k]['visual']['selective_success'] and not bycase[k]['skill_continue']['selective_success']]
    key=(eligible or keys)[0];group=bycase[key];runs=[]
    for mode in MODES:
        r=group[mode];reader=imageio_ffmpeg.read_frames(str(S/r['directory']/'rollout.mp4'));meta=next(reader)
        frames=[np.frombuffer(f,np.uint8).reshape(meta['size'][1],meta['size'][0],3).copy() for f in reader]
        assert len(frames)==r['steps']+1;runs.append((r,frames))
    target=S/'选择性搬运_七组实际回合.mp4';width,height=1600,1050;fps=20;count=max(len(f) for _,f in runs)
    process=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-f','rawvideo','-vcodec','rawvideo','-s',f'{width}x{height}','-pix_fmt','rgb24','-r',str(fps),'-i','-','-an','-vcodec','libx264','-crf','20','-pix_fmt','yuv420p',str(target)],stdin=subprocess.PIPE,stderr=subprocess.DEVNULL)
    fonts={n:ImageFont.truetype(FONT,n) for n in [19,23,28]}
    c=group['visual']['case'];title=c['instruction']
    for step in range(count):
        canvas=Image.new('RGB',(width,height),'#eef3f7');draw=ImageDraw.Draw(canvas)
        draw.text((15,10),title,font=fonts[28],fill='#17354c')
        for i,(r,frames) in enumerate(runs):
            ox=(i%4)*400+10;oy=(i//4)*490+60
            draw.text((ox,oy),LABEL[r['mode']],font=fonts[23],fill='#17354c')
            canvas.paste(Image.fromarray(frames[min(step,len(frames)-1)]).resize((355,355)),(ox+12,oy+40))
            label=('结束 · '+('选择性成功' if r['selective_success'] else '未满足全部条件')) if step>=len(frames)-1 else f'第{step}步 · '+('停止后保持' if step>=r['decision_step'] else '策略执行')
            draw.text((ox,oy+405),label,font=fonts[19],fill='#17354c')
        draw.text((1220,590),'只搬指定一件',font=fonts[23],fill='#17354c')
        draw.text((1220,635),'原Long成功不触发停止',font=fonts[19],fill='#17354c')
        draw.text((1220,675),'20fps仿真时间',font=fonts[19],fill='#17354c')
        draw.text((1220,715),'不含推理墙钟等待',font=fonts[19],fill='#17354c')
        draw.text((1220,755),'已结束组保留真实末帧',font=fonts[19],fill='#17354c')
        if step==min(group['visual']['decision_step']+15,count-1):canvas.save(S/'figures/video_keyframe.png')
        process.stdin.write(canvas.tobytes())
    process.stdin.close();assert process.wait()==0
    reader=imageio_ffmpeg.read_frames(str(target));next(reader);assert sum(1 for _ in reader)==count
    record=dict(case=key,frames=count,fps=fps,seconds=count/fps,decoded=True,selection='first sorted eval case where visual succeeds and continued skill fails; illustrative only, no change to statistics')
    (S/'VIDEO.json').write_text(json.dumps(record,indent=2));return record


def package():
    D.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(S/'REPORT.pdf',D/'选择性搬运与视觉停止_阶段报告.pdf')
    shutil.copyfile(S/'选择性搬运_七组实际回合.mp4',D/'选择性搬运_七组实际回合.mp4')
    protocol=json.loads((S/'PROTOCOL.json').read_text())
    files=list((S/'scripts').glob('*.py'))+list((S/'tests').glob('*.py'))+list((S/'archive').rglob('*.py'))+list((S/'figures').glob('*.png'))
    files += [S/name for name in ['README.md','RESEARCH.md','requirements-offline.txt','PROTOCOL.json','RESOURCES.json','ENVIRONMENT.json','CALIBRATION.json','ANALYSIS_PLAN.json','EVALUATION_LOCK.json','EVALUATION_PREPARED.json','SUMMARY_pilot.json','SUMMARY_eval.json','VERIFICATION_skills.json','VERIFICATION_pilot.json','VERIFICATION_eval.json','DECISION_FRAMES.json','DECISION_SUMMARY.json','VIDEO.json','REPORT.md']]
    files += list((S/'plans').rglob('*.json'))+list((S/'episodes').glob('*/*/*/result.json'))
    files += [ROOT/name for name in protocol['sources']]
    files += [V/'scripts/python_shared',V/'third_party/sources.json',V/'third_party/openpi/LICENSE',V/'third_party/libero/LICENSE',V/'checkpoints/pi05_libero/download_verified.json',V/'README.md',V/'environment.json',ROOT/'perception/qwen_scene/requirements-local.lock.txt']
    files += list((V/'configs').rglob('*.json'))+list((V/'configs').rglob('*.yaml'))
    if (S/'CLEAN_ACCEPTANCE.json').exists():files.append(S/'CLEAN_ACCEPTANCE.json')
    missing=[str(p) for p in files if not p.is_file()];assert not missing,missing
    content={str(p.relative_to(ROOT)):(p,p.read_bytes()) for p in sorted(set(files))}
    manifest={name:hashlib.sha256(data).hexdigest() for name,(_,data) in content.items()}
    with zipfile.ZipFile(D/'选择性搬运_源码.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name,(path,data) in content.items():
            info=zipfile.ZipInfo(name);info.create_system=3;info.external_attr=(path.stat().st_mode&0xffff)<<16;info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,data)
        z.writestr('PACKAGE_SHA256.json',json.dumps(manifest,indent=2))
    (D/'文件清单.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in D.iterdir() if p.is_file() and p.name!='文件清单.json'},ensure_ascii=False,indent=2))


def main():
    summary=json.loads((S/'SUMMARY_eval.json').read_text());pilot=json.loads((S/'SUMMARY_pilot.json').read_text())
    verification=json.loads((S/'VERIFICATION_eval.json').read_text());pv=json.loads((S/'VERIFICATION_pilot.json').read_text())
    assert summary['episodes']==140 and pilot['episodes']==28 and verification['episodes']==140
    rows=read_rows('eval');skills=read_rows('skills');g=summary['groups']
    figures(summary);make_examples();vid=video(rows)
    outcomes=[[LABEL[m],f"{g[m]['success']}/20",f"{g[m]['target_stable']}/20",f"{g[m]['protected_ok']}/20",g[m]['premature_stop']] for m in MODES]
    costs=[[LABEL[m],g[m]['model_checks'],f"{g[m]['decision_steps_mean']:.1f}",g[m]['self_stops'],g[m]['missed_valid_window']] for m in MODES]
    direction=[]
    for t in [0,1]:
        for target in [0,1]:
            direction.append([f"{OBJECTS[t][target][2]}入篮；保留{OBJECTS[t][1-target][2]}",*[f"{summary['directions'][f'task{t}_target{target}'][m]}/5" for m in MODES]])
    calibration=json.loads((S/'CALIBRATION.json').read_text())['thresholds']
    skill_table=[[r['case']['target'],'是' if r['target_ever_reached'] else '否',r['first_valid_step'] if r['first_valid_step'] is not None else '无',calibration[r['case']['target']]] for r in skills]
    paired=[[LABEL[r['other']],r['n'],r['visual_gains'],r['visual_losses'],f"{r['descriptive_exact_p']:.3f}"] for r in summary['paired']]
    sensitivity=[[LABEL[m],*[f"{summary['posthoc_position_tolerance_sensitivity'][str(limit)][m]}/20" for limit in [.01,.02,.03,.05]]] for m in MODES]
    plans=[json.loads(p.read_text()) for p in (S/'plans/eval').glob('*.json')]
    correct=sum(p['selection']['target']==p['case']['target'] for p in plans)
    prefix_ok=all(r['initial_equal'] and r['instruction_equal'] and r['input_hashes_equal'] and r['max_action_difference']==0 for r in summary['prefixes'])
    count=lambda m:g[m]['success']
    timer_conclusion=(
        f"视觉停止在本批比开发定时多成功{count('visual')-count('timer_calibrated')}个条件；这只支持两场景的局部结果，尚不足推广。"
        if count('visual')>count('timer_calibrated') else
        f"视觉停止比开发定时少成功{count('timer_calibrated')-count('visual')}个条件，当前方案没有超过这个简单基线。"
        if count('visual')<count('timer_calibrated') else
        "视觉停止与开发定时成功数相同，当前没有总体成功数上的优势，还应看逐条件的增失。"
    )
    text=f'''# 选择性搬运与视觉停止

2026年11月13日 · π0.5 / Qwen 闭环对照阶段报告

## 摘要

上轮五组原Long任务均28/30，保持第一条子目标也能完成。这轮把同一入篮场景改为“只搬指定一件、另一件保持原位”，让停止是否及时影响结果。系统不读取原任务成功信号来停止。Qwen选择目标，π0.5预测实际动作，视觉检查可结束执行。

开发阶段4个参考技能、28个小试；固定参数后，在初态3–7执行七组配对，共140回合、每组20条件。视觉停止{g['visual']['success']}/20，单技能持续{g['skill_continue']['success']}/20，仅翻译{g['translated_en']['success']}/20，开发定时{g['timer_calibrated']['success']}/20，固定150步{g['timer150']['success']}/20，冻结检查图{g['frozen']['success']}/20。本批视觉停止未超过两种定时规则。

四个方向全部保留。开发时只有{sum(r['first_valid_step'] is not None for r in skills)}/4方向出现“目标已放好、保留物体未超限”的窗口；低层不能可靠按目标选物，是本系统的另一限制。Qwen在评测条件中选对目标{correct}/{len(plans)}，这不等于低层一定按选择执行。

![七组结果](figures/outcomes.png)

## 目录

1. 任务设计与评分
2. 开发小试与冻结设置
3. 配对结果与对照解释
4. 失败发生在哪一层
5. 执行与物理状态核验
6. 结论与下一步
7. 代码目录与复现

## 1. 任务设计与评分

沿用LIBERO-Long的汤罐/番茄酱、奶酪/黄油两个入篮场景，分别交换目标和保留物体，共四种指令。例：“只把字母汤罐头放进篮子里，番茄酱罐头保持原位。放好后停止，不要继续搬其他物体。”不移动物体初态，不改变场景或动力学，不使用mocap代替π0.5动作。

这是自定义选择性任务，不是新的原LIBERO-Long成绩。所有方法均忽略step返回的原长任务done。目标是否在篮子、物体位置和接触真值仅写入日志与离线评分，既不决定停止，也不送进模型。

每回合最多执行300步。自主停止或预算到期后，保持25步，末10帧目标都在原In篮子区域、都没有与夹爪接触，且整个过程中保留物体中心位移最大不超过30毫米，才算选择性成功。中心位移容差允许小幅移动，不代表无接触或无旋转。停止保持时沿用最后的夹爪命令，不自动松爪给任何一组补成功。

flat策略没有独立结束token，其预算终止不计自主停止；主要比较仍是所有方法共同的选择性成功条件。不能因为另一组接了停止接口，就只凭自主停止次数宣称优越。

## 2. 开发小试与冻结设置

先在初态0跑四个固定英文参考技能。可行窗口指连续10步目标入篮、不接触夹爪、保留物体尚未超限。时刻标定只使用这里的数据：将窗口末步向上取整到10步；没有可行窗口时使用预设150步。

{table(['目标','曾入篮','首个可行窗口末步','开发定时阈值'],skill_table)}

然后初态0跑七组小试（28回合），不计入评测成绩。正式初态为3–7，两场景×两方向×五初态=20条件；七组140回合。参数、标定和执行源码在评测前冻结，不按正式结果选时刻或改提示词。这些是本轮未用于调参的初态，场景仍为已知公开场景，不能称为对新世界泛化。

| 方法 | 策略指令 | 结束方式 |
|---|---|---|
| 完整中文 | 含目标、保留物体与停止要求 | 300步预算 |
| 仅翻译 | Qwen实际翻译的完整英文 | 300步预算 |
| 单技能持续 | Qwen选目标后转换的固定英文技能 | 300步预算 |
| 固定150步 | 同一技能 | 150步后门控两票 |
| 开发定时 | 同一技能 | 开发时标定的时刻后门控两票 |
| 视觉停止 | 同一技能 | Qwen开始/当前帧判断，门控两票 |
| 冻结检查图 | 同一技能 | 只将检查当前图换成开始图 |

五个同技能组完全共享目标选择与英文指令；模型选错不会替它改成正确答案。Qwen选择的是给定双物体词表里的ID，技能文字用固定模板生成，尚不是开放词汇定位或自由多步规划。

停止检查每10步一次，只有上一条命令为张开且夹爪宽度超过0.07时才可查询。连续两票确认；闭爪或票间隔超过20步就清空旧票。两票不假定统计独立。所有方法低层均读取当前224像素RGB/腕部图和8维机器人状态；Qwen读取单独512像素图。每次预测10步、执行前5步，显式提供配对噪声。

## 3. 配对结果与对照解释

{table(['方法','选择性成功','目标末10帧入篮','保留物体位移未超限','过早停止'],outcomes)}

各条件对应的四个方向如下，每格分母为5。四个方向全部计入总分，开发时已证实与未证实可行的方向另列。

{table(['指令',*[LABEL[m] for m in MODES]],direction)}

![开发可行方向与反向方向](figures/directions.png)

以视觉停止为参照，按同一场景、目标、初态计算增加与丢失的成功：

{table(['另一方法','配对数','视觉增加成功','视觉丢失成功','描述性精确p'],paired)}

只有两个场景、每方向五初态，且使用共同噪声、多重对照。p仅描述本批配对，不作为普遍显著性或泛化结论。应分别看：视觉比持续执行好多少、真实当前图比冻结图好多少，以及是否超过开发定时规则。完整中文/仅翻译同时改变策略指令，不能将差异单独归因为视觉检查。

另做事后容差敏感性分析，只重算保留物体的允许位移，不改变主评分的30毫米标准，也不据此调模型或选择最有利的阈值：

{table(['方法','10毫米','20毫米','30毫米（主）','50毫米'],sensitivity)}

## 4. 失败发生在哪一层

“目标放稳”与“保留物体未超限”分开记录，避免把两件都搬完算作成功。还记录是否出现过合法完成窗口、之后是否继续搬动保留物体、是否过早停止。

{table(['方法','模型检查次数','平均决策步数','自主停止回合','错过可行窗口后超限'],costs)}

平均决策步数包含失败，不能单独当作效率排名。检查调用不包含每条件共用的目标选择/翻译（各一次，正式共40次）；完整中文执行时不用共用计划。模拟器在推理时暂停，不据此宣称真机实时频率。

Qwen选择目标正确率为{correct}/{len(plans)}。低层能否遵从目标另行判断：开发参考技能的可行窗口和两个方向的正式结果都保留。如果保留物体先被移动，再搬对目标，停止检查无法撤销这次违反约束；如果已经存在可行窗口，却继续执行到超限，才需要进一步检查视觉判断、门控和票数延迟。

开发小试有两个具体错误：奶酪在第160步已入篮且没有接触夹爪，模型仍回答未完成；反向番茄酱指令在第180步得到完成确认，但番茄酱目标没有入篮。下图为这两个开发回合的实际模型输入与末帧。

篮内目标存在遮挡，模拟器完成标签不等于这张图足够判断。可见性尚需人工审核，不能把所有漏确认画面直接当成高置信训练正例，也不能据此认定换更大模型就能解决。

![开发小试的两个检查错误](figures/development_errors.png)

演示选取{vid['case']}。选择规则是按编号选第一个“视觉成功、持续失败”的正式条件（若不存在则取首个条件），只为展示，不改变统计。各组结束后保留真实末帧；视频20fps是仿真时间，不包含推理等待。

![实际七组回合截图](figures/video_keyframe.png)

## 5. 执行与物理状态核验

{table(['核验项目','结果'],[['正式完整回合',verification['episodes']],['开发小试完整回合',pv['episodes']],['正式视频完整解码帧',verification['video_frames']],['正式模型策略查询',verification['policy_queries']],['正式检查机会',verification['decisions']],['原Long成功后仍继续的回合',verification['original_success_continued']],['同技能停止前配对',len(summary['prefixes'])],['停止前输入/动作配对检查','通过' if prefix_ok else '未通过'],['恢复物理状态数',verification['physics_states']],['恢复后物体位置最大差(m)',f"{verification['max_restored_object_position_error_m']:.6f}"],['恢复后关系/接触标签瞬时差异',f"{verification['restored_predicate_disagreements']} / {verification['restored_contact_disagreements']}"],['恢复后最终成功判分差异',len(verification['restored_final_score_disagreements'])],['自包含CPU检查','12项通过']])}

逐次核对实际策略输入、噪声、动作块与执行动作；停止后是固定保持命令，未补轨迹。冻结图组核对实际送入模型的像素和提示词记录。还将保存的模拟器状态重新forward，独立重算目标关系、夹爪接触和整段保留物体位移。状态重算与调用时缓存存在瞬时差异，但140个正式回合的最终判分一致。

原始逐步状态、输入图和视频保存在实验目录。源码包仅包含代码、配置、结果元信息和报告插图，不包含基座权重、原始大数组或全部图像；完整重算需原实验数据或重新运行。CPU测试自带数据，不依赖这些文件。

开发时曾将最终夹爪张开误作物体释放，已将该版小试与源码保存在development/open_gripper_score，随后改为离线接触标签并重新运行。本报告不混入该版数字。原有Long对照与历史结果未修改。

## 6. 结论与下一步

本轮检验指定物体的搬运、保留物体约束和停止时机。视觉停止{count('visual')}/20，对照持续执行{count('skill_continue')}/20、冻结检查图{count('frozen')}/20。{timer_conclusion}三类对照分别衡量停止干预、当前视觉信息和相对定时规则的收益。

当前仍需区分两类改进。已有可行窗口但未停好的条件，用实际决策帧与接触标签审核检查器；低层先动错物体的条件，需要先提高目标条件化技能，不能期待完成检查器补救。这里的低层结论限于当前检查点及技能提示模板，不表示任何提示或适配后的π0.5都无法完成反向目标。下一轮优先处理这两类具体失败，再考虑中途改目标和恢复，不同时扩展模型规模、场景数量与新功能。

接口上还要保留全局限制：目前单技能模板只传递正向搬运要求，未把“另一件保持原位”继续交给低层。完整中文与该模板同时改变语言、措辞和约束，不能单独归因；下一项有针对性的对照应是在同一完整指令上加/不加停止，而不是再扩大模型参数。

## 7. 代码目录与复现

```text
experiments/vla_libero/selective_execution/
  PROTOCOL.json / CALIBRATION.json / EVALUATION_LOCK.json
  ANALYSIS_PLAN.json / RESOURCES.json
  scripts/run.py                 仿真、模型动作与逐步记录
  scripts/worker.py              Qwen选目标，复用已有翻译/检查
  scripts/control.py             门控、票据与停止，不接收任务真值
  scripts/protocol.py            指令构造、开发标定与离线评分
  scripts/analyze.py / verify.py  配对统计与状态核验
  scripts/build.py               图表、报告、实际视频与源码包
  tests/test_selective.py        12项自包含CPU检查
  plans/<pilot|eval>/<条件>.json  实际模型选择/翻译及提示词版本
  episodes/<阶段>/<条件>/<方法>/
    result.json / trace.npz / trace.jsonl / events.jsonl
    rollout.mp4 / images/ / policy_inputs/ / policy_queries.jsonl
```

相关代码在scripts/run.py、control.py、protocol.py；模型加载/通信用项目已有perception/qwen_scene/与vla_libero/scripts/模块。固定上游和许可见RESOURCES.json、RESEARCH.md。

```bash
bash experiments/vla_libero/scripts/python_shared experiments/vla_libero/selective_execution/scripts/run.py --stage pilot
bash experiments/vla_libero/scripts/python_shared experiments/vla_libero/selective_execution/scripts/run.py --stage eval
.venvs/openpi-libero/bin/python experiments/vla_libero/selective_execution/scripts/analyze.py --phase eval
bash experiments/vla_libero/scripts/python_shared experiments/vla_libero/selective_execution/scripts/verify.py --phase eval --physics
```

沿用本地openpi-libero和qwen-scene环境，4090共驻设置为XLA_PYTHON_CLIENT_MEM_FRACTION=0.40、PREALLOCATE=false。完整仿真需公开资产及指定模型缓存；无GPU测试见README。没有连接实体机械臂，也没有训练新模型。
'''
    (S/'REPORT.md').write_text(text)
    html=MarkdownIt().enable('table').render(text).replace('<h2>目录</h2>','<h2 class="toc">目录</h2>')
    css='''@page{size:A4;margin:17mm 16mm 18mm}body{font:10pt/1.7 "Noto Sans CJK SC","Noto Sans CJK JP",sans-serif;color:#24374a}h1{font-size:24pt;color:#153b58}h2{font-size:16pt;color:#153b58;border-bottom:1px solid #ccd8e3;padding-top:12px;break-after:avoid}h3{break-after:avoid}.toc{break-before:page}table{width:100%;border-collapse:collapse;font-size:8pt;margin:12px 0;break-inside:avoid}td,th{border:1px solid #ccd8e3;padding:5px}th{background:#edf3f8}img{width:100%;height:auto}p:has(img){break-inside:avoid}pre{font:8pt/1.6 "DejaVu Sans Mono","Noto Sans CJK SC",monospace;white-space:pre-wrap;overflow-wrap:anywhere;padding:10px;background:#f1f5f8;break-inside:avoid}code{overflow-wrap:anywhere}'''
    (S/'REPORT.html').write_text('<!doctype html><meta charset="utf-8"><title>选择性搬运与视觉停止</title><style>'+css+'</style>'+html)
    with tempfile.TemporaryDirectory(prefix='selective_pdf_') as profile:
        subprocess.run(['google-chrome','--headless','--no-sandbox','--disable-gpu',f'--user-data-dir={profile}','--no-pdf-header-footer',f'--print-to-pdf={S/"REPORT.pdf"}',(S/'REPORT.html').as_uri()],check=True,timeout=90,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    assert (S/'REPORT.pdf').read_bytes().startswith(b'%PDF-')
    package();print(D)


if __name__=='__main__':main()
