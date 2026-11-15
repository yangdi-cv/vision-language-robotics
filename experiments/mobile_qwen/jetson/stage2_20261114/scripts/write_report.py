"""Generate phase report from completed outputs; never invent missing measurements."""
from pathlib import Path
import json,re,subprocess,sys
S=Path(__file__).resolve().parents[1];Q=Path(__file__).resolve().parents[3]
sys.path.insert(0,'/tmp/mobile-qwen-doclibs')
import markdown
x=json.loads((S/'SUMMARY.json').read_text()); train=json.loads((S/'lora_2b/summary.json').read_text())
required=['pc_2b_base_bf16_512','pc_2b_lora_bf16_512','pc_2b_lora_nf4_512','pc_4b_lora_nf4_512','jetson_2b_lora_nf4_512_noswap']
assert all(k in x for k in required), 'Incomplete core comparison'
j=x[required[-1]];assert j['resources']['accuracy_usable_without_swap'], 'Jetson swap condition failed'
assert j['pc_agreement']['inputs_all_identical'], 'PC/Jetson input mismatch'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
plot_keys=['pc_2b_lora_bf16_512','pc_2b_lora_nf4_512','jetson_2b_lora_nf4_512_noswap','pc_4b_lora_bf16_512','pc_4b_lora_nf4_512']
plot_labels=['2B PC BF16','2B PC NF4','2B Jetson NF4','4B PC BF16','4B PC NF4']
positions=np.arange(len(plot_keys))
fig,ax=plt.subplots(figsize=(9,3.6))
for delta,metric,label,color in [(-.18,'strict_rate','Strict mask hit','#315b80'),(.18,'near_rate','4-pixel neighborhood hit','#d89a46')]:
    vals=[x[k]['metrics']['handle'][metric]*100 for k in plot_keys]
    ax.bar(positions+delta,vals,.34,label=label,color=color)
    for pos,v in zip(positions+delta,vals):ax.text(pos,v+1,f'{v:.1f}',ha='center',fontsize=8)
ax.set_xticks(positions,plot_labels,fontsize=9)
ax.set_ylim(0,113);ax.set_ylabel('Handle hit rate (%)')
ax.set_title('Matched 350-frame affordance test at 512 pixels')
ax.legend(loc='lower right',fontsize=8);ax.spines[['top','right']].set_visible(False)
fig.tight_layout();fig.savefig(S/'precision_comparison.png',dpi=170);plt.close(fig)
lines=[]
for k in ['pc_2b_base_bf16_512','pc_2b_lora_bf16_512','pc_4b_base_bf16_512','pc_4b_lora_bf16_512','pc_2b_lora_nf4_512','pc_4b_lora_nf4_512','jetson_2b_lora_nf4_512_noswap']:
 m=x[k]['metrics'];h=m['handle'];c=m['cabinet'];d=h['median_3d_cm']
 label=k.replace('_512_noswap','').replace('_512','').replace('_',' / ')
 lines.append(f"| {label} | {h['strict_rate']*100:.1f}% | {h['near_rate']*100:.1f}% | {'—' if d is None else f'{d:.3f}'} | {c['mean_iou']:.3f} | {h['valid_3d']}/350 |")
r=j['resources'];a=j['pc_agreement'];diag=x['diagnostic_partial_coordinates_base_2b']['metrics'];dh=diag['handle']; dc=diag['cabinet']
mem=r['stable_samples']; rss=max(v.get('VmHWM',0) for v in mem)/2**30;cg=max(v.get('cgroup_memory_current') or 0 for v in mem)/2**30
tegra=S/'evaluations/jetson_2b_lora_nf4_512_noswap.tegrastats.log'
ram=[int(v) for v in re.findall(r'RAM (\d+)/',tegra.read_text())]
probe=S/'probe_4b_result.json'
optional='4B板端全量350帧未执行；本阶段的完整板端精度与逐条一致率结论仅属于2B。'
if probe.exists():
 p=json.loads(probe.read_text()); optional+=f" 可选加载探针已通过，加载{p['resources']['load_seconds']:.2f}秒，并在验证集图片上完成一次查询；这不是4B准确率或延迟评测。"
if not probe.exists() and (S/'probe_4b_exit_code.txt').exists():
    optional+=' 4B可选资格检查没有完成，退出码见probe_4b_exit_code.txt，原始日志保留，不视为加载通过。'
text=f'''# 2B可供性LoRA与Jetson部署：阶段报告

2026年11月14日启动的实验。PC为RTX4090，端侧为8GB级Orin Nano、JetPack6.2.1/CUDA12.6、15W模式0；本阶段不接机械臂、里程碑检查或π0.5执行。

## 摘要

使用与既有4B实验完全相同的训练/验证划分和超参完成2B可供性LoRA，随后按PC BF16、PC NF4、Jetson NF4分步比较。测试固定512像素、350帧/1014问答，检查点只由验证集选择。2B模型已通过板端完整评测，输入一致率和输出一致率分别核对，而不是只比较总分。

稳定评测前停用了swap，本次2B正式板端结果满足无swap条件。首次启动时发生过系统交换读写的31条不完整尝试单独保留为无效记录，没有混入正式分数。384/256、MAXN SUPER、P50/P95时延对照和联合机器人闭环留到下一阶段。

## 目录

1. 实验设置与训练
2. 精度对照
3. 输出协议与诊断
4. PC/Jetson逐条一致性
5. 加载与内存
6. 工程问题与处理
7. 代码、复现与后续

## 1. 实验设置与训练

2B基座为Qwen/Qwen3-VL-2B-Instruct，revision 89644892e4d85e24eaac8bacfd4f463576704203；权重SHA256与官方文件元数据一致。直接调用既有affordance/train_lora.py，仅替换模型源。

训练初态17–49、验证13–16的同48条样例、测试0–12的同350帧。780训练帧、600优化步、梯度累积4、学习率1e-4、r8/alpha16/dropout0.05、同任务采样权重和种子20261114，每150步验证。验证把手像素误差中位数选点；两份protocol.json逐字段相等。两者都是初态隔离、动作模板共享，不是跨场景或独立动作分布泛化。

2B选中第{train['best_step']}步，验证误差中位数{train['val_median_px']:.3f}像素，可训练参数{train['trainable_parameters']:,}，训练记录墙钟{train['wall_seconds']:.1f}秒、峰值CUDA分配{train['peak_mib']:.1f}MiB。训练未读取测试集选点。

正式评测调用原eval_qwen.py的单次main流程，沿用原适用性、提示词、解析和评分；每个完成的配置只做一遍完整测试。所有生成均为贪心、最多40新token、BF16计算和SDPA。NF4仅量化语言线性层，视觉塔和共享词嵌入的输出头保持BF16，PEFT直接加载、不合并。

4B BF16沿用已经保存的同350帧结果，并未重新挑选。NF4导出后在PC重载评测，再将同一文件和适配器送到Jetson并校验SHA256。

## 2. 精度对照

| 配置（均512像素） | 把手严格命中 | 4像素邻域命中 | 把手3D中位误差/cm | 柜体框IoU | 有效3D样本 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(lines)}

![可供性定位精度对照](precision_comparison.png)

严格命中要求预测像素本身落入目标掩码；邻域命中为半径4像素的方形窗口与掩码相交。解析失败按失败计数，不从命中率分母剔除；表中3D只在可投影输出上计算，同时列出覆盖数。

3D来自仿真真值深度与相机参数反投影，仍有约1.4cm表面点/几何中心误差下限，不能解释为真实测距或抓取精度。完整碗/放置点指标见SUMMARY.json；350帧包含同回合相关画面，不能当作350次独立机器人任务。

## 3. 输出协议与诊断

2B基座在长定位提示词下常输出额外label说明，在40-token上限内截断：把手350条、放置点143条、柜体框350条均解析失败，碗171条能正常解析。因此基座正式表中零命中是当前接口不合格，不能直接解释为完全没有视觉定位能力。

为了区分格式与定位，另在CPU读取保存的原始文本，提取其中已经完整输出的坐标数组，没有重调模型。这个非正式诊断得到：基座把手严格命中{dh['strict_rate']*100:.1f}%、邻域命中{dh['near_rate']*100:.1f}%、3D中位误差{dh['median_3d_cm']:.2f}cm，柜体IoU{dc['mean_iou']:.3f}。诊断不替代生产解析，也没有参与检查点选择。LoRA收益同时包含输出格式对齐和定位改善，不能将全部收益归于视觉能力。

## 4. PC/Jetson逐条一致性

2B NF4对照共{a['n']}条：原始文本完全相同{a['raw_equal']}条（{a['raw_rate']*100:.2f}%），解析后的坐标/框完全相同{a['parsed_equal']}条（{a['parsed_rate']*100:.2f}%）。实际输入token、图像张量和网格哈希的匹配数量为{a['input_and_token_equal_counts']}。

相同输入不意味着不同GPU内核必须逐bit相同；若有不同，保留逐条输出。PC与板端Transformers/PEFT版本相同，Torch、torchvision和bitsandbytes使用各自兼容构建，故平台对照也包含软件栈差异，不能把所有差异只归因于GPU硬件。量化前后与设备变化分步比较，不混为一个变化。本轮四个PC配置的1014条实际输入token、图像张量和网格均逐条相同，记录在PC_INPUT_CHECK.json；历史4B BF16未保存同样的逐条输入元数据，只能核对原始协议、图片与提示词，不能补造历史哈希。

## 5. 加载与内存

2B板端本次进程首次加载（处理器、模型、适配器及GPU同步）{r['load_seconds']:.2f}秒。操作系统文件缓存未清空，不是冷盘启动基准。

进程VmHWM峰值{rss:.3f}GiB，稳定期cgroup采样最高{cg:.3f}GiB，CUDA分配峰值{r['peak_cuda_allocated']/2**30:.3f}GiB，CUDA保留峰值{r['peak_cuda_reserved']/2**30:.3f}GiB；tegrastats全阶段整机RAM最高{max(ram) if ram else '未取得'}MB。以上统计口径不能相加；整机RAM包含系统和缓存，CUDA内存也可能与共享物理内存重叠。

板端内核不提供cgroup memory.peak，故该字段为null，没有拿稳定期采样最大值冒充完整加载峰值。保留VmHWM、CUDA峰值和从加载前开始的tegrastats原始记录。

正式评测模型cgroup最大swap为{r['stable_max_swap_bytes']}字节，稳定段系统交换I/O页变化为{r['global_stable_swap_io_pages']}；结束时/proc/swaps没有活动设备。加载时保留交换空间，正式评测前swapoff，结束后恢复原zram及8GiB SD swap。安装与推理均有systemd内存上限，板端MemoryMax=6G，15W且未锁jetson_clocks。

{optional}

## 6. 工程问题与处理

PC第一次NF4导出出现CPU内存膨胀并触发systemd保护，训练和两份BF16完整评测未丢失、未重跑。之后改用独立受限单元，并显式保留共享输出头，成功保存NF4文件；后续限制编译线程并逐配置运行，不提高内存上限。PC驱动脚本因运行中编辑发生收尾解析错误，但四份本轮PC评测均已保存1014条唯一问答和完整资源记录；驱动错误单列留档，没有重跑测试或补造输出。

板端第一次启动推理虽然模型自身swap为0，系统仍有交换I/O。31条未完成尝试标为无效。正式新tag增加加载/无swap门控，swapoff确认后才开始350帧单遍评测。原始失败文件仍保留，不以成功结果覆盖。

保存后的tokenizer曾给出通用正则告警。本轮没有根据告警擅自改tokenizer；实际输入token和图像哈希用于核实对照输入。正式结论只限当前图像、提示词、单场景和检查点。

## 7. 代码、复现与后续

相关代码在jetson/stage2_20261114/scripts：train_2b.py调用原训练配方；runtime.py固定加载和量化口径；evaluate_once.py调用原评测；quantize.py导出；summarize.py复算严格命中与逐条一致率；run_jetson_noswap.sh管理加载期与正式期的swap；test_protocol.py检查划分。

权重：lora_2b/best_adapter；预量化模型：nf4_2b、nf4_4b。原始测试结果和逐条输入/输出在evaluations/，资源在*.resources.json与*.tegrastats.log；无效启动和内存保护记录在diagnostics/及对应日志。SUMMARY.json为完整指标汇总。

本阶段尚未测正式端到端P50/P95，没有进入MAXN SUPER或里程碑问法对照，也没有机械臂动作或联合闭环。下一阶段才比较512/384/256的精度与延迟，每个新尺寸单独复测精度。保留既定≤3秒可供性目标，不能用本阶段“可以运行”替代达标。旧π0.5 LoRA闭环复测仍最后单列。
'''
(S/'REPORT.md').write_text(text)
style='''@page{size:A4;margin:17mm}body{font-family:"Noto Sans CJK SC",sans-serif;font-size:9.5pt;line-height:1.65;color:#263448}h1{font-size:23pt;line-height:1.4}h2{font-size:15pt;break-after:avoid;margin-top:7mm}table{border-collapse:collapse;width:100%;font-size:8pt;break-inside:avoid}th,td{border:1px solid #bfcbd8;padding:5px}th{background:#e9eff5}p{orphans:3;widows:3}code{overflow-wrap:anywhere}img{max-width:100%;max-height:90mm}'''
html=S/'REPORT.html';html.write_text('<!doctype html><meta charset="utf-8"><style>'+style+'</style>'+markdown.markdown(text,extensions=['tables','fenced_code','sane_lists']))
subprocess.run(['/usr/bin/google-chrome','--headless','--no-sandbox','--disable-gpu','--no-pdf-header-footer','--print-to-pdf='+str(S/'REPORT.pdf'),html.as_uri()],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
print(S/'REPORT.pdf')

log=Q.parents[1]/'PROJECT_LOG.md'
marker='### Jetson可供性第二阶段核心对照完成（自动收尾）'
if marker not in log.read_text():
    h=j['metrics']['handle']
    with log.open('a') as stream:
        stream.write(f"\n{marker}\n\n正式2B板端350帧/1014问答已完成，无swap资源检查通过。严格命中{h['strict_hits']}/350，邻域命中{h['near_hits']}/350，3D中位{h['median_3d_cm']:.5f}cm；原始输出一致{a['raw_equal']}/{a['n']}，坐标一致{a['parsed_equal']}/{a['n']}，输入哈希全一致。报告见mobile_qwen/jetson/stage2_20261114/REPORT.pdf及SUMMARY.json。4B仅可选加载探针，无全量板端精度；384/256、时延/MAXN SUPER尚未执行，不连接机械臂。\n")
