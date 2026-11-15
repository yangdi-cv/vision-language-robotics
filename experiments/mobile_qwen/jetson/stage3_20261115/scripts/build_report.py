"""Report the small Jetson latency screen and separate PC-physics service tests."""
from pathlib import Path
import json,sys,subprocess,zipfile,re,hashlib
import numpy as np
S=Path(__file__).resolve().parents[1];ROOT=S.parents[3];C=ROOT/'experiments/mobile_qwen/closed_loop_20261115'
x=json.loads((S/'results/SUMMARY.json').read_text());assert x['valid_without_swap']
records=[json.loads(l) for l in (S/'results/records.jsonl').read_text().splitlines()];assert len(records)==60
rows=[]
for size in [512,384,256]:
 for query in ['all','handle','cabinet']:
  m=x['latency'][f'{size}_{query}'];rows.append(f"| {size} | {query} | {m['n']} | {m['wall_p50']:.2f} | {m['wall_p95']:.2f} | {m['mean_output_tokens']:.0f} | {'达标' if m['meets_3s_p95'] else '未达标'} |")
report='''# Jetson可供性服务：时延与PC仿真联调

2026年11月15日 · Orin Nano 8GB / 15W

## 摘要

复用已完成350帧精度对照的2B NF4可供性LoRA，先筛查图像尺寸对时延的影响，再将板端感知结果接入PC中的Nav2与MoveIt仿真。没有安装新环境、下载新模型或连接实体机械臂。

512、384、256三个尺寸均未达到既定可供性单次≤3秒目标。缩图只带来小幅收益，因此没有再投入新尺寸的全量350帧精度评测，也没有把小样本输出当作准确率结论。功能可用与实时达标分别判断。

## 1. 时延设置与结果

输入为冻结验证集排序前5帧，未使用test350。每帧询问handle、bowl、place、cabinet四类问题，每档20次，另各暖机一次不计入。尺寸按图像轮换顺序，减少固定顺序影响；P95只是探索性小样本分位数。max_new_tokens为40，贪心解码，未缩短原提示词。

墙钟时间从PC/板端测试脚本读取和处理图像开始，到生成文本结束；不含网络传输。模型内时间另存，不能将其替代端到端时延。输出长度一并报告，因为不同问题生成token数不同。

| 尺寸 | 查询 | 次数 | P50 秒 | P95 秒 | 平均输出token | ≤3秒P95 |
|---:|---|---:|---:|---:|---:|---|
'''+ '\n'.join(rows)+f'''

首次加载{x['load_seconds']:.2f}秒，未清除OS页缓存，不称冷盘启动。稳定段模型swap与全局交换I/O均为0。NF4保持视觉塔及共享lm_head为BF16，其余量化设置与上一阶段一致。

## 2. 功耗、频率与资源

使用15W模式0，未调用jetson_clocks锁频。实测GPU下限306MHz、上限624.75MHz，nvhost_podgov动态调频；CPU0上下限729.6/1510.4MHz。tegrastats逐100ms记录实际负载与频率。本板当前启动配置不提供MAXN SUPER模式，本次没有切换或刷写固件。

模型进程MemoryHigh5GiB、MemoryMax6GiB。加载时允许SD/zram作储备，加载完成后关闭swap再开始稳定测量；结束恢复原SD和zram。不能把SD交换造成的慢速结果算作有效实时测试。

缩图后时延仍接近5秒，且25-token的柜体框输出明显慢于17-token的点输出，说明继续只减像素的收益有限。时延与输出长度相关只是观察结果，不能代替CUDA profiler对预填充、逐token解码和内核开销的分解；不据此武断归因。

## 3. 视觉服务与PC联合仿真

服务只绑定Jetson的127.0.0.1，通过SSH隧道向PC的127.0.0.1转发，串行处理请求，不在PC加载Qwen。使用512尺寸，因为该配置已有上一阶段350帧精度核对；未选择尚未验证精度的新尺寸。服务返回二维把手点，PC用仿真深度反投影并交给MoveIt实际驱动同一世界中的关节。

机器人在模型返回期间暂停仿真推进，所以联合回合通过只能说明功能链路可用，不能替代≤3秒实时要求。联合回合只做预接触，不执行抓取或原LIBERO整任务。
'''
joint=C/'results/jetson_joint_v1_summary.json'
if joint.exists():
 rr=json.loads(joint.read_text());report+=f"\n当前记录{len(rr)}回合，预接触通过{sum(r['success'] for r in rr)}/{len(rr)}。\n\n| 初态 | 导航成功 | 预接触通过 | 真值误差 mm | 视觉残差 mm |\n|---:|---|---|---:|---:|\n"
 for r in rr:
  audit=r.get('final_audit',r['attempts'][-1]['audit'] if r['attempts'] else {});e=audit.get('shadow_true_precontact_error_m');res=r['attempts'][-1].get('visual_residual_m') if r['attempts'] else None
  et='—' if e is None else f'{e*1000:.1f}';rt='—' if res is None else f'{res*1000:.1f}'
  report+=f"| {r['initial']} | {r.get('navigation',{}).get('navigation_success',False)} | {r['success']} | {et} | {rt} |\n"
 service=S/'service_run/resources.json'
 if service.exists():
  r=json.loads(service.read_text());report+=f"\n服务稳定段无swap检查：{r['valid_without_swap']}。模型首加载{r['load_seconds']:.2f}秒。\n"
else:report+='\n联合测试尚未完成，不填写成功率。\n'
resource_rows=[]
for label,resource,tegra in [('时延筛查',x,S/'results/tegrastats.log'),('服务联调',json.loads((S/'service_run/resources.json').read_text()),S/'service_run/tegrastats.log')]:
 samples=resource['stable_samples'];rss=max(v.get('VmHWM',0) for v in samples)/2**20;cg=max(v.get('cgroup_memory_current') or 0 for v in samples)/2**20
 ram=[int(v) for v in re.findall(r'RAM (\d+)/',tegra.read_text())] if tegra.exists() else []
 resource_rows.append(f"| {label} | {rss:.0f} | {cg:.0f} | {resource['peak_cuda_allocated']/2**20:.0f} | {max(ram) if ram else '—'} |")
report+='\n资源摘要（MiB，不同口径不能相加，Jetson使用共享物理内存）：\n\n| 阶段 | 进程VmHWM | cgroup采样最大 | CUDA分配峰值 | tegrastats整板RAM最大 |\n|---|---:|---:|---:|---:|\n'+'\n'.join(resource_rows)+'\n'
profile=S/'profile_one_root_v2/SUMMARY.json'
if profile.exists():
 pr=json.loads(profile.read_text());report+='\n## 4. 单次算子剖析\n\n'
 report+=f"在正式时延之外，另用一张验证图做诊断，生成{pr['output_tokens']}个token，捕获{pr['kernel_events']}个CUDA内核事件。内核时长累加{pr['summed_kernel_microseconds']/1e6:.2f}秒；此值可能含重叠，不能与CPU时间直接相加。开启profiler后的推理时间不混入上表P50/P95。\n\n"
 report+='普通用户首次受CUPTI权限限制，没有GPU事件；保留该记录。随后只给这一受限剖析进程管理员权限，没有修改驱动全局权限。第一次管理员启动因门控文件权限未开始剖析，另存失败；修复门控后完成。\n\n'
 report+='按CPU表，cudaLaunchKernel共41431次、自耗时约1.08秒；4bit矩阵向量计算约1.11秒CUDA时间，另有解量化、拷贝与大量小算子。量化解码与调度开销是下一步候选优化点，但不能直接扣掉这些时间宣称已达标。当前ali-jetson未安装Triton。已找到官方Jetson AI Lab匹配的3.4.0预编译包，但安装前核对到当前Transformers的NF4量化器没有覆盖is_compileable=False，现有自动编译入口会拒绝该组合；因此没有安装或绕过检查。该限制不等于所有NF4模型都无法手工编译，需另做兼容性验证。\n\n'
 report+='后续可在独立环境验证静态KV缓存与编译解码，先确认输出一致，再测速度和完整精度；不再仅靠换更大模型或缩图。接口参考[PyTorch 2.8 profiler](https://docs.pytorch.org/docs/2.8/profiler.html)、[NVIDIA CUPTI权限说明](https://developer.nvidia.com/nvidia-development-tools-solutions-err-nvgpuctrperm-cupti)、[Transformers静态缓存与编译](https://huggingface.co/docs/transformers/v4.57.1/en/llm_optims)。\n'

report+='''
## 5. 文件与结论

时延原始结果在results/records.jsonl，统计和资源在results/SUMMARY.json，实际负载在results/tegrastats.log。服务请求与资源在service_run/。联合PC仿真在../../closed_loop_20261115/results/jetson_joint_v1_*，每回合含视频、预测、轨迹反馈及实际误差。相关代码在scripts/；PC的remote_grounding_worker.py是无模型客户端。

本阶段已经验证或测试的是NF4服务能否运行、尺寸对延迟的影响以及PC/板端功能联通。未完成MAXN SUPER、CUDA内核优化、新尺寸350帧精度、里程碑板端准确率与实时联合控制，也没有把当前结果写成“端侧实时部署完成”。下一轮优先定位解码开销与部署运行时，再决定是否值得做完整精度/性能回归。
'''
(S/'REPORT.md').write_text(report)
sys.path.insert(0,'/tmp/mobile-qwen-doclibs');import markdown
css='@page{size:A4;margin:17mm}body{font-family:"Noto Sans CJK SC",sans-serif;font-size:9.5pt;line-height:1.65;color:#263448}h1{font-size:23pt}h2{font-size:15pt;break-after:avoid}table{border-collapse:collapse;width:100%;font-size:8pt}td,th{border:1px solid #bbc8d3;padding:5px}th{background:#e9eff5}code{overflow-wrap:anywhere}'
h=S/'REPORT.html';h.write_text('<meta charset="utf-8"><style>'+css+'</style>'+markdown.markdown(report,extensions=['tables','fenced_code']))
subprocess.run(['/usr/bin/google-chrome','--headless','--no-sandbox','--disable-gpu','--no-pdf-header-footer','--print-to-pdf='+str(S/'REPORT.pdf'),h.as_uri()],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
with zipfile.ZipFile(S/'SOURCE.zip','w',zipfile.ZIP_DEFLATED) as z:
 for f in (S/'scripts').glob('*.py'):z.write(f,str(f.relative_to(ROOT)))
 for f in [S/'README.md',S/'input_manifest.json',S.parent/'stage2_20261114/scripts/runtime.py',C/'scripts/remote_grounding_worker.py']:
  z.write(f,str(f.relative_to(ROOT)))
 z.writestr('DEPENDENCIES.md','需要已安装ali-jetson环境、stage2 NF4模型及LoRA；权重与环境不在代码包中。PC仿真依赖closed_loop_20261115源码包及其README列出的依赖。SSH密码仅从运行环境提供。')
print(S/'REPORT.pdf')
