"""Build one combined progress PDF and one source-only archive."""
from pathlib import Path
import ast,sys,json,re,subprocess,zipfile,hashlib
sys.path.insert(0,'/tmp/vla-report-tools')
import markdown
T=Path(__file__).resolve().parents[1];R=T.parents[1];O=R/'deliverables/20261024_阿里增强版';O.mkdir(exist_ok=True)
style=next(ast.literal_eval(n.value) for n in ast.parse((R/'experiments/alibaba_pc_revision/scripts/build_delivery.py').read_text()).body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='style' for x in n.targets))
def render(md,html,pdf,title):
 body=markdown.markdown(md,extensions=['tables','fenced_code']);html.write_text('<!doctype html><meta charset="utf-8"><title>'+title+'</title><style>'+style+'pre{break-inside:avoid}</style>'+body)
 subprocess.run(['/usr/bin/google-chrome','--headless','--no-sandbox','--disable-gpu','--no-pdf-header-footer','--print-to-pdf='+str(pdf),html.as_uri()],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
def pages(p):return int(re.search(r'Pages:\s+(\d+)',subprocess.check_output(['pdfinfo',str(p)],text=True)).group(1))
md=(T/'REPORT.md').read_text()
for image in re.findall(r'!\[[^\]]*\]\(([^)]+)\)',md):assert (T/image).exists(),image
render(md,T/'REPORT.html',T/'REPORT.pdf','端侧视觉部署与策略验证扩展')
old=R/'experiments/alibaba_pc_revision/REPORT.pdf';new=T/'REPORT.pdf';front=2
summary=json.loads((T/'RESULT_SUMMARY.json').read_text());speed=summary['paired_vision'][0]['p50_ms']/summary['paired_vision'][2]['p50_ms']
cover='''# 具身视觉与策略算法阶段进展

阿里方向 · 2026年10月24日

## 摘要

本次把上一阶段PC端改进与本轮Jetson实测合在一起。10月20日的PC部分处理进程超时、图像传输、LoRA验证与ACT选模；10月24日的端侧部分接入真实腕部相机，完成检测和深度的TensorRT部署、Qwen连续观察以及4B目标指令适配器的端侧迁移，并扩大ACT验证布局与训练种子。

固定输入下，TensorRT检测＋深度处理的中位速度约为同一Jetson上的PyTorch FP32后端的{speed:.2f}倍。这里比较的是Jetson上的两种后端，不是工作站与Jetson的速度。Qwen、目标指令和策略仿真的结果分别列出，真实场景判断仍有失败用例。

本次真机提供真实相机图像，没有启用机械臂电机。视觉部署、语言目标与ACT仿真是三个已经测试的模块，还没有组成真实机械臂自主抓取闭环。

## 目录

| 部分 | 内容 | 汇总PDF页序 |
|---|---|---|
| 阅读说明 | 摘要、目录及源码索引 | 1–{front} |
| 第一部分 | PC端推理链路与训练评估改进 | {a_start}–{a_end} |
| 第二部分 | 端侧视觉部署与策略验证扩展 | {b_start}–{b_end} |

原阶段报告正文保留，页脚沿用原编号；上表按PDF阅读器显示的连续页序查阅。

<div style="break-before:page"></div>

## 源码目录与报告对应关系

```text
README.md
experiments/
  alibaba_pc_revision/
    scripts/                 LoRA训练验证、ACT选模及公共执行模块
    README.md
  alibaba_jetson_revision/
    scripts/
      jetson_vision.py        相机、视觉推理及引擎导出
      paired_vision.py        同图像同尺寸后端对照
      jetson_qwen.py          最新帧读取与连续视觉观察
      quantize_qwen.py        独立NF4模型转换
      jetson_adapter_eval.py  原4B适配器的端侧测试
      bridge_eval.py         端侧输出到同场景仿真策略
      bench_journal.py       日志编码开销对照
      train_act_expanded.py   扩大布局及重复训练
      run_act_expanded.sh
    README.md
    requirements-jetson-reference.txt
  desktop_pickplace/          训练评测使用的仿真与任务代码
  alibaba_baseline/           历史兼容入口
  alibaba_extension/          历史兼容入口
perception/
  qwen_scene/                worker客户端、共享帧与视觉模块
  web_workbench/             既有工作台
SO101/code/lerobot/           原版本依赖及许可
sensors/imu/                 既有传感器读取模块
```

相关代码在上述同名目录。报告正文各节另列具体入口；执行环境与模型路径见对应README。

源码包包含程序、配置、必要仿真资产及上游许可，不包含环境、权重、TensorRT引擎、训练数据、输入照片或逐步结果。PC、Jetson、ROS与仿真环境分开使用；现有Jetson的an-project环境未修改。模型权重和完整实验结果保留在本地，不把参考版本表称为跨平台一键部署。
'''
for _ in range(3):
 text=cover.format(speed=speed,front=front,a_start=front+1,a_end=front+pages(old),b_start=front+pages(old)+1,b_end=front+pages(old)+pages(new))
 (T/'SUMMARY.md').write_text(text);render(text,T/'SUMMARY.html',T/'SUMMARY.pdf','阶段进展摘要与目录')
 n=pages(T/'SUMMARY.pdf')
 if n==front:break
 front=n
else:raise RuntimeError('pagination did not settle')
pdf=O/'阿里_增强版阶段进展.pdf';subprocess.run(['pdfunite',str(T/'SUMMARY.pdf'),str(old),str(new),str(pdf)],check=True)
with zipfile.ZipFile(R/'deliverables/20261020_PC_阿里与字节/阿里_PC源码.zip') as z:entries={n:z.read(n) for n in z.namelist()}
# All experiment implementations, but no report-generation or agent orchestration helpers.
for p in (T/'scripts').iterdir():
 if p.is_file() and p.suffix in ['.py','.sh'] and p.name not in ['build_delivery.py','write_report.py']:entries[str(p.relative_to(R))]=p.read_bytes()
for name in ['README.md','requirements-jetson-reference.txt']:p=T/name;entries[str(p.relative_to(R))]=p.read_bytes()
entries['README.md']='''# 具身视觉与策略算法阶段源码

对应《阿里_增强版阶段进展.pdf》。

- experiments/alibaba_pc_revision：PC推理链路、训练验证与公共执行模块。
- experiments/alibaba_jetson_revision：端侧相机、TensorRT、Qwen与4B适配器测试，以及扩大ACT验证的训练入口。
- experiments/desktop_pickplace：沿用的仿真任务与数据处理代码。

请先阅读两个实验目录的README。解压根目录包含experiments、perception和SO101，保留这些相对目录结构。权重、引擎、缓存和数据不随纯源码包提供；需要按说明准备。不同平台使用不同环境，勿把Jetson和PC依赖混装。脚本保留模型和输出目录参数；个别既有启动脚本需要调整本机环境路径。
'''.encode()
code=O/'阿里_增强版源码.zip'
with zipfile.ZipFile(code,'w',zipfile.ZIP_DEFLATED) as z:
 for name,data in sorted(entries.items()):
  assert not name.startswith('/') and '..' not in Path(name).parts
  assert not name.endswith(('.pyc','.npz','.pt','.safetensors','.engine','.csv','.bin','.pdf','.png'))
  if name.endswith('.py'):compile(data,name,'exec')
  z.writestr(name,data)
with zipfile.ZipFile(code) as z:assert z.testzip() is None
manifest={'pages':pages(pdf),'front_pages':front,'pc_pages':pages(old),'new_pages':pages(new),'source_entries':len(entries),'files':[{'name':p.name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in [pdf,code]]}
(O/'清单.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2));print(json.dumps(manifest,ensure_ascii=False,indent=2))
