"""Keep detailed stage reports intact; prepend a cover and combine source bundles."""
from pathlib import Path
import subprocess,sys,re,zipfile,hashlib,json,shutil
C=Path(__file__).resolve().parents[1];ROOT=C.parents[2];S=ROOT/'experiments/mobile_qwen/jetson/stage3_20261115';D=ROOT/'deliverables/20261115_阿里阶段更新';D.mkdir(exist_ok=True)
def pages(file):
 text=subprocess.check_output(['pdfinfo',str(file)],text=True);return int(re.search(r'Pages:\s+(\d+)',text)[1])
n=pages(C/'REPORT.pdf');m=pages(S/'REPORT.pdf')
cover=f'''# 阿里具身项目：本次阶段更新

2026年11月15日

## 摘要

本次把Qwen可供性定位从离线评测接入同场景动作执行：Nav2停靠、图像定位、深度反投影、MoveIt规划和MuJoCo实际关节反馈已经贯通。随后测试无重置交接π0.5完成整任务，并部署Jetson视觉服务，与PC仿真联调。

| 内容 | 结果 | 范围 |
|---|---|---|
| 可供性驱动预接触 | LoRA 4/4，基座0/4，真值定位4/4 | 2个已有初态×2种相机平移；单厨房 |
| 预接触后交接π0.5 | 整任务2/4 | 含1个开发探针、3个重复性回合 |
| 失败初态原生对照 | 1/1通过，320步 | 事后单例诊断，未证明因果 |
| Jetson视觉→PC物理执行 | 预接触3/3 | 2B NF4 LoRA，512像素 |
| 15W端侧时延筛查 | 三种尺寸均未达≤3秒目标 | 验证集5帧，每尺寸20次；非精度评测 |

LoRA到位误差中位10.4mm，基座73.3mm。把手预接触接口可用，但不适合默认前置到所有π0.5任务，现有原生闭环继续保留。Qwen仍出现中间里程碑与几何区域标注不一致，不能因最终未误停就认为判断可靠。

端侧缩图收益有限。一次独立剖析显示大量小内核调度与量化解码开销，下一轮应先验证部署运行时兼容性，再投入全量精度/速度回归。本次没有连接实体机械臂，没有修改已有正式结果或安装新的板端依赖。

## 目录

1. 可供性定位接入同场景导航与机械臂执行 …… 第2页起，共{n}页
2. Jetson可供性服务：时延与PC仿真联调 …… 第{n+2}页起，共{m}页

后续两部分保留本次详细阶段报告正文。源码包保留项目目录层级，不含模型、环境及全量数据；依赖和运行方式见各目录README。videos/另放完整操作成功示例和Jetson联合预接触示例。
'''
(C/'DELIVERY_COVER.md').write_text(cover)
sys.path.insert(0,'/tmp/mobile-qwen-doclibs');import markdown
html=C/'DELIVERY_COVER.html';html.write_text('<meta charset="utf-8"><style>@page{size:A4;margin:17mm}body{font-family:"Noto Sans CJK SC",sans-serif;font-size:10pt;line-height:1.65;color:#253448}h1{font-size:24pt}h2{font-size:16pt}table{width:100%;border-collapse:collapse;font-size:9pt}td,th{border:1px solid #bbc7d4;padding:7px}th{background:#eaf0f5}</style>'+markdown.markdown(cover,extensions=['tables']))
coverpdf=C/'DELIVERY_COVER.pdf';subprocess.run(['/usr/bin/google-chrome','--headless','--no-sandbox','--disable-gpu','--no-pdf-header-footer','--print-to-pdf='+str(coverpdf),html.as_uri()],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE);assert pages(coverpdf)==1
pdf=D/'阿里_阶段进展汇总.pdf';subprocess.run(['pdfunite',str(coverpdf),str(C/'REPORT.pdf'),str(S/'REPORT.pdf'),str(pdf)],check=True);assert pages(pdf)==1+n+m
with zipfile.ZipFile(D/'阿里_阶段源码.zip','w',zipfile.ZIP_DEFLATED) as out:
 entries={}
 for file in [C/'SOURCE.zip',S/'SOURCE.zip']:
  with zipfile.ZipFile(file) as src:
   for name in src.namelist():
    data=src.read(name)
    if name in entries:assert entries[name]==data,name
    else:entries[name]=data
 for name,data in sorted(entries.items()):out.writestr(name,data)
 out.writestr('STAGE_README.md','本次只包含新增阶段及其调用到的源码。PC主入口：experiments/mobile_qwen/closed_loop_20261115/README.md；Jetson主入口：experiments/mobile_qwen/jetson/stage3_20261115/README.md。模型、虚拟环境、第三方LIBERO/openpi、完整数据与TF2二进制overlay均需另备。验证图片由原验证集input_manifest.json指定，不含在代码包中。本包不声明脱离依赖即可运行。')
 assert out.testzip() is None
videos=D/'videos';videos.mkdir(exist_ok=True)
for src,name in [(C/'results/combined_probe1_nominal_i00_lora/rollout.mp4','PC_完整任务成功示例.mp4'),(C/'results/jetson_joint_v1_nominal_i02_lora/rollout.mp4','Jetson感知_PC预接触示例.mp4')]:shutil.copyfile(src,videos/name)
manifest={str(f.relative_to(D)):hashlib.sha256(f.read_bytes()).hexdigest() for f in D.rglob('*') if f.is_file() and f.name!='SHA256.json'};(D/'SHA256.json').write_text(json.dumps(manifest,indent=2));print(D);print('PDF pages',1+n+m,'source entries',len(entries)+1)
