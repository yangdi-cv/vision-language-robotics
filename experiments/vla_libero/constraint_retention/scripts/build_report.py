"""Build this stage's report and a minimal, testable source delivery."""
import hashlib,json,shutil,subprocess,tempfile,zipfile
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from markdown_it import MarkdownIt
S=Path(__file__).resolve().parents[1];V=S.parent;ROOT=V.parents[1]
D=ROOT/'deliverables/20261113_阿里_约束与多视角检查'
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'

def main():
    D.mkdir(parents=True,exist_ok=True)
    summaries={n:json.loads((S/n).read_text()) for n in ['PILOT_GATE.json','VIEW_SUMMARY.json','LIVE_SUMMARY.json','MINIMAL_VIEW_SUMMARY.json','AUDIT.json']}
    assert summaries['AUDIT.json']['passed']
    # Actual same-state camera pair, no bounding-box or object-position hints.
    folder=S/'view_audit/images/development/task01_target0_init00'
    canvas=Image.new('RGB',(1060,600),'#eef2f5');draw=ImageDraw.Draw(canvas);font=ImageFont.truetype(FONT,23)
    for i,(name,label) in enumerate([('agent_0160.png','第三人称：篮内目标被边缘遮挡'),('wrist_0160.png','腕部视角：篮内蓝色目标可见')]):
        canvas.paste(Image.open(folder/name).convert('RGB').resize((512,512)),(12+530*i,50));draw.text((12+530*i,10),label,font=font,fill='#203648')
    draw.text((12,565),'同一恢复状态 · 开发初态0 · 第160步 · 原始相机图像',font=font,fill='#203648')
    canvas.save(S/'figures/camera_pair.png')
    text='''# 选择性搬运：约束透传与多视角完成检查

阿里具身项目 · 阶段补充报告  
2026年11月13日 · PC / RTX 4090 · LIBERO 仿真

## 摘要

上一阶段将原本的双物体入篮任务改为“只搬指定一件，另一件保持原位”，使停止时机成为实际考核内容。本阶段继续检查两个可能的改进方向：完整任务约束是否应直接传给低层策略，以及腕部视角能否缓解完成检查中的遮挡。

完成了 **32个新的策略闭环回合**：12个约束透传对照、12个新初态观察回合、8个多视角在线回合。另外复用了16条旧结果作为对应参照，没有重复计算成新增回合。完成615次冻结方案的离线视觉检查，以及100次开发集提示词复查。

目前两条改动都没有取得可靠的闭环收益。完整约束对照由3/12变为2/12；严格检查下，重复第三人称图与腕部图两组在线试验均为0/4。腕部相机确实改善部分目标的可见性，但更严格的检查问法同时压低了完成召回。保留原问法的小试恢复了部分召回，也增加了误报。由此，本轮保留原有实现，不替换正式检查器，也不扩大无收益的实验分支。

## 目录

1. 问题与实验范围
2. 完整约束透传
3. 多视角离线对照
4. 多视角在线验证
5. 提示词复查与解释
6. 实现、复现与后续工作

## 1. 问题与实验范围

目标物体分别来自汤罐头/番茄酱、奶酪/黄油两组场景，并交换指定搬运的目标。最终计分同时要求：目标物体在篮内稳定放置且不接触夹爪，保留物体的累计最大位移不超过30毫米。停止后继续保持25步，使用最后10步检查稳定性。此处为自定义选择性任务得分，不是原 LIBERO-Long 的整任务成功率。

原Long的成功信号不触发控制停止。策略每次预测10步动作，只执行前5步；Qwen检查采用夹爪张开门控、间隔10步、连续两票。物体关系和接触真值只用于记录、评分与离线标签，不传给模型。两组在线视角试验使用相同的正向技能指令、动作噪声、策略输入和停止规则。

相关代码在 `scripts/design.py`、`scripts/episode.py`；评分与控制复用 `../selective_execution/scripts/protocol.py`、`control.py`。

## 2. 完整约束透传

原指令只给低层策略“拿起指定物体并放入篮子”。新指令保留此前缀，并追加“另一件保持原位、不要移动其他物体、放好后停止”。为避免把截断误认为策略不理解约束，先调用实际策略分词器验证：最长38词元，长度上限200，全部约束均被保留。

开发集含4个场景条件，每个条件比较持续执行、定时停止、视觉停止，共12个配对条件。正向指令结果取自已冻结旧回合，新约束指令运行12个新回合。

| 指令形式 | 选择性成功 | 配对新增成功 | 配对损失 |
|---|---:|---:|---:|
| 正向技能指令 | 3/12 | — | — |
| 追加全局约束 | 2/12 | 0 | 1 |

追加约束没有改善本次开发样本，反向目标也没有新增成功。按运行前声明的条件，不启动72回合的新初态扩展。这是开发筛选结果，样本规模不足以支持普遍的语言能力判断。

相关代码在 `scripts/run.py`、`execution.py`、`token_preflight.py`，设置与结果见 `PROTOCOL.json`、`TOKEN_PREFLIGHT.json`、`PILOT_GATE.json`。

## 3. 多视角离线对照

从开发初态0的4条持续执行回合中取50个有效检查时刻；另外在初态8–10运行12条新观察回合，得到155个检查时刻。每个时刻恢复相同仿真状态，分别渲染512像素第三人称与腕部图像，保证两个候选方案看到同一物理状态。标签对应模型所选目标；若选择目标错误，不用正确答案暗中替换。

![同一状态的相机比较](figures/camera_pair.png)

原两帧检查作为参考。主要视角对照均输入三张图，提示词完全一致：开始时第三人称、当前第三人称、当前辅助视图。对照组的第三张重复当前第三人称，实验组替换为腕部图。因而图像数量、尺寸、生成上限和提示词一致，差别仅为第三张图的像素。

严格提示词要求确认目标已经放到目的地并释放；看不清时回答unknown。原两帧参考使用此前问法和布尔输出，其与严格三帧方案的差别不只在相机，不能将差值全部归因于腕部视角。

| 数据 | 检查方式 | 真正确认/正例 | 误确认/负例 | 召回 | 误报率 | unknown |
|---|---|---:|---:|---:|---:|---:|
| 开发50帧 | 原两帧参考 | 6/18 | 4/32 | 33.3% | 12.5% | 0 |
| 开发50帧 | 严格三帧·重复图 | 0/18 | 0/32 | 0% | 0% | 2 |
| 开发50帧 | 严格三帧·腕部 | 0/18 | 0/32 | 0% | 0% | 1 |
| 新初态155帧 | 原两帧参考 | 38/66 | 11/89 | 57.6% | 12.4% | 0 |
| 新初态155帧 | 严格三帧·重复图 | 0/66 | 0/89 | 0% | 0% | 0 |
| 新初态155帧 | 严格三帧·腕部 | 1/66 | 0/89 | 1.5% | 0% | 6 |

615次检查全部返回可解析结果。严格方案的低误报伴随几乎为零的召回，主要输出为not_done，不能作为有效停止器。腕部图能展示部分遮挡物体，但可见不等于模型能稳定确认。

同一回合中的多帧高度相关；表中的帧数用于描述分类表现，不等同于205个独立任务回合。标签来自物理谓词，并未逐帧做人类可见性标注。恢复后重新计算物理缓存可能与在线瞬间略有差别，因此在线表现仍须单独验证。

相关代码在 `scripts/prepare_views.py`、`view_checker.py`、`view_worker.py`、`view_run.py`；数据索引、原始回复与统计在 `view_audit/`、`VIEW_SUMMARY.json`。

## 4. 多视角在线验证

离线结果没有成为阻止在线小试的门槛。两种三帧检查器实际接入相同策略与停止控制，各运行4个配对回合，比较真实的选择性得分。

| 方式 | 新运行回合 | 选择性成功 | 目标最终稳定入篮 | 保留物体未超限 |
|---|---:|---:|---:|---:|
| 严格三帧·重复图 | 4 | 0/4 | 3/4 | 0/4 |
| 严格三帧·腕部 | 4 | 0/4 | 3/4 | 0/4 |
| 定时停止参照（旧回合） | 0，复用4 | 2/4 | 2/4 | 2/4 |

两组新回合在分歧前的策略输入哈希相同，动作最大差为0。重复图与腕部方案都没有及时阻止对保留物体的后续动作。腕部方案没有净增成功，按既定条件不再扩到36个新回合。旧定时结果仅作为参照，没有声明其与新进程逐步完全一致。

原始回合保存在 `episodes/live_pilot_duplicate_control/`、`episodes/live_pilot_wrist/`；在线入口在 `scripts/live.py`，汇总见 `LIVE_SUMMARY.json`。

## 5. 提示词复查与解释

严格问法可能把“看清完成”变成了过强的判定条件。为区分这一影响，另做一次有界开发复查：尽量保留原有的目的地前后变化问法与布尔输出，仅补充第三张辅助图的说明。仍使用相同的三图重复对照，不换模型，不训练新参数。

| 开发50帧 | 真正确认/18 | 误确认/32 | 召回 | 误报率 |
|---|---:|---:|---:|---:|
| 原问法扩展·重复图 | 3 | 1 | 16.7% | 3.1% |
| 原问法扩展·腕部 | 5 | 2 | 27.8% | 6.3% |

这100次调用只用开发数据。腕部图带来2个额外正确确认，也带来1个额外误报；未在新初态上验证，未接入正式闭环。结果说明提示词严格程度会显著影响完成召回，也提示视角扩展需要与判断标准一起校准。继续随机改提示词不足以证明实际收益，本轮到此收尾。

相关代码在 `scripts/minimal_view.py`，完整提示词与调用数在 `minimal_view/PROTOCOL.json`，统计在 `MINIMAL_VIEW_SUMMARY.json`。

## 6. 实现、复现与后续工作

此次沿用 openpi 的π0.5策略接口与LIBERO相机/状态接口，依赖、许可与此前设计比较沿用上一阶段 `RESEARCH.md` 和项目 `third_party/sources.json`。没有另写底层策略或仿真控制器。

代码分为三个冻结实验：约束对照、多视角离线检查、多视角在线检查。每个协议记录源码哈希和提示词版本；当前实现与归档源码逐项核对。输入审计覆盖32条新增回合、1808次策略调用、615次检查图像；八项CPU接口测试通过。本次没有另做独立物理重评分，最终任务分数使用保存的逐步轨迹评分。

目录结构：

```text
constraint_retention/
  README.md、REPORT.md、requirements-offline.txt
  scripts/          运行、采样、检查、分析与构建
  tests/            不依赖GPU的接口测试
  PROTOCOL.json     约束透传方案
  VIEW_PROTOCOL.json / LIVE_PROTOCOL.json
  *_SUMMARY.json    分类与闭环统计
  *_GATE.json       是否扩大实验的决定
  archive*/         对应冻结源码
  plans/            目标选择记录
  episodes/         本地完整视频、输入、动作与结果
  view_audit/       同状态图像、逐条回复与离线标签
  minimal_view/     开发集提示词复查
```

本阶段缩小了后续方向：仅增加约束文字或相机数量，尚不能让当前停止器可靠工作。下一步应先选可明确识别的放置事件，在独立回合上同时检查误切换与停止延迟，再决定是否扩大数据或微调。低层反向目标的失败也需与停止器分开处理，避免把执行技能不足归因于视觉判断。

模型仍是原Qwen3-VL-4B基座与π0.5，没有新增LoRA权重。本阶段不包含端侧部署、导航与抓取统一场景或真机测试。
'''
    (S/'REPORT.md').write_text(text)
    html=MarkdownIt().enable('table').render(text)
    css='@page{size:A4;margin:17mm}body{font:10pt/1.65 "Noto Sans CJK SC","Noto Sans CJK JP",sans-serif;color:#24374a}h1{font-size:23pt;color:#153b58}h2{font-size:15pt;break-after:avoid;border-bottom:1px solid #ccd8e3}table{width:100%;border-collapse:collapse;font-size:8pt;break-inside:avoid}td,th{padding:5px;border:1px solid #ccd8e3}th{background:#edf3f8}img{width:100%}pre{font-size:8pt;white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f5f8;padding:10px}code{overflow-wrap:anywhere}p:has(img){break-inside:avoid}'
    (S/'REPORT.html').write_text('<!doctype html><meta charset="utf-8"><style>'+css+'</style>'+html)
    with tempfile.TemporaryDirectory() as temp:
        subprocess.run(['google-chrome','--headless','--no-sandbox','--disable-gpu',f'--user-data-dir={temp}','--no-pdf-header-footer',f'--print-to-pdf={S/"REPORT.pdf"}',(S/'REPORT.html').as_uri()],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=90)
    shutil.copyfile(S/'REPORT.pdf',D/'约束透传与多视角检查_阶段报告.pdf')
    source=set(S.glob('*.json'))|set(S.glob('*.md'))|set(S.glob('*.txt'))|set((S/'scripts').glob('*.py'))|set((S/'tests').glob('*.py'))|set((S/'figures').glob('*.png'))
    for archive in ['archive','archive_view','archive_live']:source.update((S/archive).rglob('*.py'))
    for name in ['PROTOCOL.json','VIEW_PROTOCOL.json','LIVE_PROTOCOL.json']:
        source.update(ROOT/p for p in json.loads((S/name).read_text())['sources'])
    source.update((S/'plans').rglob('*.json'));source.update((S/'episodes').glob('*/*/*/result.json'))
    source.update(S/'minimal_view'/n for n in ['PROTOCOL.json','STATUS.json','predictions.jsonl'])
    source.update(S/'view_audit'/n for n in ['development.json','test.json','predictions.jsonl','DATA_LOCK.json'])
    source.update([V/'scripts/python_shared',V/'third_party/sources.json',V/'third_party/openpi/LICENSE',V/'third_party/libero/LICENSE',V/'environment.json',V/'README.md',V/'selective_execution/PROTOCOL.json',V/'selective_execution/RESOURCES.json',V/'selective_execution/RESEARCH.md',ROOT/'perception/qwen_scene/requirements-local.lock.txt'])
    manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source)}
    with zipfile.ZipFile(D/'约束与多视角检查_源码.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(source):z.write(p,str(p.relative_to(ROOT)))
        z.writestr('PACKAGE_SHA256.json',json.dumps(manifest,indent=2))
    (D/'文件清单.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in D.iterdir() if p.is_file() and p.name!='文件清单.json'},ensure_ascii=False,indent=2))
    print(D)

if __name__=='__main__':main()
