"""Curate archived stage sections into editable LaTeX; build.sh never reruns this migration."""
from pathlib import Path
import re,json,hashlib,subprocess,shutil,math
R=Path(__file__).resolve().parents[3];D=R/'reports/alibaba_engineering';PANDOC=Path('/home/yd/.local/share/vla-report-tools/pandoc/pandoc')
stages=json.loads((D/'outline.json').read_text());assets=json.loads((D/'asset_manifest.json').read_text());coverage=[];missing=[];stats={};commands=[]
old_assets=list(assets);known={(a['stage'],a['sha256']):a['local'] for a in assets};used_figures={};used_videos={}
for s in stages:used_figures[s['id']]=set();used_videos[s['id']]=set()
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def tex(s):
 return s.replace('\\',r'\textbackslash{}').replace('&',r'\&').replace('%',r'\%').replace('#',r'\#').replace('_',r'\_').replace('$',r'\$').replace('π0.5',r'\pifive{}')
def plain(inlines):
 if isinstance(inlines,list):return ''.join(plain(x) for x in inlines)
 if isinstance(inlines,dict):
  if inlines.get('t')=='Str':return inlines['c']
  if inlines.get('t') in ['Space','SoftBreak','LineBreak']:return ' '
  if inlines.get('t')=='Code':return inlines['c'][1]
  return plain(inlines.get('c',[]))
 return ''
def section_blocks(path):
 text=(R/path).read_text();lines=text.splitlines(keepends=True);heads=[];fence=False
 for i,l in enumerate(lines):
  if re.match(r'^\s*(```|~~~)',l):fence=not fence
  if not fence and re.match(r'^## (?!#)',l):heads.append((i,l[3:].strip()))
 blocks=[]
 for j,(i,h) in enumerate(heads):blocks.append((h,''.join(lines[i:(heads[j+1][0] if j+1<len(heads) else len(lines))])))
 return text,blocks
parts={s['id']:[] for s in stages}
def manual(sid,text,title):
 parts[sid].append({'source':None,'title':title,'markdown':text})
def select(sid,path,keys=None,title=None):
 full,blocks=section_blocks(path)
 if keys is None:
  keys=[h for h,_ in blocks if h not in ['摘要','目录'] and not re.search(r'结论.*(后续|下一步)|下一阶段|代码目录|文件与运行|复现与源码目录|文件目录和启动方式|代码、权重与后续|代码、复现与后续',h)]
 selected=[]
 for key in keys:
  match=[(h,b) for h,b in blocks if h==str(key) or re.match(r'^'+re.escape(str(key))+r'[.．、 ]',h)]
  if not match:raise ValueError((sid,path,key,[h for h,_ in blocks]))
  for h,b in match:
   selected.append((h,b));coverage.append({'stage':sid,'source':path,'section':h,'source_sha256':digest(R/path)})
 text='\n\n'.join(b for _,b in selected)
 if title:
  text='## '+title+'\n\n'+re.sub(r'^(#{2,5}) ',lambda m:'#'+m[1]+' ',text,flags=re.M)
 parts[sid].append({'source':path,'title':title or path,'markdown':text})
def excerpt(sid,path,start,end,title):
 full=(R/path).read_text();a=full.index(start);b=full.index(end,a) if end else len(full);body=full[a:b]
 body=re.sub(r'^#{2,5} .*\n','',body,flags=re.M)
 coverage.append({'stage':sid,'source':path,'section':title,'source_sha256':digest(R/path),'selection_start':start,'selection_end':end})
 parts[sid].append({'source':path,'title':title,'markdown':'## '+title+'\n\n'+body})
manual('s01',r'''## 硬件、环境与可重复启动

最早的软件验证基于Ubuntu 22.04和RTX 4090工作站。厂家LeRobot 0.3.4工程保留，运行包装脚本隔离ROS2的Python搜索路径。环境名为hiwonder-so101；环境曾随旧目录迁移，部分入口仍引用旧绝对路径，因此采用明确解释器路径，不假定普通conda或pip入口都能工作。

| 项目 | 2026年10月12日记录 |
|---|---|
| 主臂 | 黑色SO101，作为遥操作输入 |
| 从臂 | 白色SO101，执行目标关节位置 |
| 黑臂稳定串口标识 | usb-1a86_USB_Single_Serial_5CE5121669-if00 |
| 白臂稳定串口标识 | usb-1a86_USB_Single_Serial_5CE5122996-if00 |
| 临时设备号 | 主臂ttyACM0、从臂ttyACM1；重插后可能变化 |
| 摄像头 | icSpring，640×480；当时图像节点为video0 |
| 代码与校准 | SO101/code/lerobot；SO101/calibration/ |

串口角色由逐个连接识别，稳定标识用于启动；设备号不写成永久对应。主从臂电机校准值与保存文件做只读核对后，再由现场操作确认遥操作。这里建立的是硬件通信与动作接口基线。

## 示教记录与回放接口

记录工具同时保存图像、实测关节状态、发送目标和主机时间戳。执行记录中的动作是实际下发的目标，不以主臂读数代替。相机先检查、记录准备后才开始采集；校准不一致则取消。一次完整示教已录制并回放，后续回放还加入起点对齐与分段确认。

不同时间有三种动作来源：人移动主臂产生目标、程序重放已录目标、模型依据当前图像与状态预测目标。三者共用串口与电机接口，但不共用“模型自主完成”的结论。固定轨迹的物体转移过程后来被反馈存在碰桌，因此不能保留为无碰撞通过案例。

## 图像旋转与手腕偏置分别记录

画面旋转改变显示及模型所见像素，手腕偏置改变从臂关节目标，两者是独立变换。临时腕部映射为负90度时，原校准文件哈希保留；真实示教记录实际下发后的动作，因此训练和推理不能再次叠加同一个偏置。鼠标垫任务的原始图像以rotation=0记录，不能只依据网页显示方向推断训练图像方向。

| 信息 | 含义 | 需要保持一致的对象 |
|---|---|---|
| 图像旋转 | 原始传感器像素到显示或模型像素的变换 | 图像、框、点、深度与相机内参 |
| 腕部偏置 | 主从关节映射中的固定角度差 | 实测状态、发送动作和归一化统计 |
| 电机校准 | 编码器零位与有效范围 | 硬件、保存文件及运行配置 |

编码器增量与角度的换算按4096计数对应一圈处理，即角度变化约为计数变化乘以360/4096。这个换算描述关节角增量，不等于末端空间高度或碰撞间隙。''','硬件记录与接口说明')
coverage.append({'stage':'s01','source':'SO101/UBUNTU.md','section':'环境、串口与校准记录','source_sha256':digest(R/'SO101/UBUNTU.md')})
manual('s02',r'''## 连续观察的取帧方式

工作台由相机采集、检测与深度处理、场景描述三类工作组成。相机持续更新最新帧；Qwen完成一次推理后，才读取当时最新的图像开始下一轮。在新结果生成前，界面继续显示上一次结果，操作人员可以阅读。该设计没有把每帧都排队送入慢模型。

设图像在主机读取返回的时刻为 $t_r$，提交模型为 $t_s$，输出为 $t_o$，则取帧时的年龄为 $t_s-t_r$，输出对应画面的年龄为 $t_o-t_r$。这些时刻不包含传感器曝光时间；即使没有排队，较慢的生成仍会使输出对应数秒前的画面。

1. 相机线程写入最新RGB与序号。
2. 快速视觉分支处理检测和深度，保留本轮所用序号。
3. 文字分支空闲后取最新帧，保存该帧和模型回复。
4. 界面分别标明处理时间与画面年龄，避免把旧结果当成当前瞬间。

## 观察界面与运动权限

早期桌面窗口后来统一为网页，保留原RGB、深度或检测视图、文字场景观察。模型返回的结构化字段用于程序读取，界面可显示自然语言。观察结果不自动转成运动权限；在后续的有限技能系统中，只有通过明确任务范围检查的请求才会进入执行。

通用检测器没有覆盖特定耳机壳时，新增任务标注和专用检测模型。单目深度图可以显示相对远近，但未标定的米制输出不能直接用于限制机械臂下降。真实场景中曾出现观察到的空间距离与预测距离明显不符，因而深度分支保留为辅助观察，而不是唯一安全依据。

## 共享功能与后续替换

网页也曾加入惯性测量单元的姿态显示，并通过相对初始参考姿态展示安装在连杆上的板卡。该功能属于观察基础设施，本文不展开另一条机器人控制课题的实验。之后的VLA网页把该区域换成动作模型输出与状态展示，但当串口被保持程序占用时，必须区分实时关节反馈与历史快照，不能把快照送入模型后称为完整实时闭环。''','连续观察与界面契约')
select('s03','experiments/alibaba_baseline/REPORT.md',[2,3,4,5])
select('s04','experiments/alibaba_extension/REPORT.md',[2,3,4,5,6,7])
select('s05','experiments/desktop_pickplace/REPORT.md',[1,2,3,4,5,6])
select('s05','experiments/alibaba_pc_revision/REPORT.md',[6,7],'后续的周期验证与选模比较')
select('s05','experiments/alibaba_jetson_revision/REPORT.md',[6],'扩大验证布局与重复训练种子')
select('s06','experiments/alibaba_jetson_revision/REPORT.md',[1,2,3,4,5,7])
# Keep the complete short detector/tracker notes, with an explicit local heading.
for sid,path,title in [('s06','experiments/earbud_detector/README.md','自采耳机壳与盒子检测模型'),('s06','experiments/earbud_tracking/README.md','目标观察与受限跟随')]:
 text=(R/path).read_text();text=re.sub(r'^# .*\n','',text,count=1);text=re.sub(r'^## ',r'### ',text,flags=re.M)
 parts[sid].append({'source':path,'title':title,'markdown':'## '+title+'\n\n'+text});coverage.append({'stage':sid,'source':path,'section':'完整短说明','source_sha256':digest(R/path)})
manual('s06',r'''## 物体转移回放的结果修正

固定摆放下曾分段执行已记录动作，实现夹住、抬起、搬运、松开和返回。动作源是示教轨迹，过程中有现场人工确认，没有YOLO或Qwen生成动作。第二遍完整流程反馈夹爪仍碰桌，随后将revision_03标为不通过无碰撞验收，不再作为已验证路径复用。

底座跟随另做过方向检查：约3度往返时测得横向像素变化约为每编码器计数负0.332像素。初版采用左右各约10度范围、单次最多2计数、35像素死区；后版改为单次4计数、20像素死区，保留其他关节目标与丢失停止规则。现场反馈为能够跟随但幅度小、速度慢，没有独立多场景成功率。关节范围限制不等于末端空间防碰桌。''','回放与追踪的实际边界')
coverage.append({'stage':'s06','source':'PROJECT_LOG.md','section':'10月25日转移与追踪结果修正','source_sha256':digest(R/'PROJECT_LOG.md')})
select('s07','experiments/smolvla_pc/REPORT.md',['已完成','条件对照','当前限制'],'基础模型输入与输出检查')
select('s07','experiments/smolvla_pc/mousepad_report/REPORT.md',['本轮完成','训练与留出检查','三条合训版本'],'真实取放示教的训练与留出评估')
select('s07','experiments/smolvla_pc/approach_report/REPORT.md',['数据与训练','离线检查','执行约束'],'短窗口接近策略')
manual('s07',r'''## 有限真机试动与动作限制的影响

| 测试 | 模型与执行方式 | 实际观察 | 结论 |
|---|---|---|---|
| 基础模型短试 | 每次只取首步方向，单轮最多2计数，总范围22计数 | 肩约0.44度、腕俯仰约1.58度，其余变化很小 | 动作接口有响应，未完成接近 |
| 微调短窗口 | 每轮4计数、总角度约8度、夹爪不驱动 | 8秒36轮；前五关节变化约负3.6、5.01、8.18、7.82、0度 | 达到限幅附近，未形成可用任务 |
| 20%单步残差 | 目标为当前实测位置加模型残差的20%，单轮最多5度 | 肘约4.04度、腕俯仰约1.41度 | 一次受控增量执行，非连续自主策略 |
| 50%单步版本 | 代码已准备 | 当日没有启动 | 不列作已测结果 |

不能仅由“模型建议肘关节再动20度”推断控制器会立即执行20度。实际动作同时受插值比例、单次计数、总包络、误差停止与通信节奏限制；例如4计数仅约0.35度。初次短试的“推理期间位置变化”检查还把机器人自身运动当成异常，后续才把该检查限于启用扭矩前，并保留执行后的跟踪误差检查。

这些问题解释了部分“几乎没动”，但并未证明解除限制后策略就能正确抓取。归一化、相机槽位、起点夹爪开度和任务数据覆盖均未被充分验证；三条真实示教不足以解决这些问题。最终记录为自主接近与拾取目标未达到，后续停止实体尝试，转向可重复的仿真基线。''','有限真机试动')
coverage.append({'stage':'s07','source':'PROJECT_LOG.md','section':'10月25日SmolVLA微调、限幅与20%实测','source_sha256':digest(R/'PROJECT_LOG.md')})
select('s08','experiments/vla_libero/REPORT.md',[1,2,3,4,5,6,7])
manual('s09',r'''## 高层与低层的输入分工

高层Qwen接收单独渲染的512像素第三人称图像和中文任务；低层π0.5仍使用官方外部、腕部图像与机器人状态转换。高层额外渲染不替换或重采样策略输入。Qwen输出英文子目标，π0.5按子目标生成动作块，模型之间没有替换内部网络层。

完成检查比较子目标开始帧与当前帧。物体入篮后可能只露出深色顶部，单帧容易把桌上相似物体误认成目标；前后对比用于关注目的区域变化。夹爪张开门控减少持物状态下的提前确认，连续两次完成答复才切换。门控用的是机器人自身状态，不是物体真值。

最后一个子目标不由Qwen提前终止，继续到官方成功或520步上限，与平坦基线保持相同终止方式。前面子目标的确认只用模型视觉结果；官方关系另存用于分析。后来的同场景视觉停止系统采用另一种停止协议，不能混用两者。

## 中文长任务的版本与强基线

| 配置 | 回合数 | 成功 | 使用范围 |
|---|---:|---:|---|
| 直接中文给π0.5 | 30 | 22/30 | 平坦中文基线 |
| 早期分层v1 | 30 | 16/30 | 无防抖且末目标可提前结束 |
| 分层v2 | 30 | 27/30 | 含一次重跑覆盖，作为开发记录 |
| 冻结分层v3首次运行 | 30 | 28/30 | 正式保留结果 |
| Qwen只翻译整句后给π0.5 | 30 | 29/30 | 必须保留的强对照 |
| 英文原指令平坦执行 | 30 | 29/30 | 早前公开任务基线 |

十个任务各三个公开初态。分层比直接中文高，但没有超过仅翻译；这组样本支持语言处理有作用，尚不支持独立的分层成功率优势。早期把收益全部归为语言理解的表述也过强，因为未对所有影响因素作独立消融。

v1到后续版本的改变包括两票确认、最后子目标不提前结束、放宽未使用提示字段的格式检查。多个改动同时发生，因此不是单因素因果实验。某任务曾首跑超时、重跑成功；后续冻结版本保留首次结果，不用重跑覆盖作为正式统计。

## 重规划与整句回退

子目标超预算时，可按当前图像重规划剩余步骤；若新计划原样重复卡住的步骤，则回退整句英文任务执行。旧正式v3默认关闭重规划，恢复试验另用新输出目录。在三个失败条件上的单例救回没有形成完整30回合回归，后续又发现恢复前缀不同，不能与固定起点试验合并。

对外报告应同时列出仅翻译、冻结分层和直接中文，区分“理解指令”“推进阶段”“修复动作”。本阶段未训练专门的指令分解适配器，Qwen仍为基座。''','分层协议与版本比较')
coverage += [{'stage':'s09','source':'PROJECT_LOG.md','section':'11月12日中文分层、公平对照与统一梳理','source_sha256':digest(R/'PROJECT_LOG.md')},{'stage':'s09','source':'experiments/vla_libero/INTEGRATION_PLAN.md','section':'已有结果与问题归纳','source_sha256':digest(R/'experiments/vla_libero/INTEGRATION_PLAN.md')}]
select('s09','experiments/vla_libero/qwen8b_comparison/REPORT.md',['范围与方法','实测结果','解释与限制'],'4B与8B完成检查诊断')
select('s10','experiments/vla_libero/recovery_study/REPORT.md',[1,2,3,4,5])
select('s10','experiments/vla_libero/closed_loop_iteration/REPORT.md',[2,3],'后续的受控故障注入与退化复查')
select('s11','experiments/vla_libero/closed_loop_iteration/REPORT.md',[4],'首版检查器：数据与阈值校准')
select('s11','experiments/vla_libero/checker_v2/REPORT.md',[1,2,3,4],'第二版检查器：近似完成数据与独立测试')
select('s12','experiments/vla_libero/checker_v2/EXPLORATORY_REPORT.md',[1,2,3,4,5],'冻结V2检查器的配对闭环')
select('s12','experiments/vla_libero/monitor_causality/REPORT.md',[1,2,3,4,5,6],'视觉信息干预与阶段切换归因')
select('s13','experiments/vla_libero/selective_execution/REPORT.md',[1,2,3,4,5])
select('s13','experiments/vla_libero/constraint_retention/REPORT.md',[1,2,3,4,5],'完整约束与腕部视角对照')
select('s14','experiments/vla_libero/skill_bootstrap/REPORT.md',[1,2,3,4,5],'目标分叉数据与策略适配')
select('s14','experiments/vla_libero/cag_reproduction/REPORT.md',[1,2,3,4],'反事实动作引导的免训练形式')
select('s14','experiments/vla_libero/point_target/REPORT.md',[1,2,3,4],'公开可运行实例与目标图接口')
select('s15','experiments/mobile_vla/README.md',['已实际完成','关键修正'])
text=(R/'experiments/mobile_vla/PROGRESS.md').read_text();text=re.sub('^# .*\n','',text,count=1);parts['s15'].append({'source':'experiments/mobile_vla/PROGRESS.md','title':'同场景回归','markdown':'## 三个初态的开发回归\n\n'+text});coverage.append({'stage':'s15','source':'experiments/mobile_vla/PROGRESS.md','section':'开发回归与局部修复','source_sha256':digest(R/'experiments/mobile_vla/PROGRESS.md')})
select('s16','experiments/mobile_qwen/reports/20261114/STAGE_REPORT.md',[1,2])
select('s17','experiments/mobile_qwen/reports/20261114/STAGE_REPORT.md',[3,4])
manual('s17',r'''## 像素、深度与世界坐标的对应

网络输出0到1000的相对坐标 $(a,b)$，先换成512像素坐标 $u=512a/1000$、$v=512b/1000$。深度取该点周围5×5窗口的有效值中位数。渲染图像与标签使用同一翻转函数，不能把其他图像方向下的内参直接套入。

当前实现的配对反投影公式是：

$$
 u_r=W-1-u,\qquad
 \boldsymbol p_c=\left[\frac{(u_r-W/2)z}{f},-\frac{(v-H/2)z}{f},-z\right]^T,
 \qquad\boldsymbol p_w=R_{wc}\boldsymbol p_c+\boldsymbol t_{wc}.
$$

其中 $W=H=512$，$z$ 为光学深度，$f=(H/2)/\tan(\mathrm{fovy}/2)$。负深度轴和横向反向与当前MuJoCo渲染约定对应。投影、深度和分割来自配对实现；图像旋转、裁剪或外参改变后需一同更新，不能只改显示。

约1.4厘米的误差下限部分来自预测到的表面点与几何参考中心之间的差别。因此三维指标是这条反投影流程的定位误差，不是单目深度网络误差，也不是机器人末端控制误差。''','反投影实现解释')
select('s18','experiments/alibaba_pc_revision/REPORT.md',[1,2,3,4,5])
select('s18','experiments/vla_libero/skill_bootstrap/REPORT.md',[6],'有界内存加载与故障处理')
excerpt('s18','PROJECT_LOG.md','### 2026-11-13 阿里反馈修订','### 2026-11-13 字节反馈','提交包与图像、日志语义修订')
select('s18','experiments/mobile_qwen/reports/20261114/STAGE_REPORT.md',[5],'视觉判断局限与增量运行记录')
select('s19','experiments/mobile_qwen/closed_loop_20261115/REPORT.md',[1,2,3,4,5])
manual('s19',r'''## 控制量与状态同步

关节驱动使用比例、阻尼和模型偏置补偿，实际实现可写为：

$$
 \boldsymbol\tau=\operatorname{clip}\left(400(\boldsymbol q_d-\boldsymbol q)-40\dot{\boldsymbol q}+\boldsymbol b(\boldsymbol q,\dot{\boldsymbol q}),\boldsymbol\tau_{\min},\boldsymbol\tau_{\max}\right).
$$

$\boldsymbol q_d$是轨迹插值后的关节目标，$\boldsymbol q$和速度来自物理仿真，$\boldsymbol b$为MuJoCo偏置项，包含重力和速度相关项。上下限采用原执行器范围；400和40是本次开发后冻结的控制参数，不是针对不同模型分别调参。

规划器的工具坐标与仿真工具点先用八个静态姿态核对。真实导航后的坐标检查又暴露积分时序问题：关节位置已更新，但派生位姿仍对应先前时刻。读取前调用forward统一缓存后，两套前向运动学的数值差降到约 $5\times10^{-12}$ 米。这表示数学模型在同一时刻一致，不是物理定位达到该精度。

执行器发布实际关节反馈，规划成功、轨迹实际完成、视觉重新确认和退出清理分别记录。当前部分MoveIt退出仍有已知析构异常；动作结果不能掩盖进程生命周期问题。''','关节控制与时序说明')
select('s20','experiments/mobile_qwen/closed_loop_20261115/REPORT.md',[6,7])
manual('s20',r'''## 为什么不默认增加前置动作

完整技能原本先搬碗、后关抽屉；把手预接近只是定位到执行接口的验证，未必是该任务开始时应做的动作。额外改变手臂姿态可能把低层策略带离熟悉状态。因此实际运行保留原生π0.5作为默认，显式规划只在有阶段条件和任务需要时调用。

交接时关节数组不变说明没有重置到位姿态；模型加载期间仿真暂停，所以该检查并未验证真实机器人在等待时的支撑控制、重力下垂和通信延迟。移到真机仍需独立的持续控制与监督试验。

此处的正确下一步是定义技能前提与阶段接口，而不是把所有自主任务都强行接上同一个把手预接近动作。''','技能兼容性讨论')
select('s21','experiments/mobile_qwen/jetson/SOFTWARE_STACK_REPORT.md',['结论','内存与运行状态','修复的兼容性问题','PC预量化与跨设备核对','本地文件'])
select('s22','experiments/mobile_qwen/jetson/stage2_20261114/REPORT.md',[1,2,3,4,5,6])
# Split the service section before its resource table; latency resources remain with S24.
path='experiments/mobile_qwen/jetson/stage3_20261115/REPORT.md';full,blocks=section_blocks(path);service=next(b for h,b in blocks if h.startswith('3.'))
service_body,resource_body=service.split('资源摘要',1)
parts['s23'].append({'source':path,'title':'视觉服务与联合回合','markdown':service_body});coverage.append({'stage':'s23','source':path,'section':'3. 视觉服务与PC联合仿真（资源表前）','source_sha256':digest(R/path)})
select('s24',path,[1,2,4]);parts['s24'].append({'source':path,'title':'资源摘要','markdown':'## 资源统计的不同口径\n\n资源摘要'+resource_body});coverage.append({'stage':'s24','source':path,'section':'3. 联合测试后的资源表','source_sha256':digest(R/path)})


manual('s14',r'''## 后续部署核对：旧适配分数的解释需要保留条件

后续检查目标图分支的权重合并时，发现注意力输出层的低秩增量不能按朴素的逐头矩阵乘法直接加入。原算子两次求和的语义需要与未合并运行一致。固定浮点32位算子检查中，朴素实现最大差约16.6893，按原算子语义修正后约 $1.91\times10^{-6}$；这是局部算子测试，不是任务成功率。

在同一适配器的六组已保存实际输入上，读取旧部署输出、原生未合并参考与修正合并输出，得到下表。数值由已存数组重新计算，没有在写报告时运行新模型。

| 输入编号 | 旧部署相对未合并最大动作差 | 修正合并相对未合并最大动作差 |
|---|---:|---:|
| 0 | 0.03208 | 0.00195 |
| 1 | 0.60550 | 0.00366 |
| 2 | 0.58115 | 0.01035 |
| 3 | 0.37429 | 0.00447 |
| 4 | 0.42658 | 0.01144 |
| 5 | 0.19806 | 0.00275 |

动作差采用接口输出单位，七维动作含不同物理量，不能统一解释为角度误差。修正合并仍存在BF16数值差；原生未合并适配器的闭环复测尚未完成，也没有新训练。

因此，前述目标图、文字适配分数需要标注为当时部署实现下的结果，不能单独用于否定训练方法或论文思路。已确认本次目标图分支受合并实现影响，其他共用路径也应逐项核对；没有适配器的官方基座不受该问题影响。当前同场景系统使用未适配π0.5，Qwen适配器由PEFT直接加载，也不受此旧JAX合并问题影响。
''','旧策略部署一致性更正')
coverage.extend([{'stage':'s14','source':'experiments/vla_libero/target_skill/PLAN.md','section':'部署语义说明','source_sha256':digest(R/'experiments/vla_libero/target_skill/PLAN.md')},{'stage':'s14','source':'experiments/vla_libero/target_skill/SEMANTIC_CHECK.json','section':'浮点算子核对','source_sha256':digest(R/'experiments/vla_libero/target_skill/SEMANTIC_CHECK.json')}])
select('s08','experiments/vla_libero/REPORT.md',[8],'原阶段复现入口')
select('s11','experiments/vla_libero/checker_v2/REPORT.md',[5],'检查器接口与文件')
select('s12','experiments/vla_libero/checker_v2/EXPLORATORY_REPORT.md',[6],'闭环复核入口')
select('s13','experiments/vla_libero/selective_execution/REPORT.md',[7],'选择性任务复现入口')
select('s14','experiments/vla_libero/point_target/REPORT.md',[5],'显式目标实例运行入口')

# Additional explicit cross-stage references discovered during expansion.
extra_sources={'s04':['experiments/alibaba_pc_revision/REPORT.md'],'s05':['experiments/alibaba_pc_revision/REPORT.md','experiments/alibaba_jetson_revision/REPORT.md'],'s06':['experiments/earbud_detector/README.md','experiments/earbud_tracking/README.md'],'s14':['experiments/vla_libero/target_skill/PLAN.md','experiments/vla_libero/target_skill/SEMANTIC_CHECK.json'],'s09':['experiments/vla_libero/qwen8b_comparison/REPORT.md','experiments/vla_libero/INTEGRATION_PLAN.md'],'s18':['experiments/alibaba_pc_revision/REPORT.md'],'s24':['experiments/mobile_qwen/jetson/stage3_20261115/compiler_preflight.json']}
extra_code={'s06':['experiments/earbud_detector','experiments/earbud_tracking'],'s05':['experiments/alibaba_pc_revision/scripts/train_act.py','experiments/alibaba_jetson_revision/scripts/train_act_expanded.py'],'s09':['experiments/vla_libero/qwen8b_comparison/scripts'],'s18':['experiments/alibaba_pc_revision/scripts']}
for s in stages:
 for p in extra_sources.get(s['id'],[]):
  if p not in s['sources']:s['sources'].append(p)
 for p in extra_code.get(s['id'],[]):
  if (R/p).exists() and p not in s['code']:s['code'].append(p)
stages[0]['limit']='稳定设备标识、校准文件和角度约定需要随运行保存。轨迹回放说明记录与执行接口可用，不计为模型自主操作。'

def normalize(text):
 # Keep exact identifiers and command blocks; edit prose only.
 chunks=re.split(r'(```[\s\S]*?```|`[^`\n]+`|\$\$[\s\S]*?\$\$|(?<!\$)\$[^$\n]+\$)',text)
 replacements={
  '参考标签来自 agent 对图像的独立视觉审阅，不是被测模型生成的标签。':'参考标签来自单独一次自动图像审阅，未采用被测模型的答案作为标签，也未完成独立人工复核。',
  '本轮':'该阶段','本次':'该次','证据':'记录','失败痕迹':'失败记录','官方检查点':'官方模型权重','自己的训练成果':'自主训练成果','本机':'工作站','全设备显式同步':'设备级显式同步',
 }
 # Device reports use 本机 for Jetson too: only make neutral wording below.
 replacements.pop('本机')
 for i,chunk in enumerate(chunks):
  if i%2:continue
  for a,b in replacements.items():chunk=chunk.replace(a,b)
  chunk=chunk.replace('真实策略回合','实际策略仿真回合').replace('真实回合','实际仿真回合')
  chunk=chunk.replace("{'input_token_sha256': 1014, 'pixel_values_sha256': 1014, 'image_grid_thw': 1014, 'output_token_ids': 774}",'输入词元、图像张量和图像网格各1014条完全一致，输出词元774条一致')
  chunk=chunk.replace("{'pswpin': 0, 'pswpout': 0}",'换入和换出页数增量均为0')
  chunk=re.sub(r'10([⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺]+)',lambda m:'$10^{'+m[1].translate(str.maketrans('⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺','0123456789-+'))+'}$',chunk)
  for a,b in [('π0.5',r'\(\pi\)0.5'),('σ',r'\(\sigma\)'),('μ',r'\(\mu\)'),('Δ',r'\(\Delta\)'),('≥',r'\(\geq\)'),('≤',r'\(\leq\)'),('≈',r'\(\approx\)'),('→',r'\(\rightarrow\)'),('×',r'\(\times\)')]:chunk=chunk.replace(a,b)
  # Original per-report headings carry numbers; LaTeX supplies the new numbering.
  chunk=re.sub(r'^(#{2,6})\s+\d+(?:\.\d+)*[.．、:]?\s*',r'\1 ',chunk,flags=re.M)
  cleaned=[]
  for line in chunk.splitlines():
   if line.startswith('|') and 'MoveIt execution returned' in line:
    cells=line.split('|');m=re.search(r'MoveIt execution returned (-?\d+)',cells[-2]);
    if m:cells[-2]=' 规划或执行返回码 '+m[1]+' ';line='|'.join(cells)
   cleaned.append(line)
  chunks[i]='\n'.join(cleaned)
 return ''.join(chunks)

def resolve(path,origin):
 path=path.split('#')[0];p=Path(path)
 if p.is_absolute():return p if p.exists() else None
 candidates=[]
 if origin:candidates.extend([(R/origin).parent/p,R/p])
 candidates.extend([D/p,R/p])
 return next((p.resolve() for p in candidates if p.is_file()),None)
def asset(s,src,kind,caption):
 sha=digest(src);key=(s['id'],sha)
 if key in known:rel=known[key]
 else:
  folder='videos' if kind=='video' else 'figures';stem=re.sub('[^A-Za-z0-9_.-]','_',src.stem)[:65] or kind
  if not re.search('[A-Za-z]',stem):stem=kind
  suffix=src.suffix.lower();rel=str(Path(folder)/f"ch{s['report_chapter']:02d}"/f"{s['id']}_{stem}_{sha[:8]}{suffix}")
  (D/rel).parent.mkdir(exist_ok=True,parents=True);shutil.copyfile(src,D/rel)
  known[key]=rel
  assets.append({'kind':kind,'stage':s['id'],'source':str(src.relative_to(R)) if src.is_relative_to(R) else str(src),'local':rel,'sha256':sha,'caption':caption,'note':'从已有运行材料迁入，未重新运行模型。'})
 return rel

def codeinline(text):
 for delim in ['|','!','+',';']:
  if delim not in text:return r'\path'+delim+text+delim
 return r'\texttt{'+tex(text)+'}'

current={}
def walk(node,s,origin):
 if isinstance(node,list):return [walk(x,s,origin) for x in node]
 if not isinstance(node,dict):return node
 t=node.get('t');c=node.get('c')
 if t=='Header':
  current['heading']=plain(c[2]);current['header_count']+=1;c[1][0]=f"{s['id']}-detail-{current['header_count']}"
 if t=='Str':
  words={'True':'是','False':'否','timeout':'超时','oracle':'真值定位','base':'基座','lora':'低秩适配','pc':'工作站','jetson':'Jetson','2b':'2B','4b':'4B','unknown':'不确定','not_done':'未完成'}
  if c in words:return {'t':'Str','c':words[c]}
  for a,b in [('worker','独立推理进程'),('rollout','执行回合'),('checkpoint','模型检查点')]:
   if c==a:return {'t':'Str','c':b}
  if re.fullmatch(r'[A-Za-z0-9_./:\-]{24,}',c):return {'t':'RawInline','c':['latex',codeinline(c)]}
 if t=='Code':
  text=c[1]
  if text.endswith('.mp4') and not any(x in text for x in ['*','<','>']):
   src=resolve(text,origin)
   if src:
    rel=asset(s,src,'video',text);used_videos[s['id']].add(rel);return {'t':'RawInline','c':['latex',codeinline(rel)]}
  if len(text)>24 or any(x in text for x in ['/', '_']):return {'t':'RawInline','c':['latex',codeinline(text)]}
 if t=='Image':
  caption=re.sub(r'^图\s*\d+\s*[.：:、]?\s*','',plain(c[1]));src=resolve(c[2][0],origin)
  if not src:
   missing.append({'stage':s['id'],'kind':'image','path':c[2][0],'source':origin});return {'t':'Str','c':'（原图位置需核对，见素材清单。）'}
  rel=asset(s,src,'figure',caption)
  if rel in used_figures[s['id']]:return {'t':'Str','c':''}
  used_figures[s['id']].add(rel);current['images']+=1
  return {'t':'RawInline','c':['latex',r'\stagefigure{'+rel+'}{'+tex(caption or s['title'])+'}{fig:'+s['id']+'-detail-'+str(current['images'])+'}']}
 if t=='Link':
  url=c[2][0]
  if not url.startswith(('http://','https://','mailto:')):
   src=resolve(url,origin)
   if src and src.suffix.lower()=='.mp4':
    rel=asset(s,src,'video',plain(c[1]));used_videos[s['id']].add(rel);return {'t':'RawInline','c':['latex',codeinline(rel)]}
   return {'t':'RawInline','c':['latex',codeinline(url)]}
 if t=='CodeBlock':
  lang=c[0][1][0] if c[0][1] else '';body=c[1]
  if lang=='mermaid':
   labels=[]
   for label in re.findall(r'[A-Za-z][\w]*\[([^\]]+)\]',body):
    if label not in labels:labels.append(label)
   if not labels:raise ValueError(('Unsupported diagram',origin,body))
   return {'t':'OrderedList','c':[[1,{'t':'Decimal'},{'t':'Period'}],[[{'t':'Plain','c':[{'t':'Str','c':label}]}] for label in labels]]}
  if len(body.splitlines())>14 or (lang in ['text','bash','sh','shell'] and len(body)>600):
   key=f"{s['id']}-command-{len(commands)+1}";commands.append({'stage':s['id'],'id':key,'language':lang,'text':body,'source':origin})
   return {'t':'RawBlock','c':['latex',r'\noindent 运行与目录示例见\hyperlink{'+key+'}{附录D的对应条目}。\par']}
  return {'t':'RawBlock','c':['latex','\\begin{Verbatim}[breaklines=true,breakanywhere=true,fontsize=\\footnotesize]\n'+body+'\n\\end{Verbatim}']}
 if t=='Table':
  current['tables']+=1;n=len(c[2]);rows=list(c[3][1])
  for body in c[4]:rows.extend(body[2]);rows.extend(body[3])
  measures=[[] for _ in range(n)]
  for row in rows:
   for j,cell in enumerate(row[1][:n]):
    text=plain(cell[4]);measures[j].append(sum(.55 if ord(ch)<128 else 1. for ch in text))
  widths=[math.sqrt(max(4.,sum(min(v,80.) for v in vals)/max(1,len(vals)))) for vals in measures]
  weights=[v/sum(widths) for v in widths]
  for _ in range(8):
   weights=[max(.065,min(.70 if n==2 else .48,v)) for v in weights];weights=[v/sum(weights) for v in weights]
  c[2]=[[col[0],{'t':'ColWidth','c':weights[j]}] for j,col in enumerate(c[2])]
  heading=current.get('heading') or s['title'];caption=re.sub(r'\s*\(.*?\)\s*$','',heading)+'：设置与结果'
  c[1]=[None,[{'t':'Plain','c':[{'t':'Str','c':caption}]}]]
  c[0][0]=f"tab:{s['id']}-{current['tables']}"
 result={k:(walk(v,s,origin) if k=='c' else v) for k,v in node.items()}
 return result

chapter_specs={1:('02_platform','原型与初始基线','基础平台先解决数据和动作接口是否可信，再讨论模型输出是否有意义。首轮Qwen适配、动作分块训练和跨模拟器流程都保留各自的条件。'),2:('03_autonomy','从演示到自主操作','早期桌面实验说明几何条件可以帮助小样本策略，但其控制条件理想。真实相机和SO101试验暴露输入、坐标、归一化及执行限制，于是后续先用公开任务权重建立可重复的动作参照。'),3:('04_hierarchy','Qwen分层执行与视觉完成检查','本章沿着一个问题展开：高层到底增加了什么？依次比较翻译、子目标分解、失败恢复、完成判断及视觉干预。不同版本有独立输出与初态，不能用后来的成功改写早期失败。'),4:('05_mobile','指定目标与同场景移动操作','只搬指定一件物体比原任务要求更细，也更能暴露目标绑定和停止问题。公开方法参考与局部策略适配没有解决全部问题；同场景导航操作则先把物理世界和坐标接口统一。'),5:('06_affordance','可供性定位与物理执行接口','在有限技能基础上，可供性学习将视觉输出变成可执行位置。工程验证依次检查数据、坐标、规划、实际关节反馈和策略交接；定位成功和完整任务成功分开统计。'),6:('07_edge','小模型与端侧部署','端侧工作先证明软件栈与量化存储可用，再做同输入精度对照，最后测时延和联合服务。恢复后的自有板子与10月借用环境不混为一套版本，速度和准确率也不混为一个结论。')}
for ch,(file,title,intro) in chapter_specs.items():
 out=[r'\chapter{'+title+'}',intro]
 for s in [x for x in stages if x['chapter']==ch]:
  sid=s['id'];out.extend([r'\section{'+tex(s['title'])+'}',r'\label{stage:'+sid+'}',tex(s['body'])])
  combined=[];current={'heading':'','header_count':0,'tables':0,'images':0}
  for i,part in enumerate(parts[sid]):
   body=normalize(part['markdown']);combined.append(body)
   doc=json.loads(subprocess.check_output([str(PANDOC),'-f','markdown+pipe_tables+tex_math_dollars+tex_math_single_backslash','-t','json'],input=body,text=True))
   doc['blocks']=walk(doc['blocks'],s,part['source'])
   rendered=subprocess.check_output([str(PANDOC),'-f','json','-t','latex','--wrap=none','--syntax-highlighting=none'],input=json.dumps(doc),text=True)
   rendered=re.sub(r'\\begin\{figure\}.*?(\\stagefigure\{[^\n]+)\n\\caption\{[^\n]*\}\n\\end\{figure\}',r'\1',rendered,flags=re.S)
   # Table widths are explicitly set; local smaller type is confined to the table group.
   rendered=rendered.replace(r'\begin{longtable}',r'\FloatBarrier\begingroup\small\setlength{\tabcolsep}{2.5pt}'+ '\n'+r'\begin{longtable}').replace(r'\end{longtable}',r'\end{longtable}\endgroup')
   out.append(rendered)
  if s.get('figure_local') and s['figure_local'] not in used_figures[sid]:
   out.append(r'\stagefigure{'+s['figure_local']+'}{'+tex(s['caption'])+'}{fig:'+sid+'-retained}')
   used_figures[sid].add(s['figure_local'])
  if s.get('video_local'):used_videos[sid].add(s['video_local'])
  out.append(r'\paragraph{阶段结论}'+tex(s['result'])+' '+tex(s['limit']))
  out.append(r'\stagecode{'+sid+'}')
  for video in sorted(used_videos[sid]):out.append(r'\stagevideo{'+video+'}')
  out.append(r'\FloatBarrier')
  (D/'source_fragments'/f'{sid}.md').write_text('\n\n'.join(combined))
  stats[sid]={'source_pieces':len(parts[sid]),'source_characters':sum(len(x) for x in combined),'tables':current['tables'],'figures':len(used_figures[sid]),'videos':len(used_videos[sid])}
 (D/'chapters'/f'{file}.tex').write_text('\n\n'.join(out)+'\n')
# Separate command/large directory fragments from the explanatory text.
app=[r'\chapter{运行与目录示例}',r'本附录保留各阶段的运行和目录片段。它们对应历史环境与数据，不应直接覆盖已有输出；再次运行应更换实验标识，并准备原项目所需权重与数据。本文写作期间没有执行这些模型、仿真或实体命令。']
for c in commands:
 app.extend([r'\section*{'+c['stage'].upper()+'：'+tex(c['source'] or '运行示例')+'}',r'\hypertarget{'+c['id']+'}{}',r'\begin{Verbatim}[breaklines=true,breakanywhere=true,fontsize=\footnotesize]',c['text'],r'\end{Verbatim}'])
(D/'appendices/reproduction.tex').write_text('\n\n'.join(app)+'\n')
(D/'outline.json').write_text(json.dumps(stages,ensure_ascii=False,indent=2));(D/'asset_manifest.json').write_text(json.dumps(assets,ensure_ascii=False,indent=2));(D/'source_coverage.json').write_text(json.dumps(coverage,ensure_ascii=False,indent=2));(D/'expansion_stats.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2));(D/'missing_assets.json').write_text(json.dumps(missing,ensure_ascii=False,indent=2))
print('Source reports:',len({x['source'] for x in coverage}),'selected sections:',len(coverage),'detail chars:',sum(x['source_characters'] for x in stats.values()),'tables:',sum(x['tables'] for x in stats.values()),'missing assets:',len(missing))
if missing:print(json.dumps(missing,ensure_ascii=False,indent=2))
