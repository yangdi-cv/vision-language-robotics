# 视觉语言引导的机器人操作：技术与工程报告 v0.3

中文版详细技术报告，白底黑字。中文标题和正文使用Fandol宋体（开源宋体系列），数学、英文及代码采用LaTeX对应字体；未附带中易SimSun。报告按技术关系组织全部24个阶段，不限制篇幅。

## 阅读入口

- `build/main_zh.pdf`：中文版PDF。
- `main_zh.tex`、`preamble.tex`、`chapters/`：完整可编辑LaTeX源文件。
- `appendices/`：章节代码导航、图片与视频清单、术语解释、运行示例。
- `code_index/ch02`至`ch07`：按24个阶段列出原实现与源报告。
- `REPOSITORY_LAYOUT.md`：私有代码仓库组织建议；原实现没有搬迁。
- `versions/main_zh_v0.1.pdf`、`versions/sources_v0.1.zip`：保留的初版。

## 内容范围

六个技术篇章：原型与初始基线、从演示到自主操作、Qwen分层与视觉检查、指定目标与同场景移动操作、可供性与物理执行接口、小模型与端侧部署。另有概述、工程整理与交付验证、阶段结论。

从34份已有报告和记录中选入142个小节，展开实验条件、划分、指标、对照和失败样例；源材料约9.6万字符，最终正文包含综合说明。详细迁入表99张，另有概述表；72张图、7段视频均为已有实验素材。所有数字按对应阶段条件解释，不将预接触到位、完整任务成功和实时性合并。

补入早期ACT/几何与量化对照、真实机械臂局限、目标接口适配的权重合并更正、同场景导航/规划/关节控制、2B/4B及PC/Jetson精度和时延。旧策略合并修正后的闭环复测仍未完成，报告明确保留这一缺口。

## 来源与核对

- `source_coverage.json`、`source_fragments/`：迁入的小节及来源记录。
- `asset_manifest.json`：图像与视频原路径、英文相对路径、SHA256。
- `RESULT_AUDIT.json`：对已保存120个官方回合、90个中文对照回合和6组权重合并数值探针的复核。不是所有历史指标的重新评测。
- `VIDEO_CHECK.json`：7段视频首末帧解码检查。
- `code_inventory.csv`：源文件清单，区分项目代码与厂家/混合来源；清单数量不代表自编代码量。

报告只是整理已有工作，没有启动训练、批量评测或连接硬件。

## 编译与检查

```bash
cd /home/yd/Documents/VLA/reports/alibaba_engineering
./tools/build.sh
python3 tools/validate.py
python3 tools/audit_saved_results.py
```

本机使用用户目录中的Tectonic 0.17.0；脚本也支持已有XeLaTeX/latexmk。首次编译需要下载TeX宏包和字体。源文件适合采用XeLaTeX的TeX Live/Overleaf，参考文献不需要biber。

`tools/expand_report.py`是本次材料迁入的记录工具，不属于日常编译步骤；生成后已对LaTeX正文作编辑。重新运行会覆盖详细章节的手工修订，修改报告应直接编辑`.tex`。Pandoc仅用于迁入过程，正常编译不需要它。

## 视频与代码

视频均复制到`videos/chXX/`，文件名为英文；PDF给出相对路径，未嵌入视频。以后英文版可以复用同一素材。完整英文版与中文版同步维护，均已纳入私有GitHub修订分支。

代码索引给出现有实现位置；模型权重、数据、环境、上游库与原始输出不应全部混入普通Git历史。未来整理私有仓库时保留共享实现和版本来源，按章节提供入口，不复制多套同名模型或控制代码。

## English edition

[Complete English PDF — 161 pages](english/build/main_en.pdf) · [LaTeX sources and build instructions](english/README.md). The English edition covers all 24 stages, 103 tables, captions for 60 figure assets, and 7 video references. It uses Times New Roman and shares the original experiment images and videos with the Chinese edition.

## v0.3 更新

2026年11月22日：中文174页，英文176页。保留24个实验阶段，新增工程维护一章；新增12组实际录屏关键帧、保存的模型输入、训练标签和历史真机照片，共72张图、7段视频。图片说明区分PC完整操作、Jetson预接触、模型输入与仿真真值。原60张图片与视频内容不变，技术正文除新增图外保留。旧中英文PDF与LaTeX源文件保存在`versions/`。

`VISUAL_ADDITIONS_V03.json`记录新图来源、帧号、裁剪范围与校验值，`REPORT_UPDATE_V03.json`记录版本对照。新增章节记录127项离线检查，不计作新模型实验。
