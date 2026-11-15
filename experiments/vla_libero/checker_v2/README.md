# 完成检查器 V2

当前状态：实验权重，未通过部署验收。见 REPORT.md。

环境：项目根目录 `.venvs/qwen-scene/bin/python` 用于 Qwen 训练与评分；`experiments/vla_libero/scripts/python_shared` 用于仿真。沿用本机既有基座缓存，不自动下载。

先运行：
```bash
.venvs/openpi-libero/bin/python -m unittest discover -s experiments/vla_libero/checker_v2/scripts -p test_v2.py
```

原独立测试已评分并冻结，勿覆盖重跑。视觉版本未通过验证，confirmation 数据未评分。语言层版本另行开展探索性闭环，最新结果见 EXPLORATORY_REPORT.md，原 REPORT.md 为闭环之前的阶段记录。evaluate_loop.py 的正式验收保留离线通过门槛；新增 --exploratory 仅用于用户授权的仿真配对，不改变原离线未通过结论，不准许实体部署。训练命令会写实验输出，复现时请先另建输出目录并核对脚本中的路径。

源码 ZIP 保留项目相对目录。权重 ZIP 仅含两个实验适配器，需要本机 Qwen3-VL-4B 基座；不含基座、环境和训练图像。它们是检查器适配器，不是机械臂动作策略。


## 探索性闭环更新

已完成基座/V2各30回合，均28/30。搬运适用范围均21/21、零误切换；平均单次检查0.501秒与0.125秒，比较的是完整生成式/分类式检查方案。V2未减少错误切换，更多达成阶段未被确认。详见EXPLORATORY_REPORT.md和results/exploratory_metrics.json。

新源码包保留相对项目目录、冻结计划、数据元信息及指标。它不是环境镜像；外部库版本见vla_libero/third_party/sources.json和requirements.lock.txt。Qwen基座仍从本机固定版本缓存加载，π0.5需要checkpoints/pi05_libero；两个检查器实验适配器沿用上一份“完成检查器V2_实验权重_未通过验收.zip”，语言层适配器应位于checker_v2/results/model/best_adapter。源码包不含基座、第三方源码、训练图像、原始视频和模型权重。

本机已有结果时evaluate_loop.py会续跑并复用回合，不是重新重复评测。新的独立复现应另设输出标签及协议文件，不删除或覆盖本批结果。单元测试不依赖GPU或图像，使用包内元信息。
