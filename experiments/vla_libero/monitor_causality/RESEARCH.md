# 视觉阶段检查的闭环作用：实验依据

采用已有固定版本openpi和LIBERO，不从零实现机器人策略或模拟器。openpi为Apache-2.0，LIBERO为MIT，版本记录沿用vla_libero/third_party/sources.json。新工作是对检查输入与切换规则做干预，不改策略权重和任务判定。

- [openpi官方LIBERO示例](https://github.com/Physical-Intelligence/openpi/tree/main/examples/libero)：沿用官方观察/状态/动作接口。已核对本地固定版本sample_actions：显式提供动作噪声时不再从策略RNG采样噪声，观察预处理使用train=False。本轮逐查询保存输入和噪声，并实际核对切换前动作，不能只凭设置相同就断言一致。
- [LIBERO官方基准](https://github.com/Lifelong-Robot-Learning/LIBERO)：使用原Long的官方成功检查和公开初态。没有改成自定义、更宽松的成功标准。中间目标谓词仅记录和离线评估，最终成功停止仍沿用官方规则。
- [SciPy binomtest](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html)：用于配对成功不一致项的精确二项检验，即McNemar精确形式。10任务×3初态存在任务内相关，多组对照且为已研究过的初态，因此只作描述性统计，不以一个p值宣称泛化显著。

实验约束：五组保持两条冻结子目标、环境初态、动作噪声、低层输入、每块执行5步及最多520步。正常/冻结/V2/固定150步组均保留释放门控、两票确认和300步阶段预算。hold_first是单独的机制诊断：保留计划记录，但关闭所有阶段推进及阶段预算，直到官方成功或520步。

固定150步是统一预设阈值，不按任务或测试成绩调节；实际切换仍等待相同门控与两次判断，因此不声称一定在150步切换。冻结组把检查器第二张图替换成第一张图，相机和策略本身仍读取当前画面。V2是原提示词+LoRA+分类阈值整个方案，未完成离线验收的历史结论保留，不称为单独LoRA增益。

本轮首先检验高层的必要性，而非预设分层一定优于简单方法。若某组没有切换却完成整项任务，须记录低层任务先验的可能影响。既有30初态用于机制诊断，不当作新的独立泛化测试；不训练、不选新权重、不连接硬件。
