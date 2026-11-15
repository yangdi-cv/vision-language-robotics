# 完成检查器LoRA首轮权重

基座：Qwen/Qwen3-VL-4B-Instruct，revision ebb281ec70b05090aa6165b016eac8ec08e71b17。基座使用条件见官方模型页：https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct 。本包是adapter，不含完整基座。

输入两张256像素RGB图和任务指令，输出完成判断。仅训练语言层q/v LoRA；rank8，alpha16，训练参数2,949,120，固定180次更新，以验证集选择第180步。

数据为既有LIBERO基线视频与步级成功判据。初态0用于训练，初态1用于验证，初态2用于测试，分割以完整回合为单位。实际训练160例；测试保留常规160例和边界80例。

常规测试准确率76.9%→86.3%，完成召回5.1%→87.2%；边界贪心误报为82.5%，验证校准后仍16.3%。完整240例贪心总准确率82.9%→63.3%，平衡准确率51.6%→72.9%。不要只取常规子集的提升代表整体效果。

该权重目前是离线完成检查适配实验，没有用于机器人动作切换。它不输出RGB-D位置、可供性或动作；不等于任务书全部感知/规划微调已完成。数据可能含执行阶段关联，后续需补与成功画面相似的失败反例。

重新加载：先在Qwen环境中按上述revision加载基座，再调用PeftModel.from_pretrained(base_model, 'best_adapter', is_trainable=False)。独立推理入口是closed_loop_iteration/scripts/check_completion.py，支持--calibrated读取同目录calibrated_summary.json的验证阈值。阈值不是精度保证，本轮未达到自动控制要求。
