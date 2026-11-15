# 官方π0.5可运行实例

本次重新运行官方pi05_libero公开权重及原LIBERO任务，初态32、33，任务0/1/9共6条，6条全部成功。没有适配器、Qwen、脚本教师动作或示教回放。动作来自实时图像和任务指令驱动的策略；官方成功信号仅用于结束与评分。

本地页面：http://127.0.0.1:8793 。视频是这6条真实仿真记录，不是实体机械臂演示，也不是任意新任务成功的承诺。首次完成后使用新tag重新运行，例如：

```bash
cd /home/yd/Documents/VLA
bash experiments/vla_libero/official_instance/start.sh --tag my_demo --tasks 9 --initials 32
```

启动器使用现有独立环境与单模型分块加载，通过systemd用户单元设置软14GiB、硬16GiB限额。日志命令在启动时输出。不改系统Python、ROS2或实体设备。

`runs/fresh_original/`包含全部结果、输入、动作块、逐步实际动作、原录像；`scripts/build_page.py`复核动作与模型块一致、末步官方谓词为真。6条新回合是可运行实例验证，不是新基准统计。原117/120结果不混入此分母。

停止网页：`systemctl --user stop ali-vla-instance-page`。再次启动网页：`cd experiments/vla_libero/official_instance && python3 -m http.server 8793 --bind 127.0.0.1`。

启动命令另以cli_smoke新tag实际执行微波炉任务1回合，1/1成功，动作块和末步判据核验通过。该命令烟测不混入前述6回合成绩，记录见CLI_CHECK.json。
