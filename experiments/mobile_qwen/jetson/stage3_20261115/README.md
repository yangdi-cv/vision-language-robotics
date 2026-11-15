# Jetson可供性时延筛查

先完成PC物理闭环实验，最后再运行本目录。复用ali-jetson环境与stage2 NF4模型/LoRA，不安装依赖，不训练、不下载新模型。Jetson15W mode0，jetson_clocks不修改。

input_manifest.json从冻结验证集val_ids排序取5帧，未使用test350。512/384/256各handle/bowl/place/cabinet四类问题、每类5次。每档暖机1次，不计入时延；按图像轮换尺寸顺序，记录生成token数、模型时间及含图像处理的墙钟时间。P95仅探索性小样本分位数，不是线上SLA。

验收目标来自既定可供性单次≤3s。该筛查若不达标，不直接启动350帧全量；任何新尺寸若要正式选择，仍须重测350帧精度。此目录不评价里程碑检查，不使用仿真同步等待充当实时达标，不提供MAXN SUPER结果。

scripts/latency_probe.py在Jetson运行，依赖相邻stage2/scripts/runtime.py。host_driver.py从PC复制本目录、开启板端6GiB受限进程，等待加载后关闭swap门控，结束恢复SD/zram并同步结果。SSH密码只由SSHPASS环境提供，不写入源码。

本次运行由systemd单元ali-jetson-latency-host-20261115管理，结果在results/records.jsonl、SUMMARY.json、tegrastats.log。稳定段模型swap和全局交换IO必须为0，否则无效。仅模型时间不含CPU图像处理，墙钟时间包含图像读取和处理，仍不含网络通信。
