# SO-ARM101 工作目录

这套设备目前在 Windows 上已完成双臂校准、串口连接、遥操作和 USB 摄像头读取。黑色是主臂（leader），白色是从臂（follower）。**切换 Ubuntu 后，先重新确认串口和摄像头编号，再运行机械臂。**

## 文件放在哪里

| 路径 | 内容 |
| --- | --- |
| `reference/videos/` | 厂商提供的安装、校准、遥操作、数据采集和训练视频 |
| `reference/manual/` | 厂商 PDF 使用文档 |
| `reference/images/` | 产品照片和校准姿态参考图 |
| `reference/source_archive/lerobot.zip` | 厂商原始源码压缩包，作为备份 |
| `code/lerobot/` | 当前主要使用的 LeRobot 源码；包含本机调试时的本地修改，不要用 zip 直接覆盖 |
| `code/hiwonder-SoArm-101/` | 另一套 Hiwonder 源码，仅作参考和旧环境兼容 |
| `code/scripts/` | 当前运行的遥操作、摄像头预览脚本 |
| `code/diagnostics/` | 之前排查电机和串口用的脚本、旧版试验脚本 |
| `calibration/` | **当前生效的**黑白双臂校准 JSON，迁移 Ubuntu 时必须带上 |
| `records/` | 既往遥操作日志、CSV 和摄像头测试截图；新 CSV 也会写在这里 |

## Windows：继续使用

打开 Anaconda Prompt：

```bat
conda activate lerobot
python "C:\Users\diyan\Desktop\VLA\SO101\code\scripts\preview_camera.py"
```

预览中按 `Q`、`Esc` 或关闭窗口退出。打开预览前先关闭 Windows“相机”等占用摄像头的软件。目前仅确认一台 USB 摄像头，编号为 `0`。最近一次截图在 `records/camera_previews/captured_images/opencv_0.png`；截图主要是天花板，录制抓取演示前要把镜头调整到能看见夹爪和桌面。

遥操作：

```bat
conda activate lerobot
python "C:\Users\diyan\Desktop\VLA\SO101\code\scripts\teleop_table_lerobot.py"
```

Windows 默认黑色主臂 `COM13`、白色从臂 `COM12`。脚本采用 10 Hz、每次命令最多偏离白色当前位置 5°，白色肩部 `P=28`、肘部 `P=24`。脚本退出时会尝试将这两个 P 值恢复为 16，并关闭白色臂扭矩；**停止前让白色臂有支撑**。每次运行会在 `records/` 新建带时间的 CSV。

串口编号如果变化，可显式传入：

```bat
python "C:\Users\diyan\Desktop\VLA\SO101\code\scripts\teleop_table_lerobot.py" --leader-port COM13 --follower-port COM12
```

另一个 Conda 环境 `lerobot-hiwonder` 对应 `code/hiwonder-SoArm-101/`，不要在同一个环境中混用两个源码版本。

## Ubuntu：迁移顺序

1. 将整个 `SO101` 文件夹复制到 Ubuntu，尤其保留 `code/lerobot/` 和 `calibration/`。Conda 环境本身不能从 Windows 直接复制使用。
2. 在 Ubuntu 上安装 Conda，进入复制后的 `SO101` 文件夹，创建新环境。当前主要源码要求 Python 3.10 或以上；Windows 已验证使用 Python 3.10。

   ```bash
   conda create -n lerobot python=3.10 -y
   conda activate lerobot
   python -m pip install -e ./code/lerobot
   ```

   如果安装依赖遇到系统库问题，查看 `code/lerobot/README.md` 和 `code/lerobot/requirements-ubuntu.txt`。上面的 Ubuntu 安装步骤尚未在这台电脑上实测。

3. 分别插拔黑白两条 USB 线，核对各自的稳定串口路径：

   ```bash
   ls -l /dev/serial/by-id/
   ```

   不要假设 Windows 的 `COM12/COM13` 对应 Ubuntu 的 `/dev/ttyUSB0/1`。若串口权限不足，将当前用户加入 `dialout` 组并重新登录。

4. 检查摄像头。Linux 上预览脚本默认使用视频设备编号 `0`，可通过 `--camera-index` 修改：

   ```bash
   ls -l /dev/video*
   python code/scripts/preview_camera.py --camera-index 0
   ```

5. 确认双臂位置、校准文件和供电无误，托稳白色臂，再运行遥操作；用第 3 步实际查到的路径替换示例：

   ```bash
   python code/scripts/teleop_table_lerobot.py \
     --leader-port /dev/serial/by-id/BLACK_ARM \
     --follower-port /dev/serial/by-id/WHITE_ARM
   ```

两个校准文件分别是 `calibration/teleoperators/so_leader/my_awesome_leader_arm.json` 和 `calibration/robots/so_follower/my_awesome_follower_arm.json`。脚本默认从项目内的 `calibration/` 读取它们，不依赖 Windows 的 `G:` 盘。

## 当前状态与注意事项

- 两条机械臂各需一只原装 12 V 5 A 电源适配器，并通过各自 USB 连接电脑。
- 白色臂曾出现肩部抬起困难；提高肩部、肘部 P 值后，用户反馈运行改善。迁移 Ubuntu 后应先做短时间、小幅度测试，再长时间运行。
- 当前相机能读到 640×480 画面。相机接入成功不等于已经录制训练数据。
- 原版 `lerobot` 源码有本地修改；迁移时复制整个 `code/lerobot/`，不要只复制 `lerobot.zip`。

### 整理时未能删除的残留

根目录的 `manual_ascii.pdf` 与 `reference/manual/` 中的正式说明书 SHA-256 完全相同，但当前被其他程序占用，无法移动或删除。根目录的 `hiwonder-SoArm-101/` 只剩一个空的隐藏 `.git` 目录，移动时被 Windows 拒绝清理。`records/empty_repo_leftover/` 和 `records/old_python_cache/` 也是这次整理留下的空目录或缓存。它们都**不是运行所需文件**；迁移 Ubuntu 时只复制上表列出的 `reference/`、`code/`、`calibration/`、需要保留的 `records/` 和本 README 即可。
