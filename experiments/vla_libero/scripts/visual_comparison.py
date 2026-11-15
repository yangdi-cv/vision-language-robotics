"""Paired live/blank/frozen image intervention, actual world and submitted inputs."""

from pathlib import Path
import json, subprocess
import numpy as np
import imageio.v2 as imageio
import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

R = Path(__file__).resolve().parents[1]
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


def font(n):
    return ImageFont.truetype(FONT, n)


def main():
    modes = ["baseline", "blank_images", "frozen_images"]
    titles = ["实时图像", "图像置零", "冻结初始图像"]
    runs = []
    for mode in modes:
        p = R / "results" / mode / "libero_spatial/task00_init00"
        r = json.loads((p / "result.json").read_text())
        reader = imageio.get_reader(p / "rollout.mp4")
        frames = [v for v in reader]
        reader.close()
        inputs = np.load(p / "query_inputs.npz")["base_rgb"]
        runs.append((r, frames, inputs))
    assert len({r["initial_state_sha256"] for r, _, _ in runs}) == 1
    width, height = 1260, 760
    fps = 20
    count = max(len(v) for _, v, _ in runs)
    video = R / "visual_intervention.mp4"
    proc = subprocess.Popen(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{width}x{height}",
            "-pix_fmt",
            "rgb24",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        stdin=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    for i in range(count):
        canvas = Image.new("RGB", (width, height), "#edf2f7")
        d = ImageDraw.Draw(canvas)
        d.text(
            (20, 10),
            "同一初态：实际场景与提交给 VLA 的图像",
            font=font(28),
            fill="#173d59",
        )
        for j, (r, frames, inputs) in enumerate(runs):
            ox = j * 420
            idx = min(i, len(frames) - 1)
            qidx = min(idx // 5, len(inputs) - 1)
            ended = i >= len(frames) - 1
            d.text((ox + 15, 57), titles[j], font=font(24), fill="#203c54")
            canvas.paste(Image.fromarray(frames[idx]).resize((400, 400)), (ox + 10, 96))
            canvas.paste(
                Image.fromarray(inputs[qidx]).resize((150, 150)), (ox + 15, 535)
            )
            d = ImageDraw.Draw(canvas)
            d.text((ox + 15, 501), "实际相机画面", font=font(18), fill="#314b61")
            d.text((ox + 180, 540), "左下：最近提交", font=font(19), fill="#314b61")
            d.text((ox + 180, 570), "给模型的图像", font=font(19), fill="#314b61")
            d.text(
                (ox + 180, 610), f"仿真 {idx / fps:.2f}s", font=font(19), fill="#314b61"
            )
            if ended:
                d.text(
                    (ox + 180, 645),
                    "成功，保留末帧" if r["success"] else "超时，保留末帧",
                    font=font(18),
                    fill="#22784d" if r["success"] else "#b44338",
                )
        d.text(
            (20, 710),
            "只改变模型的视觉输入；状态、指令、初态和动作采样按相同协议配对。",
            font=font(21),
            fill="#425b70",
        )
        if i == 100:
            canvas.save(R / "figures/visual_intervention.png")
        proc.stdin.write(canvas.tobytes())
    proc.stdin.close()
    assert proc.wait() == 0
    out = R.parents[1] / "deliverables/20261112_阿里_VLA统一仿真/视频"
    out.mkdir(exist_ok=True, parents=True)
    import shutil

    shutil.copy2(video, out / "实时视觉与输入干预对比.mp4")
    (R / "visual_comparison_metadata.json").write_text(
        json.dumps(
            {
                "cases": [r for r, _, _ in runs],
                "fps": fps,
                "frames": count,
                "fixed_after_episode_ends": True,
            },
            indent=2,
        )
    )
    print(video)


if __name__ == "__main__":
    main()
