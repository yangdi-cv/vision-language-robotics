"""Five panels from actual episode videos; successful end frames remain labeled."""

import json, subprocess
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg
from interventions import MODES
from figures import LABEL, FONT

S = Path(__file__).resolve().parents[1]


def main():
    runs = []
    for mode in MODES:
        root = S / f"episodes/primary/{mode}/libero_10/task00_init00"
        result = json.loads((root / "result.json").read_text())
        reader = imageio_ffmpeg.read_frames(str(root / "rollout.mp4"))
        metadata = next(reader)
        frames = [
            np.frombuffer(frame, np.uint8)
            .reshape(metadata["size"][1], metadata["size"][0], 3)
            .copy()
            for frame in reader
        ]
        assert len(frames) == result["steps"] + 1
        runs.append((result, frames))
    width, height = 1440, 1040
    fps = 20
    count = max(len(frames) for _, frames in runs)
    target = S / "五组检查干预_实际回合对照.mp4"
    encoder = subprocess.Popen(
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
            "-vcodec",
            "libx264",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            str(target),
        ],
        stdin=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    fonts = {n: ImageFont.truetype(FONT, n) for n in [21, 25, 30]}
    for step in range(count):
        im = Image.new("RGB", (width, height), "#eef3f7")
        draw = ImageDraw.Draw(im)
        draw.text(
            (22, 12),
            "同一任务、初态和动作噪声：只干预阶段检查与切换",
            font=fonts[30],
            fill="#17354c",
        )
        for index, (result, frames) in enumerate(runs):
            col = index % 3
            row = index // 3
            ox = col * 480 + 10
            oy = row * 470 + 60
            im.paste(
                Image.fromarray(frames[min(step, len(frames) - 1)]).resize((365, 365)),
                (ox + 38, oy + 42),
            )
            draw = ImageDraw.Draw(im)
            draw.text((ox + 5, oy), LABEL[MODES[index]], font=fonts[25], fill="#17354c")
            ends = [
                e
                for e in result["events"]
                if e["kind"] == "subgoal_end" and e["step"] <= step
            ]
            text = (
                "已成功停止，保留实际末帧"
                if step >= result["steps"] and result["success"]
                else f"第{min(step, result['steps'])}步 · 子目标{len(ends) + 1}"
            )
            draw.text((ox + 5, oy + 412), text, font=fonts[21], fill="#17354c")
        draw.text((985, 570), "π0.5仍读取当前RGB与状态", font=fonts[21], fill="#17354c")
        draw.text((985, 610), "冻结的只是Qwen检查画面", font=fonts[21], fill="#17354c")
        draw.text((985, 650), "20fps为仿真回放时间", font=fonts[21], fill="#17354c")
        draw.text(
            (985, 690), "不包含模型推理的墙钟等待", font=fonts[21], fill="#17354c"
        )
        if step == 180:
            im.save(S / "figures/video_keyframe.png")
        encoder.stdin.write(im.tobytes())
    encoder.stdin.close()
    assert encoder.wait() == 0
    decoder = imageio_ffmpeg.read_frames(str(target))
    next(decoder)
    assert sum(1 for _ in decoder) == count
    (S / "VIDEO.json").write_text(
        json.dumps(
            dict(
                frames=count,
                fps=fps,
                seconds=count / fps,
                source_task=0,
                source_initial=0,
                decoded=True,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
