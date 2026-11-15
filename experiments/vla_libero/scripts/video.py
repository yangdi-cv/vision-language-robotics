"""Compose actual rollout footage and labeled keyframes, without synthetic motion."""

from pathlib import Path
import numpy as np
import json, subprocess, textwrap, shutil
import imageio.v2 as imageio
import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

R = Path(__file__).resolve().parents[1]
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


def font(n):
    return ImageFont.truetype(FONT, n)


def main():
    pair = Image.new("RGB", (600, 340), "#edf2f7")
    pd = ImageDraw.Draw(pair)
    for j, (name, title) in enumerate(
        [("initial_scene.png", "外部相机"), ("initial_wrist.png", "腕部相机")]
    ):
        pair.paste(
            Image.open(R / "figures" / name).convert("RGB").resize((290, 290)),
            (j * 300 + 5, 40),
        )
        pd.text((j * 300 + 12, 5), title, font=font(21), fill="#21384e")
    pair.save(R / "figures/observation_views.png")
    s = json.loads((R / "summary.json").read_text())
    rows = s["rows"]
    selected = []
    for suite in ["libero_spatial", "libero_object", "libero_goal", "libero_10"]:
        vals = [
            r
            for r in rows
            if r["mode"] == "baseline"
            and r["suite"] == suite
            and r["status"] != "error"
        ]
        if vals:
            selected.append(next((v for v in vals if v["success"]), vals[0]))
    if not selected:
        raise RuntimeError("No completed baseline rollout")
    sheet = Image.new("RGB", (1200, 350 * len(selected)), "#edf2f7")
    d = ImageDraw.Draw(sheet)
    frames = []
    for j, r in enumerate(selected):
        p = R / r["directory"] / "rollout.mp4"
        reader = imageio.get_reader(p)
        images = [frame for frame in reader]
        reader.close()
        frames.append(images)
        for k, fraction in enumerate([0.0, 0.5, 1.0]):
            idx = round((len(images) - 1) * fraction)
            im = Image.fromarray(images[idx]).resize((300, 300))
            sheet.paste(im, (300 + k * 300, j * 350 + 40))
        d.text((12, j * 350 + 10), r["suite"], font=font(20), fill="#21384e")
        d.text(
            (12, j * 350 + 55),
            f"任务 {r['task_id']} / 初态 {r['initial_state_id']}",
            font=font(18),
            fill="#21384e",
        )
        d.text(
            (12, j * 350 + 90),
            "成功" if r["success"] else "超时",
            font=font(22),
            fill="#22784d" if r["success"] else "#ab453d",
        )
        for k, line in enumerate(textwrap.wrap(r["original_prompt"], width=23)[:7]):
            d.text((12, j * 350 + 140 + k * 23), line, font=font(14), fill="#31495f")
    sheet.save(R / "figures/rollout_examples.png")
    video = R / "vla_rollout_comparison.mp4"
    fps = 20
    w, h = 1200, 850
    n = max(len(v) for v in frames)
    proc = subprocess.Popen(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{w}x{h}",
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
    for i in range(n):
        canvas = Image.new("RGB", (w, h), "#edf2f7")
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (20, 10),
            "π0.5 / LIBERO：模型预测动作，物理仿真执行",
            font=font(26),
            fill="#183d5a",
        )
        for j, (r, imgs) in enumerate(zip(selected, frames)):
            ox = (j % 2) * 600
            oy = 55 + (j // 2) * 390
            idx = min(i, len(imgs) - 1)
            canvas.paste(
                Image.fromarray(imgs[idx]).resize((350, 350)), (ox + 8, oy + 28)
            )
            draw = ImageDraw.Draw(canvas)
            draw.text(
                (ox + 12, oy),
                f"{r['suite']} / task {r['task_id']}",
                font=font(19),
                fill="#21384e",
            )
            for k, line in enumerate(textwrap.wrap(r["original_prompt"], width=24)[:8]):
                draw.text(
                    (ox + 370, oy + 40 + k * 24), line, font=font(14), fill="#31495f"
                )
            draw.text(
                (ox + 370, oy + 245),
                f"仿真 {idx / fps:.2f} s",
                font=font(19),
                fill="#21384e",
            )
            if i >= len(imgs) - 1:
                draw.text(
                    (ox + 370, oy + 280),
                    "完成，保留末帧" if r["success"] else "超时，保留末帧",
                    font=font(18),
                    fill="#22784d" if r["success"] else "#ab453d",
                )
        proc.stdin.write(canvas.tobytes())
    proc.stdin.close()
    assert proc.wait() == 0
    meta = {
        "selected": selected,
        "selection": "First successful baseline in each suite, else first completed trial; presentation only, all trials counted separately.",
        "fps": fps,
        "frames": n,
        "simulation_time_not_wall_time": True,
    }
    (R / "video_metadata.json").write_text(json.dumps(meta, indent=2))
    out = R.parents[1] / "deliverables/20261112_阿里_VLA统一仿真/视频"
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(video, out / "VLA四类任务闭环演示.mp4")
    failures = [r for r in rows if not r["success"] and r["status"] != "error"]
    for i, r in enumerate(failures[:3]):
        src = R / r["directory"] / "rollout.mp4"
        if src.exists():
            shutil.copy2(
                src, out / f"失败_{i + 1}_{r['mode']}_{r['suite']}_{r['task_id']}.mp4"
            )
    release_dir = R / "release_checks/libero_spatial/task00_init00"
    if (release_dir / "frames.npy").exists():
        released = np.load(release_dir / "frames.npy")
        original_reader = imageio.get_reader(
            R / "results/baseline/libero_spatial/task00_init00/rollout.mp4"
        )
        original = [im for im in original_reader]
        original_reader.close()
        panel = Image.new("RGB", (960, 380), "#edf2f7")
        pen = ImageDraw.Draw(panel)
        for j, (frame, title) in enumerate(
            [
                (original[0], "模型执行前"),
                (original[-1], "官方谓词通过时"),
                (released[-1], "追加松爪保持1秒后"),
            ]
        ):
            panel.paste(Image.fromarray(frame).resize((310, 310)), (j * 320 + 5, 45))
            pen.text((j * 320 + 10, 5), title, font=font(21), fill="#21384e")
        panel.save(R / "figures/release_check.png")
        writer = imageio.get_writer(
            out / "黑碗任务_追加松爪检查.mp4", fps=20, codec="libx264", quality=7
        )
        for frame in released:
            writer.append_data(frame)
        writer.close()
    print(video)


if __name__ == "__main__":
    main()
