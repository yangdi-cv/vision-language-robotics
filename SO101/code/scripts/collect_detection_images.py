"""Collect manually selected wrist-camera images, grouped by scene layout."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import tkinter as tk

import cv2
from PIL import Image, ImageTk

from camera_view import rotate_frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--camera-index', type=int, default=0)
    parser.add_argument('--rotation', type=int, choices=(0, 90, 180, 270), default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    images = args.output / 'images'
    images.mkdir()
    (args.output / 'session.json').write_text(json.dumps({
        'classes': ['earbud_case', 'container'],
        'rotation_clockwise': args.rotation,
        'split_policy': 'Keep each physical layout group in one split.',
        'created_utc': datetime.now(timezone.utc).isoformat(),
    }, ensure_ascii=False, indent=2))
    camera = cv2.VideoCapture(args.camera_index, cv2.CAP_V4L2)
    if not camera.isOpened():
        raise RuntimeError('Camera is unavailable; close other camera readers.')
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    root = tk.Tk()
    root.title('数据采集 · Space 保存 · N 新场景组 · Q 退出')
    view = tk.Label(root)
    view.pack()
    status = tk.StringVar()
    tk.Label(root, textvariable=status, font=('Noto Sans CJK SC', 14)).pack(pady=8)
    latest = None
    received = 0.0
    count = 0
    group = 1
    closed = False
    last_saved = 0.0

    def refresh_status(message=''):
        status.set(f'已保存 {count} 张 | 场景组 {group:03d} | 空格保存，N 换组\n{message}')

    def save(_event=None):
        nonlocal count, last_saved
        if latest is None or time.monotonic() - received > 0.5:
            refresh_status('画面尚未就绪，请稍候')
            return
        if time.monotonic() - last_saved < 0.7:
            return
        filename = f'group_{group:03d}_{count + 1:05d}.jpg'
        if not cv2.imwrite(str(images / filename), latest, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise OSError('Image save failed')
        with (args.output / 'manifest.jsonl').open('a') as stream:
            stream.write(json.dumps({
                'image': f'images/{filename}', 'group': group,
                'saved_utc': datetime.now(timezone.utc).isoformat(),
                'frame_received_monotonic': received,
                'rotation_clockwise': args.rotation,
                'width': latest.shape[1], 'height': latest.shape[0],
            }) + '\n')
        count += 1
        last_saved = time.monotonic()
        refresh_status(f'已保存 {filename}，改变摆放后再拍下一张')

    def next_group(_event=None):
        nonlocal group
        group += 1
        refresh_status('新场景组：请改变盒子位置或背景布置')

    def close(_event=None):
        nonlocal closed
        if not closed:
            closed = True
            camera.release()
            root.destroy()

    def update():
        nonlocal latest, received
        if closed:
            return
        ok, frame = camera.read()
        if not ok:
            refresh_status('相机读取失败；未保存图片')
            root.after(100, update)
            return
        latest = rotate_frame(frame, args.rotation)
        received = time.monotonic()
        photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(latest, cv2.COLOR_BGR2RGB)))
        view.configure(image=photo)
        view.image = photo
        root.after(30, update)

    for key in ('<space>',):
        root.bind(key, save)
    for key in ('n', 'N'):
        root.bind(key, next_group)
    for key in ('q', 'Q', '<Escape>'):
        root.bind(key, close)
    root.protocol('WM_DELETE_WINDOW', close)
    refresh_status('物体摆好、手离开画面后，按空格')
    root.after(0, update)
    try:
        root.mainloop()
    finally:
        if not closed:
            close()


if __name__ == '__main__':
    main()
