"""Read-only live detection: no robot connections or motion commands."""
import argparse
from pathlib import Path
import time
import tkinter as tk
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO

p = argparse.ArgumentParser()
p.add_argument('--model', required=True)
p.add_argument('--output', required=True, type=Path)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
model = YOLO(a.model)
model.predict(source=__import__('numpy').zeros((480,640,3), dtype='uint8'), device=0, imgsz=640, verbose=False)
cap = cv2.VideoCapture('/dev/v4l/by-id/usb-icSpring_icspring_camera-video-index0', cv2.CAP_V4L2)
if not cap.isOpened():
    raise RuntimeError('Cannot open wrist camera')
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
root = tk.Tk()
root.title('耳机壳与盒子 · 实时检测 · Q 退出')
view = tk.Label(root)
view.pack()
status = tk.StringVar()
tk.Label(root,textvariable=status,font=('Noto Sans CJK SC',14)).pack(pady=8)
closed = False
count = 0

def close(event=None):
    global closed
    if not closed:
        closed=True
        cap.release()
        root.destroy()

def update():
    global count
    if closed:return
    ok, frame = cap.read()
    if not ok:
        status.set('相机读取失败')
        root.after(100,update)
        return
    t = time.perf_counter()
    result = model.predict(frame, device=0, imgsz=640, conf=0.35, verbose=False)[0]
    drawn = result.plot()
    count += 1
    if count == 1 or count % 30 == 0:
        cv2.imwrite(str(a.output/'latest_detection.jpg'),drawn)
        (a.output/'status.txt').write_text(f'frames={count}\nrotation=0\nobjects={len(result.boxes)}\n')
    photo=ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(drawn,cv2.COLOR_BGR2RGB)))
    view.configure(image=photo);view.image=photo
    status.set(f'耳机壳 / 盒子 | 检测到 {len(result.boxes)} 个目标 | 处理 {(time.perf_counter()-t)*1000:.0f} ms')
    root.after(1,update)

for key in ('q','Q','<Escape>'):root.bind(key,close)
root.protocol('WM_DELETE_WINDOW',close)
root.after(0,update)
try:root.mainloop()
finally:
    if not closed:close()
