"""Camera-only target tracking. Does not import serial or send robot commands."""
import argparse
from collections import deque
import json
from pathlib import Path
import time
import tkinter as tk
import cv2
import numpy as np
from PIL import Image,ImageTk
from ultralytics import YOLO

p=argparse.ArgumentParser()
p.add_argument('--model',required=True)
p.add_argument('--output',required=True,type=Path)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
model=YOLO(a.model)
assert model.names[0]=='earbud_case'
model.predict(np.zeros((480,640,3),dtype=np.uint8),device=0,imgsz=640,verbose=False)
cap=cv2.VideoCapture('/dev/v4l/by-id/usb-icSpring_icspring_camera-video-index0',cv2.CAP_V4L2)
if not cap.isOpened():raise RuntimeError('Camera unavailable')
cap.set(3,640);cap.set(4,480);cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
root=tk.Tk();root.title('蓝色耳机壳追踪 · 观察模式 · 不驱动机械臂')
view=tk.Label(root);view.pack()
status=tk.StringVar()
tk.Label(root,textvariable=status,font=('Noto Sans CJK SC',14),wraplength=650).pack(pady=8)
closed=False;frame_id=0;hits=0;center=None;trail=deque(maxlen=30)
journal=(a.output/'observations.jsonl').open('a')

def publish(row):
    temp=a.output/'latest.json.tmp';temp.write_text(json.dumps(row,ensure_ascii=False))
    temp.replace(a.output/'latest.json')
    journal.write(json.dumps(row,ensure_ascii=False)+'\n');journal.flush()

def close(event=None):
    global closed
    if not closed:
        closed=True
        publish({'state':'stopped','motion_enabled':False,'host_monotonic':time.monotonic(),'target':None})
        cap.release();journal.close();root.destroy()

def update():
    global frame_id,hits,center
    if closed:return
    ok,frame=cap.read();received=time.monotonic()
    if not ok:
        hits=0;center=None;trail.clear()
        publish({'state':'camera_error','motion_enabled':False,'host_monotonic':received,'target':None})
        status.set('相机读取失败，目标无效');root.after(100,update);return
    result=model.predict(frame,classes=[0],conf=.55,imgsz=640,device=0,verbose=False)[0]
    boxes=result.boxes
    h,w=frame.shape[:2];reference=np.array([w/2,h/2],dtype=float)
    drawn=result.plot();target=None;state='lost';text='未发现耳机壳'
    if len(boxes)==1:
        box=boxes[0];xyxy=box.xyxy[0].cpu().numpy();raw=(xyxy[:2]+xyxy[2:])/2
        if center is None or np.linalg.norm(raw-center)>120:
            center=raw;hits=1;trail.clear()
        else:
            center=.35*raw+.65*center;hits+=1
        error=center-reference
        if hits>=3:
            state='centered' if abs(error[0])<=40 and abs(error[1])<=40 else 'tracking'
            trail.append(tuple(center.astype(int)))
            target={'bbox_xyxy':xyxy.tolist(),'center_px':center.tolist(),'error_px':error.tolist(),
                    'normalized_error':(error/np.array([w/2,h/2])).tolist(),
                    'confidence':float(box.conf.item())}
            horizontal='居中' if abs(error[0])<=40 else ('右侧' if error[0]>0 else '左侧')
            vertical='居中' if abs(error[1])<=40 else ('下方' if error[1]>0 else '上方')
            text=f'已锁定：目标在画面{horizontal} / {vertical}，偏差 ({error[0]:.0f}, {error[1]:.0f}) 像素'
            cv2.circle(drawn,tuple(center.astype(int)),6,(0,255,255),-1)
            cv2.line(drawn,tuple(reference.astype(int)),tuple(center.astype(int)),(0,255,255),2)
        else:state='acquiring';text=f'正在确认目标 {hits}/3'
    else:
        hits=0;center=None;trail.clear()
        if len(boxes)>1:state='ambiguous';text='发现多个候选目标，暂不锁定'
    cv2.drawMarker(drawn,tuple(reference.astype(int)),(255,255,0),cv2.MARKER_CROSS,24,2)
    cv2.rectangle(drawn,(w//2-40,h//2-40),(w//2+40,h//2+40),(255,255,0),1)
    for left,right in zip(list(trail),list(trail)[1:]):cv2.line(drawn,left,right,(0,180,255),1)
    frame_id+=1;finished=time.monotonic()
    publish({'frame':frame_id,'state':state,'target':target,'host_monotonic':finished,
             'frame_received_monotonic':received,'processing_ms':(finished-received)*1000,
             'motion_enabled':False,'rotation_clockwise':0,'height_guard_validated':False})
    if frame_id==1 or frame_id%15==0:cv2.imwrite(str(a.output/'latest_view.jpg'),drawn)
    photo=ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(drawn,cv2.COLOR_BGR2RGB)))
    view.configure(image=photo);view.image=photo
    status.set(text+'\n观察模式：机械臂不运动；Q退出')
    root.after(10,update)
for key in ('q','Q','<Escape>'):root.bind(key,close)
root.protocol('WM_DELETE_WINDOW',close)
root.after(0,update)
try:root.mainloop()
finally:
    if not closed:close()
