"""Lightweight local RGB + joint JSONL recording, without training dependencies."""
import hashlib
import json
from pathlib import Path
import time
import cv2
from camera_view import rotate_frame

class DemonstrationRecorder:
    def __init__(self, root, task, camera, joints, calibration_root, ports, fps=10, rotation=0, preview=False):
        self.root=Path(root)
        self.root.mkdir(parents=True,exist_ok=False)
        (self.root/'images').mkdir()
        self.joints=joints;self.rotation=rotation;self.frames=0;self.preview_stop=False
        self.preview_window=None
        self.camera=cv2.VideoCapture(str(camera),cv2.CAP_V4L2)
        self.camera.set(3,640);self.camera.set(4,480);self.camera.set(cv2.CAP_PROP_BUFFERSIZE,1)
        if not self.camera.isOpened():raise RuntimeError('Camera unavailable')
        for _ in range(5):
            ok,_frame=self.camera.read()
            if not ok:self.camera.release();raise RuntimeError('Camera warmup failed')
        self.stream=(self.root/'trajectory.jsonl').open('w')
        self.manifest={'format':'so101-rgb-joints-jsonl-v1','status':'prepared','task':task,'joint_order':list(joints),
            'fps_nominal':fps,'rotation_clockwise_degrees':rotation,'ports':ports,
            'units':'degrees for five arm joints; percent for gripper',
            'action_semantics':'absolute goals actually sent; review actual measured states before replay',
            'calibration_sha256':{str(p.relative_to(calibration_root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(calibration_root).rglob('*.json')},
            'timestamp_note':'host monotonic receipt timestamps, not camera exposure times'}
        self._manifest()
        if preview:
            import tkinter as tk
            self.preview_window=tk.Tk();self.preview_window.title('空爪遥操作录制 · 停止前托稳白臂 · Q停止')
            self.label=tk.Label(self.preview_window);self.label.pack()
            tk.Label(self.preview_window,text='先等待启动；运行后 Q 停止并关闭扭矩，停止前务必托稳白臂',font=('Noto Sans CJK SC',12)).pack()
            self.preview_window.bind('q',self._stop);self.preview_window.bind('Q',self._stop)
            self.preview_window.protocol('WM_DELETE_WINDOW',self._stop)
            self.preview_window.update()
    def _stop(self,event=None):self.preview_stop=True
    def _manifest(self):(self.root/'session.json').write_text(json.dumps(self.manifest,indent=2,ensure_ascii=False))
    def capture(self):
        ok,image=self.camera.read();received=time.monotonic_ns()
        if not ok:raise RuntimeError('Camera read failed')
        rgb=cv2.cvtColor(rotate_frame(image,self.rotation),cv2.COLOR_BGR2RGB)
        if self.preview_window:
            from PIL import Image,ImageTk
            photo=ImageTk.PhotoImage(Image.fromarray(rgb),master=self.preview_window)
            self.label.configure(image=photo);self.label.image=photo
            self.preview_window.update()
            if self.preview_stop:raise KeyboardInterrupt
        return rgb,received
    def add(self,actual,sent,image,state_ns,image_ns,sent_ns):
        path=f'images/{self.frames:06d}.jpg'
        if not cv2.imwrite(str(self.root/path),cv2.cvtColor(image,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,90]):raise OSError('Image write failed')
        self.stream.write(json.dumps({'frame_index':self.frames,'state':[float(actual[f'{j}.pos']) for j in self.joints],
            'action':[float(sent[f'{j}.pos']) for j in self.joints],'image':path,'state_ns':state_ns,'image_ns':image_ns,'sent_ns':sent_ns})+'\n')
        self.stream.flush();self.frames+=1
    def close(self):
        self.camera.release()
        if self.preview_window:self.preview_window.destroy()
        self.stream.close();self.manifest.update(status='review_required' if self.frames else 'empty',frames=self.frames)
        self._manifest();print(f'Saved {self.frames} frames: {self.root}',flush=True)
