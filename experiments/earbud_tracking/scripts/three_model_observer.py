"""Latest-frame YOLO + monocular depth + Qwen-VL observation, no serial access."""
import argparse,json,os,sys,time,threading
from pathlib import Path
from unittest.mock import patch
p=argparse.ArgumentParser()
p.add_argument('--root',required=True,type=Path)
p.add_argument('--qwen-model',required=True,type=Path)
a=p.parse_args();out=a.root/'three_model_live';out.mkdir(parents=True,exist_ok=True)
os.environ['YOLO_AUTOINSTALL']='false';os.environ['YOLO_CONFIG_DIR']=str(a.root/'settings')
def save(name,value):
 tmp=out/(name+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False));tmp.replace(out/name)
def phase(name):save('status.json',{'phase':name,'motion_enabled':False,'pid':os.getpid()});print(name,flush=True)
phase('imports')
import torch,cv2,numpy as np,bitsandbytes as bnb
from PIL import Image,ImageTk
from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
import tkinter as tk
phase('loading_qwen')
pool=[torch.empty(640*1024*1024,dtype=torch.uint8,device='cuda')]+[torch.empty(64*1024*1024,dtype=torch.uint8,device='cuda') for _ in range(22)]
for block in pool:block.zero_()
del block;torch.cuda.synchronize();del pool
processor=AutoProcessor.from_pretrained(a.qwen_model,local_files_only=True)
with patch('transformers.modeling_utils.caching_allocator_warmup',lambda *args,**kwargs:None):
 vlm=Qwen3VLForConditionalGeneration.from_pretrained(a.qwen_model,dtype=torch.bfloat16,device_map={'':0},local_files_only=True).eval()
sys.path.insert(0,str(a.root.parents[0]/'alibaba_jetson_revision/deps'))
sys.path.append('/usr/lib/python3.10/dist-packages')
from ultralytics import YOLO
phase('warming_detection_depth')
det=YOLO(str(a.root.parents[0]/'earbud_detector/models/earbud_container.engine'))
depth=YOLO(str(a.root.parents[0]/'alibaba_jetson_revision/models/yolo26n-depth.engine'))
blank=np.zeros((480,640,3),dtype=np.uint8)
# Finish TensorRT static-graph warmup before starting concurrent Qwen generation.
torch.cuda.empty_cache()
for _ in range(3):
 det(blank,imgsz=640,rect=False,device=0,conf=.55,verbose=False)
torch.cuda.synchronize();torch.cuda.empty_cache()
for _ in range(3):
 depth(blank,imgsz=640,rect=False,device=0,verbose=False)
torch.cuda.synchronize();torch.cuda.empty_cache()
stop=threading.Event();lock=threading.Lock();latest={};vision={};language={};errors=[]

def camera_loop():
 cap=cv2.VideoCapture('/dev/v4l/by-id/usb-icSpring_icspring_camera-video-index0',cv2.CAP_V4L2)
 cap.set(3,640);cap.set(4,480);cap.set(cv2.CAP_PROP_BUFFERSIZE,1);seq=0
 try:
  if not cap.isOpened():raise RuntimeError('Camera unavailable')
  while not stop.is_set():
   ok,frame=cap.read();stamp=time.monotonic()
   if not ok:raise RuntimeError('Camera read failed')
   with lock:latest.update(frame=frame,stamp=stamp,seq=seq)
   seq+=1
 except Exception as exc:
  with lock:errors.append('camera: '+repr(exc))
 finally:cap.release()

def vision_loop():
 last=-1;hits=0;previous_center=None
 try:
  with (out/'vision.jsonl').open('a') as journal:
   while not stop.is_set():
    with lock:data=latest.copy()
    if not data or data['seq']==last:stop.wait(.02);continue
    last=data['seq'];frame=data['frame'];began=time.monotonic()
    result=det(frame,imgsz=640,rect=False,device=0,conf=.55,verbose=False)[0]
    d=depth(frame,imgsz=640,rect=False,device=0,verbose=False)[0].depth.data.float().cpu().numpy().squeeze()
    h,w=frame.shape[:2]
    if d.shape!=(h,w):raise RuntimeError(f'Depth shape mismatch {d.shape}')
    valid=np.isfinite(d)&(d>0);fraction=float(valid.mean())
    values=d[valid]
    if values.size:
     lo,hi=np.percentile(values,[5,95]);scaled=np.clip((d-lo)/max(float(hi-lo),1e-6),0,1)
     color=cv2.applyColorMap(np.nan_to_num(255-scaled*255).astype(np.uint8),cv2.COLORMAP_TURBO)
     color[~valid]=0
    else:color=np.zeros_like(frame)
    roi=d[int(h*.3):int(h*.7),int(w*.25):int(w*.75)]
    roi_valid=roi[np.isfinite(roi)&(roi>0)]
    predicted_near=float(np.percentile(roi_valid,10)) if roi_valid.size else None
    targets=[]
    for box in result.boxes:
     if int(box.cls.item())!=0:continue
     bounds=box.xyxy[0].cpu().numpy();center=(bounds[:2]+bounds[2:])/2
     targets.append({'bbox':bounds.tolist(),'center_px':center.tolist(),'confidence':float(box.conf.item()),'error_px':(center-np.array([w/2,h/2])).tolist()})
     x1,y1,x2,y2=np.round(bounds).astype(int)
     cv2.rectangle(color,(x1,y1),(x2,y2),(255,255,255),2)
    if len(targets)==1:
     center=np.array(targets[0]['center_px'])
     hits=hits+1 if previous_center is not None and np.linalg.norm(center-previous_center)<120 else 1
     previous_center=center
    else:hits=0;previous_center=None
    state='tracking' if hits>=3 else ('acquiring' if len(targets)==1 else ('ambiguous' if targets else 'lost'))
    finished=time.monotonic()
    # This flag may request a stop; it never authorizes lowering.
    advisory_stop=fraction<.95 or predicted_near is None or predicted_near<.25
    row={'frame':last,'state':state,'target':targets[0] if state=='tracking' else None,
         'frame_received_monotonic':data['stamp'],'finished_monotonic':finished,'processing_ms':(finished-began)*1000,
         'depth_roi_p10_model_units':predicted_near,'depth_finite_positive_fraction':fraction,
         'depth_advisory_stop':advisory_stop,'depth_metric_calibrated':False,
         'allow_lowering':False,'motion_enabled':False,'inhibit_reason':'Cartesian height floor not calibrated; observation only'}
    journal.write(json.dumps(row)+'\n');journal.flush();save('vision_latest.json',row)
    with lock:vision.update(row=row,boxes=result.plot(),depth=color)
    if last%30==0:cv2.imwrite(str(out/'latest_view.jpg'),np.hstack([frame,result.plot(),color]))
    stop.wait(.1)
 except Exception as exc:
  with lock:errors.append('vision: '+repr(exc))

def language_loop():
 prompt='请仅根据这张图，用两句简短中文描述：能看到什么，是否能看到深蓝色耳机壳和桌面。看不清就说明看不清。不要输出JSON，不要给机械臂运动指令，不要推断图外物体。'
 try:
  with (out/'qwen.jsonl').open('a') as journal:
   while not stop.is_set():
    with lock:data=latest.copy()
    if not data:stop.wait(.05);continue
    image=Image.fromarray(cv2.cvtColor(data['frame'],cv2.COLOR_BGR2RGB))
    messages=[{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':prompt}]}]
    began=time.monotonic()
    batch=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt').to(vlm.device)
    with torch.inference_mode():tokens=vlm.generate(**batch,max_new_tokens=96,do_sample=False)
    text=processor.decode(tokens[0][len(batch.input_ids[0]):],skip_special_tokens=True)
    finished=time.monotonic()
    row={'frame':data['seq'],'frame_received_monotonic':data['stamp'],'finished_monotonic':finished,
         'inference_s':finished-began,'text':text,'model':'Qwen3-VL-2B NF4','motion_enabled':False}
    journal.write(json.dumps(row,ensure_ascii=False)+'\n');journal.flush();save('qwen_latest.json',row)
    with lock:language.update(row)
 except Exception as exc:
  with lock:errors.append('qwen: '+repr(exc))

root=tk.Tk();root.title('耳机壳寻找与追踪 · YOLO / 深度 / Qwen · 观察模式')
view=tk.Label(root);view.pack()
state_text=tk.StringVar(value='正在接收相机画面')
tk.Label(root,textvariable=state_text,font=('Noto Sans CJK SC',13),wraplength=1250,justify='left').pack(pady=6)
qwen_text=tk.StringVar(value='Qwen正在理解当前画面……')
tk.Label(root,textvariable=qwen_text,font=('Noto Sans CJK SC',14),wraplength=1250,justify='left',bg='#edf0f4',padx=12,pady=12).pack(fill='x')
closed=False

def close(event=None):
 global closed
 if not closed:closed=True;stop.set();root.destroy()

def refresh():
 if closed:return
 with lock:data=latest.copy();v=vision.copy();l=language.copy();err=errors.copy()
 if data:
  frame=data['frame'];parts=[cv2.resize(frame,(400,300))]
  if v:
   parts.extend([cv2.resize(v['boxes'],(400,300)),cv2.resize(v['depth'],(400,300))])
   row=v['row'];age=time.monotonic()-row['frame_received_monotonic']
   distance=row['depth_roi_p10_model_units'];value='无效' if distance is None else f'{distance:.2f}（未标定预测）'
   state_text.set(f"实时RGB | 目标检测：{row['state']} | 深度：{value} | 视觉结果年龄 {age:.2f}s | 机械臂运动未启用")
  else:parts.extend([np.zeros((300,400,3),np.uint8)]*2)
  photo=ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(np.hstack(parts),cv2.COLOR_BGR2RGB)))
  view.configure(image=photo);view.image=photo
 if l:qwen_text.set(l['text']+f"\n推理 {l['inference_s']:.1f}s · 对应 {time.monotonic()-l['frame_received_monotonic']:.1f}s前画面")
 if err:state_text.set('；'.join(err)+' · 运动禁止')
 root.after(50,refresh)
for key in ('q','Q','<Escape>'):root.bind(key,close)
root.protocol('WM_DELETE_WINDOW',close)
threads=[threading.Thread(target=f,daemon=True) for f in [camera_loop,vision_loop,language_loop]]
for thread in threads:thread.start()
phase('running');root.after(0,refresh)
try:root.mainloop()
finally:
 stop.set()
 for thread in threads:thread.join(timeout=8)
 save('status.json',{'phase':'stopped','motion_enabled':False,'errors':errors})
