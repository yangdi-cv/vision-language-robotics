"""Live wrist-camera benchmark. No robot serial interfaces are imported."""
import argparse,json,time,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--trt',action='store_true');a=p.parse_args();r=a.root;r.mkdir(parents=True,exist_ok=True)
os.environ['YOLO_AUTOINSTALL']='false';os.environ['YOLO_CONFIG_DIR']=str(r/'settings')
import cv2,numpy as np,torch,ultralytics
from ultralytics import YOLO
if a.trt:
 import sys
 sys.path.append('/usr/lib/python3.10/dist-packages')
out=r/('trt' if a.trt else 'vision');out.mkdir(exist_ok=True)
cap=cv2.VideoCapture('/dev/v4l/by-id/usb-icSpring_icspring_camera-video-index0',cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640);cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480);cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
frames=[];start=time.perf_counter()
try:
 for i in range(45):
  ok,f=cap.read()
  if not ok:raise RuntimeError('Camera read failed')
  if i>=15:
   f=cv2.rotate(f,cv2.ROTATE_90_COUNTERCLOCKWISE);frames.append(f)
   cv2.imwrite(str(out/f'frame_{len(frames)-1:03}.jpg'),f)
finally:cap.release()
meta={'torch':torch.__version__,'ultralytics':ultralytics.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'capture_elapsed_s':time.perf_counter()-start,'frames':len(frames),'rotation_clockwise':270,'robot_motion':False}
(out/'environment.json').write_text(json.dumps(meta,indent=2));rows=[]
for name,size in [('yolo26n',640),('yolo26n-depth',640)]:
 model=YOLO(str(r/'models'/(name+'.pt')))
 if a.trt:
  engine=model.export(format='engine',quantize=16,imgsz=size,device=0,workspace=1,simplify=False,verbose=False)
  model=YOLO(engine)
 for half in ([True] if a.trt else [False,True]):
  for i,f in enumerate(frames):
   torch.cuda.synchronize();t=time.perf_counter()
   result=model(f,imgsz=size,device=0,quantize=16 if half and not a.trt else None,verbose=False)[0]
   torch.cuda.synchronize();elapsed=time.perf_counter()-t
   item={'model':name,'backend':'TensorRT' if a.trt else 'PyTorch','half':half,'frame':i,'warmup':i<3,'seconds':elapsed,'peak_cuda_bytes':torch.cuda.max_memory_allocated()}
   if hasattr(result,'depth') and result.depth is not None:
    d=result.depth.data.float().cpu().numpy();item.update(depth_finite_fraction=float(np.isfinite(d).mean()),depth_median=float(np.median(d)),depth_scale_validated=False)
    if i==5:
     np.save(out/('depth_'+str(half)+'.npy'),d)
     c=cv2.normalize(d,None,0,255,cv2.NORM_MINMAX).astype(np.uint8);cv2.imwrite(str(out/('depth_'+str(half)+'.jpg')),cv2.applyColorMap(255-c,cv2.COLORMAP_TURBO))
   else:
    item['objects']=[{'class':result.names[int(b.cls.item())],'xyxy':b.xyxy[0].cpu().tolist(),'confidence':float(b.conf.item())} for b in result.boxes]
    if i==5:cv2.imwrite(str(out/('boxes_'+str(half)+'.jpg')),result.plot())
   rows.append(item)
   (out/'measurements.json').write_text(json.dumps(rows,indent=2))
  print('COMPLETE',name,half,flush=True)
summary=[]
for name in ['yolo26n','yolo26n-depth']:
 for half in ([True] if a.trt else [False,True]):
  times=[v['seconds'] for v in rows if v['model']==name and v['half']==half and not v['warmup']]
  summary.append({'model':name,'half':half,'n':len(times),'p50_ms':float(np.median(times)*1000),'p95_ms':float(np.percentile(times,95)*1000)})
(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
