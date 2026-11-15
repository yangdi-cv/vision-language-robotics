"""Identical saved camera inputs and 640-square letterboxing across three backends."""
import argparse,sys,json,time,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
os.environ['YOLO_AUTOINSTALL']='false';os.environ['YOLO_CONFIG_DIR']=str(r/'settings')
import cv2,numpy as np,torch
from ultralytics import YOLO
sys.path.append('/usr/lib/python3.10/dist-packages')
out=r/'paired_vision';out.mkdir(exist_ok=True)
paths=sorted((r/'scene').glob('frame_*.jpg'))[::3];frames=[cv2.imread(str(p)) for p in paths];records=[];predictions={};summary=[]
assert len(frames)==10
for backend,half in [('pytorch',False),('pytorch',True),('tensorrt',True)]:
 suffix='.engine' if backend=='tensorrt' else '.pt'
 models=[YOLO(str(r/'models'/(name+suffix))) for name in ['yolo26n','yolo26n-depth']]
 key=f'{backend}_'+('fp16' if half else 'fp32');predictions[key]=[]
 def infer(f):
  kw=dict(imgsz=640,rect=False,device=0,verbose=False,conf=.35)
  if backend=='pytorch' and half:kw['quantize']=16
  b=models[0](f,**kw)[0];d=models[1](f,**kw)[0].depth.data.float().cpu().numpy()
  boxes=[{'class':int(v.cls.item()),'xyxy':v.xyxy[0].cpu().tolist(),'confidence':float(v.conf.item())} for v in b.boxes]
  return b,d,boxes
 for _ in range(3):infer(frames[0])
 if backend=='pytorch':
  for m in models:
   expected=torch.float16 if half else torch.float32
   assert next(m.predictor.model.model.parameters()).dtype==expected

 for rep in range(3):
  for i,f in enumerate(frames):
   torch.cuda.synchronize();t=time.perf_counter();b,d,boxes=infer(f);torch.cuda.synchronize();elapsed=time.perf_counter()-t
   records.append({'backend':key,'repeat':rep,'frame':paths[i].name,'seconds':elapsed})
   if rep==0:
    assert np.isfinite(d).all()
    predictions[key].append({'depth':d.copy(),'boxes':boxes})
    if i==1:
     cv2.imwrite(str(out/(key+'_boxes.jpg')),b.plot());c=cv2.normalize(d,None,0,255,cv2.NORM_MINMAX).astype(np.uint8);cv2.imwrite(str(out/(key+'_depth.jpg')),cv2.applyColorMap(255-c,cv2.COLORMAP_TURBO))
 vals=[v['seconds'] for v in records if v['backend']==key]
 summary.append({'backend':key,'n':len(vals),'p50_ms':float(np.median(vals)*1000),'p95_ms':float(np.percentile(vals,95)*1000),'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'pytorch_parameter_dtypes':[[str(v.dtype) for v in list(m.predictor.model.model.parameters())[:1]] for m in models] if backend=='pytorch' else None})
 (out/'measurements.json').write_text(json.dumps(records,indent=2));print('COMPLETE',key,summary[-1],flush=True)
 del models;torch.cuda.empty_cache()
reference=predictions['pytorch_fp32'];agreement=[]
for key,values in predictions.items():
 if key=='pytorch_fp32':continue
 rel=[];counts=[]
 for i,v in enumerate(values):
  ref=reference[i];rel.append(float(np.mean(np.abs(v['depth']-ref['depth'])/np.maximum(np.abs(ref['depth']),1e-6))));counts.append({'image':paths[i].name,'reference_boxes':len(ref['boxes']),'variant_boxes':len(v['boxes'])})
 agreement.append({'backend':key,'depth_relative_difference_to_fp32_mean':float(np.mean(rel)),'detection_counts':counts,'interpretation':'numeric agreement only, not accuracy or metric depth calibration'})
(out/'summary.json').write_text(json.dumps({'latency':summary,'agreement':agreement,'shape':[1,3,640,640],'same_inputs':True,'includes':'sequential detection+depth, preprocessing and CPU output copies; excludes camera acquisition/file decode'},indent=2))
