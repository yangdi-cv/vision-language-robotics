"""Quantized Qwen on latest wrist frames; visual observations only, no motor interface."""
import argparse,json,os,time,threading,hashlib,resource
from pathlib import Path
from unittest.mock import patch
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--cycles',type=int,default=4);p.add_argument('--tokens',type=int,default=64);p.add_argument('--integrated-cycles',type=int,default=4);p.add_argument('--skip-saved',action='store_true');p.add_argument('--output-name',default='qwen_live');a=p.parse_args()
r=a.root/a.output_name;r.mkdir(parents=True,exist_ok=False)
os.environ['YOLO_AUTOINSTALL']='false';os.environ['YOLO_CONFIG_DIR']=str(a.root/'settings')
def write(n,v):(r/n).write_text(json.dumps(v,ensure_ascii=False,indent=2))
write('status.json',{'phase':'imports','pid':os.getpid()})
import torch,cv2,numpy as np,bitsandbytes as bnb
from PIL import Image
from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
# Process-local allocator preparation reused from the user's existing Jetson loader.
t=time.perf_counter();pool=[torch.empty(640*1024*1024,dtype=torch.uint8,device='cuda')]+[torch.empty(64*1024*1024,dtype=torch.uint8,device='cuda') for _ in range(22)]
for block in pool:block.zero_()
del block;torch.cuda.synchronize();del pool
write('status.json',{'phase':'loading'})
processor=AutoProcessor.from_pretrained(a.model,local_files_only=True)
with patch('transformers.modeling_utils.caching_allocator_warmup',lambda *args,**kwargs:None):
 model=Qwen3VLForConditionalGeneration.from_pretrained(a.model,dtype=torch.bfloat16,device_map={'':0},local_files_only=True).eval()
write('runtime.json',{'host_wall_s':time.time(),'model_path':str(a.model),'model_type':model.config.model_type,'text_parameters_hidden_size':model.config.text_config.hidden_size,'quantized_linear_count':sum(isinstance(m,bnb.nn.Linear4bit) for m in model.modules()),'adapter_loaded':False,'torch':torch.__version__,'model_load_s':time.perf_counter()-t,'config_sha256':hashlib.sha256((a.model/'config.json').read_bytes()).hexdigest(),'note':'Qwen3-VL-2B NF4 existing prequantized model; different from PC 4B, no apples-to-apples quality claim'})
lock=threading.Lock();stop=threading.Event();latest={};errors=[]
def camera():
 cap=cv2.VideoCapture('/dev/v4l/by-id/usb-icSpring_icspring_camera-video-index0',cv2.CAP_V4L2);cap.set(cv2.CAP_PROP_BUFFERSIZE,1);seq=0
 try:
  while not stop.is_set():
   ok,f=cap.read()
   if not ok:raise RuntimeError('camera read failed')
   stamp=time.monotonic_ns();f=cv2.rotate(f,cv2.ROTATE_90_COUNTERCLOCKWISE)
   with lock:latest.update(frame=f,stamp=stamp,seq=seq)
   seq+=1
 except BaseException as e:errors.append(str(e))
 finally:cap.release()
thread=threading.Thread(target=camera,daemon=True);thread.start();results=[]
vision_results=[];vision_thread=None;vision_models=None

def prepare_vision():
 import sys
 sys.path.insert(0,str(a.root/'deps'))
 sys.path.append('/usr/lib/python3.10/dist-packages')
 from ultralytics import YOLO
 global vision_models
 det=YOLO(str(a.root/'models/yolo26n.engine'))
 depth=YOLO(str(a.root/'models/yolo26n-depth.engine'))
 deadline=time.monotonic()+10
 while not latest and time.monotonic()<deadline:time.sleep(.02)
 with lock:frame=latest['frame'].copy()
 # TensorRT's static-graph capture must finish while no Qwen CUDA work is in flight.
 for m in [det,depth]:
  for _ in range(3):m(frame,imgsz=640,rect=False,device=0,conf=.35,verbose=False)
 torch.cuda.synchronize()
 vision_models=(det,depth)

def vision_loop():
 det,depth=vision_models
 try:
  while not stop.is_set():
   with lock:
    if not latest:continue
    frame=latest['frame'].copy();stamp=latest['stamp'];seq=latest['seq']
   t=time.perf_counter()
   boxes=det(frame,imgsz=640,rect=False,device=0,conf=.35,verbose=False)[0]
   d=depth(frame,imgsz=640,rect=False,device=0,verbose=False)[0].depth.data.float().cpu().numpy()
   elapsed=time.perf_counter()-t
   vision_results.append({'frame_sequence':seq,'host_frame_receipt_ns':stamp,'seconds':elapsed,'result_age_ms':(time.monotonic_ns()-stamp)/1e6,'boxes':len(boxes.boxes),'depth_finite_fraction':float(np.isfinite(d).mean())})
   with (r/'vision_results.jsonl').open('a') as journal:journal.write(json.dumps(vision_results[-1])+'\n')
   if len(vision_results)==5:
    color=cv2.applyColorMap(255-cv2.normalize(d,None,0,255,cv2.NORM_MINMAX).astype(np.uint8),cv2.COLORMAP_TURBO)
    cv2.imwrite(str(r/'integrated_rgb_boxes_depth.jpg'),np.hstack([frame,boxes.plot(),color]))
   stop.wait(.15)
 except BaseException as e:
  errors.append('vision: '+repr(e));write('vision_error.json',{'error':repr(e)})
 finally:
  write('vision_results.json',vision_results)
prompt='Observe only this image. Return a compact JSON object with keys visible_objects (short English list), gripper_visible (boolean), target_visible (boolean or null). The target is a dark blue plastic earbud case. Do not infer objects outside the image. Do not output commands.'
try:
 cases=[(a.root/folder/f'frame_{index:03}.jpg',visible) for folder,visible in [('scene',True),('scene_absent',False),('scene_shifted',True),('scene_partial',True)] for index in [5,10] if (a.root/folder/f'frame_{index:03}.jpg').exists()]
 if a.skip_saved:cases=[]
 for cycle in range(a.cycles+len(cases)+a.integrated_cycles):
  expected=None;source='live_camera'
  integrated=cycle>=a.cycles+len(cases)
  if integrated and vision_thread is None:
   prepare_vision()
   vision_thread=threading.Thread(target=vision_loop,daemon=True);vision_thread.start()
  if cycle<a.cycles or integrated:
   deadline=time.monotonic()+10
   while not latest and time.monotonic()<deadline:time.sleep(.02)
   with lock:
    if not latest:raise RuntimeError('No camera frame: '+str(errors))
    frame=latest['frame'].copy();capture=latest['stamp'];seq=latest['seq']
   chosen=time.monotonic_ns();image=r/f'input_{cycle:02}.png';cv2.imwrite(str(image),frame);image_content=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
  else:
   image,expected=cases[cycle-a.cycles];source='saved_camera_case';capture=chosen=time.monotonic_ns();seq=-1;image_content=Image.open(image).convert('RGB')
  messages=[{'role':'user','content':[{'type':'image','image':image_content},{'type':'text','text':prompt}]}]
  torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();wall_start=time.time();t=time.perf_counter()
  batch=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt').to(model.device)
  prep=time.perf_counter()-t
  with torch.inference_mode():output=model.generate(**batch,max_new_tokens=a.tokens,do_sample=False)
  torch.cuda.synchronize();elapsed=time.perf_counter()-t;tokens=output[0][len(batch.input_ids[0]):];text=processor.decode(tokens,skip_special_tokens=True)
  try:parsed=json.loads(text[text.index('{'):text.rindex('}')+1]);valid=isinstance(parsed.get('visible_objects'),list) and isinstance(parsed.get('gripper_visible'),bool) and parsed.get('target_visible') in [True,False,None]
  except Exception:parsed=None;valid=False
  item={'integrated_vision':integrated,'inference_start_wall_s':wall_start,'inference_end_wall_s':time.time(),'process_peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'cycle':cycle,'source':source,'image':str(image),'expected_target_visible':expected,'frame_sequence':seq,'host_frame_receipt_ns':capture,'selected_ns':chosen,'input_age_at_selection_ms':(chosen-capture)/1e6 if source=='live_camera' else None,'result_frame_age_s':(time.monotonic_ns()-capture)/1e9 if source=='live_camera' else None,'preprocess_s':prep,'total_inference_s':elapsed,'tokens':len(tokens),'max_new_tokens':a.tokens,'raw':text,'parsed':parsed,'schema_valid':valid,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(),'warmup':cycle==0 or cycle==a.cycles+len(cases),'target_visibility_correct':(parsed.get('target_visible') is expected) if parsed is not None and expected is not None else None}
  results.append(item);write('results.json',results);print(json.dumps(item,ensure_ascii=False),flush=True)
  write('status.json',{'phase':'running','completed':len(results)})
finally:
 stop.set();thread.join(timeout=5)
 if vision_thread is not None:vision_thread.join(timeout=10)
write('status.json',{'phase':'complete','completed':len(results),'camera_errors':errors,'frame_selection':'latest host-received frame at start of each inference; no pending inference queue'})
