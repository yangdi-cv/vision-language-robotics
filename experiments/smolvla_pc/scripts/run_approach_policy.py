"""Live fine-tuned SmolVLA approach. Defaults to prediction-only; explicit execute gate."""
import argparse,json,time,sys,select,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'SO101/code/lerobot/src'),str(ROOT/'experiments/so101_pickplace/scripts')]
import numpy as np,torch,cv2
from PIL import Image
from read_follower import PORT,NAMES,read_register,serial
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy,load_smolvla

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run',type=Path,default=ROOT/'experiments/smolvla_pc/runs/approach_refined_v1');ap.add_argument('--execute',action='store_true');ap.add_argument('--resume-hold',type=Path);ap.add_argument('--seconds',type=float,default=8);ap.add_argument('--max-excursion',type=float,choices=(8,12),default=8);ap.add_argument('--confirmed-extension',action='store_true');args=ap.parse_args()
 if not 0<args.seconds<=8:raise ValueError('Trial limited to eight seconds')
 if args.max_excursion>8 and (not args.resume_hold or not args.confirmed_extension):raise ValueError('Extension requires resumed hold and explicit onsite confirmation')
 max_ticks=round(args.max_excursion*4095/360)
 run=args.run;norm=json.loads((run/'normalization.json').read_text());stats={k:{s:torch.tensor(v,dtype=torch.float32) for s,v in d.items()} for k,d in norm.items()}
 config=json.loads((run/'config.json').read_text());checkpoint=run/'best_checkpoint'
 if not (run/'offline_gate.json').exists():raise RuntimeError('Offline gate has not been reviewed')
 gate=json.loads((run/'offline_gate.json').read_text())
 if hashlib.sha256((checkpoint/'model.safetensors').read_bytes()).hexdigest()!=gate.get('weights_sha256'):raise RuntimeError('Checkpoint changed after offline audit; repeat audit')
 if args.execute and gate.get('bounded_trial_allowed') is not True:raise RuntimeError('Offline gate rejected execution')
 cfg=PreTrainedConfig.from_pretrained(str(checkpoint));cfg.load_vlm_weights=False
 model=SmolVLAPolicy(cfg,dataset_stats=stats);load_smolvla(model,checkpoint/'model.safetensors');model.cuda().eval()
 cal=json.loads((ROOT/'SO101/calibration/robots/so_follower/my_awesome_follower_arm.json').read_text())
 out=ROOT/'experiments/smolvla_pc/runs'/('approach_live_'+str(time.time_ns()));out.mkdir()
 def emit(event,**kw):
  r={'event':event,'monotonic':time.monotonic(),**kw}
  with (out/'motion.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
  print(json.dumps(r),flush=True)
 def read(b,i,a,n):return read_register(b,i,a,n)
 def pos(b):return [read(b,i,56,2) for i in range(1,7)]
 def write(b,i,a,v,n):
  if not args.execute:raise RuntimeError('Prediction-only mode')
  assert i in range(1,6) and a in (21,40,42)
  body=bytes([i,n+3,3,a])+int(v).to_bytes(n,'little');b.reset_input_buffer();b.write(b'\xff\xff'+body+bytes([(~sum(body))&255]));r=b.read(6)
  if len(r)!=6 or r[:2]!=b'\xff\xff' or r[2]!=i or r[3]!=2 or r[4] or sum(r[2:])&255!=255:raise RuntimeError('Motor acknowledgement failed')
 def normalized(raw):return [(v-(cal[n]['range_min']+cal[n]['range_max'])/2)*360/4095 if j<5 else (v-cal[n]['range_min'])*100/(cal[n]['range_max']-cal[n]['range_min']) for j,(n,v) in enumerate(zip(NAMES,raw))]
 cap=cv2.VideoCapture('/dev/v4l/by-id/usb-icSpring_icspring_camera-video-index0',cv2.CAP_V4L2);cap.set(3,640);cap.set(4,480);cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
 if not cap.isOpened():raise RuntimeError('Camera unavailable')
 torch.manual_seed(42);noise=torch.randn(1,cfg.chunk_size,cfg.max_action_dim,device='cuda')
 with serial.Serial(PORT,1000000,timeout=.3,exclusive=True) as b:
  raw=pos(b);origin=raw[:5];previous=origin.copy();enabled=[];gains={};lo=[];hi=[]
  torques=[read(b,i,40,1) for i in range(1,7)]
  if args.resume_hold:
   if torques!=[1,1,1,1,1,0]:raise RuntimeError('Expected arm holding torque, gripper off')
   preflight=next(json.loads(x) for x in (args.resume_hold/'motion.jsonl').read_text().splitlines() if json.loads(x).get('event')=='preflight')
   origin=preflight['raw'][:5];previous=[read(b,i,42,2) for i in range(1,6)]
   if max(abs(a-c) for a,c in zip(previous,origin))>91:raise RuntimeError('Previous command outside original trial envelope')
   if max(abs(a-c) for a,c in zip(raw[:5],previous))>68:raise RuntimeError('Following error before resume')
   enabled=list(range(1,6));gains={2:16,3:16}
  elif torques!=[0]*6:raise RuntimeError('Expected torque off')
  for i,v in enumerate(origin,1):
   lo.append(read(b,i,9,2));hi.append(read(b,i,11,2))
   if not lo[-1]<=v<=hi[-1]:raise RuntimeError('Current pose outside motor limits')
   if read(b,i,63,1)>=55:raise RuntimeError('Temperature limit')
  if args.execute:input('Confirm arm supported, workspace clear, power cutoff ready. ENTER starts bounded model control: ')
  initial=np.array(normalized(origin+[raw[5]])[:5]);starts=np.array(gate.get('recorded_start_states',[]))
  if args.execute and (starts.size==0 or np.any(initial<starts.min(axis=0)-8) or np.any(initial>starts.max(axis=0)+8)):
   raise RuntimeError('Place arm near the recorded upright start envelope before model control')
  emit('preflight',raw=origin+[raw[5]],measured_raw=raw,resume=bool(args.resume_hold),execute=args.execute,checkpoint=str(checkpoint),max_total_degrees=args.max_excursion,gripper_driven=False)
  try:
   started=time.monotonic();iteration=0
   while time.monotonic()-started<args.seconds:
    raw=pos(b)
    if max(abs(a-c) for a,c in zip(raw[:5],previous))>68:raise RuntimeError('Following error exceeds 6 degrees')
    ok,frame=cap.read();stamp=time.monotonic()
    if not ok:raise RuntimeError('Camera read failed')
    im=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));im.thumbnail((256,256));canvas=Image.new('RGB',(256,256));canvas.paste(im,((256-im.width)//2,(256-im.height)//2))
    tensor=torch.from_numpy(np.asarray(canvas).copy()).permute(2,0,1)[None].float().cuda()/255
    model.reset()
    with torch.inference_mode():prediction=model.predict_action_chunk({'observation.images.camera1':tensor,'observation.state':torch.tensor([normalized(raw)],device='cuda',dtype=torch.float32),'task':[config['task']]},noise=noise.clone())
    a=prediction[0,0].float().cpu().numpy()
    if not np.isfinite(a).all() or time.monotonic()-stamp>.8:raise RuntimeError('Invalid or stale prediction')
    latest=pos(b)
    if not enabled and max(abs(x-y) for x,y in zip(latest[:5],raw[:5]))>6:
     raise RuntimeError('Arm moved before torque enable; re-evaluate pose')
    if enabled and max(abs(x-y) for x,y in zip(latest[:5],previous))>68:
     raise RuntimeError('Following error during prediction exceeds 6 degrees')
    targets=[round(float(v)*4095/360+(cal[n]['range_min']+cal[n]['range_max'])/2) for v,n in zip(a,NAMES)]
    command=[max(l,o-max_ticks,min(h,o+max_ticks,p+max(-4,min(4,t-p)))) for o,p,t,l,h in zip(origin,previous,targets,lo,hi)]
    if args.execute:
     if not enabled:
      for i,v in enumerate(raw[:5],1):write(b,i,42,v,2)
      for i,g in [(2,28),(3,24)]:gains[i]=read(b,i,21,1);write(b,i,21,g,1)
      for i in range(1,6):enabled.append(i);write(b,i,40,1,1)
     for i,(old,new) in enumerate(zip(previous,command),1):
      if old!=new:write(b,i,42,new,2)
     previous=command
    emit('prediction',index=iteration,state=normalized(raw),model_first_action=a.tolist(),command=command,executed=args.execute)
    iteration+=1;time.sleep(.1)
   emit('trial_ended',actual=pos(b),goal=previous)
  except BaseException as e:emit('stopped',error=str(e))
  finally:
   cap.release()
   if enabled:
    while input('Holding. Support white arm then type OFF: ').strip()!='OFF':pass
    for i in enabled:write(b,i,40,0,1)
    for i,g in gains.items():write(b,i,21,g,1)
    emit('torque_off',state=[read(b,i,40,1) for i in range(1,7)])
if __name__=='__main__':main()
