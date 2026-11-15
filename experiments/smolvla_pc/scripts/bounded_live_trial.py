"""Five-second operator-authorized SmolVLA trial, bounded to 22 ticks per arm joint."""
import sys,time,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'SO101/code/lerobot/src'),str(ROOT/'experiments/so101_pickplace/scripts'),str(ROOT/'perception/qwen_scene')]
import numpy as np,torch
from frame_transport import SharedFrame
from read_follower import PORT,NAMES,read_register,serial
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy,load_smolvla
OUT=ROOT/'experiments/smolvla_pc/runs'/('bounded_live_'+str(time.time_ns()));OUT.mkdir(parents=True)
TELEMETRY=ROOT/'experiments/smolvla_pc/runs/joint_latest.json'
cal=json.loads((ROOT/'SO101/calibration/robots/so_follower/my_awesome_follower_arm.json').read_text())
def emit(event,**kw):
 row={'event':event,'monotonic':time.monotonic(),**kw}
 with (OUT/'motion.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
 print(json.dumps(row),flush=True)
def state(raw):return [(v-(cal[n]['range_min']+cal[n]['range_max'])/2)*360/4095 if j<5 else (v-cal[n]['range_min'])*100/(cal[n]['range_max']-cal[n]['range_min']) for j,(v,n) in enumerate(zip(raw,NAMES))]
def publish(raw,torque):
 r={'monotonic':time.monotonic(),'raw':raw,'state':state(raw),'torque':torque};p=TELEMETRY.with_suffix('.tmp');p.write_text(json.dumps(r));p.replace(TELEMETRY)
def write(b,i,a,v,n):
 assert 1<=i<=5 and a in (40,42)
 body=bytes([i,n+3,3,a])+int(v).to_bytes(n,'little');b.reset_input_buffer();b.write(b'\xff\xff'+body+bytes([(~sum(body))&255]));r=b.read(6)
 if len(r)!=6 or r[:2]!=b'\xff\xff' or r[2]!=i or r[3]!=2 or r[4] or sum(r[2:])&255!=255:raise RuntimeError('Motor acknowledgement failed')
def positions(b):return [read_register(b,i,56,2) for i in range(1,7)]
rows=[json.loads(x) for x in (ROOT/'experiments/so101_pickplace/recordings/grasp_rehearsal_20261025_03/trajectory.jsonl').read_text().splitlines()]
stats={k:{'mean':torch.tensor(np.array([r[f] for r in rows]).mean(0),dtype=torch.float32),'std':torch.tensor(np.array([r[f] for r in rows]).std(0).clip(.01),dtype=torch.float32)} for k,f in [('observation.state','state'),('action','action')]}
cfg=PreTrainedConfig.from_pretrained(str(ROOT/'models/smolvla_base'));cfg.load_vlm_weights=False
model=SmolVLAPolicy(cfg,dataset_stats=stats);load_smolvla(model,ROOT/'models/smolvla_base/model.safetensors');model.to('cuda').eval()
session=max((ROOT/'records/scene_understanding').glob('web_*'),key=lambda x:x.stat().st_mtime);frame=SharedFrame(session)
torch.manual_seed(42);noise=torch.randn(1,cfg.chunk_size,cfg.max_action_dim,device='cuda')
with serial.Serial(PORT,1000000,timeout=.3,exclusive=True) as b:
 raw=positions(b);origin=raw[:5];previous=origin.copy();enabled=[]
 assert [read_register(b,i,40,1) for i in range(1,7)]==[0]*6
 assert max(abs(a-c) for a,c in zip(raw,[2041,1117,3570,2643,975,1580]))<=10,'Pose changed since authorization'
 for i,v in enumerate(origin,1):
  assert read_register(b,i,9,2)<=v-22 and v+22<=read_register(b,i,11,2)
  assert read_register(b,i,63,1)<55
 emit('preflight',raw=raw,max_excursion_ticks=22,max_step_ticks=2,gripper_driven=False)
 try:
  started=None
  for k in range(10):
   raw=positions(b);publish(raw,[1 if i in enabled else 0 for i in range(1,7)])
   if max(abs(a-c) for a,c in zip(raw[:5],previous))>34:raise RuntimeError('Following error >3 degrees')
   image,meta=frame.read()
   if not 0<=time.monotonic_ns()-meta['captured_ns']<600_000_000:raise RuntimeError('Stale camera')
   q=json.loads((session/'latest_result.json').read_text())
   if q.get('scene',{}).get('target_visible') is not True:raise RuntimeError('Target not visible')
   pic=torch.from_numpy(np.asarray(image).copy()).permute(2,0,1)[None].float().to('cuda')/255
   model.reset()
   with torch.inference_mode():prediction=model.predict_action_chunk({'observation.images.camera1':pic,'observation.state':torch.tensor([state(raw)],device='cuda',dtype=torch.float32),'task':['Pick up the blue earbud case and place it in the box.']},noise=noise.clone())
   a=prediction[0,0,:5].float().cpu().numpy()
   if not np.isfinite(a).all():raise RuntimeError('Nonfinite model action')
   if time.monotonic_ns()-meta['captured_ns']>1_000_000_000:raise RuntimeError('Prediction too old')
   actual=positions(b)
   if max(abs(x-y) for x,y in zip(actual[:5],raw[:5]))>6:raise RuntimeError('Arm moved during inference')
   if started is None:
    for i,v in enumerate(origin,1):write(b,i,42,v,2)
    for i in range(1,6):enabled.append(i);write(b,i,40,1,1)
    started=time.monotonic()
   if time.monotonic()-started>=5:break
   targets=[round(float(v)*4095/360+(cal[n]['range_min']+cal[n]['range_max'])/2) for v,n in zip(a,NAMES)]
   command=[max(o-22,min(o+22,p+max(-2,min(2,t-p)))) for o,p,t in zip(origin,previous,targets)]
   for i,(old,new) in enumerate(zip(previous,command),1):
    if old!=new:write(b,i,42,new,2)
   emit('model_step',index=k,actual=actual,model_first_action=a.tolist(),command=command,frame=meta['frame_id']);previous=command
   time.sleep(.25)
  emit('trial_stopped_holding',actual=positions(b),goal=previous)
 except BaseException as e:emit('stopped',error=str(e))
 finally:
  if enabled:
   # Keep fresh measured feedback while waiting; no further trajectory commands.
   import select
   print('Holding. Support white arm then type OFF:',flush=True)
   while True:
    try:publish(positions(b),[1,1,1,1,1,0])
    except Exception as e:emit('telemetry_error',error=str(e))
    if select.select([sys.stdin],[],[],.25)[0]:
     if sys.stdin.readline().strip()=='OFF':break
   for i in enabled:write(b,i,40,0,1)
   t=[read_register(b,i,40,1) for i in range(1,7)];publish(positions(b),t);emit('torque_off',state=t)
frame.close()
