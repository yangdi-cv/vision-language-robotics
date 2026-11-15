"""One operator-gated 20% VLA residual step, then hold. No automatic rollout."""
import sys,json,time,hashlib,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'SO101/code/lerobot/src'),str(ROOT/'experiments/so101_pickplace/scripts')]
import torch,numpy as np,cv2
from PIL import Image
from read_follower import PORT,NAMES,read_register,serial
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy,load_smolvla
fraction=float(os.environ.get('VLA_INITIAL_FRACTION','0.2'))
if fraction not in (.2,.5):raise ValueError('Allowed fractions: 0.2, 0.5')
RUN=ROOT/'experiments/smolvla_pc/runs/approach_refined_v1';OUT=ROOT/'experiments/smolvla_pc/runs'/('fractional_'+str(time.time_ns()));OUT.mkdir()
def emit(event,**kw):
 r={'event':event,'monotonic':time.monotonic(),**kw}
 with (OUT/'motion.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
 print(json.dumps(r),flush=True)
def write(b,i,addr,value,size):
 assert 1<=i<=5 and addr in (21,40,42)
 body=bytes([i,size+3,3,addr])+int(value).to_bytes(size,'little');b.reset_input_buffer();b.write(b'\xff\xff'+body+bytes([(~sum(body))&255]));r=b.read(6)
 if len(r)!=6 or r[:2]!=b'\xff\xff' or r[2]!=i or r[3]!=2 or r[4] or sum(r[2:])&255!=255:raise RuntimeError('Write acknowledgement failed')
def pos(b):return [read_register(b,i,56,2) for i in range(1,7)]
cal=json.loads((ROOT/'SO101/calibration/robots/so_follower/my_awesome_follower_arm.json').read_text())
def state(raw):return [(v-(cal[n]['range_min']+cal[n]['range_max'])/2)*360/4095 if j<5 else (v-cal[n]['range_min'])*100/(cal[n]['range_max']-cal[n]['range_min']) for j,(n,v) in enumerate(zip(NAMES,raw))]
gate=json.loads((RUN/'offline_gate.json').read_text());checkpoint=RUN/'best_checkpoint'
assert hashlib.sha256((checkpoint/'model.safetensors').read_bytes()).hexdigest()==gate['weights_sha256']
d=json.loads((RUN/'normalization.json').read_text());stats={k:{n:torch.tensor(v,dtype=torch.float32) for n,v in r.items()} for k,r in d.items()}
cfg=PreTrainedConfig.from_pretrained(str(checkpoint));cfg.load_vlm_weights=False
model=SmolVLAPolicy(cfg,dataset_stats=stats);load_smolvla(model,checkpoint/'model.safetensors');model.cuda().eval()
cap=cv2.VideoCapture('/dev/v4l/by-id/usb-icSpring_icspring_camera-video-index0',cv2.CAP_V4L2);cap.set(3,640);cap.set(4,480);cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
if not cap.isOpened():raise RuntimeError('Camera unavailable')
task=json.loads((RUN/'config.json').read_text())['task'];torch.manual_seed(42);noise=torch.randn(1,cfg.chunk_size,cfg.max_action_dim,device='cuda')
with serial.Serial(PORT,1000000,timeout=.3,exclusive=True) as b:
 if [read_register(b,i,40,1) for i in range(1,7)]!=[1,1,1,1,1,0]:raise RuntimeError('Expected current holding state')
 lo=[read_register(b,i,9,2) for i in range(1,6)];hi=[read_register(b,i,11,2) for i in range(1,6)]
 emit('connected_holding',actual=pos(b),fraction=fraction,maximum_step_degrees=10 if fraction==.5 else 5,gripper_driven=False)
 off_authorized=False
 try:
  while True:
   try:
    for _ in range(4):
     ok,picture=cap.read()
     if not ok:raise RuntimeError('Camera read failed')
    stamp=time.monotonic();raw=pos(b);q=state(raw);goals=[read_register(b,i,42,2) for i in range(1,6)]
    if max(abs(v-g) for v,g in zip(raw[:5],goals))>68:raise RuntimeError('Following error before inference')
    health=[(read_register(b,i,63,1),read_register(b,i,13,1)) for i in range(1,6)]
    if any(temp>=min(60,limit-10) for temp,limit in health):raise RuntimeError('Temperature reaches trial limit (<=60C, >=10C below configured shutdown)')
    im=Image.fromarray(cv2.cvtColor(picture,cv2.COLOR_BGR2RGB));im.thumbnail((256,256));canvas=Image.new('RGB',(256,256));canvas.paste(im,((256-im.width)//2,(256-im.height)//2))
    tensor=torch.from_numpy(np.asarray(canvas).copy()).permute(2,0,1)[None].float().cuda()/255;model.reset()
    with torch.inference_mode():a=model.predict_action_chunk({'observation.images.camera1':tensor,'observation.state':torch.tensor([q],device='cuda',dtype=torch.float32),'task':[task]},noise=noise.clone())[0,0].cpu().float().numpy()
    delta=fraction*(a[:5]-np.array(q[:5]))
    if not np.isfinite(a).all() or time.monotonic()-stamp>.8:raise RuntimeError('Invalid or stale prediction')
    limit=10 if fraction==.5 else 5
    if max(abs(delta))>limit:raise RuntimeError(f'{fraction:.0%} model residual exceeds {limit}-degree single-step envelope')
    if max(abs(v-w) for v,w in zip(pos(b)[:5],raw[:5]))>6:raise RuntimeError('Pose changed during inference')
    target=[round(r+d*4095/360) for r,d in zip(raw[:5],delta)]
    if not all(l<=v<=h for l,v,h in zip(lo,target,hi)):raise RuntimeError('Target outside motor limits')
    emit('proposal',fraction=fraction,state=q,model_target=a.tolist(),fractional_delta_deg=delta.tolist(),target_raw=target)
    canvas.save(OUT/f'input_{time.time_ns()}.jpg')
    steps=max(1,int(np.ceil(max(abs(t-r) for t,r in zip(target,raw[:5]))/3)));previous=raw[:5]
    for k in range(1,steps+1):
     actual=pos(b)
     if max(abs(v-g) for v,g in zip(actual[:5],previous))>34:raise RuntimeError('Following error during interpolation >3 degrees')
     command=[round(r+(t-r)*k/steps) for r,t in zip(raw[:5],target)]
     for i,(old,new) in enumerate(zip(previous,command),1):
      if old!=new:write(b,i,42,new,2)
     previous=command;time.sleep(.1)
    time.sleep(.3);emit('step_completed_holding',before=raw,actual=pos(b),goal=target)
   except BaseException as exc:emit('stopped_holding',error=str(exc))
   command=input('Holding. NEXT=20%, NEXT50=50%; OFF only after support: ').strip()
   if command=='OFF':off_authorized=True;break
   if command not in ('NEXT','NEXT50'):
    while command not in ('NEXT','NEXT50','OFF'):command=input('Type NEXT, NEXT50 or OFF: ').strip()
    if command=='OFF':off_authorized=True;break
   fraction=.5 if command=='NEXT50' else .2
 finally:
  cap.release()
  # Normal OFF follows explicit support confirmation; unexpected process kill
  # leaves current servo goals holding and requires physical power cutoff.
  if off_authorized:
   for i in range(1,6):write(b,i,40,0,1)
   for i in (2,3):write(b,i,21,16,1)
   emit('torque_off',state=[read_register(b,i,40,1) for i in range(1,7)])
  else:emit('exit_holding_torque_on')
