"""Continuous latest-frame SmolVLA predictions; no serial or motor writes."""
import sys,time,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];RUN=Path(sys.argv[1]);sys.path[:0]=[str(ROOT/'SO101/code/lerobot/src'),str(ROOT/'perception/qwen_scene')]
import torch,numpy as np
from frame_transport import SharedFrame
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy,load_smolvla

def publish(r):
 t=RUN/'latest.tmp';t.write_text(json.dumps(r,ensure_ascii=False));t.replace(RUN/'latest.json')
try:
 publish({'state':'loading'})
 record=ROOT/'experiments/so101_pickplace/recordings/grasp_rehearsal_20261025_03'
 rows=[json.loads(x) for x in (record/'trajectory.jsonl').read_text().splitlines()]
 stats={k:{'mean':torch.tensor(np.array([r[f] for r in rows]).mean(0),dtype=torch.float32),'std':torch.tensor(np.array([r[f] for r in rows]).std(0).clip(.01),dtype=torch.float32)} for k,f in [('observation.state','state'),('action','action')]}
 cfg=PreTrainedConfig.from_pretrained(str(ROOT/'models/smolvla_base'));cfg.load_vlm_weights=False
 model=SmolVLAPolicy(cfg,dataset_stats=stats);load_smolvla(model,ROOT/'models/smolvla_base/model.safetensors');model.to('cuda').eval()
 session=max((ROOT/'records/scene_understanding').glob('web_*'),key=lambda x:x.stat().st_mtime);frame=SharedFrame(session)
 # The existing holding controller owns serial. Use its last measured snapshot,
 # label it explicitly, and never present these states as live telemetry.
 logs=sorted((ROOT/'experiments/smolvla_pc/runs').glob('direction_test_*/motion.jsonl'))
 measured=None
 for line in logs[-1].read_text().splitlines():
  x=json.loads(line)
  if x.get('event')=='probe_completed_holding':measured=x
 if measured is None:raise RuntimeError('No measured joint snapshot available')
 cal=json.loads((ROOT/'SO101/calibration/robots/so_follower/my_awesome_follower_arm.json').read_text())
 names=['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper']
 audit=max((ROOT/'experiments/smolvla_pc/runs').glob('live_*/audit.json'),key=lambda x:x.stat().st_mtime)
 prior=json.loads(audit.read_text());raw=measured['actual']+[prior['raw_position'][5]]
 state=[(v-(cal[n]['range_min']+cal[n]['range_max'])/2)*360/4095 if j<5 else (v-cal[n]['range_min'])*100/(cal[n]['range_max']-cal[n]['range_min']) for j,(n,v) in enumerate(zip(names,raw))]
 state_t=torch.tensor([state],device='cuda',dtype=torch.float32)
 torch.manual_seed(42);noise=torch.randn(1,cfg.chunk_size,cfg.max_action_dim,device='cuda')
 iteration=0;last=-1
 while True:
  telemetry=ROOT/'experiments/smolvla_pc/runs/joint_latest.json'
  source='snapshot';state_age=time.monotonic()-measured['monotonic']
  if telemetry.exists():
   live=json.loads(telemetry.read_text());age=time.monotonic()-live['monotonic']
   if 0<=age<1:
    state=live['state'];state_t=torch.tensor([state],device='cuda',dtype=torch.float32);source='live';state_age=age
  image,meta=frame.read()
  if meta['frame_id']==last or time.monotonic_ns()-meta['captured_ns']>1e9:time.sleep(.05);continue
  last=meta['frame_id'];taskfile=RUN/'task.json'
  task=json.loads(taskfile.read_text())['task'] if taskfile.exists() else 'Pick up the blue earbud case and place it in the box.'
  pic=torch.from_numpy(np.asarray(image).copy()).permute(2,0,1)[None].float().to('cuda')/255
  model.reset();torch.cuda.synchronize();start=time.perf_counter()
  with torch.inference_mode():pred=model.predict_action_chunk({'observation.images.camera1':pic,'observation.state':state_t,'task':[task]},noise=noise.clone())
  torch.cuda.synchronize();elapsed=time.perf_counter()-start;a=pred.cpu().float().numpy()[0]
  r={'state':'ready','iteration':iteration,'task':task,'seconds':elapsed,'frame_received_monotonic_ns':meta['captured_ns'],'frame_id':last,'joints':names,'joint_state':state,'first_action':a[0].tolist(),'delta':(a[0]-state).tolist(),'trajectory':a.tolist(),'finite':bool(np.isfinite(a).all()),'motion_enabled':False,'joint_state_source':source,'joint_state_age_s':state_age,'normalization_validated':False}
  publish(r)
  with (RUN/'predictions.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
  iteration+=1;time.sleep(.2)
except BaseException as exc:
 publish({'state':'error','error':str(exc),'motion_enabled':False});raise
