"""Read current joints and latest RGB; evaluate VLA actions without motor writes."""
import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'SO101/code/lerobot/src'),str(ROOT/'experiments/so101_pickplace/scripts'),str(ROOT/'perception/qwen_scene')]
import torch
import numpy as np
from frame_transport import SharedFrame
from read_follower import PORT,NAMES,read_register,serial
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy,load_smolvla

def main():
 import argparse
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--task',default='Pick up the blue earbud case and place it in the box.')
 args=ap.parse_args()
 out=ROOT/'experiments/smolvla_pc/runs'/('live_'+str(time.time_ns()));out.mkdir(parents=True)
 cal=json.loads((ROOT/'SO101/calibration/robots/so_follower/my_awesome_follower_arm.json').read_text())
 # No WRITE instruction (0x03), torque-enable or position-goal call exists here.
 with serial.Serial(PORT,1000000,timeout=.4,exclusive=True) as bus:
  raw=[read_register(bus,i,56,2) for i in range(1,7)]
  torque=[read_register(bus,i,40,1) for i in range(1,7)]
  temps=[read_register(bus,i,63,1) for i in range(1,7)]
 if torque!=[0]*6:raise RuntimeError('Expected all motors torque off for shadow evaluation')
 if max(temps)>=60:raise RuntimeError('Motor temperature >=60C')
 state=[]
 for i,name in enumerate(NAMES):
  c=cal[name];v=raw[i]
  if not c['range_min']<=v<=min(4095,c['range_max']):raise RuntimeError(f'{name}: measured raw position outside saved bounds')
  state.append((v-c['range_min'])*100/(c['range_max']-c['range_min']) if i==5 else (v-(c['range_min']+c['range_max'])/2)*360/4095)
 record=ROOT/'experiments/so101_pickplace/recordings/grasp_rehearsal_20261025_03'
 rows=[json.loads(s) for s in (record/'trajectory.jsonl').read_text().splitlines()]
 stats={k:{'mean':torch.tensor(np.array([r[f] for r in rows]).mean(0),dtype=torch.float32),'std':torch.tensor(np.array([r[f] for r in rows]).std(0).clip(.01),dtype=torch.float32)} for k,f in [('observation.state','state'),('action','action')]}
 cfg=PreTrainedConfig.from_pretrained(str(ROOT/'models/smolvla_base'));cfg.load_vlm_weights=False
 model=SmolVLAPolicy(cfg,dataset_stats=stats);load_smolvla(model,ROOT/'models/smolvla_base/model.safetensors');model.to('cuda').eval()
 session=max((ROOT/'records/scene_understanding').glob('web_*'),key=lambda x:x.stat().st_mtime)
 frame=SharedFrame(session)
 try:image,meta=frame.read()
 finally:frame.close()
 if not 0<time.monotonic_ns()-meta['captured_ns']<1_000_000_000:raise RuntimeError('Camera frame stale')
 # Re-read joints after model loading; use the state contemporaneous with RGB.
 with serial.Serial(PORT,1000000,timeout=.4,exclusive=True) as bus:
  raw_now=[read_register(bus,i,56,2) for i in range(1,7)]
  if max(abs(a-b) for a,b in zip(raw_now,raw))>3:raise RuntimeError('Arm moved during model load; repeat with stable support')
 image.save(out/'input.jpg')
 picture=torch.from_numpy(np.asarray(image).copy()).permute(2,0,1)[None].float().to('cuda')/255
 torch.manual_seed(42)
 with torch.inference_mode():a=model.predict_action_chunk({'observation.images.camera1':picture,'observation.state':torch.tensor([state],device='cuda'),'task':[args.task]})
 action=a.float().cpu().numpy()[0];np.save(out/'actions.npy',action)
 targets=[]
 for j,n in enumerate(NAMES):
  c=cal[n];targets.append(action[:,j]*(c['range_max']-c['range_min'])/100+c['range_min'] if j==5 else action[:,j]*4095/360+(c['range_min']+c['range_max'])/2)
 targets=np.stack(targets,axis=1)
 bounds_ok=all(np.all((targets[:,j]>=cal[n]['range_min'])&(targets[:,j]<=min(4095,cal[n]['range_max']))) for j,n in enumerate(NAMES))
 result={'task':args.task,'frame':meta,'raw_position':raw,'state':state,'first_action':action[0].tolist(),'first_step_delta':(action[0]-state).tolist(),'finite':bool(np.isfinite(action).all()),'all_chunk_targets_within_saved_limits':bool(bounds_ok),'motion_enabled':False,'execution_allowed':False,'reason':'Pretraining-to-local coordinates and task policy not validated; shadow output only'}
 (out/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False,indent=2));print('Saved:',out)
if __name__=='__main__':main()
