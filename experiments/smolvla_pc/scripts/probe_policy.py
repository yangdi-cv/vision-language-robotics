"""SmolVLA offline input/action probe. No serial or actuator interface."""
import json,time,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'SO101/code/lerobot/src'))
import numpy as np
import torch
from PIL import Image
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy,load_smolvla
OUT=ROOT/'experiments/smolvla_pc/runs/base_probe';OUT.mkdir(parents=True,exist_ok=True)
record=ROOT/'experiments/so101_pickplace/recordings/grasp_rehearsal_20261025_03'
rows=[json.loads(s) for s in (record/'trajectory.jsonl').read_text().splitlines()]
# Temporary local normalization is an experiment condition, not a validated
# conversion of the checkpoint's heterogeneous pretraining robot coordinates.
stats={key:{'mean':torch.tensor(np.array([r[field] for r in rows]).mean(0),dtype=torch.float32),'std':torch.tensor(np.array([r[field] for r in rows]).std(0).clip(.01),dtype=torch.float32)} for key,field in [('observation.state','state'),('action','action')]}
cfg=PreTrainedConfig.from_pretrained(str(ROOT/'models/smolvla_base'))
cfg.load_vlm_weights=False
model=SmolVLAPolicy(cfg,dataset_stats=stats)
load_smolvla(model,ROOT/'models/smolvla_base/model.safetensors')
model=model.to('cuda').eval()
print('Loaded pretrained SmolVLA',flush=True)
row=rows[340];image=Image.open(record/row['image']).convert('RGB')
image.save(OUT/'input.jpg')
image_tensor=torch.from_numpy(np.asarray(image).copy()).permute(2,0,1).unsqueeze(0).float().to('cuda')/255
state=torch.tensor([row['state']],device='cuda',dtype=torch.float32)
torch.manual_seed(42);noise=torch.randn(1,cfg.chunk_size,cfg.max_action_dim,device='cuda')
results=[];actions={}
for name,prompt,picture in [('pick','Pick up the blue earbud case and place it in the box.',image_tensor),('different_language','Move away from the blue earbud case.',image_tensor),('blank_image','Pick up the blue earbud case and place it in the box.',torch.zeros_like(image_tensor))]:
 model.reset();torch.cuda.synchronize();start=time.perf_counter()
 with torch.inference_mode():a=model.predict_action_chunk({'observation.images.camera1':picture,'observation.state':state,'task':[prompt]},noise=noise.clone())
 torch.cuda.synchronize();elapsed=time.perf_counter()-start
 arr=a.float().cpu().numpy();actions[name]=arr
 result={'condition':name,'prompt':prompt,'seconds':elapsed,'shape':list(arr.shape),'finite':bool(np.isfinite(arr).all()),'first_action':arr[0,0].tolist(),'max_first_step_from_recorded_state':float(abs(arr[0,0]-np.array(row['state'])).max())}
 results.append(result);print(json.dumps(result),flush=True)
np.savez(OUT/'actions.npz',**actions)
report={'checkpoint':'lerobot/smolvla_base','revision':'5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6','inference_only':True,'motion_enabled':False,'normalization':'mean/std from single local demonstration; coordinate/domain compatibility unvalidated','camera_mapping':'single wrist camera mapped to camera1; pretraining viewpoint compatibility unvalidated','conditions':results,'language_change_mean_abs_action_difference':float(abs(actions['pick']-actions['different_language']).mean()),'image_change_mean_abs_action_difference':float(abs(actions['pick']-actions['blank_image']).mean()),'same_diffusion_noise':True,'robot_success_rate_measured':False}
(OUT/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print('Saved',OUT,flush=True)
# Paired repeats distinguish conditioning sensitivity from sampling randomness.
paired=[]
for seed in [42,43,44]:
 torch.manual_seed(seed);shared_noise=torch.randn_like(noise);preds={}
 for name,prompt,picture in [('pick','Pick up the blue earbud case and place it in the box.',image_tensor),('different_language','Move away from the blue earbud case.',image_tensor),('blank_image','Pick up the blue earbud case and place it in the box.',torch.zeros_like(image_tensor))]:
  model.reset();torch.cuda.synchronize();start=time.perf_counter()
  with torch.inference_mode():prediction=model.predict_action_chunk({'observation.images.camera1':picture,'observation.state':state,'task':[prompt]},noise=shared_noise.clone())
  torch.cuda.synchronize();preds[name]=prediction.cpu().numpy()
  paired.append({'seed':seed,'condition':name,'seconds':time.perf_counter()-start,'finite':bool(np.isfinite(preds[name]).all())})
 for name in ['different_language','blank_image']:
  # Unitless difference scaled by this local record's per-joint action std.
  paired.append({'seed':seed,'comparison':name,'mean_abs_difference_in_local_std_units':float((abs(preds['pick']-preds[name])/stats['action']['std'].numpy()).mean())})
(OUT/'paired_repeats.json').write_text(json.dumps(paired,indent=2))
print('Paired seed repeats saved',flush=True)
