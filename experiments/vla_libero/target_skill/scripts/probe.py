"""One model per process: compare actual LoRA runtime with old/fixed merged deployment."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
S=Path(__file__).resolve().parents[1];V=S.parent;ROOT=V.parents[1]
sys.path[:0]=[str(V/'point_target/scripts'),str(V/'skill_bootstrap/scripts'),str(V/'scripts')]
from episode_journal import atomic_json
from goal_inputs import transforms_for
from train_lora import config as lora_config
import importlib.util
spec=importlib.util.spec_from_file_location('fixed_loader',S/'scripts/loader.py');loader=importlib.util.module_from_spec(spec);sys.path.insert(0,str(S/'scripts'));spec.loader.exec_module(loader)

def load(backend,adapter):
 import jax.numpy as jnp
 from flax.traverse_util import flatten_dict,unflatten_dict
 from openpi.training import config
 from openpi.policies.policy import Policy
 if backend=='unmerged':
  cfg=lora_config();params,_=loader.restore(V/'checkpoints/pi05_libero/params');flat=flatten_dict(params,sep='/');del params
  with np.load(adapter) as f:flat.update({k:jnp.asarray(f[k]) for k in f.files})
  model=cfg.model.load(unflatten_dict(flat,sep='/'));del flat
 else:
  cfg=config.get_config('pi05_libero')
  if backend=='old':
   from bounded_restore import restore
   params,_=restore(V/'checkpoints/pi05_libero/params',adapter)
  else:params,_=loader.restore(V/'checkpoints/pi05_libero/params',adapter)
  model=cfg.model.load(params);del params
 transform,outputs=transforms_for(cfg)
 return Policy(model,transforms=[transform],output_transforms=outputs,sample_kwargs={'num_steps':10})

def cases():
 for variant,task,target,q in [('goal',0,1,0),('goal',0,1,20),('goal',1,1,0),('goal',1,1,20),('text',1,1,0),('text',1,1,20)]:
  yield V/f'point_target/evaluation/{variant}/normal/episodes/task{task:02d}_init41/target{target}/actual_inputs/query{q:03d}.npz'

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--backend',choices=['old','fixed','unmerged'],required=True);a=ap.parse_args();out=S/'deployment_probe'/a.backend;out.mkdir(parents=True);adapter=V/'point_target/training/adapter_best.npz'
 atomic_json(out/'PROTOCOL.json',dict(backend=a.backend,adapter_sha256=hashlib.sha256(adapter.read_bytes()).hexdigest(),cases=[str(p.relative_to(ROOT)) for p in cases()],no_training=True,no_rollout=True))
 policy=load(a.backend,adapter);rows=[]
 for i,p in enumerate(cases()):
  with np.load(p) as f:
   raw={'observation/image':f['image'],'observation/wrist_image':f['wrist'],'observation/state':f['state'],'prompt':str(f['prompt'])};noise=f['noise'];stored=f['actions']
   if bool(f['goal_mask']):raw['goal_ref']=f['goal_ref']
  start=time.monotonic();pred=policy.infer(raw,noise=noise)['actions'];assert pred.shape==(10,7) and np.isfinite(pred).all();np.savez_compressed(out/f'case{i}.npz',actions=pred,stored_old=stored)
  if a.backend=='old':np.testing.assert_array_equal(pred,stored)
  rows.append(dict(source=str(p.relative_to(ROOT)),seconds=time.monotonic()-start,max_old_difference=float(abs(pred-stored).max())));atomic_json(out/'STATUS.json',dict(state='running',complete=len(rows),rows=rows))
 atomic_json(out/'STATUS.json',dict(state='complete',complete=len(rows),rows=rows))
if __name__=='__main__':main()
