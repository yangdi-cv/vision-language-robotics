"""Fresh layout test: same co-trained weights, text vs explicit initial reference goal."""
import argparse,copy,hashlib,importlib.util,json,sys,time
from pathlib import Path
import numpy as np
S=Path(__file__).resolve().parents[1];V=S.parent;ROOT=V.parents[1]
sys.path[:0]=[str(V/'scripts'),str(V/'selective_execution/scripts')]
from episode_journal import atomic_json,append_jsonl
from visual_prompt import mark,GENERIC
from policy_runtime import load
from prepare import Capture,OriginalEnv
import libero.libero.envs as envs
spec=importlib.util.spec_from_file_location('old_engine',V/'instruction_response/scripts/run.py');engine=importlib.util.module_from_spec(spec);spec.loader.exec_module(engine)

class GoalPolicy:
 def __init__(self,policy,variant,folder,target):self.policy=policy;self.variant=variant;self.folder=folder;self.target=target;self.n=0;self._input_transform=policy._input_transform;self.initial=None
 def infer(self,obs,*,noise):
  if self.initial is None:
   with np.load(self.folder/'reference.npz') as f:self.initial=f['image'].copy();self.boxes=f['boxes'].copy()
  raw=copy.deepcopy(obs)
  if self.variant=='goal':raw.update(prompt=GENERIC,goal_ref=mark(self.initial,self.boxes[self.target]))
  x=self.policy._input_transform(copy.deepcopy(raw));assert bool(x['image_mask']['right_wrist_0_rgb'])==(self.variant=='goal')
  t=time.monotonic();out=self.policy.infer(raw,noise=noise)
  if self.variant=='goal' and self.n==0:
   alternate=copy.deepcopy(raw);alternate['goal_ref']=mark(self.initial,self.boxes[1-self.target]);other=self.policy.infer(alternate,noise=noise)
   np.savez_compressed(self.folder/'box_counterfactual.npz',requested=out['actions'],other_box=other['actions'],noise=noise,requested_reference=raw['goal_ref'],other_reference=alternate['goal_ref'])
  actual=self.folder/'actual_inputs';actual.mkdir(exist_ok=True)
  np.savez_compressed(actual/f'query{self.n:03d}.npz',image=raw['observation/image'],wrist=raw['observation/wrist_image'],state=raw['observation/state'],goal_ref=raw.get('goal_ref',np.zeros((224,224,3),np.uint8)),prompt=raw['prompt'],noise=noise,actions=out['actions'],goal_mask=x['image_mask']['right_wrist_0_rgb'])
  append_jsonl(actual/'queries.jsonl',dict(query=self.n,prompt=raw['prompt'],goal_mask=bool(x['image_mask']['right_wrist_0_rgb']),seconds=time.monotonic()-t,oracle_reference=self.variant=='goal'))
  self.n+=1;return out

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--variant',choices=['base','text','goal'],required=True);a=ap.parse_args()
 assert json.loads((S/'training/STATUS.json').read_text())['state']=='complete'
 out=S/'evaluation'/a.variant;out.mkdir(parents=True);assert not (out/'PROTOCOL.json').exists()
 files=[Path(__file__),S/'scripts/policy_runtime.py',S/'scripts/goal_inputs.py',S/'scripts/layout.py',S/'scripts/visual_prompt.py',S/'scripts/prepare.py',V/'instruction_response/scripts/run.py']
 protocol=dict(variant=a.variant,tasks=[0,1],initials=[41],layouts=['normal','swap'],targets=[0,1],episodes=8,policy_steps=300,hold_steps=25,oracle_box_only_upper_bound=a.variant=='goal',current_images_unchanged=True,box_fixed_at_episode_start=True,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},adapter_sha256=None if a.variant=='base' else hashlib.sha256((S/'training/adapter_best.npz').read_bytes()).hexdigest())
 atomic_json(out/'PROTOCOL.json',protocol);version=hashlib.sha256((out/'PROTOCOL.json').read_bytes()).hexdigest();policy=load(a.variant);engine.probes=lambda *args,**kwargs:None;n=0
 for task in [0,1]:
  for layout in ['normal','swap']:
   engine.S=out/layout
   for target in [0,1]:
    folder=engine.S/f'episodes/task{task:02d}_init41/target{target}';folder.mkdir(parents=True,exist_ok=True)
    envs.OffScreenRenderEnv=lambda *args,**kwargs:Capture(task,layout,folder,*args,**kwargs)
    try:r=engine.rollout(GoalPolicy(policy,a.variant,folder,target),task,41,'target'+str(target),version)
    finally:envs.OffScreenRenderEnv=OriginalEnv
    n+=1;atomic_json(out/'STATUS.json',dict(state='running',complete=n,expected=8));print(json.dumps(dict(variant=a.variant,task=task,layout=layout,target=target,score=r['scores'][r['mode']])),flush=True)
 atomic_json(out/'STATUS.json',dict(state='complete',complete=n,expected=8))
if __name__=='__main__':main()
