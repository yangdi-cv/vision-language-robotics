"""Same native policy, sequential conditional/null-task queries, explicit shared noise."""
import copy,hashlib,json,time
from pathlib import Path
import numpy as np
from mixing import combine

def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def same(a,b):
 if isinstance(a,dict):return set(a)==set(b) and all(same(a[k],b[k]) for k in a)
 return np.array_equal(a,b)

class CAGPolicy:
 def __init__(self,policy,scale):
  self.policy=policy;self.scale=scale;self._input_transform=policy._input_transform;self.folder=None;self.index=0
 def start(self,folder):
  self.folder=Path(folder)/'guidance';self.folder.mkdir(parents=True,exist_ok=True);self.index=0
 def branches(self,obs,noise):
  if noise is None:raise ValueError('Explicit shared noise required')
  null=copy.deepcopy(obs);null['prompt']=''
  x=self.policy._input_transform(copy.deepcopy(obs));y=self.policy._input_transform(copy.deepcopy(null))
  assert all(same(x[k],y[k]) for k in x if k not in ['tokenized_prompt','tokenized_prompt_mask'])
  ta=np.asarray(x['tokenized_prompt'])[np.asarray(x['tokenized_prompt_mask'],bool)]
  tb=np.asarray(y['tokenized_prompt'])[np.asarray(y['tokenized_prompt_mask'],bool)]
  assert not np.array_equal(ta,tb) and len(tb)>0
  c=np.asarray(self.policy.infer(obs,noise=noise)['actions']);u=np.asarray(self.policy.infer(null,noise=noise)['actions'])
  return c,u,dict(conditional_tokens=ta.tolist(),null_tokens=tb.tolist(),normalized_state_sha256=digest(x['state']),noise_sha256=digest(noise),nonlanguage_inputs_identical=True)
 def infer(self,obs,*,noise=None):
  start=time.monotonic()
  if self.scale==1:
   c=np.asarray(self.policy.infer(obs,noise=noise)['actions']);a=c.copy();u=None;meta={'branch_calls':1}
  else:
   c,u,meta=self.branches(obs,noise);a=combine(c,u,self.scale);meta['branch_calls']=2
  assert np.isfinite(a).all()
  if self.folder:
   values=dict(conditional=c,guided=a,noise=noise)
   if u is not None:values['unconditional']=u
   np.savez_compressed(self.folder/f'query{self.index:03d}.npz',**values)
   meta.update(index=self.index,scale=self.scale,seconds=time.monotonic()-start,max_abs_action=float(abs(a).max()),components_outside_unit=int((abs(a)>1).sum()))
   with (self.folder/'queries.jsonl').open('a') as f:f.write(json.dumps(meta)+'\n')
   self.index+=1
  return dict(actions=a)
