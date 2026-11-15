"""Bounded Point-VLA-inspired co-training, balanced goal pairs and reset-only swaps."""
import dataclasses,functools,hashlib,importlib.util,json,sys,time
from pathlib import Path
import numpy as np
S=Path(__file__).resolve().parents[1];V=S.parent;ROOT=V.parents[1]
sys.path[:0]=[str(V/'skill_bootstrap/scripts'),str(V/'scripts'),str(V/'selective_execution/scripts')]
from episode_journal import atomic_json,append_jsonl
from protocol import OBJECTS,skill
from visual_prompt import mark,GENERIC
from goal_inputs import transforms_for

def main():
 import jax,jax.numpy as jnp,flax.nnx as nnx
 from flax.traverse_util import flatten_dict,unflatten_dict
 from train_lora import config as old_config,save_adapter
 from bounded_restore import restore
 from openpi.training import optimizer,utils,sharding
 from openpi.models.model import Observation
 assert json.loads((S/'DATA_STATUS.json').read_text())['state']=='complete'
 assert json.loads((S/'TRANSFORM_CHECK.json').read_text())['passed']
 out=S/'training';out.mkdir(exist_ok=True);assert not (out/'PROTOCOL.json').exists(),'Refuse overwrite'
 cfg=dataclasses.replace(old_config(),batch_size=2,num_train_steps=1000,seed=20261116,lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=30,peak_lr=1e-4,decay_steps=1000,decay_lr=1e-5))
 transform,_=transforms_for(cfg);pairs=json.loads((S/'PAIRS.json').read_text())
 for pair in pairs:
  for row in pair['rows']:
   with np.load(ROOT/row['trajectory']) as f:row['arrays']={k:f[k].copy() for k in ['image','wrist','state','action','phase']}
   with np.load(ROOT/row['reference']) as f:row['reference_image']=f['image'].copy();row['box']=f['boxes'][row['target']].copy()
 def batch(split,rng,grounded,branch=None,augment=True):
  task=int(rng.integers(2));layout=['normal','swap'][int(rng.integers(2))];options=[p for p in pairs if p['split']==split and p['task']==task and p['layout']==layout];pair=options[int(rng.integers(len(options)))];shared=bool(rng.random()<.4) if branch is None else branch;items=[];indices=[]
  if shared:
   index=max(0,pair['branch']-int(rng.integers(3)))
   assert all(np.array_equal(pair['rows'][0]['arrays'][k][index],pair['rows'][1]['arrays'][k][index]) for k in ['image','wrist','state'])
  phase=['clear','hover','approach','close','lift','transfer','lower','release','retreat','hold'][int(rng.integers(10))]
  for row in pair['rows']:
   a=row['arrays'];ids=np.flatnonzero(a['phase']==phase);i=index if shared else int(ids[int(rng.integers(len(ids)))])
   raw={'observation/image':a['image'][i],'observation/wrist_image':a['wrist'][i],'observation/state':a['state'][i].astype(np.float32),'prompt':GENERIC if grounded else skill(OBJECTS[task][row['target']][1]),'actions':a['action'][np.minimum(np.arange(i,i+10),len(a['action'])-1)].astype(np.float32)}
   if grounded:raw['goal_ref']=mark(row['reference_image'],row['box'],rng if augment else None)
   x=transform(raw);items.append(x);indices.append(i)
  b=jax.tree.map(lambda *v:jnp.asarray(np.stack(v)),*items)
  return (Observation.from_dict(b),b['actions']),dict(task=task,layout=layout,initial=pair['initial'],grounded=grounded,shared=shared,frames=indices)
 protocol=dict(steps=1000,batch_size=2,seed=20261116,start='native pi05_libero with zero-B LoRA; no previous trained adapter',trainable='only LoRA A/B in language and action expert; float32 master parameters; frozen base and vision',interface='unused right_wrist image slot carries fixed marked initial image; mask true only in grounded mode',language='generic marked-object instruction in grounded mode, object name in text mode',sampling='alternate grounded/text batches exactly 1:1; paired opposite targets per batch; 40% exact shared-observation branch,60% phase',augmentation='grounded reference RGB and box translated together by at most12px; current camera unchanged',selection='minimum equal-weight text/grounded fixed validation loss at steps200/600/1000; never use test rollouts',oracle_reference=True,test_initials=[41,42],scope='local Point-VLA-inspired interface, not official code/weights reproduction',source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),S/'scripts/goal_inputs.py',S/'scripts/visual_prompt.py',S/'PAIRS.json',S/'DATA_PROTOCOL.json']})
 atomic_json(out/'PROTOCOL.json',protocol);atomic_json(out/'STATUS.json',dict(state='loading',step=0))
 params,info=restore(V/'checkpoints/pi05_libero/params');flat=flatten_dict(params,sep='/');del params
 with np.load(V/'skill_bootstrap/lora_pilot/adapter_step0000.npz') as f:
  for k in f.files:flat[k]=jnp.asarray(np.zeros_like(f[k]) if k.endswith('lora_b') else f[k])
 model=cfg.model.load(unflatten_dict(flat,sep='/'));del flat
 values=nnx.state(model);tx=optimizer.create_optimizer(cfg.optimizer,cfg.lr_schedule,weight_decay_mask=None)
 state=utils.TrainState(step=0,params=values,model_def=nnx.graphdef(model),tx=tx,opt_state=tx.init(values.filter(cfg.trainable_filter)),ema_decay=None,ema_params=None);del model,values
 leaves=flatten_dict(state.params.filter(cfg.trainable_filter).to_pure_dict(),sep='/');assert leaves and all('lora' in k and v.dtype==jnp.float32 for k,v in leaves.items());del leaves
 spec=importlib.util.spec_from_file_location('official_train',V/'third_party/openpi/scripts/train.py');train=importlib.util.module_from_spec(spec);spec.loader.exec_module(train)
 stepfn=jax.jit(functools.partial(train.train_step,cfg),donate_argnums=(1,));mesh=sharding.make_mesh(1);graph=state.model_def
 @jax.jit
 def loss(params,obs,actions):return jnp.mean(nnx.merge(graph,params).compute_loss(jax.random.key(1729),obs,actions,train=False))
 validation={}
 for grounded in [False,True]:
  rng=np.random.default_rng(77);validation[str(grounded)]=[batch('validation',rng,grounded,branch=i%2==0,augment=False)[0] for i in range(8)]
 def validate(step):
  scores={k:float(np.mean([float(loss(state.params,*b)) for b in vv])) for k,vv in validation.items()};r=dict(step=step,text_loss=scores['False'],grounded_loss=scores['True'],mean_loss=float(np.mean(list(scores.values()))));append_jsonl(out/'validation.jsonl',r);return r
 initial=validate(0);save_adapter(state,out/'adapter_zero.npz');best=float('inf');rng=np.random.default_rng(20261116);start=time.monotonic()
 for i in range(1,1001):
  b,meta=batch('train',rng,grounded=bool(i%2));
  with sharding.set_mesh(mesh):state,metrics=stepfn(jax.random.key(20261116),state,b)
  numeric={k:float(v) for k,v in jax.device_get(metrics).items()};assert all(np.isfinite(v) for v in numeric.values());append_jsonl(out/'train.jsonl',dict(step=i,**numeric,**meta))
  if i%10==0:atomic_json(out/'STATUS.json',dict(state='running',step=i,seconds=time.monotonic()-start,loss=numeric['loss']));print(json.dumps(dict(step=i,**numeric)),flush=True)
  if i in [200,600,1000]:
   r=validate(i)
   if r['mean_loss']<best:
    best=r['mean_loss'];metadata=save_adapter(state,out/'adapter_best.npz');atomic_json(out/'BEST.json',dict(**r,initial=initial,**metadata))
 atomic_json(out/'STATUS.json',dict(state='complete',step=1000,best=json.loads((out/'BEST.json').read_text()),seconds=time.monotonic()-start))
if __name__=='__main__':main()
