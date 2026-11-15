"""Bounded continuation emphasizing exact same-state, opposite-goal branch points."""
import dataclasses,functools,hashlib,importlib.util,json,sys,time,traceback
from pathlib import Path
import numpy as np
from train_lora import S,V,ROOT,config,load_data,sample,save_adapter
sys.path[:0]=[str(V/'scripts')]
from episode_journal import atomic_json,append_jsonl


def main():
    import jax,jax.numpy as jnp,flax.nnx as nnx
    from flax.traverse_util import flatten_dict,unflatten_dict
    from openpi.training import sharding,optimizer,utils
    from openpi.models.model import Observation
    from bounded_restore import restore
    from protocol import OBJECTS,skill
    out=S/'lora_paired';out.mkdir(exist_ok=True);assert not (out/'TRAIN_STATUS.json').exists(),'Refuse overwrite'
    cfg=dataclasses.replace(config(),batch_size=2,num_train_steps=600,lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=20,peak_lr=5e-5,decay_steps=600,decay_lr=5e-6),seed=20261115)
    sets,transform,data=load_data(cfg);branches=json.loads((S/'PAIRED_BRANCH_AUDIT.json').read_text());assert all(r['exact_shared_observation'] for r in branches)
    def at_frame(row,i):
        a=row['arrays'];m=row['meta'];ids=np.minimum(np.arange(i,i+10),len(a['action'])-1)
        raw={'observation/image':a['image'][i],'observation/wrist_image':a['wrist'][i],'observation/state':a['state'][i].astype(np.float32),'prompt':skill(OBJECTS[m['task']][m['target']][1]),'actions':a['action'][ids].astype(np.float32)}
        x=transform(raw);b=jax.tree.map(lambda v:jnp.asarray(v)[None],x)
        return (Observation.from_dict(b),b['actions'])
    def pair_batch(split,rng):
        t=int(rng.integers(2));eligible=[r for r in branches if r['split']==split and r['task']==t];branch=eligible[int(rng.integers(len(eligible)))];i=max(0,branch['branch_frame']-int(rng.integers(3)))
        rows=[next(r for r in sets[split] if r['meta']['task']==t and r['meta']['initial']==branch['initial'] and r['meta']['target']==g) for g in [0,1]]
        assert all(np.array_equal(rows[0]['arrays'][k][i],rows[1]['arrays'][k][i]) for k in ['image','wrist','state'])
        a,b=[at_frame(r,i) for r in rows]
        batch=jax.tree.map(lambda x,y:jnp.concatenate([x,y],axis=0),a,b)
        return batch,dict(kind='paired_branch',task=t,initial=branch['initial'],frame=i)
    source=[Path(__file__),S/'scripts/train_lora.py',S/'scripts/bounded_restore.py',S/'scripts/chunk_math.py',S/'PAIRED_BRANCH_AUDIT.json',S/'MANIFEST.json']
    spec=dict(steps=600,batch_size=2,start_adapter_sha256=hashlib.sha256((S/'lora_pilot/adapter_best.npz').read_bytes()).hexdigest(),sampling='50% paired shared-observation opposite-goal batches; 50% phase-balanced original batches',teacher_data_unchanged=True,seed=20261115,validation='same16 phase samples plus all8 held-out exact branch-goal samples; selection0.5phase+0.5branch loss',new_policy_test_initials=[30,31],old_v1_test_initials=[28,29],not_a_sampling_only_ablation='also more updates and optimizer restart',source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source})
    atomic_json(out/'PROTOCOL.json',spec);atomic_json(out/'TRAIN_STATUS.json',dict(state='loading',steps=0))
    params,loadinfo=restore(V/'checkpoints/pi05_libero/params',progress=lambda r:atomic_json(out/'LOAD_STATUS.json',r))
    flat=flatten_dict(params,sep='/');del params
    with np.load(S/'lora_pilot/adapter_best.npz') as f:flat.update({k:jnp.asarray(f[k]) for k in f.files})
    model=cfg.model.load(unflatten_dict(flat,sep='/'));del flat
    values=nnx.state(model);tx=optimizer.create_optimizer(cfg.optimizer,cfg.lr_schedule,weight_decay_mask=None)
    state=utils.TrainState(step=0,params=values,model_def=nnx.graphdef(model),tx=tx,opt_state=tx.init(values.filter(cfg.trainable_filter)),ema_decay=None,ema_params=None)
    trainable=flatten_dict(values.filter(cfg.trainable_filter).to_pure_dict(),sep='/');assert trainable and all('lora' in k for k in trainable)
    del model,values,trainable
    script=importlib.util.spec_from_file_location('upstream_train',V/'third_party/openpi/scripts/train.py');train=importlib.util.module_from_spec(script);script.loader.exec_module(train)
    stepfn=jax.jit(functools.partial(train.train_step,cfg),donate_argnums=(1,));mesh=sharding.make_mesh(1);train_rng=jax.random.key(20261115);graph=state.model_def
    @jax.jit
    def loss(params,obs,actions):return jnp.mean(nnx.merge(graph,params).compute_loss(jax.random.key(1729),obs,actions,train=False))
    vrng=np.random.default_rng(77);phase_validation=[sample(sets,'validation',transform,vrng)[0] for _ in range(16)]
    branch_validation=[]
    for branch in [r for r in branches if r['split']=='validation']:
        for g in [0,1]:
            row=next(r for r in sets['validation'] if r['meta']['task']==branch['task'] and r['meta']['initial']==branch['initial'] and r['meta']['target']==g);branch_validation.append(at_frame(row,branch['branch_frame']))
    assert len(branch_validation)==8
    def evaluate(step):
        phase=float(np.mean([float(loss(state.params,*b)) for b in phase_validation]));branch=float(np.mean([float(loss(state.params,*b)) for b in branch_validation]));r=dict(step=step,phase_loss=phase,branch_loss=branch,combined=.5*(phase+branch));append_jsonl(out/'validation.jsonl',r);return r
    initial=evaluate(0);save_adapter(state,out/'adapter_step0000.npz');best=float('inf');beststep=None;drng=np.random.default_rng(20261115);start=time.monotonic()
    for i in range(1,601):
        if drng.random()<.5:batch,meta=pair_batch('train',drng)
        else:
            a,ma=sample(sets,'train',transform,drng);b,mb=sample(sets,'train',transform,drng);batch=jax.tree.map(lambda x,y:jnp.concatenate([x,y],axis=0),a,b);meta=dict(kind='phase',samples=[ma,mb])
        with sharding.set_mesh(mesh):state,info=stepfn(train_rng,state,batch)
        numeric={k:float(v) for k,v in jax.device_get(info).items()};assert all(np.isfinite(v) for v in numeric.values());append_jsonl(out/'train.jsonl',dict(step=i,**numeric,sample=meta))
        if i%10==0:atomic_json(out/'TRAIN_STATUS.json',dict(state='running',steps=i,loss=numeric['loss'],seconds=time.monotonic()-start));print(json.dumps(dict(step=i,**numeric)),flush=True)
        if i in [100,200,400,600]:
            result=evaluate(i)
            if result['combined']<best:
                best=result['combined'];beststep=i;metadata=save_adapter(state,out/'adapter_best.npz');atomic_json(out/'BEST.json',dict(**result,initial=initial,**metadata))
    atomic_json(out/'TRAIN_STATUS.json',dict(state='complete',steps=600,best_step=beststep,best_combined_loss=best,initial=initial,seconds=time.monotonic()-start))

if __name__=='__main__':
    try:main()
    except BaseException as e:
        if (S/'lora_paired').exists():atomic_json(S/'lora_paired/ERROR.json',dict(error=repr(e),traceback=traceback.format_exc()))
        raise
