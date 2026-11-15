"""Small native pi0.5 LoRA adaptation; official openpi train_step and transforms."""
import argparse,dataclasses,functools,hashlib,importlib.util,json,sys,time,traceback
from pathlib import Path
import numpy as np
S=Path(__file__).resolve().parents[1];V=S.parent;ROOT=V.parents[1]
sys.path[:0]=[str(V/'scripts'),str(V/'selective_execution/scripts')]
from episode_journal import atomic_json,append_jsonl
from protocol import OBJECTS,skill


def config():
    import flax.nnx as nnx
    from openpi.training import config as cfg,weight_loaders,optimizer
    from openpi.models import pi0_config
    from openpi.shared import nnx_utils
    base=cfg.get_config('pi05_libero')
    model=pi0_config.Pi0Config(pi05=True,action_horizon=10,discrete_state_input=False,paligemma_variant='gemma_2b_lora',action_expert_variant='gemma_300m_lora')
    return dataclasses.replace(base,model=model,batch_size=1,ema_decay=None,freeze_filter=nnx.Not(nnx_utils.PathRegex('.*lora.*')),weight_loader=weight_loaders.CheckpointWeightLoader(str(V/'checkpoints/pi05_libero/params')),lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=20,peak_lr=1e-4,decay_steps=200,decay_lr=1e-5),num_train_steps=200,seed=20261114,fsdp_devices=1)


def load_data(cfg):
    from openpi.training import checkpoints
    from openpi import transforms
    data=cfg.data.create(cfg.assets_dirs,cfg.model)
    norm=checkpoints.load_norm_stats(V/'checkpoints/pi05_libero/assets',data.asset_id)
    transform=transforms.compose([*data.data_transforms.inputs,transforms.Normalize(norm,use_quantiles=data.use_quantile_norm),*data.model_transforms.inputs])
    manifest=json.loads((S/'MANIFEST.json').read_text());sets={'train':[],'validation':[]}
    for row in manifest:
        if not row['accepted']:continue
        with np.load(ROOT/row['trajectory']) as f:a={k:f[k].copy() for k in ['image','wrist','state','action','phase']}
        sets[row['split']].append(dict(meta=row,arrays=a))
    for split,rows in sets.items():
        groups={(t,g):[r for r in rows if (r['meta']['task'],r['meta']['target'])==(t,g)] for t in [0,1] for g in [0,1]}
        assert all(groups.values()),f'Missing task/target in {split}'
    return sets,transform,data


def sample(sets,split,transform,rng):
    import jax
    import jax.numpy as jnp
    from openpi.models.model import Observation
    t,g=[int(x) for x in rng.integers(0,2,size=2)]
    choices=[r for r in sets[split] if (r['meta']['task'],r['meta']['target'])==(t,g)]
    row=choices[int(rng.integers(len(choices)))];a=row['arrays'];phases=np.unique(a['phase']);phase=phases[int(rng.integers(len(phases)))];indices=np.flatnonzero(a['phase']==phase);i=int(indices[int(rng.integers(len(indices)))])
    ids=np.minimum(np.arange(i,i+10),len(a['action'])-1)
    raw={'observation/image':a['image'][i],'observation/wrist_image':a['wrist'][i],'observation/state':a['state'][i].astype(np.float32),'prompt':skill(OBJECTS[t][g][1]),'actions':a['action'][ids].astype(np.float32)}
    x=transform(raw);batched=jax.tree.map(lambda v:jnp.asarray(v)[None],x)
    return (Observation.from_dict(batched),batched['actions']),dict(task=t,target=g,initial=row['meta']['initial'],frame=i,phase=str(phase))


def save_adapter(state,path):
    import jax
    import flax.traverse_util as traverse
    flat=traverse.flatten_dict(state.params.to_pure_dict(),sep='/');values={k:np.asarray(jax.device_get(v),np.float32) for k,v in flat.items() if 'lora' in k}
    assert values
    np.savez_compressed(path,**values)
    return dict(tensors=len(values),parameters=sum(x.size for x in values.values()),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    import jax,jax.numpy as jnp,flax.nnx as nnx
    from openpi.training import sharding
    from openpi.shared import nnx_utils
    assert json.loads((S/'DATA_STATUS.json').read_text())['state']=='complete'
    out=S/'lora_pilot';out.mkdir(exist_ok=True)
    assert not (out/'TRAIN_STATUS.json').exists(),'Refuse training overwrite'
    cfg=config();sets,transform,data=load_data(cfg)
    spec=importlib.util.spec_from_file_location('upstream_train',V/'third_party/openpi/scripts/train.py');train=importlib.util.module_from_spec(spec);spec.loader.exec_module(train)
    source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),V/'third_party/openpi/scripts/train.py',V/'third_party/openpi/src/openpi/training/weight_loaders.py',S/'DATA_PROTOCOL.json',S/'MANIFEST.json']}
    protocol=dict(steps=200,batch_size=1,seed=20261114,trainable='only lora leaves; vision and all base weights frozen',weight_source='existing pi05_libero',source_hashes=source_hashes,validation='fixed16 phase-balanced samples from initials26/27; fixed loss RNG; check0/50/100/150/200',selection='minimum validation flow loss among trained steps; test initials28/29 untouched',evaluation='exploratory paired policy rollouts required regardless of validation result',data_counts={k:len(v) for k,v in sets.items()},teacher_privileged=True,no_hardware=True)
    atomic_json(out/'PROTOCOL.json',protocol);atomic_json(out/'TRAIN_STATUS.json',dict(state='initializing',steps=0))
    mesh=sharding.make_mesh(1);rng=jax.random.key(20261114);train_rng,init_rng=jax.random.split(rng)
    state,ss=train.init_train_state(cfg,init_rng,mesh,resume=False);jax.block_until_ready(state)
    flat=state.params.to_pure_dict();from flax.traverse_util import flatten_dict
    leaves=flatten_dict(flat,sep='/');trainable=nnx.state(nnx.merge(state.model_def,state.params),cfg.trainable_filter).to_pure_dict();trainleaves=flatten_dict(trainable,sep='/')
    assert trainleaves and all('lora' in k for k in trainleaves)
    atomic_json(out/'PARAMETERS.json',dict(total_parameters=sum(v.size for v in leaves.values()),trainable_parameters=sum(v.size for v in trainleaves.values()),trainable_paths=list(trainleaves)))
    save_adapter(state,out/'adapter_step0000.npz')
    trainstep=jax.jit(functools.partial(train.train_step,cfg),donate_argnums=(1,))
    graph=state.model_def
    @jax.jit
    def loss(params,observation,actions):
        model=nnx.merge(graph,params)
        return jnp.mean(model.compute_loss(jax.random.key(1729),observation,actions,train=False))
    vrng=np.random.default_rng(77);validation=[sample(sets,'validation',transform,vrng) for _ in range(16)]
    atomic_json(out/'VALIDATION_SAMPLES.json',[meta for _,meta in validation])
    def evaluate(step):
        values=[float(loss(state.params,*batch)) for batch,_ in validation];value=float(np.mean(values));append_jsonl(out/'validation.jsonl',dict(step=step,loss=value,individual=values));return value
    baseline=evaluate(0);best=float('inf');beststep=None;drng=np.random.default_rng(20261114);start=time.monotonic()
    for i in range(1,201):
        batch,meta=sample(sets,'train',transform,drng)
        if i==1:atomic_json(out/'TRAIN_STATUS.json',dict(state='compiling_train_step',steps=0,baseline_validation_loss=baseline))
        with sharding.set_mesh(mesh):state,info=trainstep(train_rng,state,batch)
        values={k:float(v) for k,v in jax.device_get(info).items()};assert all(np.isfinite(v) for v in values.values())
        append_jsonl(out/'train.jsonl',dict(step=i,**values,sample=meta,seconds=time.monotonic()-start))
        if i%10==0:atomic_json(out/'TRAIN_STATUS.json',dict(state='running',steps=i,loss=values['loss'],seconds=time.monotonic()-start));print(json.dumps(dict(step=i,**values)),flush=True)
        if i%50==0:
            value=evaluate(i);meta=save_adapter(state,out/f'adapter_step{i:04d}.npz')
            if value<best:
                best=value;beststep=i;meta=save_adapter(state,out/'adapter_best.npz');atomic_json(out/'BEST.json',dict(step=i,validation_loss=value,baseline_validation_loss=baseline,**meta))
    atomic_json(out/'TRAIN_STATUS.json',dict(state='complete',steps=200,best_step=beststep,best_validation_loss=best,baseline_validation_loss=baseline,seconds=time.monotonic()-start))

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        print(traceback.format_exc(),flush=True)
        if (S/'lora_pilot').exists():atomic_json(S/'lora_pilot/ERROR.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
