"""Merge adapter increments into the original architecture before deployment."""
from pathlib import Path
import sys
import numpy as np
S=Path(__file__).resolve().parents[1];V=S.parent;ROOT=V.parents[1]
sys.path[:0]=[str(V/'scripts')]

def load_policy(adapter=None):
    from openpi.training import config,checkpoints
    from openpi.policies import policy_config,policy as policy_module
    from openpi.models import model as model_module
    from openpi import transforms
    if adapter is None:return policy_config.create_trained_policy(config.get_config('pi05_libero'),V/'checkpoints/pi05_libero',sample_kwargs={'num_steps':10})
    import jax,jax.numpy as jnp
    from flax.traverse_util import flatten_dict,unflatten_dict
    from openpi.models import gemma
    assert gemma.get_config('gemma_2b_lora').lora_configs['attn'].scaling_value==1.
    assert gemma.get_config('gemma_300m_lora').lora_configs['attn'].scaling_value==1.
    cfg=config.get_config('pi05_libero');data=cfg.data.create(cfg.assets_dirs,cfg.model)
    params=model_module.restore_params(V/'checkpoints/pi05_libero/params',restore_type=np.ndarray,dtype=jnp.bfloat16)
    flat=flatten_dict(params,sep='/');del params
    with np.load(adapter) as values:
        assert values.files and all('lora' in k for k in values.files)
        akeys=[k for k in values.files if k.endswith('lora_a')];assert len(akeys)*2==len(values.files)
        for akey in akeys:
            bkey=akey[:-1]+'b';a=values[akey];b=values[bkey]
            key=akey[:-len('/lora_a')]+'/w' if akey.endswith('/lora_a') else akey[:-len('_lora_a')]
            assert key in flat,(akey,key)
            assert np.matmul(a[:1],b[:1]).shape[1:]==flat[key].shape[1:]
            if not np.any(b):continue
            # Match the forward pass's BF16 adapter weights, merge one scanned layer at a time.
            # This avoids a multi-gigabyte FP32 temporary or a second GPU model.
            w=flat[key].copy()
            for layer in range(w.shape[0]):
                aa=a[layer].astype(jnp.bfloat16).astype(np.float32);bb=b[layer].astype(jnp.bfloat16).astype(np.float32)
                w[layer]=(w[layer].astype(np.float32)+np.matmul(aa,bb)).astype(w.dtype)
            flat[key]=w
    merged=jax.tree.map(jnp.asarray,unflatten_dict(flat,sep='/'));del flat
    model=cfg.model.load(merged)
    norm=checkpoints.load_norm_stats(V/'checkpoints/pi05_libero/assets',data.asset_id)
    return policy_module.Policy(model,transforms=[transforms.InjectDefaultPrompt(None),*data.data_transforms.inputs,transforms.Normalize(norm,use_quantiles=data.use_quantile_norm),*data.model_transforms.inputs],output_transforms=[*data.model_transforms.outputs,transforms.Unnormalize(norm,use_quantiles=data.use_quantile_norm),*data.data_transforms.outputs],sample_kwargs={'num_steps':10})
