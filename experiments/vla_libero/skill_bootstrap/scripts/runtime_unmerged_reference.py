"""Reload a small native openpi LoRA file over the immutable LIBERO checkpoint."""
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
    import jax.numpy as jnp
    from flax.traverse_util import flatten_dict,unflatten_dict
    from train_lora import config as lora_config
    cfg=lora_config();data=cfg.data.create(cfg.assets_dirs,cfg.model)
    params=model_module.restore_params(V/'checkpoints/pi05_libero/params',dtype=jnp.bfloat16)
    flat=flatten_dict(params,sep='/')
    with np.load(adapter) as a:
        assert a.files and all('lora' in k for k in a.files)
        flat.update({k:jnp.asarray(a[k]) for k in a.files})
    model=cfg.model.load(unflatten_dict(flat,sep='/'))
    norm=checkpoints.load_norm_stats(V/'checkpoints/pi05_libero/assets',data.asset_id)
    return policy_module.Policy(model,transforms=[transforms.InjectDefaultPrompt(None),*data.data_transforms.inputs,transforms.Normalize(norm,use_quantiles=data.use_quantile_norm),*data.model_transforms.inputs],output_transforms=[*data.model_transforms.outputs,transforms.Unnormalize(norm,use_quantiles=data.use_quantile_norm),*data.data_transforms.outputs],sample_kwargs={'num_steps':10})
