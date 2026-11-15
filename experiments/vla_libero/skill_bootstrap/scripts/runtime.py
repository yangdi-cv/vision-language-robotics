"""Same native pi0.5 architecture for base and merged LoRA, bounded chunk loading."""
from pathlib import Path
import sys,json
S=Path(__file__).resolve().parents[1];V=S.parent;ROOT=V.parents[1]
sys.path[:0]=[str(V/'scripts')]
LAST_LOAD_INFO={}

def load_policy(adapter=None):
    global LAST_LOAD_INFO
    from openpi.training import config,checkpoints
    from openpi.policies import policy as policy_module
    from openpi import transforms
    from openpi.models import gemma
    from bounded_restore import restore
    from episode_journal import atomic_json
    assert gemma.get_config('gemma_2b_lora').lora_configs['attn'].scaling_value==1.
    assert gemma.get_config('gemma_300m_lora').lora_configs['attn'].scaling_value==1.
    cfg=config.get_config('pi05_libero');data=cfg.data.create(cfg.assets_dirs,cfg.model)
    progress=lambda row:atomic_json(S/'resource_checks/LOAD_STATUS.json',row)
    params,LAST_LOAD_INFO=restore(V/'checkpoints/pi05_libero/params',adapter,progress)
    model=cfg.model.load(params)
    norm=checkpoints.load_norm_stats(V/'checkpoints/pi05_libero/assets',data.asset_id)
    progress(dict(stage='model_ready',**{k:v for k,v in LAST_LOAD_INFO.items() if k!='chunk_sha256'}))
    return policy_module.Policy(model,transforms=[transforms.InjectDefaultPrompt(None),*data.data_transforms.inputs,transforms.Normalize(norm,use_quantiles=data.use_quantile_norm),*data.model_transforms.inputs],output_transforms=[*data.model_transforms.outputs,transforms.Unnormalize(norm,use_quantiles=data.use_quantile_norm),*data.data_transforms.outputs],sample_kwargs={'num_steps':10})
