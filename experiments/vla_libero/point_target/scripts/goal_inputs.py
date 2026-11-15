"""Train and infer with exactly the same extra image slot and mask convention."""
import numpy as np
class GoalInputs:
 def __init__(self,model_type):
  from openpi.policies.libero_policy import LiberoInputs
  self.base=LiberoInputs(model_type=model_type)
 def __call__(self,raw):
  out=self.base(raw)
  if raw.get('goal_ref') is not None:
   goal=np.asarray(raw['goal_ref'],np.uint8);assert goal.shape==(224,224,3)
   out['image']['right_wrist_0_rgb']=goal;out['image_mask']['right_wrist_0_rgb']=np.True_
  return out

def transforms_for(cfg):
 from openpi.training import checkpoints
 from openpi import transforms
 V=__import__('pathlib').Path(__file__).resolve().parents[2]
 data=cfg.data.create(cfg.assets_dirs,cfg.model);norm=checkpoints.load_norm_stats(V/'checkpoints/pi05_libero/assets',data.asset_id)
 chain=[GoalInputs(cfg.model.model_type),transforms.Normalize(norm,use_quantiles=data.use_quantile_norm),*data.model_transforms.inputs]
 return transforms.compose(chain),[ *data.model_transforms.outputs,transforms.Unnormalize(norm,use_quantiles=data.use_quantile_norm),*data.data_transforms.outputs]
