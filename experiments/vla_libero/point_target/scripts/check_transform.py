"""Check the added goal slot does not change original camera/state inputs."""
import copy,json,sys
from pathlib import Path
import numpy as np
from goal_inputs import transforms_for
from visual_prompt import mark,GENERIC
S=Path(__file__).resolve().parents[1];V=S.parent
from openpi.training import config,checkpoints
from openpi import transforms

def same(a,b):
 if isinstance(a,dict):return set(a)==set(b) and all(same(a[k],b[k]) for k in a)
 return np.array_equal(a,b)
def main():
 cfg=config.get_config('pi05_libero');data=cfg.data.create(cfg.assets_dirs,cfg.model);norm=checkpoints.load_norm_stats(V/'checkpoints/pi05_libero/assets',data.asset_id)
 original=transforms.compose([*data.data_transforms.inputs,transforms.Normalize(norm,use_quantiles=data.use_quantile_norm),*data.model_transforms.inputs]);custom,_=transforms_for(cfg)
 with np.load(V/'instruction_response/episodes/task00_init00/target0/inputs/step0000.npz') as f:raw={'observation/image':f['image'],'observation/wrist_image':f['wrist'],'observation/state':f['state'],'prompt':GENERIC}
 a=original(copy.deepcopy(raw));b=custom(copy.deepcopy(raw));assert same(a,b)
 with np.load(S/'preflight/task0_normal.npz') as f:ref=f['image'];boxes=f['boxes']
 c=custom(dict(copy.deepcopy(raw),goal_ref=mark(ref,boxes[0])));d=custom(dict(copy.deepcopy(raw),goal_ref=mark(ref,boxes[1])))
 for k in a:
  if k not in ['image','image_mask']:assert same(a[k],c[k]) and same(c[k],d[k])
 for k in ['base_0_rgb','left_wrist_0_rgb']:assert same(a['image'][k],c['image'][k]) and same(c['image'][k],d['image'][k])
 assert not bool(a['image_mask']['right_wrist_0_rgb']) and bool(c['image_mask']['right_wrist_0_rgb']);assert not same(c['image']['right_wrist_0_rgb'],d['image']['right_wrist_0_rgb'])
 (S/'TRANSFORM_CHECK.json').write_text(json.dumps(dict(passed=True,text_branch_identical_to_original_transform=True,only_goal_image_and_mask_change=True,current_cameras_and_state_unchanged=True,task_tokens_same_when_only_box_changes=True,discrete_state_input=cfg.model.discrete_state_input),indent=2))
if __name__=='__main__':main()
