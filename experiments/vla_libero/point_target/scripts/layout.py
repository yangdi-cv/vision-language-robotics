"""Reset-only object swaps and segmentation boxes. Oracle boxes are explicitly labelled."""
import hashlib,sys
from pathlib import Path
import numpy as np
from PIL import Image
S=Path(__file__).resolve().parents[1];V=S.parent
sys.path[:0]=[str(V/'scripts'),str(V/'selective_execution/scripts')]
import evaluate as ev
from protocol import OBJECTS

def swap_at_reset(env,task,layout):
 if layout=='normal':return
 if layout!='swap':raise ValueError(layout)
 joints=[env.env.get_object(n[0]).joints[0] for n in OBJECTS[task]]
 q=[env.sim.data.get_joint_qpos(n).copy() for n in joints]
 for i,name in enumerate(joints):
  q[i][:2]=q[1-i][:2].copy() if i==0 else original_xy
  if i==0:original_xy=env.sim.data.get_joint_qpos(joints[0])[:2].copy()
  env.sim.data.set_joint_qpos(name,q[i])
 state=env.sim.get_state().flatten().copy();env.regenerate_obs_from_state(state)

def reference(env,obs,task):
 from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING
 from robosuite import macros
 convention=IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]
 # Verify raw RGB convention before applying exactly the policy's two-axis flip.
 raw=env.sim.render(camera_name='agentview',width=256,height=256)
 sensor=obs['agentview_image'].astype(np.float32);base=raw[::convention]
 errors=[float(np.mean(abs(x.astype(np.float32)-sensor))) for x in [base,base[::-1],base[:,::-1],base[::-1,::-1]]]
 assert errors[0]==min(errors) and errors[0]<5.,f'RGB orientation mismatch: {errors}'
 seg=env.sim.render(camera_name='agentview',width=256,height=256,segmentation=True)[::convention][::-1,::-1]
 boxes=[];counts=[]
 for name,_,_ in OBJECTS[task]:
  root=env.env.obj_body_id[name];bodies={root};changed=True
  while changed:
   expanded=bodies|{i for i,p in enumerate(env.sim.model.body_parentid) if int(p) in bodies};changed=expanded!=bodies;bodies=expanded
  ids=np.array([i for i,b in enumerate(env.sim.model.geom_bodyid) if int(b) in bodies])
  mask=(seg[...,0]==5)&np.isin(seg[...,1],ids);small=np.asarray(Image.fromarray(mask).resize((224,224),resample=Image.Resampling.NEAREST));yy,xx=np.where(small)
  assert len(xx)>=6,f'Target invisible: {name}'
  boxes.append([max(0,int(xx.min())-2),max(0,int(yy.min())-2),min(223,int(xx.max())+2),min(223,int(yy.max())+2)]);counts.append(len(xx))
 image=ev.make_input(obs,'')['observation/image']
 return dict(rgb_alignment_mae=np.array(errors),image=image,boxes=np.array(boxes,np.int32),visible_pixels=np.array(counts),settled_state=env.sim.get_state().flatten().copy())
