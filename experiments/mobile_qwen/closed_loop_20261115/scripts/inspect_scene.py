from pathlib import Path
import json,sys
import numpy as np
Q=Path(__file__).resolve().parents[2];ROOT=Q.parents[1]
sys.path.insert(0,str(ROOT/'experiments/mobile_vla/scripts'))
from scene import MobileScene
s=MobileScene(task=3,initial=0,mobile_start=(0,0,0),stow=False)
m,d=s.model,s.data
out={'joint_names':m.joint_names,'q':d.qpos[s.arm_indices].tolist(),'arm_indices':np.asarray(s.arm_indices).tolist(),'pose':s.pose().tolist(),'dock':s.dock.tolist(),'ee_obs':s.obs['robot0_eef_pos'].tolist(),'robot_bodies':{},'sites':{}}
for n in m.body_names:
 if n and (n.startswith('robot0') or n.startswith('gripper0')):
  i=m.body_name2id(n);out['robot_bodies'][n]={'p':d.body_xpos[i].tolist(),'R':d.body_xmat[i].reshape(3,3).tolist()}
for n in m.site_names:
 if n and ('grip' in n or 'eef' in n):
  i=m.site_name2id(n);out['sites'][n]={'p':d.site_xpos[i].tolist(),'R':d.site_xmat[i].reshape(3,3).tolist()}
p=Path(__file__).resolve().parents[1]/'research'
(p/'scene_geometry.json').write_text(json.dumps(out,indent=2));(p/'scene.xml').write_text(m.get_xml())
s.close();print(json.dumps(out,indent=2))
