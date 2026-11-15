"""Independent static kinematic calibration, not motion execution."""
from pathlib import Path
import json,sys
import numpy as np
from scipy.spatial.transform import Rotation
Q=Path(__file__).resolve().parents[2];ROOT=Q.parents[1]
sys.path.insert(0,str(ROOT/'experiments/mobile_vla/scripts'))
from scene import MobileScene
from kinematics import forward
s=MobileScene(task=3,initial=0,mobile_start=(0,0,0),stow=False)
rng=np.random.default_rng(11015);rows=[]
for i in range(8):
    q=s.arm_q_initial+rng.uniform(-.12,.12,7)
    s.data.qpos[s.arm_indices]=q;s.sim.forward()
    b=s.model.body_name2id('robot0_link0');site=s.model.site_name2id('gripper0_grip_site')
    B=np.eye(4);B[:3,:3]=s.data.body_xmat[b].reshape(3,3);B[:3,3]=s.data.body_xpos[b]
    predicted=B@forward(q)
    pe=float(np.linalg.norm(predicted[:3,3]-s.data.site_xpos[site]));re=float(Rotation.from_matrix(predicted[:3,:3].T@s.data.site_xmat[site].reshape(3,3)).magnitude())
    rows.append({'q':q.tolist(),'position_error_m':pe,'rotation_error_rad':re})
assert max(x['position_error_m'] for x in rows)<1e-6
assert max(x['rotation_error_rad'] for x in rows)<1e-6
out=Path(__file__).resolve().parents[1]/'research/FK_CHECK.json';out.write_text(json.dumps({'static_only':True,'passed':True,'cases':rows},indent=2));print(out);s.close()
