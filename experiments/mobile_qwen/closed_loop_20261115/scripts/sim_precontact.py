"""Single-world mobile scene with a finite-torque joint driver and camera grounding.
Only initialization sets qpos. Runtime targets are actuator commands, never resets.
"""
from pathlib import Path
import argparse,json,socketserver,sys,types,traceback,hashlib
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image
D=Path(__file__).resolve().parents[1];Q=D.parent;ROOT=Q.parents[1]
sys.path[:0]=[str(ROOT/'experiments/mobile_vla/scripts'),str(Q/'affordance'),str(Q/'scripts')]
import sim_server as old
from scene import MobileScene
from gen_dataset import Labeller,HANDLE_GEOM
from durable_log import Journal,atomic_json

OFFSET=np.array([-.12,-.10,.10]) # fixed skill prior in WORLD coordinates, not a learned 6D pose

class JointScene(MobileScene):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.joint_mode=False
        original=self.e.env._pre_action
        def pre(env,action,policy_step=False):
            original(action,policy_step=policy_step)
            if self.joint_mode:
                tau=400*(self.arm_target-self.data.qpos[self.arm_indices])-40*self.data.qvel[self.arm_vel]+self.data.qfrc_bias[self.arm_vel]
                ranges=self.model.actuator_ctrlrange[self.arm_act]
                self.data.ctrl[self.arm_act]=np.clip(tau,ranges[:,0],ranges[:,1])
        self.e.env._pre_action=types.MethodType(pre,self.e.env)
    def activate_joint(self):
        self.park();self.mode='manip';self.arm_target=self.data.qpos[self.arm_indices].copy();self.joint_mode=True

class Engine(old.Engine):
    def __init__(self,a):
        self.control_rows=[];self.ready=False
        old.MobileScene=lambda **kw:JointScene(**kw,mobile_start=(0,0,0) if a.start_docked else (-.7,-.8,0),stow=not a.start_docked)
        super().__init__(a)
        self.journal=Journal(self.out)
        self.labeller=Labeller(self.s)
        self.start_physics_id=id(self.s.sim)
        self.commanded_goal=None
        self.tcp=self.s.model.site_name2id('gripper0_grip_site')
        cid=self.s.model.camera_name2id('agentview')
        camR=self.s.data.cam_xmat[cid].reshape(3,3).copy()
        shift={'nominal':np.zeros(3),'camera_right_3cm':camR[:,0]*.03,'camera_up_3cm':camR[:,1]*.03}.get(a.variant)
        if shift is None:raise ValueError(a.variant)
        self.s.model.cam_pos[cid]+=shift
        self.s.sim.forward()
        if a.start_docked:
            self.s.activate_joint();self.phase='arm_ready';self.ready=True
        atomic_json(self.out/'initial_state.json',self.arm_state())
        self.manifest.update(experiment='camera grounding to MoveIt precontact',controller={'kp':400,'kd':40,'torque_limits':'original MuJoCo actuator_ctrlrange'},goal_offset_world=OFFSET.tolist(),variant=a.variant,depth_source='simulator true depth',collision_map_source='simulator geometry',target_source='predicted pixel except explicit oracle ablation',no_runtime_qpos_assignment=True,source_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (D/'scripts').glob('*.py')})
        atomic_json(self.out/'protocol.json',self.manifest)

    def record(self):
        super().record()
        if hasattr(self,'journal'):
            self.journal.append('physics_step',**self.arm_state())

    def arm_state(self):
        s=self.s;m=s.model;d=s.data
        # mj_step integrates qpos after computing derived poses: synchronize before sampling.
        s.sim.forward()
        b=m.body_name2id('robot0_link0');site=m.site_name2id('gripper0_grip_site')
        contacts=[];self_contacts=[]
        for c in d.contact[:d.ncon]:
            names=[m.geom_id2name(int(c.geom1)) or '',m.geom_id2name(int(c.geom2)) or '']
            arm=[n.startswith(('robot0','gripper0')) for n in names]
            if sum(arm)==2 and c.dist<-.0005:self_contacts.append({'geoms':names,'depth_m':float(c.dist)})
            if sum(arm)==1 and c.dist<-.0005:
                other=names[1 if arm[0] else 0]
                if not other.startswith(('mount0','mobile_')):
                    contacts.append({'geoms':names,'depth_m':float(c.dist)})
        return {'ok':True,'time':float(d.time),'phase':self.phase,'joint_positions':d.qpos[s.arm_indices].tolist(),
                'joint_velocities':d.qvel[s.arm_vel].tolist(),'effort':d.ctrl[s.arm_act].tolist(),'gravity_bias':d.qfrc_bias[s.arm_vel].tolist(),'actual_generalized_actuation':d.qfrc_actuator[s.arm_vel].tolist(),'constraint_force':d.qfrc_constraint[s.arm_vel].tolist(),
                'finger_positions':np.abs(s.obs['robot0_gripper_qpos']).tolist(),
                'base_p':d.body_xpos[b].tolist(),'base_R':d.body_xmat[b].reshape(3,3).tolist(),
                'tcp_p':d.site_xpos[site].tolist(),'tcp_R':d.site_xmat[site].reshape(3,3).tolist(),
                'contacts':contacts,'self_contacts':self_contacts,'physics_id_unchanged':id(s.sim)==s.initial_model_id}

    def capture(self):
        rgb,depth,cls=self.labeller.render()
        index=getattr(self,'capture_index',0);self.capture_index=index+1
        image=self.out/f'grounding_{index:03d}.png';Image.fromarray(rgb).save(image)
        np.savez_compressed(self.out/f'grounding_depth_{index:03d}.npz',depth=depth)
        atomic_json(self.out/f'shadow_labels_{index:03d}.json',self.labeller.labels(cls))
        self.rgb,self.depth,self.cls=rgb,depth,cls
        return {'ok':True,'image':str(image),'state':self.arm_state(),'collision_geometries':self.geometry()}

    def geometry(self):
        s=self.s;m=s.model;d=s.data;objects=[]
        for i in range(m.ngeom):
            name=m.geom_id2name(i) or f'geom_{i}';body=m.body_id2name(int(m.geom_bodyid[i])) or ''
            if body.startswith(('robot0','gripper0','mount0')):continue
            if not (m.geom_contype[i] or m.geom_conaffinity[i]):continue
            typ=int(m.geom_type[i]);size=np.asarray(m.geom_size[i]);row={'id':name,'p':d.geom_xpos[i].tolist(),'R':d.geom_xmat[i].reshape(3,3).tolist()}
            if typ==6:row.update(type='box',dimensions=(2*size).tolist())
            elif typ==2:row.update(type='sphere',dimensions=[float(size[0])])
            elif typ==5:row.update(type='cylinder',dimensions=[float(2*size[1]),float(size[0])])
            elif typ==0:row.update(type='box',dimensions=[8.,8.,.04],p=[0.,0.,-.02])
            elif typ==7:
                j=int(m.geom_dataid[i]);v0=int(m.mesh_vertadr[j]);n=int(m.mesh_vertnum[j]);f0=int(m.mesh_faceadr[j]);nf=int(m.mesh_facenum[j])
                row.update(type='mesh',vertices=np.asarray(m.mesh_vert[v0:v0+n]).tolist(),triangles=np.asarray(m.mesh_face[f0:f0+nf]).tolist())
            else:
                # Conservative bounding box for rare capsule/ellipsoid scene geometry.
                half=size.copy()
                if typ==3:half=np.array([size[0],size[0],size[0]+size[1]])
                row.update(type='box',dimensions=(2*half).tolist(),approximation='conservative_primitive_bounds')
            objects.append(row)
        return objects

    def request(self,q):
        op=q['op']
        if op=='dock_for_arm':
            self.phase='unfold';self.s.engage_manipulation();self.s.activate_joint();self.phase='arm_ready';self.ready=True
            return self.arm_state()
        if op=='arm_state':return self.arm_state()
        if op=='capture_grounding':
            if not self.ready:raise ValueError('Not docked')
            return self.capture()
        if op=='ground_target':
            if not self.ready or not hasattr(self,'depth'):raise ValueError('No observed frame')
            if q.get('mode')=='oracle':
                # Explicit privileged upper-bound ablation only, never used for base/LoRA.
                point=self.s.data.geom_xpos[self.s.model.geom_name2id(HANDLE_GEOM)].copy()
                uv,_=self.labeller.project(point)
            else:
                uv=np.asarray(q['point_2d'],float)/1000*512
                if uv.shape!=(2,) or not np.isfinite(uv).all() or not ((uv>=0)&(uv<512)).all():raise ValueError('Invalid predicted pixel')
                u,v=map(int,uv);z=float(np.median(self.depth[max(0,v-2):v+3,max(0,u-2):u+3]))
                if not .05<z<3:raise ValueError('Invalid depth')
                point=np.asarray(self.labeller.unproject(*uv,z))
            target=point+OFFSET
            state=self.arm_state();B=np.array(state['base_R']);bp=np.array(state['base_p'])
            local=B.T@(target-bp)
            if np.linalg.norm(local)>.95 or not -.1<local[2]<.9:raise ValueError('Predicted target outside validated reach envelope')
            self.commanded_goal=target.copy()
            response={'ok':True,'target_position':local.tolist(),'target_orientation':Rotation.from_matrix(B.T@np.array(state['tcp_R'])).as_quat().tolist(),
                      'predicted_world':point.tolist(),'predicted_uv':np.asarray(uv).tolist(),'offset_world':OFFSET.tolist(),
                      'base_state':state,'prediction_mode':q.get('mode')}
            atomic_json(self.out/'goal.json',response);return response
        if op=='joint_step':
            if not self.s.joint_mode:raise ValueError('Joint controller inactive')
            goal=np.asarray(q['positions'],float)
            if goal.shape!=(7,) or not np.isfinite(goal).all():raise ValueError('Bad joint command')
            if np.max(np.abs(goal-self.s.arm_target))>.15:raise ValueError('Trajectory command discontinuity')
            self.s.arm_target=goal;self.phase='joint_execution';self.s.step();state=self.arm_state()
            self.control_rows.append({'command':goal.tolist(),**state});return state
        if op=='evaluate_precontact':
            state=self.arm_state();truth=self.s.data.geom_xpos[self.s.model.geom_name2id(HANDLE_GEOM)].copy()+OFFSET
            result={'ok':True,'state':state,'target_tracking_error_m':float(np.linalg.norm(np.array(state['tcp_p'])-self.commanded_goal)) if self.commanded_goal is not None else None,
                    'shadow_true_precontact_error_m':float(np.linalg.norm(np.array(state['tcp_p'])-truth)),
                    'any_arm_environment_contact':any(x['contacts'] for x in self.control_rows),'any_self_contact':any(x.get('self_contacts') for x in self.control_rows),
                    'physics_id_unchanged':id(self.s.sim)==self.start_physics_id,'control_steps':len(self.control_rows)}
            atomic_json(self.out/'precontact_result.json',result);return result
        if op=='close_episode':
            self.phase='stopped';self.finish();return self.arm_state()
        return super().request(q)

    def finish(self):
        if hasattr(self,'control_rows'):atomic_json(self.out/'joint_execution.json',self.control_rows)
        super().finish()

def main():
    p=argparse.ArgumentParser();p.add_argument('--socket',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--task',type=int,default=3);p.add_argument('--initial',type=int,default=0);p.add_argument('--start-docked',action='store_true');p.add_argument('--variant',default='nominal')
    a=p.parse_args();assert not a.socket.exists();e=Engine(a)
    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            try:r=e.request(json.loads(self.rfile.readline(16*1024*1024)))
            except Exception:r={'ok':False,'error':traceback.format_exc()}
            self.wfile.write((json.dumps(r)+'\n').encode())
    try:
        with socketserver.UnixStreamServer(str(a.socket),Handler) as server:
            print('READY',flush=True);server.serve_forever()
    except KeyboardInterrupt:pass
    finally:
        e.writer.close();e.s.close();a.socket.unlink(missing_ok=True)
if __name__=='__main__':main()
