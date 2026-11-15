"""Two privileged simulation-only OSC demonstrations; never a VLA evaluation."""
import argparse,hashlib,json,sys,time,traceback
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
S=Path(__file__).resolve().parents[1];V=S.parent;ROOT=V.parents[1]
sys.path[:0]=[str(V/'scripts'),str(V/'selective_execution/scripts')]
import evaluate as ev
from run_hierarchical import upright
from protocol import OBJECTS,score,skill
from episode_journal import atomic_json,append_jsonl


def grasp_reference(task):
    p=V/f'instruction_response/episodes/task{task:02d}_init00/target0'
    r=json.loads((p/'result.json').read_text())
    with np.load(p/'trace.npz') as f:
        positions=f['object_positions'];origin=np.asarray(r['origins']);mask=(positions[:,0,2]>origin[0,2]+.02)&f['contact'][:,0]&(f['applied'][:,6]>0)
        i=int(np.flatnonzero(mask)[0]);offset=f['eef'][i]-positions[i,0]
    with np.load(p/f'inputs/step{i//5*5:04d}.npz') as f:orientation=Rotation.from_rotvec(f['state'][3:6])
    return offset,orientation,dict(source=str(p.relative_to(ROOT)),frame=i,offset=offset.tolist(),orientation=orientation.as_rotvec().tolist())

def trial(task_id,initial,target,tag,grasp_adjust=0.):
    import imageio.v2 as imageio
    from libero.libero import benchmark,get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    offset,desired_ori,reference=grasp_reference(task_id);offset=offset.copy();offset[2]+=grasp_adjust
    folder=S/f'episodes/{tag}/task{task_id:02d}_target{target}_init{initial:02d}'
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'result.json').exists():raise RuntimeError('Refuse overwrite')
    record=dict(status='running',task=task_id,initial=initial,target=target,tag=tag,privileged_teacher=True,uses_object_positions=True,policy_inference=False,reference=reference,grasp_adjust_m=grasp_adjust,steps=0,phases=[])
    atomic_json(folder/'result.json',record);env=writer=None;rows=[];obs=None
    try:
        suite=benchmark.get_benchmark_dict()['libero_10']();task=suite.get_task(task_id)
        env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file),camera_heights=256,camera_widths=256,ignore_done=True)
        env.seed(7);env.reset();obs=env.set_init_state(np.asarray(ev.initial_states(suite,task_id)[initial]))
        for _ in range(10):obs,_,_,_=env.step(ev.DUMMY.tolist())
        ids=[env.env.obj_body_id[o[0]] for o in OBJECTS[task_id]];origins=env.sim.data.body_xpos[ids].copy();record['origins']=origins.tolist()
        basket=env.sim.data.body_xpos[env.env.obj_body_id['basket_1']].copy()
        grasp=origins[target]+offset;hover=grasp+np.array([0,0,.13]);lift=grasp+np.array([0,0,.20])
        transit=np.array([basket[0]+offset[0],basket[1]+offset[1],max(lift[2],basket[2]+.23)])
        release=transit.copy();release[2]=basket[2]+.15+offset[2]
        record.update(basket=basket.tolist(),grasp=grasp.tolist(),release=release.tolist(),controller_output_max=np.asarray(env.env.robots[0].controller.output_max).tolist())
        scale=np.asarray(env.env.robots[0].controller.output_max)[:6];assert np.allclose(scale,[.05,.05,.05,.5,.5,.5])
        writer=imageio.get_writer(folder/'rollout.mp4',fps=20,codec='libx264',quality=7,macro_block_size=16);writer.append_data(upright(obs))
        def apply(action,phase):
            nonlocal obs
            element=ev.make_input(obs,skill(OBJECTS[task_id][target][1]))
            obs,_,done,_=env.step(action.tolist());positions=env.sim.data.body_xpos[ids].copy();contact=np.array([env.env.check_contact(env.env.robots[0].gripper.contact_geoms,env.env.get_object(o[0]).contact_geoms) for o in OBJECTS[task_id]],bool)
            atoms=np.array([env.env._eval_predicate(a) for a in env.env.parsed_problem['goal_state']],bool)
            row=dict(action=action.copy(),phase=phase,image=element['observation/image'],wrist=element['observation/wrist_image'],state=element['observation/state'],object_positions=positions,contact=contact,atoms=atoms,eef=obs['robot0_eef_pos'].copy(),gripper=obs['robot0_gripper_qpos'].copy(),sim_state=env.sim.get_state().flatten().copy())
            rows.append(row);writer.append_data(upright(obs));append_jsonl(folder/'trace.jsonl',dict(step=len(rows),phase=phase,eef=row['eef'],positions=positions,contact=contact,atoms=atoms))
        def go(name,position,grip,max_steps,minimum=8):
            start=len(rows);reached=False
            for k in range(max_steps):
                delta=position-obs['robot0_eef_pos'];angular=(desired_ori*Rotation.from_quat(obs['robot0_eef_quat']).inv()).as_rotvec()
                action=np.r_[np.clip(delta/.05*.45,-.45,.45),np.clip(angular/.5*.35,-.5,.5),grip]
                apply(action,name)
                if k>=minimum and np.linalg.norm(delta)<.004 and np.linalg.norm(angular)<.08:reached=True;break
            record['phases'].append(dict(name=name,start=start,end=len(rows),reached=reached,final_error_m=float(np.linalg.norm(position-obs['robot0_eef_pos']))));record['steps']=len(rows);atomic_json(folder/'result.json',record)
        clear=obs['robot0_eef_pos'].copy();clear[2]=max(clear[2],hover[2])
        go('clear',clear,-1,20)
        go('hover',hover,-1,65)
        go('approach',grasp,-1,40)
        go('close',grasp,1,30,minimum=29)
        go('lift',lift,1,55)
        go('transfer',transit,1,65)
        go('lower',release,1,40)
        go('release',release,-1,35,minimum=34)
        go('retreat',transit,-1,45)
        teacher_steps=len(rows)
        for _ in range(25):apply(np.r_[np.zeros(6),-1.],'hold')
        arrays={k:np.asarray([r[k] for r in rows]) for k in rows[0]};np.savez_compressed(folder/'trajectory.npz',**arrays)
        a=dict(target_in=arrays['atoms'][:,target],target_free=~arrays['contact'][:,target],protected_displacement=np.linalg.norm(arrays['object_positions'][:,1-target]-origins[1-target],axis=1))
        record.update(score(a,teacher_steps,'teacher_end'),status='complete',steps=len(rows),teacher_steps=teacher_steps)
        atomic_json(folder/'result.json',record);print(json.dumps(record),flush=True)
        return record
    except BaseException as exc:
        if rows:np.savez_compressed(folder/'partial.npz',**{k:np.asarray([r[k] for r in rows]) for k in rows[0]})
        record.update(status='error',error=repr(exc),traceback=traceback.format_exc());atomic_json(folder/'result.json',record);raise
    finally:
        if writer is not None:writer.close()
        if env is not None:env.close()


def main():
    a=argparse.ArgumentParser();a.add_argument('--task',type=int,choices=[0,1],required=True);a.add_argument('--init',type=int,default=0);a.add_argument('--target',type=int,choices=[0,1],default=1);a.add_argument('--tag',default='pilot_v1');a.add_argument('--grasp-adjust',type=float,default=0.);args=a.parse_args()
    trial(args.task,args.init,args.target,args.tag,args.grasp_adjust)

if __name__=='__main__':main()
