"""Check final scoring after restoring every recorded policy simulation state."""
import json,sys
from pathlib import Path
import numpy as np
S=Path(__file__).resolve().parents[1];V=S.parent;ROOT=V.parents[1]
sys.path[:0]=[str(V/'scripts'),str(V/'selective_execution/scripts')]
from episode_journal import atomic_json
from protocol import OBJECTS,score

def main(evaluation_root='policy_eval',variants=('base','lora'),output='POLICY_PHYSICS_AUDIT.json'):
    from libero.libero import benchmark,get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    rows=[];frames=0;max_error=0.;env=None
    try:
        for taskid in [0,1]:
            suite=benchmark.get_benchmark_dict()['libero_10']();task=suite.get_task(taskid)
            env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file),camera_heights=256,camera_widths=256,ignore_done=True);env.reset();ids=[env.env.obj_body_id[o[0]] for o in OBJECTS[taskid]]
            for variant in variants:
                for p in sorted((S/evaluation_root/variant/'episodes').glob(f'task{taskid:02d}*/*/result.json')):
                    r=json.loads(p.read_text());assert r['status']=='complete';target=int(r['mode'][-1]);origin=np.array(r['origins'])
                    with np.load(p.parent/'trace.npz') as f:states=f['sim_state'];expected=f['object_positions']
                    inside=[];free=[];displacements=[]
                    for state,positions in zip(states,expected,strict=True):
                        env.sim.set_state_from_flattened(state);env.sim.forward();pos=env.sim.data.body_xpos[ids].copy();max_error=max(max_error,float(np.max(abs(pos-positions))))
                        inside.append(bool(env.env._eval_predicate(env.env.parsed_problem['goal_state'][target])))
                        free.append(not env.env.check_contact(env.env.robots[0].gripper.contact_geoms,env.env.get_object(OBJECTS[taskid][target][0]).contact_geoms))
                        displacements.append(float(np.linalg.norm(pos[1-target]-origin[1-target])))
                    result=score(dict(target_in=inside,target_free=free,protected_displacement=displacements),300,'budget');rows.append(dict(variant=variant,task=taskid,initial=r['initial'],target=target,live=r['scores'][r['mode']]['selective_success'],restored=result['selective_success'],restored_score=result));frames+=len(states)
            env.close();env=None
        assert len(rows)==16
        atomic_json(S/output,dict(episodes=16,states=frames,max_position_error_m=max_error,final_disagreements=sum(r['live']!=r['restored'] for r in rows),rows=rows))
    finally:
        if env is not None:env.close()

if __name__=='__main__':
    main('policy_eval_paired',('base','paired'),'POLICY_PHYSICS_AUDIT_PAIRED.json') if '--paired' in sys.argv else main()
