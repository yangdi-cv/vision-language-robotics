"""Independently rescore saved teaching states, including rejected trajectories."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
S=Path(__file__).resolve().parents[1];V=S.parent;ROOT=V.parents[1]
sys.path[:0]=[str(V/'scripts'),str(V/'selective_execution/scripts')]
from protocol import OBJECTS,score
from episode_journal import atomic_json

def main():
    from libero.libero import benchmark,get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    cfg=json.loads((S/'DATA_PROTOCOL.json').read_text());manifest=json.loads((S/'MANIFEST.json').read_text())
    for path,digest in cfg['sources'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
    assert len(manifest)==32
    assert set(cfg['train_initials']).isdisjoint(cfg['validation_initials'])
    assert set(cfg['policy_test_initials']).isdisjoint(cfg['train_initials']+cfg['validation_initials'])
    rows=[];pair_inputs={};max_error=0.;frames=0;env=None
    try:
        for taskid in [0,1]:
            suite=benchmark.get_benchmark_dict()['libero_10']();task=suite.get_task(taskid)
            env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file),camera_heights=256,camera_widths=256,ignore_done=True);env.reset()
            ids=[env.env.obj_body_id[o[0]] for o in OBJECTS[taskid]]
            for row in [r for r in manifest if r['task']==taskid]:
                p=ROOT/row['trajectory'];r=json.loads((ROOT/row['source']).read_text())
                with np.load(p) as f:
                    states=f['sim_state'];positions=f['object_positions'];actions=f['action'];atoms=f['atoms'];contacts=f['contact'];origins=np.array(r['origins']);steps=len(actions)
                    assert actions.shape==(steps,7) and np.isfinite(actions).all()
                    assert np.max(abs(actions[:,:3]))<=.450001 and np.max(abs(actions[:,3:6]))<=.500001
                    assert all(x in [-1.,1.] for x in actions[:,6])
                    key=(taskid,row['initial']);initial=tuple(f[k][0].copy() for k in ['image','wrist','state'])
                    if key in pair_inputs:assert all(np.array_equal(a,b) for a,b in zip(initial,pair_inputs[key],strict=True))
                    else:pair_inputs[key]=initial
                    target=row['target'];target_in=[];target_free=[];displacement=[]
                    for state,expected in zip(states,positions,strict=True):
                        env.sim.set_state_from_flattened(state);env.sim.forward();pos=env.sim.data.body_xpos[ids].copy();max_error=max(max_error,float(np.max(abs(pos-expected))))
                        target_in.append(bool(env.env._eval_predicate(env.env.parsed_problem['goal_state'][target])))
                        target_free.append(not env.env.check_contact(env.env.robots[0].gripper.contact_geoms,env.env.get_object(OBJECTS[taskid][target][0]).contact_geoms))
                        displacement.append(float(np.linalg.norm(pos[1-target]-origins[1-target])))
                    result=score(dict(target_in=target_in,target_free=target_free,protected_displacement=displacement),r['teacher_steps'],'teacher_end')
                    assert result['selective_success']==row['accepted'],(row,result)
                    rows.append(dict(task=taskid,initial=row['initial'],target=target,accepted=row['accepted'],rescore=result['selective_success'],frames=len(states),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
                    frames+=len(states)
                    atomic_json(S/'DATA_AUDIT.json',dict(state='running',episodes=len(rows),frames=frames))
            env.close();env=None
        atomic_json(S/'DATA_AUDIT.json',dict(state='complete',episodes=len(rows),frames=frames,paired_initial_inputs=len(pair_inputs),max_restored_position_error_m=max_error,all_final_labels_agree=True,rows=rows))
        print(json.dumps(dict(episodes=len(rows),frames=frames,max_error=max_error)))
    finally:
        if env is not None:env.close()

if __name__=='__main__':main()
