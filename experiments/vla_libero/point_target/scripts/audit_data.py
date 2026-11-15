"""Rescore all 40 new teacher candidates, including rejected ones."""
import json,sys
from pathlib import Path
import numpy as np
S=Path(__file__).resolve().parents[1];V=S.parent
sys.path[:0]=[str(V/'scripts'),str(V/'selective_execution/scripts')]
from episode_journal import atomic_json
from protocol import OBJECTS,score
from libero.libero import benchmark,get_libero_path
from libero.libero.envs import OffScreenRenderEnv

def main():
 candidates=[r for r in json.loads((S/'MANIFEST_CANDIDATES.json').read_text()) if r['origin']=='new_reset_swap_teacher'];assert len(candidates)==40
 rows=[];count=0;maxerr=0.
 for t in [0,1]:
  suite=benchmark.get_benchmark_dict()['libero_10']();task=suite.get_task(t);env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file),camera_heights=256,camera_widths=256,ignore_done=True);env.reset();ids=[env.env.obj_body_id[o[0]] for o in OBJECTS[t]]
  try:
   for row in [r for r in candidates if r['task']==t]:
    path=V.parents[1]/row['trajectory'];r=json.loads((path.parent/'result.json').read_text());orig=np.array(r['origins']);target=row['target'];inside=[];free=[];dis=[]
    with np.load(path) as f:states=f['sim_state'];expected=f['object_positions']
    for state,pos in zip(states,expected):
     env.sim.set_state_from_flattened(state);env.sim.forward();p=env.sim.data.body_xpos[ids];maxerr=max(maxerr,float(abs(p-pos).max()));inside.append(bool(env.env._eval_predicate(env.env.parsed_problem['goal_state'][target])));free.append(not env.env.check_contact(env.env.robots[0].gripper.contact_geoms,env.env.get_object(OBJECTS[t][target][0]).contact_geoms));dis.append(float(np.linalg.norm(p[1-target]-orig[1-target])));count+=1
    sc=score(dict(target_in=inside,target_free=free,protected_displacement=dis),r['teacher_steps'],'teacher_end');rows.append(dict(task=t,initial=row['initial'],layout=row['layout'],target=target,live=row['accepted'],restored=sc['selective_success']))
  finally:env.close()
 atomic_json(S/'DATA_PHYSICS_AUDIT.json',dict(candidates=40,states=count,max_position_error_m=maxerr,final_disagreements=sum(r['live']!=r['restored'] for r in rows),rows=rows))
if __name__=='__main__':main()
