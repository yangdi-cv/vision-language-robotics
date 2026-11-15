"""Independent forward/goal/contact rescore of held-out test trajectories."""
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
 rows=[];count=0;maxerr=0.
 for t in [0,1]:
  suite=benchmark.get_benchmark_dict()['libero_10']();task=suite.get_task(t);env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file),camera_heights=256,camera_widths=256,ignore_done=True);env.reset();ids=[env.env.obj_body_id[o[0]] for o in OBJECTS[t]]
  try:
   for v in ['base','text','goal']:
    for layout in ['normal','swap']:
     for p in sorted((S/'evaluation'/v/layout/'episodes').glob(f'task{t:02d}*/*/result.json')):
      r=json.loads(p.read_text());target=int(r['mode'][-1]);orig=np.array(r['origins']);inside=[];free=[];displacement=[]
      with np.load(p.parent/'trace.npz') as f:states=f['sim_state'];expected=f['object_positions']
      for state,pos in zip(states,expected):
       env.sim.set_state_from_flattened(state);env.sim.forward();positions=env.sim.data.body_xpos[ids];maxerr=max(maxerr,float(abs(positions-pos).max()));inside.append(bool(env.env._eval_predicate(env.env.parsed_problem['goal_state'][target])));free.append(not env.env.check_contact(env.env.robots[0].gripper.contact_geoms,env.env.get_object(OBJECTS[t][target][0]).contact_geoms));displacement.append(float(np.linalg.norm(positions[1-target]-orig[1-target])));count+=1
      sc=score(dict(target_in=inside,target_free=free,protected_displacement=displacement),300,'budget');rows.append(dict(variant=v,layout=layout,task=t,target=target,live=r['scores'][r['mode']]['selective_success'],restored=sc['selective_success']))
  finally:env.close()
 assert len(rows)==24
 atomic_json(S/'PHYSICS_AUDIT.json',dict(episodes=len(rows),states=count,final_disagreements=sum(r['live']!=r['restored'] for r in rows),max_position_error_m=maxerr,rows=rows))
if __name__=='__main__':main()
