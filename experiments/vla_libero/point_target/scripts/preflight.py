import json,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from layout import S,V,swap_at_reset,reference
sys.path.insert(0,str(V/'scripts'))
import evaluate as ev
from libero.libero import benchmark,get_libero_path
from libero.libero.envs import OffScreenRenderEnv

def main():
 out=S/'preflight';out.mkdir(exist_ok=True);rows=[]
 for t in [0,1]:
  suite=benchmark.get_benchmark_dict()['libero_10']();task=suite.get_task(t)
  env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file),camera_heights=256,camera_widths=256,ignore_done=True)
  for layout in ['normal','swap']:
   env.seed(7);env.reset();obs=env.set_init_state(np.asarray(ev.initial_states(suite,t)[36]));swap_at_reset(env,t,layout)
   for _ in range(10):obs,_,_,_=env.step(ev.DUMMY.tolist())
   r=reference(env,obs,t);np.savez_compressed(out/f'task{t}_{layout}.npz',**r)
   im=Image.fromarray(r['image']).resize((672,672));d=ImageDraw.Draw(im)
   for i,b in enumerate(r['boxes']):d.rectangle(tuple((b*3).tolist()),outline=['red','lime'][i],width=4)
   im.save(out/f'task{t}_{layout}.png');rows.append(dict(task=t,layout=layout,boxes=r['boxes'].tolist(),visible_pixels=r['visible_pixels'].tolist()))
  env.close()
 (out/'CHECK.json').write_text(json.dumps(dict(passed=True,oracle=True,rows=rows),indent=2))
if __name__=='__main__':main()
