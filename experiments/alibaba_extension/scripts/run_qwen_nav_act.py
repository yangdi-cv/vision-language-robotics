"""Current MuJoCo frame -> Qwen -> contract -> Nav2 -> public ACT -> return (separate worlds)."""
import os,json,subprocess,argparse,time
from pathlib import Path
from task_contract import authorize
ROOT=Path(__file__).resolve().parents[3];TASK=ROOT/'experiments/alibaba_extension';p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
import sys
sys.path.insert(0,str(ROOT/'perception/qwen_scene'))
from worker_client import read_json_line,close_json_worker
env=dict(os.environ)
for key in ('PYTHONPATH','PYTHONHOME','LD_LIBRARY_PATH'):env.pop(key,None)
with (a.out/'qwen_worker.log').open('w') as log:
 worker=subprocess.Popen([str(ROOT/'.venvs/qwen-scene/bin/python'),str(ROOT/'experiments/alibaba_baseline/scripts/qwen_goal_worker.py')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,env=env)
 try:
  __import__('atexit').register(close_json_worker,worker)
  assert read_json_line(worker,timeout=90)['ready']
  subprocess.run([str(ROOT/'experiments/alibaba_baseline/scripts/act-python'),str(TASK/'scripts/capture_act_scene.py'),'--out',str(a.out/'initial.png')],check=True)
  instruction='抓起红色方块，并把它交给另一只机械臂。';sent=time.monotonic();worker.stdin.write(json.dumps({'image':str((a.out/'initial.png').resolve()),'instruction':instruction},ensure_ascii=False)+'\n');worker.stdin.flush();decision=read_json_line(worker,timeout=90)
  approval=authorize(instruction,decision.get('goal') if decision['valid'] else None,age_s=time.monotonic()-sent);(a.out/'authorization.json').write_text(json.dumps({'instruction':instruction,'decision':decision,'authorization':approval,'source':'actual initial MuJoCo frame seed1003; current inference, not stored prediction','scene_shared_with_navigation':False},ensure_ascii=False,indent=2))
 finally:close_json_worker(worker)
if approval['execute']:
 cmd=['/usr/bin/python3',str(TASK/'scripts/run_nav2.py'),'--out',str(a.out/'workflow'),'--with-act']
 with (a.out/'orchestration.log').open('w') as log:proc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
 nav=json.loads((a.out/'workflow/navigation_results.json').read_text());act=json.loads((a.out/'workflow/handoff.json').read_text());success=all(x['succeeded'] for x in nav) and len(nav)==2 and act['act_successes']==1
else:success=False
(a.out/'summary.json').write_text(json.dumps({'authorized':approval['execute'],'workflow_success':success,'real_robot':False,'same_physical_scene':False,'act_model':'public reference','seed':'known-success1003 smoke test','notes':'End-to-end process/interface demonstration across independent simulators, not a mobile manipulator in one physical world'},indent=2));print('Workflow success',success,flush=True)
