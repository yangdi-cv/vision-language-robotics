"""Qwen skill selection -> public ACT -> actual simulator feedback; no physical robot."""
import os,json,subprocess,time,argparse,re
from pathlib import Path
import numpy as np,torch,gymnasium as gym,gym_aloha,av
from PIL import Image
from lerobot.policies.act.modeling_act import ACTPolicy
ROOT=Path(__file__).resolve().parents[3];TASK=ROOT/'experiments/alibaba_baseline'
parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=TASK/'runs/integrated_demo_001');parser.add_argument('--guarded',action='store_true');args=parser.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=False)

def allowed_skill(instruction):
 # Bounded single-skill grammar, independent of evaluation labels and model output.
 if any(word in instruction for word in ('不要','别','不抓','不能','停止')):return 'observe'
 return 'transfer_red_cube' if re.fullmatch(r'(?:请)?(?:抓起|抓取|拿起)红色(?:方块|立方体)(?:[，,].*)?[。.!！]?',instruction) else 'observe'
import sys
sys.path.insert(0,str(ROOT/'perception/qwen_scene'))
from worker_client import read_json_line,close_json_worker
worker_env=dict(os.environ)
for name in ('PYTHONPATH','PYTHONHOME','LD_LIBRARY_PATH'):worker_env.pop(name,None)
worker_log=(out/'qwen.log').open('w')
worker=subprocess.Popen([str(ROOT/'.venvs/qwen-scene/bin/python'),str(TASK/'scripts/qwen_goal_worker.py')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=worker_log,text=True,env=worker_env)
__import__('atexit').register(close_json_worker,worker)
assert read_json_line(worker,timeout=90)['ready']
policy=ACTPolicy.from_pretrained(TASK/'data/official_act').to('cuda').eval();torch.set_num_threads(8)
env=gym.make('gym_aloha/AlohaTransferCube-v0',obs_type='pixels_agent_pos',max_episode_steps=400)
cases=[{'seed':seed,'instruction':'抓起红色方块，并把它交给另一只机械臂。','expected_skill':'transfer_red_cube'} for seed in range(1010,1015)]+[{'seed':seed,'instruction':'请抓起蓝色球。','expected_skill':'observe'} for seed in range(1015,1017)]
results=[]
try:
 for case in cases:
  obs,info=env.reset(seed=case['seed']);policy.reset();image_path=out/f"initial_{case['seed']}.png";Image.fromarray(obs['pixels']['top']).save(image_path)
  worker.stdin.write(json.dumps({'image':str(image_path.resolve()),'instruction':case['instruction']},ensure_ascii=False)+'\n');worker.stdin.flush()
  decision=read_json_line(worker,timeout=90)
  permission=allowed_skill(case['instruction']) if args.guarded else None
  execute=decision['valid'] and decision['goal']['target_visible'] and decision['goal']['selected_skill']=='transfer_red_cube' and (not args.guarded or permission=='transfer_red_cube')
  row={**case,'decision':decision,'guarded':args.guarded,'authorized_skill':permission,'executed':bool(execute),'max_reward':0.,'success':False,'steps':0}
  if execute:
   sys.path.insert(0,str(ROOT/'experiments/alibaba_pc_revision/scripts'))
   from aloha_runner import rollout_episode
   episode=rollout_episode(policy,env,obs,out/f"rollout_{case['seed']}.mp4")
   row.update({k:episode[k] for k in ('max_reward','steps','success')})
  row['skill_correct']=decision['valid'] and decision['goal']['selected_skill']==case['expected_skill']
  row['effective_skill_correct']=('transfer_red_cube' if execute else 'observe')==case['expected_skill']
  results.append(row);(out/f"case_{case['seed']}.json").write_text(json.dumps(row,ensure_ascii=False,indent=2))
  print('seed',case['seed'],'selected',decision.get('goal'),'success',row['success'],flush=True)
finally:
 env.close();close_json_worker(worker);worker_log.close()
summary={'cases':len(results),'correct_skill_selections':sum(r['skill_correct'] for r in results),'effective_correct_after_guard':sum(r['effective_skill_correct'] for r in results),'guarded':args.guarded,'requested_red_cube_cases':5,'red_cube_successes':sum(r['success'] for r in results if r['expected_skill']=='transfer_red_cube'),'negative_cases':2,'negative_cases_executed':sum(r['executed'] for r in results if r['expected_skill']=='observe'),'act_source':'public pretrained model, not own trained model','qwen_source':'base model, not pilot LoRA adapter','notes':'Constrained VLM skill gate plus ACT in MuJoCo. Not Nav2 navigation, not end-to-end VLA, not real robot evaluation.'}
(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
