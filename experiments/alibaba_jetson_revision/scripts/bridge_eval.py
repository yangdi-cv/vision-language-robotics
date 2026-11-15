"""Dispatch saved Jetson language goals into the matching PC simulated layouts; no real motor access."""
from pathlib import Path
import sys,json,argparse,numpy as np,torch
from PIL import Image
R=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(R/'experiments/alibaba_pc_revision/scripts'))
from act_runner import rollout
from lerobot.policies.act.modeling_act import ACTPolicy
from task import make,image
p=argparse.ArgumentParser();p.add_argument('--predictions',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--policy',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
rows=json.loads(a.predictions.read_text());cases={c['id']:c for c in json.loads((a.data/'cases.json').read_text())}
policy=ACTPolicy.from_pretrained(a.policy).to('cuda').eval();records=[];checks={}
for row in rows:
 case=cases[row['id']];seed=int(case['id'].split('_')[0]);goal=row.get('parsed');schema=isinstance(goal,dict) and isinstance(goal.get('should_execute'),bool)
 execute=bool(schema and goal['should_execute'] and goal.get('target_color') in ['red','blue','green'] and goal.get('destination') in ['left','right'])
 record={'id':row['id'],'seed':seed,'goal':goal,'expected_execution':case['should_execute'],'executed':execute,'task_success':False}
 if execute:
  # Compare the initialized scene to the actual language input, before any policy motion.
  env,physics=make(seed)
  try:
   for _ in range(50):env.step(np.r_[physics.data.mocap_pos[1].copy(),[1,0,0,0],1])
   current=image(physics);reference=np.asarray(Image.open(a.data/case['image']).convert('RGB'))
   exact=bool(np.array_equal(current,reference));record['initial_image_exact_match']=exact
   if not exact:
    Image.fromarray(current).save(a.out/(row['id']+'_actual.png'))
    Image.fromarray(reference).save(a.out/(row['id']+'_reference.png'))
    delta=float(np.abs(current.astype(float)-reference.astype(float)).mean())
    print('IMAGE_DIFFERENCE',row['id'],delta,flush=True)
    raise ValueError('Language image does not match policy scene')
  finally:physics.free()
  result=rollout(policy,seed,goal['target_color'],goal['destination'],a.out/row['id'])
  record['task_success']=bool(result['success'] and goal['target_color']==case['target_color'] and goal['destination']==case['destination'])
 records.append(record);(a.out/'results.json').write_text(json.dumps(records,ensure_ascii=False,indent=2));print(record,flush=True)
positive=[x for x in records if x['expected_execution']];negative=[x for x in records if not x['expected_execution']]
(a.out/'summary.json').write_text(json.dumps({'positive_cases':len(positive),'positive_successes':sum(x['task_success'] for x in positive),'negative_cases':len(negative),'negative_executions':sum(x['executed'] for x in negative),'source':'saved actual Jetson 4B NF4+LoRA predictions; PC ACT simulation','clock':'offline staged handoff; not live network latency; simulator initialized after inference','same_initial_rgb_verified':all(x.get('initial_image_exact_match',True) for x in records),'robot_commands_sent':0},indent=2))
