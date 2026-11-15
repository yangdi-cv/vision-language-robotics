"""Prespecified validation tuning, then untouched rollout seeds for paired comparisons."""
from pathlib import Path
import subprocess,time,json
ROOT=Path(__file__).resolve().parents[3];TASK=ROOT/'experiments/alibaba_extension'
# Keep our expensive GPU profiling separate from ACT evaluations.
while not (TASK/'runs/profile_bf16_no_cache/metrics.json').exists():time.sleep(2)
python=ROOT/'experiments/alibaba_baseline/scripts/act-python';script=TASK/'scripts/evaluate_act_extended.py';model=TASK/'runs/act_continue_001/best_checkpoint'
def run(name,model,steps,episodes,seeds):
 cmd=[str(python),str(script),'--model',str(model),'--out',str(TASK/'runs'/name),'--action-steps',str(steps),'--episodes',str(episodes),'--seed-start',str(seeds)]
 with (TASK/'setup_logs'/(name+'.log')).open('w') as f:proc=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
 if proc.returncode:raise RuntimeError(name+' failed')
 return json.loads((TASK/'runs'/name/'metrics.json').read_text())
tuned=[]
for steps in (10,25,100):
 name='act_tune_steps_'+str(steps);metrics=run(name,model,steps,5,2000);episodes=json.loads((TASK/'runs'/name/'rollout_2000.json').read_text())
 rewards=[json.loads((TASK/'runs'/name/f'rollout_{seed}.json').read_text())['summary']['max_reward'] for seed in range(2000,2005)]
 tuned.append({'n_action_steps':steps,'successes':metrics['successes'],'mean_max_reward':sum(rewards)/len(rewards)})
# Highest success, then reward progression, then larger chunk (less frequent inference) on ties.
selected=max(tuned,key=lambda r:(r['successes'],r['mean_max_reward'],r['n_action_steps']))['n_action_steps']
(TASK/'runs/act_selection.json').write_text(json.dumps({'tuning_seeds':list(range(2000,2005)),'test_seeds':list(range(3000,3030)),'checkpoint_selected_by':'offline validation normalized L1 only','tuning':tuned,'selected_n_action_steps':selected,'rule':'successes; mean max reward; larger n_action_steps on ties'},indent=2))
run('act_improved_test30',model,selected,30,3000)
run('act_original_test30',ROOT/'experiments/alibaba_baseline/runs/own_act_train_001/final_checkpoint',100,30,3000)
run('act_public_test30',ROOT/'experiments/alibaba_baseline/data/official_act',100,30,3000)
print('ACT suite complete',flush=True)
