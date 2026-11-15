"""Bounded exploratory reproduction, paper's training-free CAG variant only."""
import hashlib,importlib.util,json,sys,time,traceback
from pathlib import Path
import numpy as np
S=Path(__file__).resolve().parents[1];V=S.parent;ROOT=V.parents[1]
sys.path[:0]=[str(V/'skill_bootstrap/scripts'),str(V/'scripts'),str(V/'selective_execution/scripts')]
from runtime import load_policy
from cag_policy import CAGPolicy
from mixing import combine
from episode_journal import atomic_json
spec=importlib.util.spec_from_file_location('frozen_engine',V/'instruction_response/scripts/run.py');engine=importlib.util.module_from_spec(spec);spec.loader.exec_module(engine)

def main():
 files=[*sorted((S/'scripts').glob('*.py')),V/'skill_bootstrap/scripts/runtime.py',V/'skill_bootstrap/scripts/bounded_restore.py',V/'skill_bootstrap/scripts/chunk_math.py',V/'instruction_response/scripts/run.py',V/'selective_execution/scripts/protocol.py',V/'scripts/evaluate.py']
 protocol=dict(method='Training-free final-action CAG, local reproduction of arXiv:2602.17659v1 Eq4; official code not verified',tasks=[0,1],initials=[0,11],targets=[0,1],scales=[1.,1.5],episodes=16,split='exploratory development layouts already used in diagnostic; not held-out',policy_steps=300,hold_steps=25,horizon=10,execute=5,same_explicit_noise=True,null_task='empty prompt before original transforms; proprioception retained',mix_space='7D unnormalized LIBERO action output; original env clipping unchanged; gripper also mixed',mix_note='Final-action linear mixing, not flow-velocity guidance. Inverse affine normalization commutes with this rule.',baseline='native base only, no adapter; one call/query; scale1 algebra control',no_training=True,no_hardware=True,no_Qwen=True,no_oracle_control=True,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
 assert not (S/'PROTOCOL.json').exists(),'Refuse overwrite'
 atomic_json(S/'PROTOCOL.json',protocol);version=hashlib.sha256((S/'PROTOCOL.json').read_bytes()).hexdigest()
 policy=load_policy()
 # Fixed saved observation, not simulator coordinates. Check actual null-task token contents.
 source=V/'instruction_response/episodes/task00_init00/target0/inputs/step0000.npz'
 with np.load(source) as f:obs={'observation/image':f['image'],'observation/wrist_image':f['wrist'],'observation/state':f['state'],'prompt':str(f['prompt'])};noise=f['noise']
 wrapper=CAGPolicy(policy,1.5);c,u,meta=wrapper.branches(obs,noise)
 reference=np.asarray(policy.infer(obs,noise=noise)['actions']);repeat=np.asarray(policy.infer(obs,noise=noise)['actions'])
 from openpi.models.tokenizer import PaligemmaTokenizer
 tok=PaligemmaTokenizer(200)._tokenizer
 ct=tok.decode(meta['conditional_tokens']);ut=tok.decode(meta['null_tokens'])
 from openpi.training import config
 discrete=config.get_config('pi05_libero').model.discrete_state_input
 if discrete:
  assert 'State:' in ut and ct.split('State:',1)[1]==ut.split('State:',1)[1]
  assert ut.split('State:',1)[0].strip()=='Task: ,',repr(ut)
 else:assert not ut.strip(),repr(ut)
 meta.update(discrete_state_input=discrete,normalized_state_retained=True)
 check=dict(**meta,conditional_text=ct,null_text=ut,scale1_max_abs=float(abs(combine(c,u,1)-reference).max()),repeat_max_abs=float(abs(reference-repeat).max()),language_delta_rms=float(np.sqrt(np.mean((c-u)**2))),passed=bool(np.array_equal(combine(c,u,1),reference) and np.array_equal(reference,repeat)),no_new_model=True)
 atomic_json(S/'CONTROL_CHECK.json',check);assert check['passed'],check
 engine.probes=lambda *args,**kwargs:None # paired rollout data, no unrelated 4-way probes
 completed=0
 for task in [0,1]:
  for initial in [0,11]:
   for mode in ['target0','target1']:
    for scale,variant in [(1.,'base'),(1.5,'cag15')]:
     engine.S=S/'development'/variant;w=CAGPolicy(policy,scale);w.start(engine.S/f'episodes/task{task:02d}_init{initial:02d}/{mode}')
     r=engine.rollout(w,task,initial,mode,version);assert w.index==60
     completed+=1;atomic_json(S/'STATUS.json',dict(state='running',complete=completed,expected=16,latest=[task,initial,mode,variant]));print(json.dumps(dict(complete=completed,task=task,initial=initial,target=mode,variant=variant,score=r['scores'][mode])),flush=True)
 atomic_json(S/'STATUS.json',dict(state='complete',complete=completed,expected=16))
if __name__=='__main__':
 try:main()
 except BaseException as e:
  atomic_json(S/'ERROR.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
