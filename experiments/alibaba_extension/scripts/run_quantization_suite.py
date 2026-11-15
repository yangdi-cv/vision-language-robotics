"""Wait for training/evaluation to finish, then profile models without our other GPU jobs."""
import os,time,subprocess,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];TASK=ROOT/'experiments/alibaba_extension';env=dict(os.environ)
for k in ('PYTHONPATH','PYTHONHOME','LD_LIBRARY_PATH'):env.pop(k,None)
for marker in (TASK/'runs/act_continue_001/training_summary.json',TASK/'runs/qwen_rgbd_lora_heldout/metrics.json'):
 while not marker.exists():time.sleep(2)
runs=[('bf16_cache',[]),('int8_cache',['--precision','int8','--save-model']),('bf16_no_cache',['--disable-kv-cache'])]
for name,extra in runs:
 cmd=[str(ROOT/'.venvs/qwen-scene/bin/python'),str(TASK/'scripts/qwen_rgbd_eval.py'),'--limit','40','--out',str(TASK/'runs'/('profile_'+name))]+extra
 with (TASK/'setup_logs'/('profile_'+name+'.log')).open('w') as log:
  proc=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
 if proc.returncode:raise RuntimeError(name+' failed; inspect log')
 print('Completed',name,flush=True)
