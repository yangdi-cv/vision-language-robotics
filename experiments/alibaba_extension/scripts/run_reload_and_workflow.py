from pathlib import Path
import time,subprocess,os
ROOT=Path(__file__).resolve().parents[3];TASK=ROOT/'experiments/alibaba_extension';env=dict(os.environ)
for key in ('PYTHONPATH','PYTHONHOME','LD_LIBRARY_PATH'):env.pop(key,None)
while not (TASK/'runs/profile_bf16_no_cache/metrics.json').exists():time.sleep(2)
with (TASK/'setup_logs/quantized-reload.log').open('w') as log:
 proc=subprocess.run([str(ROOT/'.venvs/qwen-scene/bin/python'),str(TASK/'scripts/reload_quantized.py')],stdout=log,stderr=subprocess.STDOUT,env=env)
if proc.returncode:raise RuntimeError('Quantized reload failed')
with (TASK/'setup_logs/qwen-nav-act-workflow.log').open('w') as log:
 proc=subprocess.run(['/usr/bin/python3',str(TASK/'scripts/run_qwen_nav_act.py'),'--out',str(TASK/'runs/qwen_nav_act_001')],stdout=log,stderr=subprocess.STDOUT)
if proc.returncode:raise RuntimeError('Qwen workflow failed')
print('Reload and Qwen workflow complete',flush=True)
