"""Stage the minimal read-only evaluator on Jetson; no model loading."""
from pathlib import Path
import subprocess
Q=Path(__file__).resolve().parents[3];S=Path(__file__).resolve().parents[1];ROOT=Q.parents[1]
host='csip@192.168.31.168';remote='/home/csip/ali-jetson/project'
files=[Q/'scripts/qwen_worker.py',Q/'affordance/eval_qwen.py',ROOT/'experiments/vla_libero/scripts/qwen_planner.py',ROOT/'perception/qwen_scene/engine.py',*list((S/'scripts').glob('*.py'))]
for directory in sorted({str(x.parent.relative_to(ROOT)) for x in files}):
 subprocess.run(['sshpass','-e','ssh',host,'mkdir','-p',remote+'/'+directory],check=True)
for f in files:
 subprocess.run(['sshpass','-e','scp',str(f),host+':'+remote+'/'+str(f.relative_to(ROOT))],check=True)
subprocess.run(['sshpass','-e','scp','-r',str(S/'test350'),host+':'+remote+'/'+str(S.relative_to(ROOT))+'/'],check=True)
print(remote+'/'+str(S.relative_to(ROOT)))
