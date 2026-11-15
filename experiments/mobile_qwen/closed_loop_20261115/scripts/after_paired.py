"""Sequential transition from the precontact comparison to one full-chain probe."""
from pathlib import Path
import subprocess,json,sys,time
D=Path(__file__).resolve().parents[1]
for _ in range(240):
 p=D/'results/paired_v1_summary.json'
 if p.exists() and len(json.loads(p.read_text()))==12:break
 state=subprocess.run(['systemctl','--user','is-active','ali-precontact-paired-20261115'],capture_output=True,text=True).stdout.strip()
 if state!='active':raise RuntimeError('Paired batch ended before all cases were recorded')
 time.sleep(15)
else:raise TimeoutError('Prior batch exceeded one hour')
# Wait until the prior Qwen worker has actually been closed by the owning driver.
while subprocess.run(['systemctl','--user','is-active','ali-precontact-paired-20261115'],capture_output=True,text=True).stdout.strip()=='active':time.sleep(3)
subprocess.run([sys.executable,str(D/'scripts/run_combined.py'),'--prefix','combined_probe1','--modes','lora','--initials','0','--variants','nominal','--navigate'],check=True)
