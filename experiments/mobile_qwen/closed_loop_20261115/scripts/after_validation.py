"""One bounded native-skill diagnostic after the fixed validation batch."""
from pathlib import Path
import time,json,subprocess,sys
D=Path(__file__).resolve().parents[1]
for _ in range(240):
 p=D/'results/combined_validation1_summary.json'
 try:r=json.loads(p.read_text())
 except (OSError,json.JSONDecodeError):r=[]
 active=subprocess.run(['systemctl','--user','is-active','ali-precontact-combined-validation-20261115'],capture_output=True,text=True).stdout.strip()=='active'
 if len(r)==3 and not active:break
 if not active:raise RuntimeError('Prior batch did not finish')
 time.sleep(15)
else:raise TimeoutError('Prior batch exceeded one hour')
case=next(x for x in r if x['initial']==4)
if not case['success']:
 subprocess.run([sys.executable,str(D/'scripts/run_native_control.py'),'--prefix','native_probe1','--modes','lora','--initials','4','--variants','nominal','--navigate'],check=True)
else:print('No diagnostic needed')
