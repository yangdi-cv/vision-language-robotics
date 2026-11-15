import subprocess,time,json
from pathlib import Path
S=Path(__file__).resolve().parents[1];V=S.parent;ROOT=V.parents[1]
def props(unit):
 r=subprocess.run(['systemctl','--user','show',unit,'-p','ActiveState','-p','Result'],capture_output=True,text=True,check=True);return dict(x.split('=',1) for x in r.stdout.splitlines())
for backend in ['fixed','unmerged']:
 unit='ali-target-probe-'+backend;cmd=f'exec bash experiments/vla_libero/official_instance/python_model.sh experiments/vla_libero/target_skill/scripts/probe.py --backend {backend} > experiments/vla_libero/target_skill/logs/probe_{backend}.log 2>&1'
 subprocess.run(['systemd-run','--user','--unit='+unit,'--property=MemoryHigh=14G','--property=MemoryMax=16G','--property=OOMPolicy=stop','--property=WorkingDirectory='+str(ROOT),'/bin/bash','-c',cmd],check=True)
 while props(unit)['ActiveState'] in ['active','activating']:time.sleep(2)
 assert props(unit)['Result']=='success',props(unit)
(S/'PROBE_COMPLETE.json').write_text(json.dumps({'complete':True,'old_reference':'stored actual-input outputs; independent re-run showed small cross-process rounding difference and was retained'},indent=2))
