import json,subprocess,time
from pathlib import Path
S=Path(__file__).resolve().parents[1];V=S.parent;ROOT=V.parents[1]
def props(unit):
 r=subprocess.run(['systemctl','--user','show',unit,'-p','ActiveState','-p','Result'],capture_output=True,text=True,check=True);return dict(x.split('=',1) for x in r.stdout.splitlines())
def wait(unit):
 while props(unit)['ActiveState'] in ['active','activating']:time.sleep(2)
 assert props(unit)['Result']=='success',props(unit)
wait('ali-point-controller');assert json.loads((S/'PIPELINE_COMPLETE.json').read_text())['complete']
for unit,script in [('ali-point-analysis','summarize.py'),('ali-point-physics','audit_physics.py')]:
 cmd=f'exec bash experiments/vla_libero/scripts/python_shared experiments/vla_libero/point_target/scripts/{script} > experiments/vla_libero/point_target/logs/{script}.log 2>&1'
 subprocess.run(['systemd-run','--user','--unit='+unit,'--property=MemoryHigh=4G','--property=MemoryMax=6G','--property=WorkingDirectory='+str(ROOT),'/bin/bash','-c',cmd],check=True);wait(unit)
(S/'VERIFIED_READY.json').write_text(json.dumps({'ready':True,'hardware':False},indent=2))
