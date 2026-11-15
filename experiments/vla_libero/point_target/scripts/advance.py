import json,subprocess,time
from pathlib import Path
S=Path(__file__).resolve().parents[1];V=S.parent;ROOT=V.parents[1]
def state(unit):
 r=subprocess.run(['systemctl','--user','show',unit,'-p','ActiveState','-p','Result'],text=True,capture_output=True,check=True);return dict(x.split('=',1) for x in r.stdout.splitlines())
def wait(unit):
 while state(unit)['ActiveState'] in ['active','activating']:time.sleep(2)
 assert state(unit)['Result']=='success',state(unit)
def launch(unit,script,log,args=()):
 import shlex
 cmd='exec bash '+shlex.quote(str(V/'skill_bootstrap/scripts/python_model.sh'))+' '+shlex.quote(str(S/'scripts'/script))+' '+' '.join(shlex.quote(x) for x in args)+' > '+shlex.quote(str(S/'logs'/log))+' 2>&1'
 subprocess.run(['systemd-run','--user','--unit='+unit,'--property=MemoryHigh=14G','--property=MemoryMax=16G','--property=OOMPolicy=stop','--property=WorkingDirectory='+str(ROOT),'/bin/bash','-c',cmd],check=True)
 subprocess.run(['systemd-run','--user','--unit='+unit+'-monitor','--property=MemoryMax=128M','/usr/bin/python3',str(V/'skill_bootstrap/scripts/watch_resources.py'),unit+'.service'],check=True)
 wait(unit)
def main():
 wait('ali-point-data');assert json.loads((S/'DATA_STATUS.json').read_text())['state']=='complete';wait('ali-vla-official-instance')
 launch('ali-point-train','train.py','train.log')
 for variant in ['base','text','goal']:launch('ali-point-eval-'+variant,'evaluate.py','eval_'+variant+'.log',['--variant',variant])
 (S/'PIPELINE_COMPLETE.json').write_text(json.dumps({'complete':True,'variants':['base','text','goal'],'episodes':24},indent=2))
if __name__=='__main__':main()
