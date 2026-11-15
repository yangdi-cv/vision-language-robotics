"""Keep each evaluation in its own memory-limited systemd service."""
import json,subprocess,time,shlex
from pathlib import Path
ROOT=Path('/home/yd/Documents/VLA');S=ROOT/'experiments/vla_libero/skill_bootstrap'

def properties(unit):
    r=subprocess.run(['systemctl','--user','show',unit,'-p','ActiveState','-p','Result'],capture_output=True,text=True,check=True)
    return dict(x.split('=',1) for x in r.stdout.splitlines() if '=' in x)

def main():
    import sys
    paired='--paired' in sys.argv
    if paired:
        assert json.loads((S/'lora_paired/TRAIN_STATUS.json').read_text())['state']=='complete'
    assert json.loads((S/'lora_pilot/ZERO_ADAPTER_CHECK.json').read_text())['passed']
    for variant in (['base','paired'] if paired else ['base','lora']):
        unit=('ali-vla-pairtest-' if paired else 'ali-vla-test-')+variant
        command='exec bash '+shlex.quote(str(S/'scripts/python_model.sh'))+' '+shlex.quote(str(S/('scripts/evaluate_paired.py' if paired else 'scripts/evaluate_policy.py')))+' --variant '+variant+' > '+shlex.quote(str(S/f'logs/test_{"paired_" if paired else ""}{variant}.log'))+' 2>&1'
        subprocess.run(['systemd-run','--user','--unit='+unit,'--property=MemoryHigh=14G','--property=MemoryMax=16G','--property=OOMPolicy=stop','--property=WorkingDirectory='+str(ROOT),'/bin/bash','-c',command],check=True)
        subprocess.run(['systemd-run','--user','--unit='+unit+'-monitor','--property=MemoryMax=128M','/usr/bin/python3',str(S/'scripts/watch_resources.py'),unit+'.service'],check=True)
        while properties(unit)['ActiveState'] in ['activating','active']:time.sleep(2)
        p=properties(unit);print(json.dumps(dict(variant=variant,**p)),flush=True)
        assert p['Result']=='success',p
        assert json.loads((S/f'{"policy_eval_paired" if paired else "policy_eval"}/{variant}/STATUS.json').read_text())['state']=='complete'
    (S/('SAFE_PAIRED_EVAL_COMPLETE.json' if paired else 'SAFE_EVAL_COMPLETE.json')).write_text(json.dumps({'complete':True,'variants':(['base','paired'] if paired else ['base','lora']),'episodes_each':8},indent=2))

if __name__=='__main__':main()
