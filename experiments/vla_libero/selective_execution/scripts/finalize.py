"""Wait for all evaluation episodes and build checked deliverables; never rerun them."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

S=Path(__file__).resolve().parents[1];V=S.parent


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pid',type=int,required=True);ap.add_argument('--cpu-python',required=True);a=ap.parse_args()
    while True:
        status=json.loads((S/'STATUS.json').read_text())
        if status['state']=='complete' and status['stage']=='eval':
            assert status['completed']==140;break
        try:os.kill(a.pid,0)
        except ProcessLookupError as exc:raise RuntimeError('Evaluation exited without completing; inspect eval.log') from exc
        time.sleep(10)
    for phase in ['pilot','eval']:
        subprocess.run([sys.executable,str(S/'scripts/analyze.py'),'--phase',phase],check=True)
    subprocess.run(['bash',str(V/'scripts/python_shared'),str(S/'scripts/verify.py'),'--phase','skills','--physics'],check=True)
    subprocess.run(['bash',str(V/'scripts/python_shared'),str(S/'scripts/verify.py'),'--phase','eval','--physics'],check=True)
    verification=json.loads((S/'VERIFICATION_eval.json').read_text())
    assert not verification['restored_final_score_disagreements'], 'Restored scores differ; inspect before reporting'
    summary=json.loads((S/'SUMMARY_eval.json').read_text())
    assert all(x['initial_equal'] and x['input_hashes_equal'] and x['max_action_difference']==0 for x in summary['prefixes'])
    subprocess.run([sys.executable,str(S/'scripts/collect.py')],check=True)
    subprocess.run([sys.executable,str(S/'scripts/build.py')],check=True)
    validate=[sys.executable,str(S/'scripts/validate_package.py'),'--python',a.cpu_python]
    subprocess.run(validate+['--record'],check=True)
    from build import package
    package()
    subprocess.run(validate,check=True)
    (S/'FINALIZATION.json').write_text(json.dumps(dict(complete=True,evaluation_episodes=140,pilot_episodes=28,reference_skills=4,cpu_acceptance=True),indent=2))


if __name__=='__main__':main()
