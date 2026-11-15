"""Freeze splits before generating target-conditioned simulated teaching data."""
import hashlib,json,shutil,sys
from pathlib import Path
from teacher import S,V,ROOT,trial
sys.path.insert(0,str(V/'scripts'))
from episode_journal import atomic_json

def main():
    files=[Path(__file__),S/'scripts/teacher.py']
    sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    spec=dict(train_initials=list(range(20,26)),validation_initials=[26,27],policy_test_initials=[28,29],training_reference_initial=0,teacher_uses_simulator_truth=True,policy_test_must_not_use_truth_for_control=True,reject_rule='Full-path protected displacement<=30mm, final10 targetIn and no gripper contact',sources=sources)
    path=S/'DATA_PROTOCOL.json'
    if path.exists():assert json.loads(path.read_text())==spec
    else:
        atomic_json(path,spec)
        for name in sources:
            p=S/'archive'/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,p)
    results=[]
    for split,initials in [('train',spec['train_initials']),('validation',spec['validation_initials'])]:
        for task in [0,1]:
            for init in initials:
                for target in [0,1]:
                    folder=S/f'episodes/{split}/task{task:02d}_target{target}_init{init:02d}'
                    if (folder/'result.json').exists():
                        r=json.loads((folder/'result.json').read_text());assert r['status']=='complete'
                    else:r=trial(task,init,target,split)
                    results.append(dict(split=split,task=task,initial=init,target=target,accepted=r['selective_success'],source=str((folder/'result.json').relative_to(ROOT)),trajectory=str((folder/'trajectory.npz').relative_to(ROOT)),steps=r['teacher_steps']))
                    atomic_json(S/'DATA_STATUS.json',dict(state='running',completed=len(results),expected=32,accepted=sum(r['accepted'] for r in results)))
                    atomic_json(S/'MANIFEST.json',results)
    atomic_json(S/'DATA_STATUS.json',dict(state='complete',completed=len(results),expected=32,accepted=sum(r['accepted'] for r in results)))

if __name__=='__main__':main()
