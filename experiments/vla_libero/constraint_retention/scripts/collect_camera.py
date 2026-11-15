"""Collect only twelve new policy observation streams if prompt expansion is rejected."""
import hashlib,json
from pathlib import Path
import sys
import numpy as np
from design import S,V,case
from execution import Worker,run_episode

ROOT=V.parents[1];sys.path.insert(0,str(V/'scripts'))
from episode_journal import atomic_json


def main():
    protocol=json.loads((S/'PROTOCOL.json').read_text())
    for name,digest in protocol['sources'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    sha=hashlib.sha256((S/'PROTOCOL.json').read_bytes()).hexdigest()
    atomic_json(S/'CAMERA_COLLECTION_PROTOCOL.json',dict(initials=[8,9,10],mode='positive_continue',episodes=12,
        purpose='New passive observation streams for a frozen multiview checker audit, not a prompt-performance expansion',executor_sha256=hashlib.sha256((S/'scripts/execution.py').read_bytes()).hexdigest()))
    worker=Worker()
    try:
        from openpi.training import config
        from openpi.policies import policy_config
        policy=policy_config.create_trained_policy(config.get_config('pi05_libero'),V/'checkpoints/pi05_libero',sample_kwargs={'num_steps':10})
        policy.infer({'observation/image':np.zeros((224,224,3),np.uint8),'observation/wrist_image':np.zeros((224,224,3),np.uint8),'observation/state':np.zeros(8,np.float32),'prompt':'put the object in the basket'},noise=np.zeros((10,32),np.float32))
        rows=[]
        for task in [0,1]:
            for initial in [8,9,10]:
                for target in [0,1]:
                    c=case(task,target,initial);r=run_episode(c,'positive_continue','camera_eval',policy,worker,{},sha)
                    path=S/f'episodes/camera_eval/{c["key"]}/positive_continue/result.json'
                    rows.append(str(path.relative_to(ROOT)))
                    atomic_json(S/'CAMERA_STATUS.json',dict(state='running',completed=len(rows),expected=12,latest=c['key']))
                    print(json.dumps(dict(case=c['key'],steps=r['decision_step'])),flush=True)
        atomic_json(S/'CAMERA_INDEX.json',rows)
        atomic_json(S/'CAMERA_STATUS.json',dict(state='complete',completed=12))
    finally:worker.close()


if __name__=='__main__':main()
