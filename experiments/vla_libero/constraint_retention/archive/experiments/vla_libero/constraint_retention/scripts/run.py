"""Bounded development screen, then preregistered fresh-initial-state validation."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
from design import S,V,PREVIOUS,MODES,SPECS,REUSE,case
from execution import Worker,run_episode

ROOT=V.parents[1]
sys.path.insert(0,str(V/'scripts'))
from episode_journal import atomic_json


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze():
    old=json.loads((PREVIOUS/'PROTOCOL.json').read_text())
    for name,digest in old['sources'].items():assert sha(ROOT/name)==digest,name
    files=[S/f'scripts/{name}.py' for name in ['run','execution','design','worker']]
    files += [ROOT/name for name in old['sources'] if not name.endswith('selective_execution/scripts/run.py')]
    # Old scoring/controller/worker are deliberately shared and immutable.
    files += [PREVIOUS/'scripts/worker.py',PREVIOUS/'scripts/protocol.py',PREVIOUS/'scripts/control.py']
    p=dict(
        scope='same positive English prefix, with/without constraint suffix × continue/timer150/Qwen stop',
        modes=list(MODES),development_initials=[0],evaluation_initials=[8,9,10],
        max_steps=300,settle_steps=25,check_interval=10,confirm=2,vote_ttl=20,
        score='unchanged selective_execution score: final10 In/no-gripper-contact and cumulative protected displacement<=30mm',
        monitor='unchanged base two-frame monitor of selected positive skill; same images and prompt across prompt families',
        planning='same Qwen target/protected selector; deterministic English skill template',
        development_reuse=REUSE,
        gate='Expand only if constraint suffix has more selective successes across the12 matched development cells, or achieves at least one new reverse-target success. Otherwise stop prompt branch.',
        no_training=True,no_hardware=True,
        token_preflight_sha256=sha(S/'TOKEN_PREFLIGHT.json'),
        previous_protocol_sha256=sha(PREVIOUS/'PROTOCOL.json'),
        resources_sha256=sha(PREVIOUS/'RESOURCES.json'),
        sources={str(f.relative_to(ROOT)):sha(f) for f in sorted(set(files))})
    path=S/'PROTOCOL.json'
    if path.exists():assert json.loads(path.read_text())==p,'Execution protocol changed'
    else:
        atomic_json(path,p)
        for name in p['sources']:
            dest=S/'archive'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
    return sha(path)


def old_result(c,mode):
    p=PREVIOUS/f'episodes/pilot/{c["key"]}/{REUSE[mode]}/result.json'
    r=json.loads(p.read_text());assert r['status']=='complete'
    assert r['case']==json.loads(json.dumps(c))
    return p,r


def gate():
    comparisons=[]
    for task in [0,1]:
        for target in [0,1]:
            c=case(task,target,0)
            for stop in ['continue','timer','visual']:
                oldp,a=old_result(c,f'positive_{stop}')
                newp=S/f'episodes/pilot/{c["key"]}/constrained_{stop}/result.json'
                b=json.loads(newp.read_text());assert b['status']=='complete'
                comparisons.append(dict(case=c['key'],target_index=target,stop=stop,
                    positive=a['selective_success'],constrained=b['selective_success'],
                    old_source=str(oldp.relative_to(ROOT)),old_sha256=sha(oldp),
                    new_source=str(newp.relative_to(ROOT)),new_sha256=sha(newp)))
    gains=sum(not x['positive'] and x['constrained'] for x in comparisons)
    losses=sum(x['positive'] and not x['constrained'] for x in comparisons)
    reverse_gains=sum(x['target_index']==1 and not x['positive'] and x['constrained'] for x in comparisons)
    result=dict(expand=(gains>losses or reverse_gains>0),gains=gains,losses=losses,reverse_gains=reverse_gains,
        positive_success=sum(x['positive'] for x in comparisons),constrained_success=sum(x['constrained'] for x in comparisons),
        comparisons=comparisons,independent_test=False)
    atomic_json(S/'PILOT_GATE.json',result);return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['pilot','eval'],default='pilot');a=ap.parse_args()
    protocol_sha=freeze()
    if a.stage=='eval':assert json.loads((S/'PILOT_GATE.json').read_text())['expand'],'Development gate did not pass'
    worker=Worker()
    try:
        from openpi.training import config
        from openpi.policies import policy_config
        atomic_json(S/'STATUS.json',dict(state='loading_policy',stage=a.stage))
        policy=policy_config.create_trained_policy(config.get_config('pi05_libero'),V/'checkpoints/pi05_libero',sample_kwargs={'num_steps':10})
        policy.infer({'observation/image':np.zeros((224,224,3),np.uint8),'observation/wrist_image':np.zeros((224,224,3),np.uint8),'observation/state':np.zeros(8,np.float32),'prompt':'put the object in the basket'},noise=np.zeros((10,32),np.float32))
        initials=[0] if a.stage=='pilot' else [8,9,10]
        if a.stage=='eval':
            lock=dict(protocol_sha256=protocol_sha,pilot_gate_sha256=sha(S/'PILOT_GATE.json'),initials=initials)
            lockpath=S/'EVALUATION_LOCK.json'
            if lockpath.exists():assert json.loads(lockpath.read_text())==lock
            else:atomic_json(lockpath,lock)
        rows=[]
        for task in [0,1]:
            for initial in initials:
                for target in [0,1]:
                    c=case(task,target,initial)
                    if a.stage=='pilot':
                        plan=S/f'plans/pilot/{c["key"]}.json';plan.parent.mkdir(parents=True,exist_ok=True)
                        source=PREVIOUS/f'plans/pilot/{c["key"]}.json'
                        if not plan.exists():shutil.copyfile(source,plan)
                    offset=(task*6+initial*2+target)%len(MODES)
                    for mode in MODES[offset:]+MODES[:offset]:
                        if a.stage=='pilot' and mode in REUSE:
                            path,r=old_result(c,mode);reused=True
                        else:
                            r=run_episode(c,mode,a.stage,policy,worker,{},protocol_sha);path=S/f'episodes/{a.stage}/{c["key"]}/{mode}/result.json';reused=False
                        row=dict(case=c['key'],mode=mode,success=r['selective_success'],protected_ok=r['protected_untouched'],target_stable=r['target_final_stable'],step=r['decision_step'],reused=reused,source=str(path.relative_to(ROOT)),source_sha256=sha(path))
                        rows.append(row);atomic_json(S/f'INDEX_{a.stage}.json',rows)
                        atomic_json(S/'STATUS.json',dict(state='running',stage=a.stage,completed=len(rows),expected=24 if a.stage=='pilot' else 72,latest=row))
                        print(json.dumps(row),flush=True)
        if a.stage=='pilot':print(json.dumps(gate()),flush=True)
        atomic_json(S/'STATUS.json',dict(state='complete',stage=a.stage,completed=len(rows),new_episodes=sum(not r['reused'] for r in rows),reused_episodes=sum(r['reused'] for r in rows)))
    finally:worker.close()


if __name__=='__main__':main()
