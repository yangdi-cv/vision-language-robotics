"""Frozen offline paired audit; no simulator truth is sent to the worker."""
import hashlib,json
from pathlib import Path
import subprocess
import sys
import time
from design import S,V
from view_checker import SCHEMES,input_images,prompt

ROOT=V.parents[1];sys.path.insert(0,str(V/'scripts'));sys.path.insert(0,str(ROOT/'perception/qwen_scene'))
from episode_journal import atomic_json,append_jsonl
from worker_client import read_json_line,close_json_worker


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze():
    files=[S/f'scripts/{n}.py' for n in ['view_run','view_worker','view_checker','prepare_views']]+[V/'scripts/qwen_planner.py',ROOT/'perception/qwen_scene/engine.py',ROOT/'perception/qwen_scene/worker_client.py']
    value=dict(scope='Offline completion audit only, not a closed-loop success experiment',schemes=list(SCHEMES),
        development='old initial0 positive continuation streams',test='fresh initial8,9,10 positive continuation streams',
        primary='duplicate_control vs wrist: SAME three-image prompt and512px dimensions; only third image differs',
        reference='base_pair uses original two-image prompt/boolean schema, reported separately; not a pure wrist-only ablation',
        prompt_template=prompt('{subgoal}'),generation={'do_sample':False,'new_schemes_max_tokens':128},
        label='In basket and no gripper contact, evaluated on same restored state as both current camera renders',
        metrics='done confirmation precision/recall/FPR; unknown and invalid separate; strict classification counts unknown/invalid incorrect',
        no_training=True,no_hardware=True,no_threshold_tuning=True,
        no_truth_input=True,sources={str(p.relative_to(ROOT)):sha(p) for p in files})
    p=S/'VIEW_PROTOCOL.json'
    if p.exists():assert json.loads(p.read_text())==value,'View protocol changed'
    else:
        atomic_json(p,value)
        for name in value['sources']:
            out=S/'archive_view'/name;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes((ROOT/name).read_bytes())
    return sha(p)


def main():
    version=freeze();folder=S/'view_audit';(folder/'logs').mkdir(exist_ok=True)
    rows=[]
    for split in ['development','test']:rows+=json.loads((folder/f'{split}.json').read_text())
    assert {r['case'].split('_init')[-1] for r in rows if r['split']=='development'}=={'00'}
    assert {r['case'].split('_init')[-1] for r in rows if r['split']=='test'}=={'08','09','10'}
    lock={name:sha(folder/f'{name}.json') for name in ['development','test']}
    path=folder/'DATA_LOCK.json'
    if path.exists():assert json.loads(path.read_text())==lock
    else:atomic_json(path,lock)
    output=folder/'predictions.jsonl';completed={}
    if output.exists():
        for line in output.read_text().splitlines():
            r=json.loads(line);assert r['protocol_sha256']==version;completed[(r['id'],r['scheme'])]=r
    log=open(folder/'logs/worker.log','w')
    proc=subprocess.Popen([str(ROOT/'.venvs/qwen-scene/bin/python'),str(S/'scripts/view_worker.py')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,start_new_session=True)
    ready=read_json_line(proc,timeout=240);atomic_json(folder/'READY.json',ready)
    try:
        count=0
        for i,row in enumerate(rows):
            shift=i%len(SCHEMES);order=SCHEMES[shift:]+SCHEMES[:shift]
            for scheme in order:
                if (row['id'],scheme) in completed:count+=1;continue
                images=[str(ROOT/name) for name in input_images(row,scheme)]
                payload=dict(scheme=scheme,image=images,subgoal=row['subgoal'])
                append_jsonl(folder/'requests.jsonl',payload)
                t0=time.monotonic();proc.stdin.write(json.dumps(payload)+'\n');proc.stdin.flush();reply=read_json_line(proc,timeout=120)
                result=dict(id=row['id'],split=row['split'],case=row['case'],step=row['step'],scheme=scheme,
                    goal_done=row['goal_done'],protocol_sha256=version,seconds=time.monotonic()-t0,
                    images=input_images(row,scheme),reply=reply)
                append_jsonl(output,result);count+=1
                atomic_json(folder/'STATUS.json',dict(state='running',completed=count,expected=len(rows)*3,latest=dict(id=row['id'],scheme=scheme,status=reply.get('status'))))
        atomic_json(folder/'STATUS.json',dict(state='complete',completed=count,queries=len(rows),per_split={split:sum(r['split']==split for r in rows) for split in ['development','test']}))
    finally:close_json_worker(proc);log.close()


if __name__=='__main__':main()
