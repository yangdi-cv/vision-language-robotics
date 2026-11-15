"""Development-only camera ablation retaining the original check question."""
import hashlib,json,subprocess,sys,time
from pathlib import Path
from design import S,V
from view_checker import input_images
ROOT=V.parents[1]
sys.path[:0]=[str(V/'scripts'),str(ROOT/'perception/qwen_scene')]
from episode_journal import atomic_json,append_jsonl
from worker_client import read_json_line,close_json_worker

def prompt(goal):
    return f'''Image 1 was taken when the robot started this subgoal; image 2 is the scene now.
Image 3 is an auxiliary current view of the same scene; it may repeat image 2.
Subgoal: "{goal}".
Look at the destination named in the subgoal. Is there a new object at the destination in image 2 or image 3 that was not there in image 1, and is it the target object? For a state change (open, close, turn on), compare that object between the images.
Return only JSON: {{"change": "short description", "done": true or false}}'''

def worker():
    from qwen_planner import Planner
    model=Planner();print(json.dumps({'ready':True}),flush=True)
    for line in sys.stdin:
        r=json.loads(line)
        if r.get('stop'):break
        assert set(r)=={'image','subgoal'}
        out=model.generate(r['image'],prompt(r['subgoal']),max_new_tokens=96)
        try:
            raw=out['raw'].strip()
            if raw.startswith('```'):raw=raw.split('\n',1)[1].rsplit('```',1)[0].strip()
            val=json.loads(raw);assert isinstance(val.get('done'),bool)
            out.update(ok=True,done=val['done'],reason=val.get('change',''))
        except Exception as exc:out.update(ok=False,error=str(exc))
        print(json.dumps(out),flush=True)

def main():
    rows=json.loads((S/'view_audit/development.json').read_text())
    assert len(rows)==50
    folder=S/'minimal_view';folder.mkdir(exist_ok=True)
    protocol={'scope':'development only; no new independent test','rows':50,'calls':100,'prompt':prompt('{subgoal}'),'max_new_tokens':96,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'data_sha256':hashlib.sha256((S/'view_audit/development.json').read_bytes()).hexdigest()}
    atomic_json(folder/'PROTOCOL.json',protocol)
    log=open(folder/'worker.log','w');p=subprocess.Popen([str(ROOT/'.venvs/qwen-scene/bin/python'),__file__,'--worker'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True)
    atomic_json(folder/'READY.json',read_json_line(p,timeout=240));output=folder/'predictions.jsonl'
    assert not output.exists(),'Do not overwrite prior run'
    try:
        for i,row in enumerate(rows):
            order=['duplicate_control','wrist'] if i%2==0 else ['wrist','duplicate_control']
            for scheme in order:
                payload={'image':[str(ROOT/n) for n in input_images(row,scheme)],'subgoal':row['subgoal']}
                t=time.monotonic();p.stdin.write(json.dumps(payload)+'\n');p.stdin.flush();reply=read_json_line(p,timeout=120)
                append_jsonl(output,dict(id=row['id'],case=row['case'],scheme=scheme,goal_done=row['goal_done'],seconds=time.monotonic()-t,reply=reply))
            atomic_json(folder/'STATUS.json',dict(state='running',completed_queries=i+1))
        atomic_json(folder/'STATUS.json',dict(state='complete',queries=50,calls=100))
    finally:close_json_worker(p);log.close()

if __name__=='__main__':
    worker() if '--worker' in sys.argv else main()
