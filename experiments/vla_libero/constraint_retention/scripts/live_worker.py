"""Online worker uses exactly the same three-image prompt/parser as offline audit."""
import hashlib,json,runpy
from pathlib import Path
import sys

V=Path(__file__).resolve().parents[2];ROOT=V.parents[1]
sys.path[:0]=[str(V/'scripts'),str(ROOT/'perception/qwen_scene')]
from qwen_planner import Planner
from view_checker import prompt,parse
select=runpy.run_path(str(V/'selective_execution/scripts/worker.py'))['select']


def main():
    model=Planner();print(json.dumps(dict(ready=True,load_seconds=model.engine.load_seconds)),flush=True)
    for line in sys.stdin:
        try:
            r=json.loads(line)
            if r.get('stop'):break
            op=r.pop('op')
            if op=='select':reply=select(model,**r)
            elif op=='translate':reply=model.translate(**r)
            elif op=='check':
                assert set(r)=={'image','subgoal'} and len(r['image'])==3
                reply=model.generate(r['image'],prompt(r['subgoal']),max_new_tokens=128)
                try:reply.update(parse(reply['raw']),ok=True)
                except (ValueError,TypeError) as exc:reply.update(ok=False,status='invalid',error=str(exc))
                reply['live_checker_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            else:raise ValueError(op)
        except Exception as exc:reply=dict(ok=False,status='invalid',error=repr(exc))
        print(json.dumps(reply,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
