"""One existing Qwen4B engine; no policy, labels, state or training interface."""
import json
from pathlib import Path
import sys

V=Path(__file__).resolve().parents[2];ROOT=V.parents[1]
sys.path[:0]=[str(V/'scripts'),str(ROOT/'perception/qwen_scene')]
from qwen_planner import Planner
from view_checker import prompt,parse


def main():
    model=Planner()
    print(json.dumps(dict(ready=True,load_seconds=model.engine.load_seconds)),flush=True)
    for line in sys.stdin:
        try:
            r=json.loads(line)
            if r.get('stop'):break
            assert set(r)=={'scheme','image','subgoal'}
            if r['scheme']=='base_pair':
                result=model.check(r['image'],r['subgoal'])
                result['status']=('done' if result['done'] else 'not_done') if result.get('ok') else 'invalid'
            else:
                result=model.generate(r['image'],prompt(r['subgoal']),max_new_tokens=128)
                try:result.update(parse(result['raw']),ok=True)
                except (ValueError,TypeError) as exc:result.update(ok=False,status='invalid',error=str(exc))
        except Exception as exc:result=dict(ok=False,status='invalid',error=repr(exc))
        print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
