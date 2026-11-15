"""Reuse the existing Qwen engine/checker; add closed-vocabulary object selection."""
import hashlib
import json
from pathlib import Path
import sys

V = Path(__file__).resolve().parents[2]
ROOT = V.parents[1]
sys.path[:0] = [str(V/'scripts'), str(ROOT/'perception/qwen_scene')]
from qwen_planner import Planner


def select(planner, image, instruction, objects):
    prompt = f"""Inspect the image and the instruction. Select exactly one object to move into the basket, and the one explicitly required to remain untouched.
Known object IDs and English/Chinese names: {json.dumps(objects, ensure_ascii=False)}
Instruction: {instruction}
Return only JSON: {{"target":"one known ID", "protected":"the other known ID"}}. Do not output coordinates or add another action."""
    reply = planner.generate(image, prompt, max_new_tokens=96)
    try:
        raw = reply['raw'].strip()
        if raw.startswith('```'):
            raw = raw.split('\n',1)[1].rsplit('```',1)[0].strip()
        selection = json.loads(raw)
        ids = {o[0] for o in objects}
        if set(selection) != {'target','protected'} or set(selection.values()) != ids:
            raise ValueError('Expected distinct known object IDs')
        reply.update(selection, ok=True)
    except (ValueError, TypeError) as exc:
        reply.update(ok=False, error=str(exc))
    reply['worker_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return reply


def main():
    planner = Planner()
    print(json.dumps({'ready':True, 'load_seconds':planner.engine.load_seconds}),flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get('stop'):
                break
            op = request.pop('op')
            reply = select(planner, **request) if op == 'select' else getattr(planner, op)(**request)
        except Exception as exc:
            reply = dict(ok=False,error=repr(exc))
        print(json.dumps(reply,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
