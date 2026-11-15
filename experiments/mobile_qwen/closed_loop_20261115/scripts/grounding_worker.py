"""PC vision worker, one 4B BF16 model with an optional, unmerged LoRA.
Input is only image + the fixed affordance query. No simulator goal coordinates.
"""
from pathlib import Path
import contextlib,json,sys
Q=Path(__file__).resolve().parents[2];ROOT=Q.parents[1]
sys.path[:0]=[str(ROOT/'perception/qwen_scene'),str(Q/'affordance')]
with contextlib.redirect_stdout(sys.stderr):
    from engine import SceneEngine
    from eval_qwen import QUERIES_V2,parse_json
    from peft import PeftModel
    e=SceneEngine();e.model=PeftModel.from_pretrained(e.model,str(Q/'affordance/lora_v1/best_adapter')).eval()
print(json.dumps({'ready':True,'model':'Qwen3-VL-4B-Instruct BF16','adapter':'affordance/lora_v1/best_adapter'}),flush=True)
from PIL import Image
for line in sys.stdin:
    try:
        q=json.loads(line)
        if q.get('stop'):break
        if q['mode'] not in ['base','lora']:raise ValueError('Unsupported grounding mode')
        with Image.open(q['image']) as im:
            context=e.model.disable_adapter() if q['mode']=='base' else contextlib.nullcontext()
            with context:
                states=[m.disable_adapters for m in e.model.modules() if isinstance(getattr(m,'disable_adapters',None),bool)]
                assert states and all(v==(q['mode']=='base') for v in states)
                r=e.generate(im.convert('RGB'),QUERIES_V2['handle'],max_new_tokens=40)
                r['inference']['adapter_enabled']=q['mode']=='lora'
        value=parse_json(r['raw'])
        if isinstance(value,list):value=value[0]
        point=value['point_2d']
        assert len(point)==2 and all(type(x) in (int,float) and 0<=x<1000 for x in point)
        r.update(ok=True,point_2d=point,mode=q['mode'])
    except Exception as exc:r={'ok':False,'error':repr(exc)}
    print(json.dumps(r),flush=True)
