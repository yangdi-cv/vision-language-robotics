"""Optional 4B load/resource qualification, using a validation image only.
Not a 350-frame accuracy evaluation and not a latency benchmark.
"""
import json,time
from pathlib import Path
from runtime import Runtime,memory
from PIL import Image
S=Path(__file__).resolve().parents[1]
r=Runtime(S/'nf4_4b',precision='nf4',adapter=S/'adapter_4b')
p=S/'probe_4b'
p.with_suffix('.load.json').write_text(json.dumps({'load_seconds':r.load_seconds,'before_load':r.before_load,'after_load':r.after_load},indent=2))
end=time.monotonic()+600
while not p.with_suffix('.swap_ready').exists():
    if time.monotonic()>end:raise TimeoutError('Swap gate')
    time.sleep(.5)
assert len(Path('/proc/swaps').read_text().splitlines())==1
r.samples.clear();r.samples.append({'time_unix':time.time(),**memory()})
request=json.loads((S/'probe_4b_request.json').read_text())
with Image.open(S/'probe_val13.png') as im:result=r.generate(im.convert('RGB'),request['prompt'],40)
resources=r.report()
(S/'probe_4b_result.json').write_text(json.dumps({'qualification_only':True,'validation_id':request['id'],'result':result,'resources':resources},indent=2))
