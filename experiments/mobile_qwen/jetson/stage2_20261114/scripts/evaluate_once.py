"""Original eval_qwen.main once per configuration; capture output/inputs/resources."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time
import subprocess
from runtime import Runtime,Q
p=argparse.ArgumentParser()
p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
p.add_argument('--model',required=True);p.add_argument('--revision');p.add_argument('--precision',choices=['bf16','nf4'],required=True)
p.add_argument('--adapter');p.add_argument('--resolution',type=int,choices=[512,384,256],default=512)
a=p.parse_args()
if a.resolution != 512 and os.environ.get('ALLOW_RESOLUTION_SWEEP') != '1':
    a.out.with_suffix('.deferred.json').write_text(json.dumps({'deferred': True, 'resolution': a.resolution, 'reason': 'Phase 2 fixed-512 single-factor comparison; remeasure accuracy with latency sweep in next phase'},indent=2))
    print('Deferred resolution sweep to next phase:',a.resolution,flush=True)
    sys.exit(0)
assert not a.out.exists(),a.out
meta_path=a.out.with_suffix('.metadata.jsonl')
assert not meta_path.exists(),meta_path
sys.path[:0]=[str(Q/'affordance'),str(Q/'scripts')]
import eval_qwen as original
from PIL import Image
telemetry=None
telemetry_file=None
if Path("/usr/bin/tegrastats").exists():
    telemetry_file=a.out.with_suffix(".tegrastats.log").open("w")
    telemetry=subprocess.Popen(["/usr/bin/tegrastats","--interval","100"],stdout=telemetry_file,stderr=subprocess.STDOUT)
try:
    runtime=Runtime(a.model,a.revision,a.precision,a.adapter,a.resolution)
except BaseException:
    if telemetry: telemetry.terminate();telemetry.wait();telemetry_file.close()
    raise
if os.environ.get('JETSON_SWAP_GATE') == '1':
    a.out.with_suffix('.load.json').write_text(json.dumps({'load_seconds':runtime.load_seconds,'before_load':runtime.before_load,'after_load':runtime.after_load,'awaiting_swapoff':True},indent=2))
    gate=a.out.with_suffix('.swap_ready')
    deadline=time.monotonic()+600
    while not gate.exists():
        if time.monotonic()>deadline: raise TimeoutError('Waiting for verified swapoff gate')
        time.sleep(.5)
    assert len(Path('/proc/swaps').read_text().splitlines()) == 1, 'Swap still enabled'
    runtime.samples.clear()
    from runtime import memory
    runtime.samples.append({'time_unix':time.time(),**memory()})
started=time.time()
# Exposes the existing generate interface, leaving applicability, prompts, parsing,
# scoring and the single pass through 350 frames entirely in eval_qwen.py.
class Proxy:
    def generate(self,image,prompt,max_new_tokens=40):
        with Image.open(image) as im: r=runtime.generate(im.convert('RGB'),prompt,max_new_tokens)
        record={'id':Path(image).stem,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),**r}
        with meta_path.open('a') as f:
            f.write(json.dumps(record)+'\n');f.flush();os.fsync(f.fileno())
        return r
original.MissionPlanner=Proxy
sys.argv=['eval_qwen.py',str(a.data),str(a.out),'--inits','0-12','--replay-every','20','--prompts','v2']
try:
    original.main()
finally:
    if telemetry:
        telemetry.terminate();telemetry.wait();telemetry_file.close()
    report=runtime.report();report['evaluation_started_unix']=started;report['evaluation_seconds']=time.time()-started
    ps=report['stable_samples']
    report['global_stable_swap_io_pages']={k:max(x[k] for x in ps)-min(x[k] for x in ps) for k in ['pswpin','pswpout'] if ps and all(k in x for x in ps)}
    report['accuracy_usable_without_swap']=report['stable_max_swap_bytes']==0 and not any(report['global_stable_swap_io_pages'].values())
    report['swap_devices_at_finish']=Path('/proc/swaps').read_text()
    report['eval_source_sha256']=hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest()
    a.out.with_suffix('.resources.json').write_text(json.dumps(report,indent=2))
