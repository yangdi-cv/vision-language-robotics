"""Small validation-only 15W latency probe; not a 350-frame accuracy evaluation."""
from pathlib import Path
import sys,json,time,os,subprocess
import numpy as np
from PIL import Image
S=Path(__file__).resolve().parents[1];STAGE2=S.parent/'stage2_20261114'
sys.path.insert(0,str(STAGE2/'scripts'))
from runtime import Runtime,memory
manifest=json.loads((S/'input_manifest.json').read_text());out=S/'results';out.mkdir(exist_ok=False)
log=(out/'tegrastats.log').open('w');tegra=subprocess.Popen(['/usr/bin/tegrastats','--interval','100'],stdout=log,stderr=subprocess.STDOUT)
runtime=None
try:
 runtime=Runtime(STAGE2/'nf4_2b',precision='nf4',adapter=STAGE2/'adapter_2b',resolution=512)
 (out/'load.json').write_text(json.dumps({'seconds':runtime.load_seconds,'before':runtime.before_load,'after':runtime.after_load},indent=2))
 gate=out/'swap_ready';deadline=time.monotonic()+600
 while not gate.exists():
  if time.monotonic()>deadline:raise TimeoutError('No verified swapoff gate')
  time.sleep(.5)
 assert len(Path('/proc/swaps').read_text().splitlines())==1
 runtime.samples.clear();runtime.samples.append({'time_unix':time.time(),**memory()})
 image=Image.open(S/'validation'/manifest['images'][0]['file']).convert('RGB')
 for size in [512,384,256]:
  runtime.resolution=size;runtime.generate(image,manifest['prompts']['handle'],40)
 rows=[]
 for i,frame in enumerate(manifest['images']):
  sizes=[512,384,256];sizes=sizes[i%3:]+sizes[:i%3]
  for size in sizes:
   runtime.resolution=size
   for query,prompt in manifest['prompts'].items():
    start=time.perf_counter()
    with Image.open(S/'validation'/frame['file']) as im:r=runtime.generate(im.convert('RGB'),prompt,40)
    row={'id':frame['id'],'resolution':size,'query':query,'wall_seconds':time.perf_counter()-start,**r};rows.append(row)
    with (out/'records.jsonl').open('a') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
    print(size,frame['id'],query,row['wall_seconds'],flush=True)
 report=runtime.report();samples=report['stable_samples'];io={k:max(x[k] for x in samples)-min(x[k] for x in samples) for k in ['pswpin','pswpout']}
 report['global_swap_io_pages']=io;report['valid_without_swap']=not any(io.values()) and report['stable_max_swap_bytes']==0
 report['power']='15W mode0; jetson_clocks unchanged';report['accuracy_test']=False
 groups={}
 for size in [512,384,256]:
  for query in ['all',*manifest['prompts']]:
   part=[r for r in rows if r['resolution']==size and (query=='all' or r['query']==query)];v=[r['wall_seconds'] for r in part]
   groups[f'{size}_{query}']={'n':len(v),'wall_p50':float(np.percentile(v,50)),'wall_p95':float(np.percentile(v,95)),'mean_output_tokens':float(np.mean([r['metadata']['output_tokens'] for r in part])),'meets_3s_p95':bool(np.percentile(v,95)<=3),'exploratory_small_sample':True}
 report['latency']=groups;(out/'SUMMARY.json').write_text(json.dumps(report,indent=2));print('PROBE_COMPLETE',flush=True)
finally:
 tegra.terminate();tegra.wait();log.close()
 if runtime and runtime.thread.is_alive():(out/'resources_partial.json').write_text(json.dumps(runtime.report(),indent=2))
