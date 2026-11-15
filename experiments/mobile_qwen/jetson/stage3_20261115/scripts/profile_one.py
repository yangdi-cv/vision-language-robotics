"""One diagnostic profile, separate from latency numbers; no shape/stack tracing."""
from pathlib import Path
import sys,json,time,gzip,collections,os
from PIL import Image
S=Path(__file__).resolve().parents[1];S2=S.parent/'stage2_20261114';sys.path.insert(0,str(S2/'scripts'))
from runtime import Runtime,memory
import torch
OUT=S/os.environ.get('ALI_PROFILE_TAG','profile_one');OUT.mkdir(exist_ok=False)
runtime=None
try:
 runtime=Runtime(S2/'nf4_2b',precision='nf4',adapter=S2/'adapter_2b',resolution=512)
 (OUT/'load.json').write_text(json.dumps({'seconds':runtime.load_seconds},indent=2));deadline=time.monotonic()+600
 while not (OUT/'swap_ready').exists():
  if time.monotonic()>deadline:raise TimeoutError('Swap gate')
  time.sleep(.5)
 assert len(Path('/proc/swaps').read_text().splitlines())==1
 runtime.samples.clear();runtime.samples.append({'time_unix':time.time(),**memory()})
 m=json.loads((S/'input_manifest.json').read_text());image=Image.open(S/'validation'/m['images'][0]['file']).convert('RGB');prompt=m['prompts']['handle'];runtime.generate(image,prompt,40)
 activities=[torch.profiler.ProfilerActivity.CPU]
 if torch.profiler.ProfilerActivity.CUDA in torch.profiler.supported_activities():activities.append(torch.profiler.ProfilerActivity.CUDA)
 with torch.profiler.profile(activities=activities,record_shapes=False,profile_memory=False,with_stack=False) as profile:
  response=runtime.generate(image,prompt,40)
 (OUT/'cpu_table.txt').write_text(profile.key_averages().table(sort_by='self_cpu_time_total',row_limit=25))
 trace=OUT/'trace.json';profile.export_chrome_trace(str(trace));events=json.loads(trace.read_text()).get('traceEvents',[]);kernels=collections.defaultdict(lambda:{'count':0,'summed_microseconds':0.})
 for e in events:
  if e.get('cat')=='kernel' and 'dur' in e:
   k=kernels[e['name']];k['count']+=1;k['summed_microseconds']+=e['dur']
 with trace.open('rb') as src,gzip.open(OUT/'trace.json.gz','wb') as dst:
  import shutil;shutil.copyfileobj(src,dst)
 trace.unlink();report={'diagnostic_only':True,'do_not_mix_with_latency_benchmark':True,'activities':[str(x) for x in activities],'raw':response['raw'],'output_tokens':response['metadata']['output_tokens'],'profiled_model_seconds':response['seconds'],'kernel_events':sum(v['count'] for v in kernels.values()),'summed_kernel_microseconds':sum(v['summed_microseconds'] for v in kernels.values()),'kernels':sorted([{'name':k,**v} for k,v in kernels.items()],key=lambda x:x['summed_microseconds'],reverse=True),'reference':'https://docs.pytorch.org/docs/2.8/profiler.html','resources':runtime.report()}
 (OUT/'SUMMARY.json').write_text(json.dumps(report,indent=2));print('PROFILE_COMPLETE',report['kernel_events'],flush=True)
except Exception as e:
 (OUT/'error.json').write_text(json.dumps({'error':repr(e)},indent=2));raise
finally:
 if runtime and runtime.thread.is_alive():runtime.report()
