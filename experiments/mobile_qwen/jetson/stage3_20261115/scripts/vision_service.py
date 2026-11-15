"""Localhost-only serial NF4 affordance service. Transport uses an SSH tunnel."""
from pathlib import Path
import sys,json,time,base64,io,hashlib,os,subprocess
from http.server import HTTPServer,BaseHTTPRequestHandler
from PIL import Image
S=Path(__file__).resolve().parents[1];S2=S.parent/'stage2_20261114';sys.path.insert(0,str(S2/'scripts'))
from runtime import Runtime,memory
OUT=S/'service_run';OUT.mkdir(exist_ok=False)
tegra_log=(OUT/'tegrastats.log').open('w');tegra=subprocess.Popen(['/usr/bin/tegrastats','--interval','100'],stdout=tegra_log,stderr=subprocess.STDOUT)
runtime=Runtime(S2/'nf4_2b',precision='nf4',adapter=S2/'adapter_2b',resolution=512)
(OUT/'load.json').write_text(json.dumps({'seconds':runtime.load_seconds,'after_load':runtime.after_load},indent=2))
deadline=time.monotonic()+600
while not (OUT/'swap_ready').exists():
 if time.monotonic()>deadline:raise TimeoutError('No swap gate')
 time.sleep(.5)
assert len(Path('/proc/swaps').read_text().splitlines())==1
runtime.samples.clear();runtime.samples.append({'time_unix':time.time(),**memory()})
prompt=json.loads((S/'input_manifest.json').read_text())['prompts']['handle'];sequence=0
class Handler(BaseHTTPRequestHandler):
 def do_GET(self):
  self.send_response(200);self.end_headers();self.wfile.write(b'{"ready":true}')
 def do_POST(self):
  global sequence
  try:
   n=int(self.headers.get('Content-Length','0'))
   if not 0<n<=2*1024*1024:raise ValueError('Invalid request length')
   q=json.loads(self.rfile.read(n));assert q.get('mode')=='lora','Only the validated LoRA endpoint is enabled'
   data=base64.b64decode(q['image_base64'],validate=True);start=time.perf_counter()
   with Image.open(io.BytesIO(data)) as image:r=runtime.generate(image.convert('RGB'),prompt,40)
   raw=r['raw'].strip()
   if raw.startswith('```'):raw=raw.split('\n',1)[1].rsplit('```',1)[0].strip()
   value=json.loads(raw);value=value[0] if isinstance(value,list) else value;point=value['point_2d'];assert len(point)==2 and all(isinstance(x,(int,float)) and 0<=x<1000 for x in point)
   r.update(ok=True,point_2d=point,mode='lora',server_wall_seconds=time.perf_counter()-start,inference={'adapter_enabled':True,'model':'Qwen3-VL-2B NF4','device':'Jetson Orin Nano 15W','prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'input_png_sha256':hashlib.sha256(data).hexdigest()})
  except Exception as exc:r={'ok':False,'error':repr(exc)}
  r['phase']=q.get('phase','case') if 'q' in locals() else 'invalid_request'
  sequence+=1
  with (OUT/'requests.jsonl').open('a') as f:f.write(json.dumps({'sequence':sequence,**r})+'\n');f.flush();os.fsync(f.fileno())
  payload=json.dumps(r).encode();self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)
 def log_message(self,*args):pass
print('SERVICE_READY',flush=True)
try:HTTPServer(('127.0.0.1',18888),Handler).serve_forever()
except KeyboardInterrupt:pass
finally:
 tegra.terminate();tegra.wait();tegra_log.close()
 report=runtime.report();samples=report['stable_samples'];report['global_swap_io_pages']={k:max(x[k] for x in samples)-min(x[k] for x in samples) for k in ['pswpin','pswpout']};report['valid_without_swap']=report['stable_max_swap_bytes']==0 and not any(report['global_swap_io_pages'].values());(OUT/'resources.json').write_text(json.dumps(report,indent=2))
