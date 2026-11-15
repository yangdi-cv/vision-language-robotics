"""Model-free JSON-lines client of the Jetson localhost service through SSH."""
from pathlib import Path
import json,sys,base64,urllib.request,time
URL='http://127.0.0.1:18889'
with urllib.request.urlopen(URL,timeout=10) as response:assert json.load(response)['ready']
print(json.dumps({'ready':True,'model':'Jetson 2B NF4 LoRA; no PC model loaded'}),flush=True)
for line in sys.stdin:
 try:
  q=json.loads(line)
  if q.get('stop'):break
  t=time.perf_counter();payload={'mode':q['mode'],'image_base64':base64.b64encode(Path(q['image']).read_bytes()).decode()}
  request=urllib.request.Request(URL,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
  with urllib.request.urlopen(request,timeout=90) as response:r=json.load(response)
  r['client_wall_seconds']=time.perf_counter()-t
 except Exception as e:r={'ok':False,'error':repr(e)}
 print(json.dumps(r),flush=True)
