"""Separate VLA observation page. Reuses :8080 camera/models; no motion API."""
import asyncio,json,sys,time,subprocess,signal,os
from pathlib import Path
from aiohttp import web,ClientSession
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
RUN=ROOT/'records/smolvla_live'/str(time.time_ns());RUN.mkdir(parents=True)
async def index(r):return web.FileResponse(HERE/'dist/index.html')
async def status(r):
 try:
  async with r.app['client'].get('http://127.0.0.1:8080/api/state') as resp:s=await resp.json()
 except Exception:s={'camera':{'valid':False},'qwen':{'text':'原工作台未连接'},'vision':{},'robot':{'active':False}}
 try:v=json.loads((RUN/'latest.json').read_text())
 except (OSError,ValueError):v={'state':'loading'}
 if 'frame_received_monotonic_ns' in v:v['frame_age_s']=(time.monotonic_ns()-v['frame_received_monotonic_ns'])/1e9
 s['smolvla']=v;return web.json_response(s)
async def task(r):
 if r.headers.get('Origin') not in ('http://127.0.0.1:8081','http://localhost:8081'):raise web.HTTPForbidden()
 d=await r.json();text=d.get('task','').strip()
 if not text or len(text)>500:raise web.HTTPBadRequest()
 tmp=RUN/'task.tmp';tmp.write_text(json.dumps({'task':text}));tmp.replace(RUN/'task.json')
 return web.json_response({'ok':True})
async def proxy(r):
 path=r.path_qs
 async with r.app['client'].get('http://127.0.0.1:8080'+path) as upstream:
  response=web.StreamResponse(status=upstream.status,headers={'Content-Type':upstream.headers.get('Content-Type','application/octet-stream'),'Cache-Control':'no-store'})
  await response.prepare(r)
  try:
   async for data in upstream.content.iter_any():await response.write(data)
  except (ConnectionResetError,asyncio.CancelledError):pass
  return response
async def lifecycle(app):
 app['client']=ClientSession(timeout=__import__('aiohttp').ClientTimeout(total=None))
 log=(RUN/'worker.log').open('w');app['log']=log
 app['worker']=subprocess.Popen([str(ROOT/'.venvs/smolvla/bin/python'),'-u',str(HERE/'worker.py'),str(RUN)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 print('VLA workbench http://127.0.0.1:8081/ | '+str(RUN),flush=True)
 try:yield
 finally:
  p=app['worker']
  if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
  await asyncio.to_thread(p.wait);log.close();await app['client'].close()
def main():
 a=web.Application(client_max_size=2048);a.cleanup_ctx.append(lifecycle)
 a.router.add_get('/',index);a.router.add_get('/api/state',status);a.router.add_post('/api/task',task)
 a.router.add_get('/stream/{kind}',proxy);a.router.add_get('/legend.png',proxy)
 a.router.add_static('/static',HERE/'dist');web.run_app(a,host='127.0.0.1',port=8081)
if __name__=='__main__':main()
