"""Local robot workbench. Inference never generates or authorizes robot commands."""
import asyncio
import io
import json
import os
import secrets
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, PngImagePlugin
from aiohttp import web, ClientSession

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'perception/qwen_scene'))
sys.path.insert(0,str(ROOT/'SO101/code/scripts'))
from camera_view import rotate_frame
from app import readable_result
from depth_display import render
from frame_transport import SharedFrame


def read_json(path):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return {}


class Workbench:
    def __init__(self):
        self.session=ROOT/'records/scene_understanding'/f'web_{time.time_ns()}'
        self.session.mkdir(parents=True)
        self.frame_store=SharedFrame(self.session,create=True)
        self.token=secrets.token_urlsafe(32)
        self.stop=threading.Event()
        self.rgb=None
        self.camera_rotation=int(os.environ.get("VLA_CAMERA_ROTATION", "270"))
        if self.camera_rotation not in (0,90,180,270):raise ValueError("Invalid camera rotation")
        self.camera_ns=0
        self.camera_error=None
        self.lock=threading.Lock()
        self.processes=[]
        self.logs=[]
        self.robot=None
        self.robot_owner=None
        self.heartbeat=0
        self.robot_lock=asyncio.Lock()
        self.visual_mtime=None
        self.visual=None
        self.render_cache={}
        self.legend={}

    def capture(self):
        cap=cv2.VideoCapture('/dev/v4l/by-id/usb-icSpring_icspring_camera-video-index0',cv2.CAP_V4L2)
        try:
            if not cap.isOpened():raise RuntimeError('相机无法打开；请关闭旧预览、Tkinter系统或采集程序。')
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,640);cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
            cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
            i=0
            while not self.stop.is_set():
                started=time.monotonic()
                ok,frame=cap.read()
                if not ok:raise RuntimeError('相机读取中断')
                if self.stop.is_set():break
                timestamp=time.monotonic_ns()
                rgb=cv2.cvtColor(rotate_frame(frame,self.camera_rotation),cv2.COLOR_BGR2RGB)
                image=Image.fromarray(rgb)
                self.frame_store.write(rgb,i,timestamp,self.camera_rotation)
                buffer=io.BytesIO();image.save(buffer,format='JPEG',quality=85)
                with self.lock:self.rgb=buffer.getvalue();self.camera_ns=timestamp
                i+=1
                self.stop.wait(max(0,.1-(time.monotonic()-started)))
        except Exception as exc:self.camera_error=str(exc)
        finally:cap.release()

    def spawn(self,script,log_name,*args):
        log=(self.session/log_name).open('w');self.logs.append(log)
        process=subprocess.Popen([sys.executable,str(script),*map(str,args)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        self.processes.append(process)
        return process

    def refresh_visual(self):
        path=self.session/'latest_vision.npz'
        try:
            modified=path.stat().st_mtime_ns
            if modified!=self.visual_mtime:
                with np.load(path,allow_pickle=False) as data:
                    self.visual={'depth':data['raw_depth'].copy(),'stats':json.loads(str(data['metadata']))}
                self.visual_mtime=modified;self.render_cache={}
        except (OSError,ValueError):pass

    def depth(self,mode):
        self.refresh_visual()
        if not self.visual:return None
        if mode not in self.render_cache:
            bgr,legend,_=render(self.visual['depth'],self.visual['stats']['objects'],mode)
            ok,jpeg=cv2.imencode('.jpg',bgr,[cv2.IMWRITE_JPEG_QUALITY,85])
            if not ok:raise RuntimeError('Depth JPEG failed')
            self.render_cache[mode]=jpeg.tobytes();self.legend[mode]=legend
        return self.render_cache[mode]

    def state(self):
        self.refresh_visual()
        qwen=read_json(self.session/'latest_result.json')
        qs=read_json(self.session/'worker_status.json')
        vs=read_json(self.session/'vision_status.json')
        rs=read_json(self.session/'robot_status.json')
        running=self.robot is not None and self.robot.poll() is None
        return {'camera':{'valid':bool(self.rgb) and time.monotonic_ns()-self.camera_ns<1_000_000_000,'error':self.camera_error},
                'qwen':{'state':qs.get('state','loading'),'error':qs.get('message'),'text':readable_result(qwen) if qwen else '模型加载后自动分析最新画面。',
                        'scene':qwen.get('scene'),'language':qwen.get('language','zh'),
                        'frame_id':qwen.get('frame_id'),'seconds':qwen.get('inference_seconds'),
                        'age_s':(time.monotonic_ns()-qwen['captured_monotonic_ns'])/1e9 if qwen else None},
                'vision':{'state':vs.get('state','loading'),'error':vs.get('message'),
                          'seconds':self.visual['stats']['inference_seconds'] if self.visual else None,
                          'age_s':(time.monotonic_ns()-self.visual['stats']['captured_ns'])/1e9 if self.visual else None},
                'robot':{'active':running,'state':rs.get('state','idle') if running else 'idle','exit_code':self.robot.poll() if self.robot else None},
                'legend':self.legend,'session':str(self.session)}

    async def start_robot(self,owner):
        async with self.robot_lock:
            if self.robot and self.robot.poll() is None:raise ValueError('主从操作已经运行。')
            state=self.state()
            if not state['camera']['valid'] or state['qwen']['state'] not in ('ready','inferencing'):raise ValueError('请等待相机与模型就绪。')
            log=(self.session/f'teleop_{time.time_ns()}.log').open('w');self.logs.append(log)
            (self.session/'robot_status.json').unlink(missing_ok=True)
            self.robot=subprocess.Popen([str(ROOT/'SO101/code/scripts/ubuntu/start-teleop'),'--require-calibration','--quiet','--status-file',str(self.session/'robot_status.json')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            self.robot_owner=owner;self.heartbeat=time.monotonic()

    async def stop_robot(self):
        async with self.robot_lock:
            if not self.robot or self.robot.poll() is not None:return
            status=read_json(self.session/'robot_status.json')
            try:
                if status.get('pid'):os.kill(status['pid'],signal.SIGINT)
                else:os.killpg(self.robot.pid,signal.SIGINT)
            except ProcessLookupError:pass
            try:await asyncio.to_thread(self.robot.wait,10)
            except subprocess.TimeoutExpired:raise ValueError('控制尚未退出，请托稳从臂并断开电源。')
            self.robot_owner=None

    async def watchdog(self):
        while not self.stop.is_set():
            if self.robot and self.robot.poll() is None and time.monotonic()-self.heartbeat>4:
                try:await self.stop_robot()
                except ValueError:pass
            await asyncio.sleep(.5)


async def index(request):return web.FileResponse(HERE/'dist/index.html')
async def session(request):return web.json_response({'token':request.app['bench'].token})
async def state(request):return web.json_response(request.app['bench'].state())

async def control(request):
    bench=request.app['bench']
    data=await request.json()
    if not secrets.compare_digest(str(data.get('token','')),bench.token):raise web.HTTPForbidden(text='Invalid session token')
    origin=request.headers.get('Origin')
    if origin and origin not in ('http://127.0.0.1:8080','http://localhost:8080'):raise web.HTTPForbidden(text='Invalid origin')
    try:
        action=request.match_info['action']
        if action=='language':
            language=data.get('language')
            if language not in ('zh','en'):raise ValueError('Unsupported language')
            path=bench.session/'language.json'
            temp=path.with_suffix('.tmp');temp.write_text(json.dumps({'language':language}));temp.replace(path)
        elif action=='heartbeat':
            if data.get('owner')==bench.robot_owner:bench.heartbeat=time.monotonic()
        elif action=='start':
            if data.get('confirmed') is not True or not data.get('owner'):raise ValueError('请先确认运动准备。')
            await bench.start_robot(data['owner'])
        elif action=='stop':await bench.stop_robot()
        else:raise web.HTTPNotFound()
        return web.json_response({'ok':True})
    except ValueError as exc:return web.json_response({'ok':False,'message':str(exc)},status=409)

async def stream(request):
    bench=request.app['bench'];kind=request.match_info['kind']
    if kind not in ('rgb','depth'):raise web.HTTPNotFound()
    mode=request.query.get('mode','adaptive')
    if mode != 'adaptive':raise web.HTTPBadRequest()
    response=web.StreamResponse(headers={'Content-Type':'multipart/x-mixed-replace; boundary=frame','Cache-Control':'no-store'})
    await response.prepare(request)
    try:
        while not bench.stop.is_set():
            image=bench.rgb if kind=='rgb' else bench.depth(mode)
            if image:await response.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '+str(len(image)).encode()+b'\r\n\r\n'+image+b'\r\n')
            await asyncio.sleep(.1)
    except (ConnectionResetError,asyncio.CancelledError):pass
    return response

async def legend_image(request):
    dummy=np.array([[.5,1]],dtype=np.float32)
    _,_,legend=render(dummy,[],request.query.get('mode','adaptive'))
    ok,data=cv2.imencode('.png',legend)
    return web.Response(body=data.tobytes(),content_type='image/png')

async def lifecycle(app):
    bench=Workbench();app['bench']=bench
    bench.thread=threading.Thread(target=bench.capture,daemon=True);bench.thread.start()
    bench.spawn(ROOT/'perception/qwen_scene/continuous_worker.py','qwen.log',bench.session)
    bench.spawn(ROOT/'perception/qwen_scene/vision_worker.py','vision.log',bench.session)
    # Reuse the sole IMU serial owner when its existing service is running.
    async with ClientSession() as client:
        try:
            async with client.get('http://127.0.0.1:8000/',timeout=2) as response:
                if response.status!=200:raise RuntimeError('IMU service unavailable')
        except Exception:bench.spawn(ROOT/'sensors/imu/server.py','imu.log')
    watchdog=asyncio.create_task(bench.watchdog())
    print(f'Workbench http://127.0.0.1:8080/ · session {bench.session}',flush=True)
    try:yield
    finally:
        try:await bench.stop_robot()
        finally:
            bench.stop.set();(bench.session/'stop').touch();watchdog.cancel()
            await asyncio.gather(watchdog,return_exceptions=True)
            for process in bench.processes:
                if process.poll() is None:
                    try:os.killpg(process.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
            for process in bench.processes:
                try:await asyncio.to_thread(process.wait,10)
                except subprocess.TimeoutExpired:process.kill()
            bench.thread.join(timeout=2)
            bench.frame_store.close()
            for log in bench.logs:log.close()


def main():
    app=web.Application(client_max_size=65536)
    app.cleanup_ctx.append(lifecycle)
    app.router.add_get('/',index)
    app.router.add_get('/api/session',session)
    app.router.add_get('/api/state',state)
    app.router.add_post('/api/control/{action}',control)
    app.router.add_get('/stream/{kind}',stream)
    app.router.add_get('/legend.png',legend_image)
    app.router.add_static('/static',HERE/'dist')
    web.run_app(app,host='127.0.0.1',port=8080,shutdown_timeout=3)

if __name__=='__main__':main()
