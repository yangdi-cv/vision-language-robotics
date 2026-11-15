"""Serve the existing yaw demo, with explicit IMU selection and read-only IO."""
import asyncio
import json
import sys
import threading
import time
from collections import deque
from pathlib import Path

from aiohttp import web
import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = Path('/home/yd/Documents/IMU/demo')
sys.path.insert(0, str(SOURCE))
from YbImuSerialLib import YbImuSerial


class IMUReader:
    def __init__(self):
        self.imu = YbImuSerial('/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0')
        self.imu._dev.timeout = .1
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.last_euler_ns = None
        self.last_quaternion_ns = None
        self.samples = deque(maxlen=150)
        self.reference = None
        self.reference_path = HERE/'reference_pose.json'
        if self.reference_path.exists():
            self.reference = json.loads(self.reference_path.read_text())['quaternion_wxyz']
        self.error = None
        original_parse = self.imu._parse_data

        def parse(kind, data):
            original_parse(kind, data)
            if kind == self.imu.FUNC_REPORT_IMU_EULER:
                self.last_euler_ns = time.monotonic_ns()
            if kind == self.imu.FUNC_REPORT_IMU_QUAT:
                self.last_quaternion_ns = time.monotonic_ns()
                self.samples.append((self.last_quaternion_ns, self.imu.get_imu_quaternion_data(), self.imu.get_gyroscope_data(), self.imu.get_accelerometer_data()))

        self.imu._parse_data = parse
        self.thread = threading.Thread(target=self.receive, daemon=True)
        self.thread.start()

    def receive(self):
        try:
            while not self.stop.is_set():
                data = self.imu._dev.read(max(1, self.imu._dev.in_waiting))
                with self.lock:
                    for byte in data:
                        self.imu._receive_data(byte)
        except Exception as exc:
            self.error = str(exc)
        finally:
            self.imu._dev.close()

    def snapshot(self):
        with self.lock:
            roll, pitch, yaw = self.imu.get_imu_attitude_data(True)
            return {'roll': roll, 'pitch': pitch, 'yaw': yaw,
                    'quaternion_wxyz': self.imu.get_imu_quaternion_data(),
                    'acceleration_g': self.imu.get_accelerometer_data(),
                    'gyro_rad_s': self.imu.get_gyroscope_data(),
                    'valid': self.last_quaternion_ns is not None and time.monotonic_ns()-self.last_quaternion_ns < 1_000_000_000,
                    'reference_quaternion_wxyz': self.reference,
                    'received_monotonic_ns': self.last_euler_ns, 'error': self.error}

    def set_reference(self):
        with self.lock:
            now = time.monotonic_ns()
            recent = [s for s in self.samples if now-s[0] < 2_000_000_000]
            if len(recent) < 30 or now-recent[-1][0] > 200_000_000:
                raise ValueError('有效数据不足，请保持静止约两秒后再试。')
            quaternions = np.array([s[1] for s in recent])
            norms = np.linalg.norm(quaternions,axis=1)
            if not np.isfinite(quaternions).all() or np.any(norms < .5):
                raise ValueError('四元数无效，不能设置参考。')
            quaternions /= norms[:,None]
            _, vectors = np.linalg.eigh(quaternions.T@quaternions)
            average = vectors[:,-1]
            if average[0]<0: average *= -1
            angles = np.degrees(2*np.arccos(np.clip(np.abs(quaternions@average),0,1)))
            gyro = np.array([s[2] for s in recent])
            accel = np.array([s[3] for s in recent])
            if np.max(angles)>2 or np.max(np.linalg.norm(gyro,axis=1))>.05 or np.max(np.std(accel,axis=0))>.02:
                raise ValueError('检测到运动，请先让机械臂与 IMU 静止两秒。')
            self.reference = average.tolist()
            record = {'quaternion_wxyz':self.reference,'samples':len(recent),'type':'display_reference_only',
                      'device_calibration_written':False,'saved_unix_ns':time.time_ns()}
            self.reference_path.write_text(json.dumps(record,indent=2))
            return record

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2)


async def index(request):
    return web.FileResponse(HERE/'viewer.html')


async def model(request):
    return web.FileResponse(SOURCE/'model/Xbot.glb')


async def reference(request):
    try:
        record = request.app['reader'].set_reference()
        return web.json_response({'ok':True,'message':'参考姿态已保存；未改动 IMU 硬件校准。','reference':record})
    except ValueError as exc:
        return web.json_response({'ok':False,'message':str(exc)},status=400)


async def clear_reference(request):
    reader=request.app['reader']
    with reader.lock:
        reader.reference=None
        reader.reference_path.unlink(missing_ok=True)
    return web.json_response({'ok':True})


async def websocket(request):
    ws = web.WebSocketResponse(heartbeat=10)
    await ws.prepare(request)
    async def send_samples():
        while not ws.closed:
            await ws.send_json(request.app['reader'].snapshot())
            await asyncio.sleep(.05)
    sender = asyncio.create_task(send_samples())
    try:
        async for _message in ws:
            pass
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    finally:
        sender.cancel()
        await asyncio.gather(sender, return_exceptions=True)
        await ws.close()
    return ws


async def lifecycle(app):
    reader = IMUReader()
    app['reader'] = reader
    try:
        yield
    finally:
        reader.close()


def main():
    app = web.Application()
    app.cleanup_ctx.append(lifecycle)
    app.router.add_get('/', index)
    app.router.add_get('/ws', websocket)
    app.router.add_post('/reference', reference)
    app.router.add_post('/reference/clear', clear_reference)
    app.router.add_get('/model/Xbot.glb', model)
    app.router.add_static('/assets', HERE/'assets')
    web.run_app(app, host='127.0.0.1', port=8000)


if __name__ == '__main__':
    main()
