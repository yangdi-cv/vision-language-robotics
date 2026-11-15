"""Shared camera, continuous Qwen observer, independently supervised teleop."""
import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageTk, PngImagePlugin

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'SO101/code/scripts'))
from camera_view import DEFAULT_ROTATION, rotate_frame


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, ValueError):
        return {}


def readable_result(result):
    scene = result.get('scene')
    if not isinstance(scene, dict):
        return '本轮回答未通过格式检查，正在等待下一轮分析。'
    visible = {True: '可见', False: '未看到', None: '无法确定'}.get(scene.get('target_visible'), '无法确定')
    gripper = {'open': '张开', 'closed': '闭合', 'unknown': '无法确定'}.get(scene.get('gripper_state'), '无法确定')
    paragraphs = [scene.get('scene_description', ''),
                  '可见物体：' + '、'.join(scene.get('visible_objects', [])),
                  f'蓝色塑胶壳：{visible}。夹爪：{gripper}。']
    if scene.get('target_relation'):
        paragraphs.append('目标位置：' + scene['target_relation'])
    if scene.get('grasp_evidence'):
        paragraphs.append('接触观察：' + scene['grasp_evidence'])
    if scene.get('uncertainties'):
        paragraphs.append('暂不能确定：\n' + '\n'.join('• ' + item for item in scene['uncertainties']))
    return '\n\n'.join(p for p in paragraphs if p)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=Path)
    parser.add_argument('--camera', default='/dev/v4l/by-id/usb-icSpring_icspring_camera-video-index0')
    parser.add_argument('--rotation', type=int, choices=(0, 90, 180, 270), default=DEFAULT_ROTATION)
    parser.add_argument('--self-test', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.image:
        from engine import SceneEngine
        result, path = SceneEngine().analyze(Image.open(args.image).convert('RGB'), ROOT / 'records/scene_understanding', None, str(args.image.resolve()))
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print(f'Saved: {path}')
        return
    import tkinter as tk
    from tkinter import font as tkfont
    from tkinter.scrolledtext import ScrolledText
    from tkinter import messagebox
    from tkinter import ttk
    session = ROOT / 'records/scene_understanding' / f'live_{time.time_ns()}'
    session.mkdir(parents=True)
    print(f'Live session: {session}', flush=True)
    window = tk.Tk()
    window.title('SO101 主从操作 · 检测与深度 · Qwen 连续理解')
    for name in ('TkDefaultFont', 'TkTextFont', 'TkMenuFont', 'TkHeadingFont', 'TkCaptionFont', 'TkTooltipFont'):
        tkfont.nametofont(name).configure(family='Noto Sans CJK SC', size=12)
    window.option_add('*Font', 'TkDefaultFont')
    window.configure(background='#f4f6f8')
    window.columnconfigure(2, weight=1)
    window.rowconfigure(4, weight=1)
    rgb_panel = tk.Frame(window, background='#f4f6f8')
    rgb_panel.grid(row=0, column=0, rowspan=5, sticky='n', padx=12, pady=12)
    tk.Label(rgb_panel, text='实时 RGB', background='#f4f6f8', font=('Noto Sans CJK SC', 14)).pack(pady=(0,10))
    view = tk.Label(rgb_panel)
    view.pack()
    vision_panel = tk.Frame(window, background='#f4f6f8')
    vision_panel.grid(row=0, column=1, rowspan=5, sticky='n', padx=6, pady=12)
    tk.Label(vision_panel, text='深度图 + 目标框', background='#f4f6f8', font=('Noto Sans CJK SC', 14)).pack(pady=(0,10))
    detection_view = tk.Label(vision_panel)
    detection_view.pack()
    legend = tk.Canvas(vision_panel, width=360, height=66, bg='#f4f6f8', highlightthickness=0)
    legend.pack(pady=(8,0))
    for x in range(320):
        # Same fixed linear 0–3 m TURBO color scale as the model overlay.
        value = round(255*(1-x/319))
        b,g,r = cv2.applyColorMap(np.array([[value]],dtype=np.uint8),cv2.COLORMAP_TURBO)[0,0]
        legend.create_line(20+x,4,20+x,22,fill=f'#{r:02x}{g:02x}{b:02x}')
    for fraction in (0,.25,.5,.75,1):
        x = 20+320*fraction
        legend.create_line(x,22,x,27,fill='#243447')
        label = f'{3*fraction:g} m' if fraction<1 else '≥3 m'
        legend.create_text(x,40,text=label,font=('Noto Sans CJK SC',10),fill='#243447')
    vision_status = tk.StringVar(value='检测与深度模型正在加载……')
    detection_status = tk.StringVar(value='预测深度未标定 · 固定 0–3 m 色阶')
    tk.Label(vision_panel,textvariable=detection_status,wraplength=360,background='#f4f6f8',justify='left').pack(pady=5)
    tk.Label(vision_panel,textvariable=vision_status,wraplength=360,background='#f4f6f8',justify='left').pack()
    status = tk.StringVar(value='正在自动加载 Qwen；相机预览独立运行。')
    robot_status = tk.StringVar(value='双臂未启动')
    age_status = tk.StringVar(value='尚无分析结果')
    for row, variable in enumerate((status, age_status, robot_status)):
        tk.Label(window, textvariable=variable, wraplength=580, justify='left', anchor='w',
                 background='#f4f6f8', foreground='#243447').grid(row=row, column=2, sticky='ew', padx=18, pady=(12 if row == 0 else 5, 5))
    controls = tk.Frame(window, background='#f4f6f8')
    controls.grid(row=3, column=2, sticky='w', padx=18, pady=10)
    text = ScrolledText(window, width=42, height=20, wrap='word', font=('Noto Sans CJK SC', 14),
                        padx=18, pady=16, spacing1=4, spacing3=8, relief='flat',
                        background='white', foreground='#172b4d', insertwidth=0)
    text.grid(row=4, column=2, sticky='nsew', padx=18, pady=(0, 18))
    text.insert('end', '正在准备场景理解。\n\n模型加载后会自动分析最新画面，结果显示在这里。')
    text.configure(state='disabled')
    camera_stop = threading.Event()
    camera_error = []
    frame_lock = threading.Lock()
    camera_frame = [None, None]
    teleop = None
    teleop_log = None
    stopping_since = None
    closed = False
    result_id = None
    latest_result = {}
    last_displayed = None
    vision_mtime = None
    latest_vision = {}

    def capture():
        camera = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
        try:
            if not camera.isOpened():
                raise RuntimeError('相机无法打开，请关闭其他预览或采集程序。')
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            counter = 0
            while not camera_stop.is_set():
                cycle = time.monotonic()
                ok, frame = camera.read()
                if not ok:
                    raise RuntimeError('相机停止返回图像')
                captured_ns = time.monotonic_ns()
                image = Image.fromarray(cv2.cvtColor(rotate_frame(frame, args.rotation), cv2.COLOR_BGR2RGB))
                metadata = PngImagePlugin.PngInfo()
                metadata.add_text('frame_id', str(counter))
                metadata.add_text('captured_ns', str(captured_ns))
                metadata.add_text('rotation', str(args.rotation))
                image.save(session / 'latest.tmp', format='PNG', pnginfo=metadata, compress_level=1)
                (session / 'latest.tmp').replace(session / 'latest.png')
                with frame_lock:
                    camera_frame[:] = [counter, image]
                counter += 1
                camera_stop.wait(max(0, .1 - (time.monotonic() - cycle)))
        except Exception as exc:
            camera_error.append(str(exc))
        finally:
            camera.release()

    capture_thread = threading.Thread(target=capture, daemon=True)
    capture_thread.start()
    worker_log = (session / 'qwen.log').open('w')
    worker_args = [sys.executable, str(Path(__file__).with_name('continuous_worker.py')), str(session)]
    if args.self_test:
        worker_args += ['--max-results', '3']
    worker = subprocess.Popen(worker_args, stdout=worker_log, stderr=subprocess.STDOUT, start_new_session=True)
    vision_log = (session / 'vision.log').open('w')
    vision_args = [sys.executable, str(Path(__file__).with_name('vision_worker.py')), str(session)]
    if args.self_test:
        vision_args += ['--max-results', '30']
    vision_worker = subprocess.Popen(vision_args, stdout=vision_log, stderr=subprocess.STDOUT, start_new_session=True)

    def start_teleop():
        nonlocal teleop, teleop_log
        if args.self_test or (teleop and teleop.poll() is None):
            return
        if not messagebox.askokcancel('启动主从操作', '确认两臂固定、空间清空、电源可随手断开，并托稳白色臂。\n启动后松开托扶，缓慢移动黑色主臂。\nQwen 只观察，不发送动作。', parent=window):
            return
        teleop_log = (session / f'teleop_{time.time_ns()}.log').open('w')
        (session / 'robot_status.json').unlink(missing_ok=True)
        teleop = subprocess.Popen([str(ROOT / 'SO101/code/scripts/ubuntu/start-teleop'), '--require-calibration', '--quiet', '--status-file', str(session / 'robot_status.json')], stdout=teleop_log, stderr=subprocess.STDOUT, start_new_session=True)
        robot_status.set('双臂启动中；若报错查看本次 teleop 日志。')
        start_button.configure(state='disabled')
        stop_button.configure(state='normal')

    def stop_teleop():
        nonlocal stopping_since
        if teleop and teleop.poll() is None and stopping_since is None:
            if not messagebox.askokcancel('停止主从操作', '请先托稳白色从臂。确认后关闭扭矩。\n异常运动请直接断开 12V 电源。', parent=window):
                return False
            control = read_json(session / 'robot_status.json')
            try:
                if control.get('pid'):
                    os.kill(control['pid'], signal.SIGINT)
                else:
                    os.killpg(teleop.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            stopping_since = time.monotonic()
            robot_status.set('正在停止双臂、关闭扭矩……')
            stop_button.configure(state='disabled')
        return True

    start_button = tk.Button(controls, text='启动主从操作', command=start_teleop, state='disabled', padx=14, pady=8)
    start_button.pack(side='left', padx=(0, 10))
    stop_button = tk.Button(controls, text='停止操作（先托稳）', command=stop_teleop, state='disabled', padx=14, pady=8)
    stop_button.pack(side='left')

    def close(*_):
        nonlocal closed
        if teleop and teleop.poll() is None:
            if stop_teleop():
                status.set('先等待双臂停止，再关闭窗口。')
            return
        closed = True
        camera_stop.set()
        (session / 'stop').touch()
        if worker.poll() is None:
            os.killpg(worker.pid, signal.SIGTERM)
        if vision_worker.poll() is None:
            os.killpg(vision_worker.pid, signal.SIGTERM)
        window.destroy()

    def update():
        nonlocal result_id, latest_result, last_displayed, stopping_since, vision_mtime, latest_vision
        if closed:
            return
        with frame_lock:
            frame_id, image = camera_frame
        if image is not None and frame_id != last_displayed:
            display_image = image.copy()
            display_image.thumbnail((360,480))
            photo = ImageTk.PhotoImage(display_image)
            view.configure(image=photo)
            view.image = photo
            last_displayed = frame_id
        visual_state = read_json(session/'vision_status.json')
        if visual_state.get('state') == 'error':
            vision_status.set('检测/深度失败：' + visual_state.get('message',''))
        elif visual_state.get('state') == 'loading':
            vision_status.set('检测与深度模型正在加载……')
        visual_path = session/'latest_vision.npz'
        try:
            modified = visual_path.stat().st_mtime_ns
            if modified != vision_mtime:
                with np.load(visual_path, allow_pickle=False) as data:
                    latest_vision = json.loads(str(data['metadata']))
                    for label, key in ((detection_view,'detection'),):
                        display_image = Image.fromarray(data[key])
                        display_image.thumbnail((360,480))
                        photo = ImageTk.PhotoImage(display_image)
                        label.configure(image=photo)
                        label.image = photo
                vision_mtime = modified
        except (FileNotFoundError, OSError, ValueError):
            pass
        if latest_vision:
            age = (time.monotonic_ns()-latest_vision['captured_ns'])/1e9
            duration = latest_vision['inference_seconds']*1000
            count = len(latest_vision['objects'])
            vision_status.set(f'检测＋深度 {duration:.0f} ms；结果画面距今 {age:.1f} 秒。')
            detection_status.set(f'检测到 {count} 个目标 · 预测深度未标定\n暖色近、冷色远 · 固定 0–3 m 色阶')
        worker_status = read_json(session / 'worker_status.json')
        state = worker_status.get('state', 'loading')
        status.set({'loading': 'Qwen 正在自动加载……', 'ready': 'Qwen 已加载，等待最新图像。',
                    'inferencing': 'Qwen 连续推理中；上一轮结果保持显示。', 'waiting_camera': '等待新鲜相机帧……',
                    'stopped': 'Qwen 已停止。', 'error': 'Qwen 分析失败：' + worker_status.get('message', '')}.get(state, state))
        if worker.poll() is not None and state not in ('stopped', 'error'):
            status.set('Qwen 进程已退出，请检查 qwen.log。')
        if camera_error:
            status.set(camera_error[-1])
            start_button.configure(state='disabled')
        elif state in ('ready', 'inferencing') and not args.self_test and (not teleop or teleop.poll() is not None):
            start_button.configure(state='normal')
        result = read_json(session / 'latest_result.json')
        if result and result.get('frame_id') != result_id:
            result_id = result['frame_id']
            latest_result = result
            text.configure(state='normal')
            text.delete('1.0', 'end')
            text.insert('end', readable_result(result))
            text.configure(state='disabled')
        if latest_result:
            elapsed = (time.monotonic_ns() - latest_result['captured_monotonic_ns']) / 1e9
            age_status.set(f"上次结果对应 {elapsed:.1f} 秒前画面；推理 {latest_result['inference_seconds']:.2f} 秒。")
        if teleop:
            code = teleop.poll()
            if code is not None:
                if teleop_log and not teleop_log.closed:
                    teleop_log.close()
                robot_status.set(f'双臂进程已结束（退出码 {code}）；正常退出关闭扭矩，异常退出请检查日志。')
                stop_button.configure(state='disabled')
                stopping_since = None
            elif stopping_since is not None and time.monotonic()-stopping_since > 5:
                robot_status.set('双臂尚未退出；托稳从臂，必要时断开电源。')
            elif stopping_since is None:
                control = read_json(session / 'robot_status.json')
                robot_status.set('主从操作已就绪：操作黑色主臂；Qwen 独立观察。' if control.get('state') == 'running' else '正在连接与配置双臂……')
        if args.self_test and worker.poll() is not None:
            close()
            return
        window.after(50, update)

    window.bind('<Escape>', close)
    window.protocol('WM_DELETE_WINDOW', close)
    window.after(0, update)
    try:
        window.mainloop()
    except KeyboardInterrupt:
        if teleop and teleop.poll() is None:
            os.killpg(teleop.pid, signal.SIGINT)
            teleop.wait(timeout=10)
    finally:
        camera_stop.set()
        (session / 'stop').touch()
        if teleop and teleop.poll() is None:
            os.killpg(teleop.pid, signal.SIGINT)
            teleop.wait(timeout=10)
        if worker.poll() is None:
            os.killpg(worker.pid, signal.SIGTERM)
            worker.wait(timeout=10)
        if vision_worker.poll() is None:
            os.killpg(vision_worker.pid, signal.SIGTERM)
            vision_worker.wait(timeout=10)
        capture_thread.join(timeout=2)
        worker_log.close()
        vision_log.close()
        if teleop_log:
            teleop_log.close()


if __name__ == '__main__':
    main()
