"""Local image-based LeRobot v2.1 recording; no robot commands or uploads."""
import csv
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from camera_view import DEFAULT_ROTATION, rotate_frame


class DemonstrationRecorder:
    def __init__(self, root, task, camera, joints, calibration_root, ports, fps=10, rotation=DEFAULT_ROTATION, preview=False):
        self.root = Path(root).resolve()
        if self.root.exists():
            raise ValueError(f"Recording path already exists: {self.root}")
        self.task = task
        self.joints = joints
        self.fps = fps
        self.rotation = rotation
        self.camera = cv2.VideoCapture(str(camera), cv2.CAP_V4L2)
        self.dataset = None
        self.timing_file = None
        self.frames = 0
        self.preview_window = None
        try:
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not self.camera.isOpened():
                raise RuntimeError(f"Cannot open camera: {camera}")
            for _ in range(5):
                ok, image = self.camera.read()
                if not ok:
                    raise RuntimeError("Camera warmup failed")
            names = [f"{joint}.pos" for joint in joints]
            image = rotate_frame(image, self.rotation)
            features = {
                "observation.state": {"dtype": "float32", "shape": (6,), "names": names},
                "action": {"dtype": "float32", "shape": (6,), "names": names},
                "observation.images.front": {
                    "dtype": "image", "shape": image.shape,
                    "names": ["height", "width", "channels"],
                },
            }
            self.dataset = LeRobotDataset.create(
                "local/desktop_pick_place", root=self.root, fps=fps,
                features=features, robot_type="so101_follower", use_videos=False,
                image_writer_threads=4,
            )
            self.manifest = {
                "task": task, "status": "recording", "fps_nominal": fps,
                "camera": str(camera), "image_shape": list(image.shape),
                "rotation_clockwise_degrees": self.rotation,
                "camera_mount": "arm-mounted; viewpoint changes with arm pose",
                "ports": ports, "joint_order": list(joints),
                "units": "degrees for five joints; percent for gripper",
                "action_semantics": "absolute follower goals actually sent after 5-unit clipping",
                "timing": "dataset uses nominal frame_index/fps; timing.csv uses host monotonic receipt times, not camera exposure timestamps",
                "calibration_sha256": {
                    str(p.relative_to(calibration_root)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in Path(calibration_root).rglob("*.json")
                },
            }
            self._write_manifest()
            self.timing_file = (self.root / "timing.csv").open("w", newline="")
            self.timing = csv.writer(self.timing_file)
            self.timing.writerow(["frame_index", "state_received_ns", "image_received_ns", "action_sent_ns"])
            if preview:
                import tkinter as tk
                self.preview_window = tk.Tk()
                self.preview_window.title("示教采集画面 · 关闭窗口停止采集（先托稳从臂）")
                self.preview_label = tk.Label(self.preview_window)
                self.preview_label.pack()
                self.preview_stop = False
                self.preview_window.protocol("WM_DELETE_WINDOW", self._request_stop)
        except BaseException:
            self.camera.release()
            if self.timing_file:
                self.timing_file.close()
            if self.dataset:
                self.dataset.stop_image_writer()
            if self.preview_window:
                self.preview_window.destroy()
            raise

    def _request_stop(self):
        self.preview_stop = True

    def capture(self):
        ok, image = self.camera.read()
        received_ns = time.monotonic_ns()
        if not ok:
            raise RuntimeError("Camera stopped: stopping teleoperation")
        rgb = cv2.cvtColor(rotate_frame(image, self.rotation), cv2.COLOR_BGR2RGB)
        if self.preview_window:
            from PIL import Image, ImageTk
            photo = ImageTk.PhotoImage(Image.fromarray(rgb), master=self.preview_window)
            self.preview_label.configure(image=photo)
            self.preview_label.image = photo
            self.preview_window.update()
            if self.preview_stop:
                raise KeyboardInterrupt
        return rgb, received_ns

    def add(self, actual, sent, image, state_ns, image_ns, sent_ns):
        self.dataset.add_frame({
            "observation.state": np.array([actual[f"{j}.pos"] for j in self.joints], dtype=np.float32),
            "action": np.array([sent[f"{j}.pos"] for j in self.joints], dtype=np.float32),
            "observation.images.front": image,
        }, task=self.task)
        self.timing.writerow([self.frames, state_ns, image_ns, sent_ns])
        self.timing_file.flush()
        self.frames += 1

    def _write_manifest(self):
        (self.root / "session.json").write_text(json.dumps(self.manifest, indent=2, ensure_ascii=False))

    def close(self):
        self.camera.release()
        if self.preview_window:
            self.preview_window.destroy()
        self.timing_file.close()
        self.manifest.update(status="saving", frames=self.frames)
        self._write_manifest()
        if self.frames:
            try:
                self.dataset.save_episode()
            finally:
                self.dataset.stop_image_writer()
        else:
            self.dataset.stop_image_writer()
        self.manifest.update(status="review_required" if self.frames else "empty")
        self._write_manifest()
        print(f"Recorded {self.frames} frames: {self.root}; review before using for training.", flush=True)
