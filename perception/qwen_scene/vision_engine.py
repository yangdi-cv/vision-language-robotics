"""Ultralytics detection and monocular depth on the same rotated RGB frame."""
import time
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
DEPTH_MIN_M = 0.0
DEPTH_MAX_M = 3.0


class VisionEngine:
    def __init__(self, half=False, depth_size=768):
        self.half=bool(half)
        self.depth_size=int(depth_size)
        if self.depth_size not in (640,768):raise ValueError("depth_size must be640 or768")
        self.detector_path = Path(os.environ.get('VLA_DETECTOR_PATH', str(ROOT / 'models/ultralytics/yolo26n.pt')))
        self.detector = YOLO(str(self.detector_path))
        self.depth_model = YOLO(str(ROOT / 'models/ultralytics/yolo26n-depth.pt'))

    def analyze(self, image):
        rgb = np.asarray(image)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        started = time.perf_counter()
        detections = self.detector(bgr, imgsz=640, device=0, conf=.35, quantize=16 if self.half else None, verbose=False)[0]
        depths = self.depth_model(bgr, imgsz=self.depth_size, device=0, quantize=16 if self.half else None, verbose=False)[0]
        raw_depth = depths.depth.data.float().cpu().numpy()
        if raw_depth.shape != rgb.shape[:2]:
            raise ValueError('Depth and RGB resolution mismatch')
        valid = np.isfinite(raw_depth) & (raw_depth > 0)
        color_input = 1 - np.clip((raw_depth-DEPTH_MIN_M)/(DEPTH_MAX_M-DEPTH_MIN_M), 0, 1)
        color_input[~valid] = 0
        depth_color = cv2.applyColorMap((color_input*255).astype(np.uint8), cv2.COLORMAP_TURBO)
        depth_color[~valid] = 0
        annotated = depth_color.copy()
        objects = []
        for box in detections.boxes:
            coords = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = coords.round().astype(int)
            x1, x2 = np.clip([x1, x2], 0, rgb.shape[1])
            y1, y2 = np.clip([y1, y2], 0, rgb.shape[0])
            region = raw_depth[y1:y2, x1:x2]
            values = region[np.isfinite(region) & (region > 0)]
            estimate = float(np.median(values)) if values.size else None
            name = detections.names[int(box.cls.item())]
            confidence = float(box.conf.item())
            objects.append({'class': name, 'confidence': confidence, 'xyxy': coords.tolist(),
                            'box_region_depth_estimate_m': estimate})
            label = f'Depth {estimate:.2f} m' if estimate is not None else 'Depth unknown'
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 255), 2)
            label_y = max(y1-7, 20)
            cv2.putText(annotated, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(annotated, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)
        elapsed = time.perf_counter()-started
        stats = {'inference_seconds': elapsed, 'objects': objects,
                 'depth_valid_fraction': float(valid.mean()),
                 'depth_percentile_5_m': float(np.percentile(raw_depth[valid],5)) if valid.any() else None,
                 'depth_percentile_95_m': float(np.percentile(raw_depth[valid],95)) if valid.any() else None,
                 'depth_scale_validated': False,
                 'color_scale': 'fixed linear depth in meters; warm near, cool far; clipped beyond range',
                 'depth_color_min_m': DEPTH_MIN_M, 'depth_color_max_m': DEPTH_MAX_M,
                 'display_mode': 'opaque depth map with detection boxes',
                 'models': [self.detector_path.name,'yolo26n-depth.pt']}
        return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB), raw_depth, stats
