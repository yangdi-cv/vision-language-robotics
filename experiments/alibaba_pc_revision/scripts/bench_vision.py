"""PC detection/depth speed and simulated-depth error, not real-camera calibration."""

from pathlib import Path
import json, time, random, itertools
import numpy as np, torch
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[3]
T = ROOT / "experiments/alibaba_pc_revision"
D = ROOT / "experiments/alibaba_extension/data/rgbd_cases"
O = T / "runs/vision_precision"
O.mkdir(exist_ok=False)
det = YOLO(str(ROOT / "models/ultralytics/yolo26n.pt"))
depth = YOLO(str(ROOT / "models/ultralytics/yolo26n-depth.pt"))
torch.set_num_threads(8)
images = sorted(D.glob("rgb_*.png"))[:8]
if not images:
    images = sorted(D.glob("*.png"))[:8]
records = []
# Separate model instances avoid FP16 state carrying into the FP32 branch.
for half, size in [(False, 768), (True, 768), (True, 640)]:
    det = YOLO(str(ROOT / "models/ultralytics/yolo26n.pt"))
    depth = YOLO(str(ROOT / "models/ultralytics/yolo26n-depth.pt"))
    for warm in range(2):
        det(
            np.asarray(Image.open(images[0]))[:, :, ::-1].copy(),
            device=0,
            half=half,
            verbose=False,
        )
        depth(
            np.asarray(Image.open(images[0]))[:, :, ::-1].copy(),
            imgsz=size,
            device=0,
            half=half,
            verbose=False,
        )
    for rep in range(2):
        for i, path in enumerate(images):
            image = np.asarray(Image.open(path).convert("RGB"))[:, :, ::-1].copy()
            torch.cuda.synchronize()
            tick = time.perf_counter()
            boxes = det(
                image, imgsz=640, device=0, half=half, conf=0.35, verbose=False
            )[0]
            pred = (
                depth(image, imgsz=size, device=0, half=half, verbose=False)[0]
                .depth.data.float()
                .cpu()
                .numpy()
            )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - tick
            key = path.stem.split("_")[-1]
            truth = np.load(D / f"depth_{key}.npy")
            mask = (
                np.isfinite(truth)
                & (truth > 0.1)
                & (truth < 3)
                & np.isfinite(pred)
                & (pred > 0)
            )
            absolute = np.mean(abs(pred[mask] - truth[mask]) / truth[mask])
            scale = np.median(truth[mask] / pred[mask])
            aligned = np.mean(abs(pred[mask] * scale - truth[mask]) / truth[mask])
            records.append(
                dict(
                    image=path.name,
                    half=half,
                    depth_size=size,
                    repeat=rep,
                    seconds=elapsed,
                    abs_rel=float(absolute),
                    scale_aligned_abs_rel=float(aligned),
                    alignment_scale=float(scale),
                    boxes=len(boxes.boxes),
                )
            )
summary = []
for half, size in [(False, 768), (True, 768), (True, 640)]:
    r = [v for v in records if v["half"] == half and v["depth_size"] == size]
    summary.append(
        dict(
            half=half,
            depth_size=size,
            frames=len(r),
            median_seconds=float(np.median([v["seconds"] for v in r])),
            p95_seconds=float(np.percentile([v["seconds"] for v in r], 95)),
            mean_abs_rel=float(np.mean([v["abs_rel"] for v in r])),
            mean_scale_aligned_abs_rel=float(
                np.mean([v["scale_aligned_abs_rel"] for v in r])
            ),
            boxes_total=sum(v["boxes"] for v in r),
        )
    )
(O / "records.json").write_text(json.dumps(records, indent=2))
(O / "summary.json").write_text(json.dumps(summary, indent=2))
print(summary, flush=True)
