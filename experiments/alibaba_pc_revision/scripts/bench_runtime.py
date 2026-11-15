"""Two-reader transport replay and bounded worker failure checks, without hardware."""

from pathlib import Path
import sys, os, json, time, subprocess, hashlib, io, tempfile
import numpy as np
from PIL import Image, PngImagePlugin

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "perception/qwen_scene"))
from frame_transport import SharedFrame, read_latest_image
from worker_client import read_json_line, close_json_worker

T = ROOT / "experiments/alibaba_pc_revision"
O = T / "runs/runtime"
if "--reader" in sys.argv:
    session = Path(sys.argv[-1])
    print('{"ready":true}', flush=True)
    for line in sys.stdin:
        if json.loads(line).get("stop"):
            break
        image, meta = read_latest_image(session)
        print(
            json.dumps(
                {
                    "frame_id": int(meta["frame_id"]),
                    "sha256": hashlib.sha256(np.asarray(image).tobytes()).hexdigest(),
                }
            ),
            flush=True,
        )
    sys.exit(0)
O.mkdir(exist_ok=False)
failures = []
for name, body in [
    ("silent", "import time;time.sleep(20)"),
    ("crash", "raise SystemExit(7)"),
    ("malformed", 'print("broken",flush=True)'),
    ("healthy", 'print("{\\"ready\\":true}",flush=True)'),
]:
    p = subprocess.Popen(
        [sys.executable, "-c", body],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    begin = time.monotonic()
    try:
        r = read_json_line(p, timeout=0.2)
        status = "ok"
    except Exception as exc:
        status = type(exc).__name__
    finally:
        close_json_worker(p)
    elapsed = time.monotonic() - begin
    assert elapsed < 5
    assert (status == "ok") == (name == "healthy")
    failures.append(
        dict(
            case=name,
            status=status,
            total_seconds=elapsed,
            process_reaped=p.poll() is not None,
        )
    )
paths = sorted((ROOT / "experiments/alibaba_baseline/data/qwen_frames").glob("*.png"))[
    :12
]
assert paths
images = [np.asarray(Image.open(p).convert("RGB")) for p in paths]
records = []
for mode in ["png", "shared"]:
    session = O / mode
    session.mkdir()
    owner = SharedFrame(session, create=True) if mode == "shared" else None
    workers = []
    try:
        for _ in range(2):
            p = subprocess.Popen(
                [sys.executable, __file__, "--reader", str(session)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            read_json_line(p, timeout=3)
            workers.append(p)
        for repeat in range(3):
            for i, rgb in enumerate(images):
                frame_id = repeat * len(images) + i
                stamp = time.monotonic_ns()
                tick = time.perf_counter()
                if owner:
                    owner.write(rgb, frame_id, stamp)
                else:
                    info = PngImagePlugin.PngInfo()
                    for k, v in dict(
                        frame_id=frame_id, captured_ns=stamp, rotation=270
                    ).items():
                        info.add_text(k, str(v))
                    Image.fromarray(rgb).save(
                        session / "latest.tmp",
                        format="PNG",
                        pnginfo=info,
                        compress_level=1,
                    )
                    (session / "latest.tmp").replace(session / "latest.png")
                published = time.perf_counter()
                for p in workers:
                    p.stdin.write("{}\n")
                    p.stdin.flush()
                replies = [read_json_line(p, timeout=3) for p in workers]
                end = time.perf_counter()
                digest = hashlib.sha256(rgb.tobytes()).hexdigest()
                assert all(
                    r["frame_id"] == frame_id and r["sha256"] == digest for r in replies
                )
                records.append(
                    dict(
                        mode=mode,
                        frame_id=frame_id,
                        write_ms=(published - tick) * 1000,
                        total_two_readers_ms=(end - tick) * 1000,
                    )
                )
        if owner:
            for i in range(100):
                owner.write(images[0], 1000 + i, time.monotonic_ns())
            _, meta = owner.read()
            assert meta["frame_id"] == "1099"
    finally:
        for p in workers:
            close_json_worker(p)
        if owner:
            owner.close()
summary = []
for mode in ["png", "shared"]:
    r = [x for x in records if x["mode"] == mode][2:]
    summary.append(
        dict(
            mode=mode,
            frames=len(r),
            write_p50_ms=float(np.median([x["write_ms"] for x in r])),
            total_p50_ms=float(np.median([x["total_two_readers_ms"] for x in r])),
            total_p95_ms=float(
                np.percentile([x["total_two_readers_ms"] for x in r], 95)
            ),
        )
    )
(O / "records.json").write_text(json.dumps(records, indent=2))
(O / "summary.json").write_text(
    json.dumps(
        {
            "transport": summary,
            "worker_failures": failures,
            "pixel_integrity_checked": len(records) * 2,
            "latest_only_checked": True,
            "camera_opened": False,
        },
        indent=2,
    )
)
print(summary, flush=True)
