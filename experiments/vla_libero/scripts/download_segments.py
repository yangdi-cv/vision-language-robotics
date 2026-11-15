"""Resumable ranged official download, followed by GCS checksum verification."""

from pathlib import Path
import urllib.request, urllib.parse, json, concurrent.futures, time

R = Path(__file__).resolve().parents[1]
root = R / "checkpoints/pi05_libero"
objects = json.loads((R / "configs/checkpoint_objects.json").read_text())["items"]
tasks = []
chunk = 32 << 20
for o in objects:
    p = root / o["name"].removeprefix("checkpoints/pi05_libero/")
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists() and p.stat().st_size == int(o["size"]):
        continue
    prefix = p.with_name(p.name + ".partial")
    start = prefix.stat().st_size if prefix.exists() else 0
    parts = p.with_name(p.name + ".parts")
    parts.mkdir(exist_ok=True)
    for a in range(start, int(o["size"]), chunk):
        tasks.append((o, a, min(a + chunk, int(o["size"])), parts / f"{a:012d}"))


def fetch(job):
    o, a, b, p = job
    if p.exists() and p.stat().st_size == b - a:
        return
    url = (
        "https://storage.googleapis.com/openpi-assets/"
        + urllib.parse.quote(o["name"])
        + f"?part={a}"
    )
    for attempt in range(6):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b - 1}"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                assert resp.status == 206
                with p.with_suffix(".tmp").open("wb") as f:
                    for block in iter(lambda: resp.read(1 << 20), b""):
                        f.write(block)
            assert p.with_suffix(".tmp").stat().st_size == b - a
            p.with_suffix(".tmp").rename(p)
            return
        except Exception as e:
            print("retry", a, attempt, str(e), flush=True)
            time.sleep(2)
    raise RuntimeError(o["name"])


print("segments", len(tasks), flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=24) as ex:
    for i, _ in enumerate(ex.map(fetch, tasks)):
        if i % 12 == 0:
            print("completed", i + 1, "/", len(tasks), flush=True)
# Verification also assembles objects. Composite GCS objects expose CRC32C,
# whereas MD5 is only available for non-composite objects.
import subprocess

subprocess.run(
    [
        str(R.parents[1] / ".venvs/openpi-libero/bin/python"),
        str(R / "scripts/verify_checkpoint.py"),
    ],
    check=True,
)
