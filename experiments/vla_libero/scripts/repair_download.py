"""Finish slow partial ranges via small resumable requests; preserve completed bytes."""

from pathlib import Path
import concurrent.futures, urllib.request, urllib.parse, json, time, shutil

R = Path(__file__).resolve().parents[1]
root = R / "checkpoints/pi05_libero"
objects = json.loads((R / "configs/checkpoint_objects.json").read_text())["items"]
jobs = []
segments = []
chunk = 32 << 20
subchunk = 1 << 20
for o in objects:
    p = root / o["name"].removeprefix("checkpoints/pi05_libero/")
    prefix = p.with_name(p.name + ".partial")
    parts = p.with_name(p.name + ".parts")
    if not parts.exists():
        continue
    start = prefix.stat().st_size if prefix.exists() else 0
    for a in range(start, int(o["size"]), chunk):
        b = min(a + chunk, int(o["size"]))
        seg = parts / f"{a:012d}"
        if seg.exists() and seg.stat().st_size == b - a:
            continue
        temp = seg.with_suffix(".tmp")
        offset = temp.stat().st_size if temp.exists() else 0
        small = parts / (seg.name + "_small")
        small.mkdir(exist_ok=True)
        segments.append((seg, temp, small, b - a))
        for x in range(a + offset, b, subchunk):
            jobs.append((o, x, min(x + subchunk, b), small / f"{x:012d}"))


def fetch(job):
    o, a, b, p = job
    for retry in range(8):
        try:
            host = (
                "openpi-assets.storage.googleapis.com"
                if retry % 2 == 0
                else "storage.googleapis.com/openpi-assets"
            )
            url = (
                "https://"
                + host
                + "/"
                + urllib.parse.quote(o["name"])
                + f"?range_repair={a}&retry={retry}"
            )
            req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b - 1}"})
            start = time.monotonic()
            with urllib.request.urlopen(req, timeout=15) as src:
                assert src.status == 206
                with p.with_suffix(".tmp").open("wb") as out:
                    while True:
                        block = src.read(64 << 10)
                        if not block:
                            break
                        if time.monotonic() - start > 25:
                            raise TimeoutError("slow range")
                        out.write(block)
            assert p.with_suffix(".tmp").stat().st_size == b - a
            p.with_suffix(".tmp").rename(p)
            return
        except Exception as e:
            print("retry", a, retry, str(e), flush=True)
    raise RuntimeError(o["name"])


print("Repair segments", len(segments), "small requests", len(jobs), flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=24) as ex:
    list(ex.map(fetch, jobs))
for seg, temp, small, size in segments:
    with temp.open("ab") as out:
        for p in sorted(small.iterdir()):
            with p.open("rb") as f:
                shutil.copyfileobj(f, out)
    assert temp.stat().st_size == size
    temp.rename(seg)
    shutil.rmtree(small)
print("Partial ranges completed", flush=True)
