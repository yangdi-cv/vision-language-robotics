"""Parallel-download large pinned public PyPI wheels; verify lockfile SHA256."""

from pathlib import Path
import tomllib, urllib.request, concurrent.futures, hashlib, time

R = Path(__file__).resolve().parents[1]
root = R / "wheelhouse"
root.mkdir(exist_ok=True)
lock = tomllib.loads((R / "third_party/openpi/uv.lock").read_text())
wanted = ["torch", "jaxlib", "jax-cuda12-pjrt", "triton"]
objects = []
tasks = []
chunk = 8 << 20
for x in lock["package"]:
    if x["name"] not in wanted:
        continue
    ws = [
        w
        for w in x.get("wheels", [])
        if ("cp311-cp311" in w["url"] or "py3-none" in w["url"])
        and "x86_64" in w["url"]
        and "musllinux" not in w["url"]
        and "macosx" not in w["url"]
    ]
    w = ws[0]
    objects.append(w)
    p = root / w["url"].split("/")[-1]
    if p.exists() and p.stat().st_size == w["size"]:
        continue
    parts = root / (p.name + ".parts")
    parts.mkdir(exist_ok=True)
    for a in range(0, w["size"], chunk):
        tasks.append((w, a, min(a + chunk, w["size"]), parts / f"{a:012d}"))


def fetch(job):
    o, a, b, p = job
    if p.exists() and p.stat().st_size == b - a:
        return
    for attempt in range(5):
        try:
            req = urllib.request.Request(
                o["url"] + f"?part={a}", headers={"Range": f"bytes={a}-{b - 1}"}
            )
            with urllib.request.urlopen(req, timeout=90) as resp:
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
    raise RuntimeError(o["url"])


with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
    for i, _ in enumerate(ex.map(fetch, tasks)):
        if i % 16 == 0:
            print("segments", i + 1, "/", len(tasks), flush=True)
for o in objects:
    p = root / o["url"].split("/")[-1]
    parts = root / (p.name + ".parts")
    if not p.exists():
        with p.open("wb") as f:
            for part in sorted(parts.iterdir()):
                with part.open("rb") as src:
                    for b in iter(lambda: src.read(8 << 20), b""):
                        f.write(b)
    assert p.stat().st_size == o["size"]
    assert hashlib.sha256(p.read_bytes()).hexdigest() == o["hash"].split(":")[1]
    print("Verified", p.name, flush=True)
    if parts.exists():
        for part in parts.iterdir():
            part.unlink()
        parts.rmdir()
