"""Download public official checkpoint objects; verify GCS MD5 before rename."""

from pathlib import Path
import json, urllib.request, urllib.parse, concurrent.futures, hashlib, base64, time

R = Path(__file__).resolve().parents[1]
objects = json.loads((R / "configs/checkpoint_objects.json").read_text())["items"]


def fetch(o):
    name = o["name"]
    p = R / "checkpoints/pi05_libero" / name.removeprefix("checkpoints/pi05_libero/")
    p.parent.mkdir(parents=True, exist_ok=True)

    def valid(path):
        if not path.exists() or path.stat().st_size != int(o["size"]):
            return False
        h = hashlib.md5()
        with path.open("rb") as f:
            for b in iter(lambda: f.read(8 << 20), b""):
                h.update(b)
        return base64.b64encode(h.digest()).decode() == o["md5Hash"]

    if valid(p):
        return name
    temp = p.with_name(p.name + ".partial")
    for attempt in range(5):
        try:
            offset = temp.stat().st_size if temp.exists() else 0
            req = urllib.request.Request(
                "https://storage.googleapis.com/openpi-assets/"
                + urllib.parse.quote(name),
                headers={"Range": f"bytes={offset}-"} if offset else {},
            )
            with urllib.request.urlopen(req, timeout=90) as src:
                mode = "ab" if offset and src.status == 206 else "wb"
                with temp.open(mode) as f:
                    for b in iter(lambda: src.read(4 << 20), b""):
                        f.write(b)
            if not valid(temp):
                raise ValueError("MD5/size mismatch")
            temp.rename(p)
            print("OK", name, flush=True)
            return name
        except Exception as e:
            print("retry", attempt, name, str(e), flush=True)
            time.sleep(2)
    raise RuntimeError(name)


if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        list(ex.map(fetch, objects))
