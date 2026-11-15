"""Assemble pending objects and verify size, CRC32C, available MD5 and SHA256."""

from pathlib import Path
import json, hashlib, base64, shutil
import google_crc32c

R = Path(__file__).resolve().parents[1]
root = R / "checkpoints/pi05_libero"
objects = json.loads((R / "configs/checkpoint_objects.json").read_text())["items"]
records = []
for o in objects:
    p = root / o["name"].removeprefix("checkpoints/pi05_libero/")
    parts = p.with_name(p.name + ".parts")
    temp = p.with_name(p.name + ".partial")
    if not p.exists() or p.stat().st_size != int(o["size"]):
        assert parts.exists(), o["name"]
        with temp.open("ab") as out:
            for part in sorted(parts.iterdir()):
                assert part.is_file() and part.suffix != ".tmp", part
                with part.open("rb") as src:
                    shutil.copyfileobj(src, out, 8 << 20)
        assert temp.stat().st_size == int(o["size"])
        temp.rename(p)
    assert p.stat().st_size == int(o["size"])
    crc = google_crc32c.Checksum()
    md5 = hashlib.md5()
    sha = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            crc.update(b)
            md5.update(b)
            sha.update(b)
    assert base64.b64encode(crc.digest()).decode() == o["crc32c"], ("CRC32C", o["name"])
    if "md5Hash" in o:
        assert base64.b64encode(md5.digest()).decode() == o["md5Hash"], (
            "MD5",
            o["name"],
        )
    records.append(
        {
            "name": o["name"],
            "bytes": p.stat().st_size,
            "crc32c": o["crc32c"],
            "md5": o.get("md5Hash"),
            "sha256": sha.hexdigest(),
        }
    )
    print("Verified", o["name"], flush=True)
    if parts.exists():
        shutil.rmtree(parts)
(root / "download_verified.json").write_text(
    json.dumps(
        {
            "objects": len(objects),
            "bytes": sum(int(o["size"]) for o in objects),
            "crc32c_all_checked": True,
            "md5_checked_where_available": True,
            "md5_object_count": sum("md5Hash" in o for o in objects),
            "files": records,
        },
        indent=2,
    )
)
