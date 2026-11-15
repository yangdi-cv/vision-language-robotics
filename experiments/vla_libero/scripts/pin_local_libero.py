"""Reconcile existing source tree with OpenPI's exact submodule, verify Git blobs."""

from pathlib import Path
import urllib.request, json, hashlib, concurrent.futures

R = Path(__file__).resolve().parents[1]
rev = "f78abd68ee283de9f9be3c8f7e2a9ad60246e95c"
root = R / "third_party/libero"
a = json.load(
    urllib.request.urlopen(
        f"https://api.github.com/repos/Lifelong-Robot-Learning/LIBERO/git/trees/{rev}?recursive=1",
        timeout=90,
    )
)
assert not a.get("truncated")
blobs = [v for v in a["tree"] if v["type"] == "blob"]
expected = {v["path"] for v in blobs}
missing = []


def check(v):
    p = root / v["path"]
    if p.exists():
        b = p.read_bytes()
        actual = hashlib.sha1(b"blob " + str(len(b)).encode() + b"\0" + b).hexdigest()
        if actual == v["sha"]:
            return True
    return False


for v in blobs:
    if not check(v):
        missing.append(v)
print("Files", len(blobs), "to replace", len(missing), flush=True)


def fetch(v):
    p = root / v["path"]
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(
        urllib.request.urlopen(
            f"https://raw.githubusercontent.com/Lifelong-Robot-Learning/LIBERO/{rev}/{v['path']}",
            timeout=90,
        ).read()
    )
    assert check(v)
    print("Updated", v["path"], flush=True)


with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
    list(ex.map(fetch, missing))
extra = [
    p
    for p in root.rglob("*")
    if p.is_file() and str(p.relative_to(root)) not in expected
]
for p in extra:
    p.unlink()
assert all(check(v) for v in blobs)
meta = json.loads((R / "third_party/sources.json").read_text())
meta["libero"] = {
    "repo": "Lifelong-Robot-Learning/LIBERO",
    "revision": rev,
    "git_blobs_verified": len(blobs),
}
meta["lerobot"] = {
    "repo": "huggingface/lerobot",
    "revision": "0cf864870cf29f4738d3ade893e6fd13fbd7cdb5",
}
(R / "third_party/sources.json").write_text(json.dumps(meta, indent=2))
p = R / "scripts/fetch_sources.py"
p.write_text(p.read_text().replace("8f1084e3132a39270c3a13ebe37270a43ece2a01", rev))
(R / "configs/libero_git_tree.json").write_text(
    json.dumps(
        {
            "revision": rev,
            "tree_sha": a["sha"],
            "verified_blobs": len(blobs),
            "changed_files": [v["path"] for v in missing],
        },
        indent=2,
    )
)
print("Exact pinned source verified", flush=True)
