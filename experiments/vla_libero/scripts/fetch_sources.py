"""Fetch immutable upstream source archives without requiring system Git."""

from pathlib import Path
import io, json, tarfile, urllib.request

R = Path(__file__).resolve().parents[1]
SOURCES = {
    "openpi": (
        "Physical-Intelligence/openpi",
        "215abfb217dbac7d5f1273282331b9b1866c0479",
    ),
    "libero": (
        "Lifelong-Robot-Learning/LIBERO",
        "f78abd68ee283de9f9be3c8f7e2a9ad60246e95c",
    ),
    "lerobot": ("huggingface/lerobot", "0cf864870cf29f4738d3ade893e6fd13fbd7cdb5"),
}
meta = {}
for name, (repo, rev) in SOURCES.items():
    target = R / "third_party" / name
    if not target.exists():
        archive = urllib.request.urlopen(
            f"https://codeload.github.com/{repo}/tar.gz/{rev}", timeout=180
        ).read()
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as t:
            prefix = t.getnames()[0]
            # Source archives are pinned; nevertheless reject path traversal and links.
            for member in t.getmembers():
                p = Path(member.name)
                if (
                    p.is_absolute()
                    or ".." in p.parts
                    or member.issym()
                    or member.islnk()
                ):
                    raise ValueError(member.name)
            t.extractall(R / "third_party")
        (R / "third_party" / prefix).rename(target)
    meta[name] = {"repo": repo, "revision": rev}
(R / "third_party/sources.json").write_text(json.dumps(meta, indent=2))
base = R / "third_party/libero/libero/libero"
cfg = R / "configs/libero"
cfg.mkdir(exist_ok=True, parents=True)
(R / "datasets").mkdir(exist_ok=True)
paths = {
    "benchmark_root": base,
    "bddl_files": base / "bddl_files",
    "init_states": base / "init_files",
    "datasets": R / "datasets",
    "assets": base / "assets",
}
(cfg / "config.yaml").write_text("".join(f"{k}: {v}\n" for k, v in paths.items()))
print("Pinned source trees and local paths ready")
