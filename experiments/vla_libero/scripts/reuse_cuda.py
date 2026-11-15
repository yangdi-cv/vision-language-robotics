"""Copy matching Python-independent CUDA distributions into the new environment.

Optional local installation acceleration. Existing environments are read only;
ordinary install.py can download the same distributions on another machine.
"""

from pathlib import Path
import importlib.metadata as metadata, json, shutil

R = Path(__file__).resolve().parents[1]
dest = R.parents[1] / ".venvs/openpi-libero/lib/python3.11/site-packages"
dest.mkdir(parents=True, exist_ok=True)
records = []
for d in metadata.distributions():
    name = d.metadata.get("Name", "")
    if not name.startswith("nvidia-"):
        continue
    wheel = d.read_text("WHEEL") or ""
    if "Tag: py3-none-" not in wheel:
        raise ValueError(f"Not Python independent: {name}")
    n = 0
    size = 0
    for f in d.files or []:
        if ".." in Path(str(f)).parts or "__pycache__" in str(f):
            continue
        src = Path(d.locate_file(f))
        dst = dest / str(f)
        if not src.is_file():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        n += 1
        size += src.stat().st_size
    records.append(
        {
            "name": name,
            "version": d.version,
            "files": n,
            "bytes": size,
            "read_only_source": str(d.locate_file("")),
        }
    )
    print("Copied", name, d.version, flush=True)
(R / "logs/local_cuda_reuse.json").write_text(json.dumps(records, indent=2))
