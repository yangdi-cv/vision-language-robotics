from pathlib import Path
import urllib.request, tarfile, io, tomllib, subprocess

R = Path(__file__).resolve().parents[1]
W = R.parents[1]
uv = R / "tools/bin/uv"
py = W / ".venvs/openpi-libero/bin/python"
repo = "huggingface/lerobot"
rev = "0cf864870cf29f4738d3ade893e6fd13fbd7cdb5"
target = R / "third_party/lerobot"
if not target.exists():
    print("Fetching pinned LeRobot dependency", flush=True)
    b = urllib.request.urlopen(
        f"https://codeload.github.com/{repo}/tar.gz/{rev}", timeout=120
    ).read()
    with tarfile.open(fileobj=io.BytesIO(b), mode="r:gz") as t:
        prefix = t.getnames()[0]
        t.extractall(R / "third_party")
    (R / "third_party" / prefix).rename(target)
proj = tomllib.loads((R / "third_party/openpi/pyproject.toml").read_text())
lock = tomllib.loads((R / "third_party/openpi/uv.lock").read_text())
constraints = []
from collections import Counter

counts = Counter(x["name"] for x in lock["package"])
for x in lock["package"]:
    if "registry" in x.get("source", {}) and counts[x["name"]] == 1:
        constraints.append(x["name"] + "==" + x["version"])
(R / "configs/upstream_constraints.txt").write_text("\n".join(sorted(set(constraints))))
deps = [
    x for x in proj["project"]["dependencies"] if x not in ["lerobot", "openpi-client"]
]
deps.extend(str(p) for p in (R / "wheelhouse").glob("*.whl"))
cmd = [
    str(uv),
    "pip",
    "install",
    "--python",
    str(py),
    "-c",
    str(R / "configs/upstream_constraints.txt"),
    *deps,
    str(R / "third_party/openpi/packages/openpi-client"),
    str(target),
    "h5py",
    "imageio-ffmpeg",
    "pytest",
    "matplotlib",
    "pypdf",
    "ruff",
]
subprocess.run(cmd, check=True)
subprocess.run(
    [
        str(uv),
        "pip",
        "install",
        "--python",
        str(py),
        "--no-deps",
        "-e",
        str(R / "third_party/openpi"),
    ],
    check=True,
)
