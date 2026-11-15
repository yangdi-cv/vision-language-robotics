"""Record exact installed versions and local source fingerprints."""

from pathlib import Path
import json, hashlib, subprocess, importlib.metadata as m, sys

R = Path(__file__).resolve().parents[1]
versions = {
    d.metadata["Name"]: d.version for d in m.distributions() if d.metadata.get("Name")
}
(R / "environment.json").write_text(
    json.dumps(
        {
            "python": sys.version,
            "packages": versions,
            "upstream": json.loads((R / "third_party/sources.json").read_text()),
            "checkpoint": json.loads(
                (R / "checkpoints/pi05_libero/download_verified.json").read_text()
            ),
            "scripts_sha256": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (R / "scripts").glob("*.py")
            },
        },
        indent=2,
    )
)
uv = R / "tools/bin/uv"
with (R / "requirements.lock.txt").open("w") as f:
    subprocess.run(
        [str(uv), "pip", "freeze", "--python", sys.executable], stdout=f, check=True
    )
print("Environment and source fingerprints saved")
