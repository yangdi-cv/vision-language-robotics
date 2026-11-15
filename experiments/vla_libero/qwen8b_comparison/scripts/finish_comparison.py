"""Wait for official checkpoint, verify SHA256, then run eight offline checks."""

from pathlib import Path
import json, time, hashlib, subprocess, sys

R = Path(__file__).resolve().parents[1]
ROOT = R.parents[2]
models = ROOT / "models/qwen3_vl_8b_instruct"
source = json.loads((R / "model_source.json").read_text())
weights = [x for x in source["siblings"] if x["rfilename"].endswith(".safetensors")]
required = [
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
]


def mark(stage, state, **extra):
    (R / "progress.json").write_text(
        json.dumps(
            {"stage": stage, "state": state, "updated_unix_s": time.time(), **extra},
            indent=2,
        )
    )
    print(stage, state, flush=True)


def ready():
    return all(
        (models / x["rfilename"]).exists()
        and (models / x["rfilename"]).stat().st_size == x["lfs"]["size"]
        for x in weights
    ) and all((models / x).exists() for x in required)


def main():
    mark("download", "waiting")
    start = time.monotonic()
    while not ready():
        if time.monotonic() - start > 10800:
            raise TimeoutError("checkpoint download")
        time.sleep(5)
    mark("checksum", "running")
    checked = []
    for x in weights:
        p = models / x["rfilename"]
        h = hashlib.sha256()
        with p.open("rb") as f:
            for b in iter(lambda: f.read(8 << 20), b""):
                h.update(b)
        assert h.hexdigest() == x["lfs"]["sha256"], p
        checked.append(
            {"name": p.name, "bytes": p.stat().st_size, "sha256": h.hexdigest()}
        )
    (R / "checkpoint_verified.json").write_text(
        json.dumps({"revision": source["sha"], "files": checked}, indent=2)
    )
    mark("8b_checks", "running")
    free = int(
        subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
    )
    if free < 21000:
        raise RuntimeError(f"GPU busy: only {free}MiB free, 8B test not started")
    with (R / "logs/8b_checks.log").open("w") as f:
        subprocess.run(
            [
                str(ROOT / ".venvs/qwen-scene/bin/python"),
                str(R / "scripts/run_checks.py"),
                "--model",
                "8b",
            ],
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
        )
    s = json.loads((R / "results/8b/summary.json").read_text())
    if s["status"] != "complete":
        raise RuntimeError(s.get("error", "8B checks failed"))
    mark("comparison", "complete")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        mark("comparison", "failed", error=repr(exc))
        raise
