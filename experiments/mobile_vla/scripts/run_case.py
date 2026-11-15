"""Run an isolated case, preserving all failed attempts and owning process cleanup."""

from pathlib import Path
import argparse
import os
import sys
import subprocess
import signal
import time
import json
import hashlib

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "scripts"))


def stop(proc):
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGINT)
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--initial", type=int, required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--domain", type=int, required=True)
    p.add_argument("--wall-clock", action="store_true")
    p.add_argument("--tf-overlay", type=Path)
    a = p.parse_args()
    out = R / "results" / a.tag
    if out.exists():
        raise FileExistsError(out)
    sock = Path(f"/tmp/mobile_vla_{a.tag}.sock")
    if sock.exists():
        raise FileExistsError(sock)
    logs = R / "logs"
    log = (logs / (a.tag + "_server.log")).open("w")
    selected = R / ("configs/nav2_wall.yaml" if a.wall_clock else "configs/nav2.yaml")
    source_manifest = {
        "domain": a.domain,
        "tf_overlay": str(a.tf_overlay) if a.tf_overlay else None,
        "wall_clock": a.wall_clock,
        "initial": a.initial,
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (R / "scripts").glob("*.py")
        },
        "config_path": str(selected.relative_to(R)),
        "config_sha256": hashlib.sha256(selected.read_bytes()).hexdigest(),
    }
    (logs / (a.tag + "_source_at_start.json")).write_text(
        json.dumps(source_manifest, indent=2)
    )
    proc = subprocess.Popen(
        [
            str(R.parent / "vla_libero/scripts/python_shared"),
            str(R / "scripts/sim_server.py"),
            "--socket",
            str(sock),
            "--out",
            str(out),
            "--initial",
            str(a.initial),
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    client = None
    try:
        end = time.monotonic() + 120
        while not sock.exists():
            if proc.poll() is not None:
                raise RuntimeError("Simulator startup failed")
            if time.monotonic() > end:
                raise TimeoutError("Simulator socket not ready")
            time.sleep(0.2)
        env = os.environ.copy()
        env["MOBILE_VLA_DOMAIN"] = str(a.domain)
        if a.tf_overlay:
            env["MOBILE_TF_OVERLAY"] = str(a.tf_overlay.resolve())
        with (logs / (a.tag + "_client.log")).open("w") as f:
            client = subprocess.Popen(
                [
                    str(R / "scripts/ros-python"),
                    str(R / "scripts/run_nav2.py"),
                    "--socket",
                    str(sock),
                    "--out",
                    str(out),
                    *(["--wall-clock"] if a.wall_clock else []),
                ],
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            client.wait(timeout=480)
        if (out / "navigation.json").exists():
            r = json.loads((out / "navigation.json").read_text())
            print(
                json.dumps(
                    {k: v for k, v in r.items() if k not in ["arrival", "final"]}
                ),
                flush=True,
            )
    finally:
        if client:
            stop(client)
        stop(proc)
        log.close()
        sock.unlink(missing_ok=True)
        if out.exists():
            (out / "run_source.json").write_text(json.dumps(source_manifest, indent=2))


if __name__ == "__main__":
    main()
