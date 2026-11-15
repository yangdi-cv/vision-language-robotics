"""One Qwen-driven mobile case: plan -> (refuse | Nav2 dock -> pi0.5 -> Qwen stop).

Adapted from mobile_vla/scripts/run_case.py; reuses its Nav2 client unchanged.
Owns and cleans up every process it starts.
"""

from pathlib import Path
import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time

from durable_log import atomic_json

Q = Path(__file__).resolve().parents[1]
ROOT = Q.parents[1]
M = ROOT / "experiments/mobile_vla"
TF_OVERLAY = M / "runtime/geometry2_0_25_24/root"


def stop(proc):
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGINT)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--initial", type=int, required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--domain", type=int, required=True)
    p.add_argument("--request", required=True)
    p.add_argument("--max-steps", type=int, default=520)
    p.add_argument("--monitor", choices=["milestone", "task"], default="milestone")
    p.add_argument("--check-every", type=int, default=10)
    a = p.parse_args()
    out = Q / "results" / a.tag
    if out.exists():
        raise FileExistsError(out)
    journal_dir = Q / "logs" / (a.tag + "_journal")
    if journal_dir.exists():
        raise FileExistsError(journal_dir)
    sock = Path(f"/tmp/mobile_qwen_{a.tag}.sock")
    if sock.exists():
        raise FileExistsError(sock)
    logs = Q / "logs"
    logs.mkdir(exist_ok=True)
    manifest = {
        "arguments": vars(a),
        "initial": a.initial,
        "domain": a.domain,
        "request": a.request,
        "tf_overlay": str(TF_OVERLAY),
        "source_sha256": {
            str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in [*(Q / "scripts").glob("*.py"), *(M / "scripts").glob("*.py"), M / "configs/nav2.yaml"]
        },
    }
    (logs / (a.tag + "_source_at_start.json")).write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    log = (logs / (a.tag + "_server.log")).open("w")
    server = subprocess.Popen(
        [
            str(ROOT / "experiments/vla_libero/scripts/python_shared"),
            str(Q / "scripts/sim_server_qwen.py"),
            "--socket", str(sock),
            "--out", str(out),
            "--initial", str(a.initial),
            "--request", a.request,
            "--max-steps", str(a.max_steps),
            "--monitor", a.monitor,
            "--check-every", str(a.check_every),
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    client = None
    summary = {"tag": a.tag, "request": a.request, "initial": a.initial,
               "monitor": a.monitor, "check_every": a.check_every}
    try:
        end = time.monotonic() + 600
        while not sock.exists():
            if server.poll() is not None:
                raise RuntimeError("Simulator startup failed")
            if time.monotonic() > end:
                raise TimeoutError("Simulator socket not ready")
            time.sleep(0.5)
        mission = json.loads((out / "mission.json").read_text())
        summary["mission"] = {k: mission.get(k) for k in ("feasible", "station", "skill", "reason", "error")}
        if mission["feasible"]:
            env = os.environ.copy()
            env["MOBILE_VLA_DOMAIN"] = str(a.domain)
            env["MOBILE_TF_OVERLAY"] = str(TF_OVERLAY)
            with (logs / (a.tag + "_client.log")).open("w") as f:
                client = subprocess.Popen(
                    [str(M / "scripts/ros-python"), str(M / "scripts/run_nav2.py"), "--socket", str(sock), "--out", str(out)],
                    env=env,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                client.wait(timeout=900)
                if client.returncode != 0:
                    raise RuntimeError(f"Navigation client exited {client.returncode}")
            if (out / "navigation.json").exists():
                nav = json.loads((out / "navigation.json").read_text())
                summary.update(navigation_success=nav.get("navigation_success"), nav_error=nav.get("error"))
                if not nav.get("navigation_success"):
                    raise RuntimeError(f"Navigation failed: {nav.get('error')}")
            if not (out / "qwen_result.json").exists():
                raise RuntimeError("Missing qwen_result.json; episode did not finish")
            if (out / "qwen_result.json").exists():
                r = json.loads((out / "qwen_result.json").read_text())
                summary.update({k: r[k] for k in ("stop_reason", "policy_steps", "official_success_at_stop", "first_official_success_step")})
                summary["qwen_checks"] = len(r["checks"])
    except Exception as e:
        summary["error"] = repr(e)
    finally:
        if client:
            stop(client)
        stop(server)
        log.close()
        sock.unlink(missing_ok=True)
        if out.exists():
            (out / "run_source.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
            atomic_json(out / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 1 if "error" in summary else 0


if __name__ == "__main__":
    sys.exit(main())
