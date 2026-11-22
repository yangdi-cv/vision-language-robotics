"""Configurable SSH-key-only Jetson/PC simulation validation.

The administrator prepares swap separately. This command never changes swap,
power modes, login credentials or a machine's system services.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from vla_runtime.config import load_config
from vla_runtime.records import Journal, atomic_json, error_record, logger, run_metadata
from vla_runtime.remote import SSH
from vla_runtime.transport import stop_process


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True, help="New run name; never reuse a result tag")
    parser.add_argument("--initials", default="2,3,4")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.tag.replace("-", "").replace("_", "").isalnum():
        parser.error("Tag must contain only letters, numbers, - and _")
    config = load_config()
    ssh = SSH(str(config.remote("host")))
    remote = str(config.remote("stage_root"))
    remote_python = str(config.remote("python"))
    if not remote.startswith("/") or not remote_python.startswith("/"):
        parser.error("Remote paths must be absolute")
    stage = Path(__file__).resolve().parents[1]
    output = stage / "results" / args.tag
    remote_output = remote + "/runs/" + args.tag
    unit = "vla-vision-" + args.tag
    service = [remote_python, remote + "/scripts/vision_service.py"]
    command = ssh.command(
        [
            "env",
            "VLA_SERVICE_OUT=" + remote_output,
            "systemd-run",
            "--user",
            "--wait",
            "--pipe",
            "--collect",
            "--unit",
            unit,
            "-p",
            "MemoryHigh=5G",
            "-p",
            "MemoryMax=6G",
            "-p",
            "OOMPolicy=stop",
            "--setenv=VLA_SERVICE_OUT=" + remote_output,
            "--setenv=VLA_SERVICE_PORT=" + str(int(config.remote("remote_port"))),
            *service,
        ]
    )
    tunnel_command = ssh.tunnel(int(config.remote("local_port")), int(config.remote("remote_port")))
    if args.dry_run:
        print(
            json.dumps(
                {
                    "service": command,
                    "tunnel": tunnel_command,
                    "output": str(output),
                    "remote_output": remote_output,
                },
                indent=2,
            )
        )
        return 0
    output.mkdir(parents=True, exist_ok=False)
    journal = Journal(output / "progress.jsonl")
    atomic_json(
        output / "metadata.json",
        run_metadata(
            config.root,
            parameters=vars(args),
            deployment="NF4 2B LoRA; remote manifest records weights and prompt",
        ),
    )
    log = logger(__name__)
    job = tunnel = None
    result = {"success": False}
    env = {
        **os.environ,
        "VLA_SERVICE_OUT": remote_output,
        "PYTHONPATH": str(config.root) + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }
    url = f"http://127.0.0.1:{int(config.remote('local_port'))}"
    with (output / "service.log").open("w") as stream:
        try:
            # This is a precondition, not an automatic privileged swapoff.
            check = ssh.command(
                [
                    "python3",
                    "-c",
                    "from pathlib import Path; import sys; sys.exit(len(Path('/proc/swaps').read_text().splitlines()) != 1)",
                ]
            )
            subprocess.run(check, check=True, timeout=15)
            journal.append("preflight_passed")
            # Deploy only code; model artifacts and input manifest must be staged separately.
            subprocess.run(
                [
                    "rsync",
                    "-a",
                    "-e",
                    shlex.join(ssh.options()),
                    "--include=*/",
                    "--include=*.py",
                    "--exclude=*",
                    str(stage / "scripts") + "/",
                    ssh.host + ":" + remote + "/scripts/",
                ],
                check=True,
                timeout=120,
            )
            job = subprocess.Popen(
                command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
            )
            tunnel = subprocess.Popen(
                tunnel_command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
            )
            deadline = time.monotonic() + 600
            while True:
                if job.poll() is not None or tunnel.poll() is not None:
                    raise RuntimeError(
                        "Service or tunnel exited during startup; inspect service.log"
                    )
                try:
                    with urllib.request.urlopen(url, timeout=2) as response:
                        if json.load(response).get("ready") is True:
                            break
                except (OSError, ValueError):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Vision service startup deadline")
                    time.sleep(1)
            journal.append("service_ready")
            manifest = json.loads((stage / "input_manifest.json").read_text())
            data = (stage / "validation" / manifest["images"][0]["file"]).read_bytes()
            request = urllib.request.Request(
                url,
                data=json.dumps(
                    {
                        "mode": "lora",
                        "phase": "warmup",
                        "image_base64": base64.b64encode(data).decode(),
                    }
                ).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=90) as response:
                warmup = json.load(response)
            if warmup.get("ok") is not True:
                raise RuntimeError(f"Warmup failed: {warmup}")
            atomic_json(output / "warmup.json", warmup)
            case = config.root / "experiments/mobile_qwen/closed_loop_20261115/scripts/run_cases.py"
            with (output / "cases.log").open("w") as cases:
                subprocess.run(
                    [
                        sys.executable,
                        str(case),
                        "--prefix",
                        args.tag,
                        "--modes",
                        "lora",
                        "--initials",
                        args.initials,
                        "--variants",
                        "nominal",
                        "--navigate",
                        "--worker-script",
                        "remote_grounding_worker.py",
                    ],
                    env=env,
                    check=True,
                    cwd=config.root,
                    stdout=cases,
                    stderr=subprocess.STDOUT,
                    timeout=1800,
                )
            result["success"] = True
            journal.append("cases_finished")
        except Exception as exc:
            result.update(error_record(exc))
            journal.append("failed", **error_record(exc))
            log.exception("Joint service run failed")
        finally:
            try:
                if job is not None:
                    subprocess.run(
                        ssh.command(["systemctl", "--user", "stop", unit]),
                        check=True,
                        timeout=30,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                    )
            except Exception as exc:
                result["cleanup_error"] = error_record(exc)
            finally:
                for process in (tunnel, job):
                    if process is not None:
                        stop_process(process)
                # Retrieve progress even after failure. A failed copy remains visible.
                try:
                    copy = subprocess.run(
                        [
                            "rsync",
                            "-a",
                            "-e",
                            shlex.join(ssh.options()),
                            ssh.host + ":" + remote_output + "/",
                            str(output / "remote") + "/",
                        ],
                        timeout=60,
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    result["artifact_copy_returncode"] = copy.returncode
                    result["artifact_copy_stderr"] = copy.stderr
                    if copy.returncode == 0:
                        resources = json.loads((output / "remote/resources.json").read_text())
                        result["valid_without_swap"] = resources.get("valid_without_swap") is True
                        if not result["valid_without_swap"]:
                            result["success"] = False
                except Exception as exc:
                    result["artifact_copy_returncode"] = -1
                    result["artifact_copy_error"] = error_record(exc)
                finally:
                    atomic_json(output / "result.json", result)
                    journal.close()
    return (
        0
        if result["success"]
        and result.get("artifact_copy_returncode") == 0
        and "cleanup_error" not in result
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
