"""Sequential, resumable simulation-only experiment pipeline."""

from pathlib import Path
import json, os, time, subprocess

R = Path(__file__).resolve().parents[1]
W = R.parents[1]
PY = W / ".venvs/openpi-libero/bin/python"
LAUNCH = R / "scripts/python"
STATUS = R / "pipeline_status.json"


def status(stage, state, **extra):
    STATUS.write_text(
        json.dumps(
            {"stage": stage, "state": state, "updated_unix_s": time.time(), **extra},
            indent=2,
        )
    )
    print(stage, state, flush=True)


def wait_for(stage, predicate, timeout=21600):
    status(stage, "waiting")
    start = time.monotonic()
    while not predicate():
        if time.monotonic() - start > timeout:
            raise TimeoutError(stage)
        time.sleep(5)


def run(stage, args, launch=False):
    status(stage, "running")
    log = R / "logs" / f"pipeline_{stage}.log"
    cmd = [str(LAUNCH), *map(str, args)] if launch else list(map(str, args))
    with log.open("w") as f:
        proc = subprocess.run(cmd, cwd=W, stdout=f, stderr=subprocess.STDOUT)
    if proc.returncode:
        raise RuntimeError(f"{stage}: exit {proc.returncode}; see {log}")
    status(stage, "complete")


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def main():
    wait_for(
        "large_wheels",
        lambda: (
            len(list((R / "wheelhouse").glob("*.whl"))) == 4
            and not list((R / "wheelhouse").glob("*.parts"))
        ),
    )
    wait_pid = int(os.environ.get("VLA_INSTALL_WAIT_PID", "0"))
    wait_for("interface_install", lambda: wait_pid == 0 or not alive(wait_pid))
    wait_for(
        "pinned_source",
        lambda: (
            json.loads((R / "third_party/sources.json").read_text())
            .get("libero", {})
            .get("revision")
            == "f78abd68ee283de9f9be3c8f7e2a9ad60246e95c"
        ),
    )
    if not os.environ.get("VLA_SKIP_BASE_INSTALL"):
        run("dependencies", [PY, R / "scripts/install.py"])
    run("simulation_dependencies", [PY, R / "scripts/install_simulation.py"])
    run("format", [W / ".venvs/openpi-libero/bin/ruff", "format", R / "scripts"])
    run("interface_tests", [R / "scripts/tests.py"], True)
    run("simulation_check", [R / "scripts/simulation_smoke.py"], True)
    wait_for(
        "official_checkpoint",
        lambda: (R / "checkpoints/pi05_libero/download_verified.json").exists(),
    )
    # Query environment within its interpreter, not the supervisor's Python.
    run("environment", [R / "scripts/environment.py"], True)
    run("policy_smoke", [R / "scripts/evaluate.py", "--phase", "smoke"], True)
    run("evaluation", [R / "scripts/evaluate.py", "--phase", "all"], True)
    run("release_stability", [R / "scripts/release_stability.py"], True)
    run("inference_benchmark", [R / "scripts/benchmark_inference.py"], True)
    run("analysis", [R / "scripts/analyze.py"], True)
    run("verification", [R / "scripts/verify.py"], True)
    run("video", [R / "scripts/video.py"], True)
    run("visual_comparison", [R / "scripts/visual_comparison.py"], True)
    run("report", [R / "scripts/report.py"], True)
    status("all", "complete")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        previous = json.loads(STATUS.read_text()) if STATUS.exists() else {}
        status(previous.get("stage", "startup"), "failed", error=str(e))
        raise
