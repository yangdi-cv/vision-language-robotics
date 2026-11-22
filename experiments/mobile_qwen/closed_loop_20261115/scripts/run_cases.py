"""Single-world camera grounding and simulated MoveIt execution. Historical outputs stay frozen."""

from pathlib import Path
from functools import partial
import argparse
import json
import os
import subprocess
import time
import shutil
import resource
import numpy as np
from vla_runtime.transport import call as rpc_call, read_json_line, close_json_worker, stop_process
from vla_runtime.config import load_config
from vla_runtime.records import Journal, atomic_json, error_record, run_metadata, logger
from vla_runtime.schemas import grounding


resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

D = Path(__file__).resolve().parents[1]
ROOT = D.parents[2]
M = ROOT / "experiments/mobile_vla"
V = ROOT / "experiments/vla_libero"

CONFIG = load_config(ROOT)
call = partial(rpc_call, timeout=CONFIG.limit("rpc_seconds"))
LOG = logger(__name__)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--prefix", required=True)
    p.add_argument("--task", choices=["precontact", "combined"], default="precontact")
    p.add_argument("--worker-script", default=None)
    p.add_argument("--modes", default="oracle,lora,base")
    p.add_argument("--initials", default="0")
    p.add_argument("--variants", default="nominal")
    p.add_argument("--navigate", action="store_true")
    p.add_argument("--max-corrections", type=int, default=1)
    a = p.parse_args()
    if a.worker_script is None:
        a.worker_script = "combined_worker.py" if a.task == "combined" else "grounding_worker.py"
    if not a.prefix.replace("-", "").replace("_", "").isalnum() or len(a.prefix) > 40:
        p.error("Prefix must be 1-40 letters, numbers, - or _")
    if a.max_corrections < 0 or a.max_corrections > 10:
        p.error("max-corrections must be in [0,10]")
    if not set(a.modes.split(",")) <= {"oracle", "lora", "base"}:
        p.error("Unknown grounding mode")
    if (
        Path(a.worker_script).name != a.worker_script
        or not (D / "scripts" / a.worker_script).is_file()
    ):
        p.error("worker-script must be an existing filename in scripts/")
    if any(x < 0 for x in map(int, a.initials.split(","))):
        p.error("Initial states must be nonnegative")
    if not all(
        x.replace("-", "").replace("_", "").isalnum() and len(x) < 20 for x in a.variants.split(",")
    ):
        p.error("Invalid variant")
    worker = None
    workerlog = None
    summary = []

    def stop(proc):
        stop_process(proc, grace=8)

    def ground(image, mode):
        nonlocal worker, workerlog
        if worker is None:
            workerlog = (D / "logs" / f"{a.prefix}_qwen.log").open("w")
            worker = subprocess.Popen(
                [str(CONFIG.path("qwen_python")), str(D / "scripts" / a.worker_script)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=workerlog,
                text=True,
                bufsize=1,
                start_new_session=True,
                env={
                    **os.environ,
                    "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
                },
            )
            ready = read_json_line(worker, CONFIG.limit("worker_ready_seconds"))
            if ready.get("ready") is not True:
                raise RuntimeError("Worker not ready")
            atomic_json(D / "logs" / f"{a.prefix}_worker_metadata.json", ready)
        worker.stdin.write(json.dumps({"image": image, "mode": mode}) + "\n")
        worker.stdin.flush()
        response = read_json_line(worker, CONFIG.limit("worker_response_seconds"))
        grounding(response)
        return response

    try:
        for initial in map(int, a.initials.split(",")):
            for variant in a.variants.split(","):
                for mode in a.modes.split(","):
                    tag = f"{a.prefix}_{variant}_i{initial:02d}_{mode}"
                    out = D / "results" / tag
                    logs = D / "logs" / tag
                    sock = Path("/tmp/" + tag + ".sock")
                    if out.exists() or sock.exists():
                        raise FileExistsError(f"Existing run: {tag}")
                    logs.mkdir(parents=True, exist_ok=False)
                    journal = Journal(logs / "progress.jsonl")
                    atomic_json(
                        logs / "run_metadata.json",
                        run_metadata(
                            ROOT,
                            parameters=vars(a),
                            worker_script=a.worker_script,
                            configuration=CONFIG.values,
                            model="worker-reported",
                            adapter="worker-reported",
                        ),
                    )
                    journal.append("starting", tag=tag)
                    env = os.environ.copy()
                    env["MOBILE_VLA_DOMAIN"] = str(int(CONFIG.limit("ros_domain")))
                    env["MOBILE_TF_OVERLAY"] = ""
                    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
                    processes = []
                    streams = []
                    row = {
                        "task": a.task,
                        "tag": tag,
                        "mode": mode,
                        "initial": initial,
                        "variant": variant,
                        "navigate": a.navigate,
                        "success": False,
                        "precontact_success": False,
                        "success_criterion": "visual residual <=25mm; true precontact error <=30mm; no contacts; single unchanged physics instance",
                        "attempts": [],
                    }

                    def launch(cmd, name, overlay=False):
                        f = (logs / (name + ".log")).open("w")
                        streams.append(f)
                        e = env.copy()
                        if overlay:
                            e["MOBILE_TF_OVERLAY"] = str(M / "runtime/geometry2_0_25_24/root")
                        z = subprocess.Popen(
                            cmd,
                            cwd=ROOT,
                            env=e,
                            stdout=f,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
                        processes.append(z)
                        return z

                    try:
                        cmd = [
                            str(CONFIG.path("policy_python")),
                            str(
                                D
                                / "scripts"
                                / (
                                    "sim_combined.py"
                                    if a.task == "combined"
                                    else "sim_precontact.py"
                                )
                            ),
                            "--socket",
                            str(sock),
                            "--out",
                            str(out),
                            "--initial",
                            str(initial),
                            "--variant",
                            variant,
                        ]
                        if not a.navigate:
                            cmd.append("--start-docked")
                        sim = launch(cmd, "sim")
                        end = time.monotonic() + 90
                        while not sock.exists():
                            if sim.poll() is not None:
                                raise RuntimeError("Simulator exited")
                            if time.monotonic() > end:
                                raise TimeoutError("Simulator startup")
                            time.sleep(0.2)
                        if a.navigate:
                            nav = launch(
                                [
                                    str(CONFIG.path("ros_python")),
                                    str(D / "scripts/nav_only.py"),
                                    "--socket",
                                    str(sock),
                                    "--out",
                                    str(out),
                                ],
                                "nav",
                                overlay=True,
                            )
                            nav.wait(timeout=350)
                            if nav.returncode:
                                raise RuntimeError(f"Navigation process exit {nav.returncode}")
                            n = json.loads((out / "navigation.json").read_text())
                            row["navigation"] = n
                            if not n.get("navigation_success"):
                                raise RuntimeError("Navigation did not complete")
                        bridge = launch(
                            [
                                str(CONFIG.path("ros_python")),
                                str(D / "scripts/physics_bridge.py"),
                                "--socket",
                                str(sock),
                                "--out",
                                str(out),
                            ],
                            "bridge",
                        )
                        moveit = launch(
                            [
                                str(CONFIG.path("ros_python")),
                                str(CONFIG.path("ros_prefix") / "bin/ros2"),
                                "launch",
                                str(D / "scripts/moveit.launch.py"),
                            ],
                            "moveit",
                        )
                        # The same model is queried again after execution; official goal geometry never decides a correction.
                        for attempt in range(a.max_corrections + 1):
                            journal.append("grounding", attempt=attempt)
                            capture = call(sock, "capture_grounding")
                            prediction = (
                                ground(capture["image"], mode)
                                if mode != "oracle"
                                else {"ok": True, "mode": "oracle"}
                            )
                            (out / f"prediction_{attempt:02d}.json").write_text(
                                json.dumps(prediction, indent=2)
                            )
                            call(
                                sock,
                                "ground_target",
                                mode=mode,
                                **(
                                    {"point_2d": prediction["point_2d"]} if mode != "oracle" else {}
                                ),
                            )
                            planout = out / "plans" / f"attempt{attempt:02d}"
                            planout.mkdir(parents=True)
                            shutil.copyfile(out / "goal.json", planout / "goal.json")
                            client = launch(
                                [
                                    str(CONFIG.path("ros_python")),
                                    str(D / "scripts/plan_execute.py"),
                                    "--socket",
                                    str(sock),
                                    "--out",
                                    str(planout),
                                ],
                                f"plan_{attempt:02d}",
                            )
                            client.wait(timeout=220)
                            if client.returncode:
                                raise RuntimeError(
                                    f"Planning process exit {client.returncode}; inspect plan log and moveit_result.json"
                                )
                            plan = json.loads((planout / "moveit_result.json").read_text())
                            audit = call(sock, "evaluate_precontact")
                            event = {"planning_success": plan.get("success"), "audit": audit}
                            row["attempts"].append(event)
                            journal.append("attempt_result", attempt=attempt, result=event)
                            atomic_json(logs / "case_snapshot.json", row)
                            if not plan.get("success"):
                                raise RuntimeError(plan.get("error", "MoveIt failed"))
                            if mode == "oracle":
                                row["visual_confirmed"] = None
                                break
                            after = call(sock, "capture_grounding")
                            check = ground(after["image"], mode)
                            (out / f"visual_check_{attempt:02d}.json").write_text(
                                json.dumps(check, indent=2)
                            )
                            checkgoal = call(
                                sock, "ground_target", mode=mode, point_2d=check["point_2d"]
                            )
                            actual = np.array(after["state"]["tcp_p"])
                            desired = np.array(checkgoal["base_state"]["base_p"]) + np.array(
                                checkgoal["base_state"]["base_R"]
                            ) @ np.array(checkgoal["target_position"])
                            residual = float(np.linalg.norm(actual - desired))
                            event["visual_residual_m"] = residual
                            row["visual_confirmed"] = residual <= CONFIG.limit("visual_residual_m")
                            if row["visual_confirmed"]:
                                break
                        audit = call(sock, "evaluate_precontact")
                        row["final_audit"] = audit
                        row["libero_original_task_success"] = call(sock, "state")["success"]
                        row["success"] = bool(
                            row.get("visual_confirmed", True) is not False
                            and audit["shadow_true_precontact_error_m"]
                            <= CONFIG.limit("precontact_error_m")
                            and not audit["any_arm_environment_contact"]
                            and not audit["any_self_contact"]
                            and audit["physics_id_unchanged"]
                        )
                        row["precontact_success"] = row["success"]
                        if a.task == "combined":
                            if not row["precontact_success"]:
                                raise RuntimeError("Precontact failed; policy handover skipped")
                            # Stop MoveIt and its bridge before handing sole actuator ownership back to OSC.
                            for proc in [moveit, bridge]:
                                stop(proc)
                            resumed = call(sock, "resume_policy", timeout=240)
                            if not resumed.get("ok", True):
                                raise RuntimeError(resumed.get("error"))
                            from vla_runtime.milestones import MilestoneMonitor
                            import runpy

                            SKILLS = runpy.run_path(str(D.parent / "scripts/skills.py"))["SKILLS"]
                            monitor = MilestoneMonitor(SKILLS["bowl_to_bottom_drawer_and_close"])
                            checks = []
                            for step in range(1, 521):
                                state = call(sock, "policy_tick", timeout=120)
                                if not state.get("ok", True):
                                    raise RuntimeError(state.get("error"))
                                if step % 10 == 0:
                                    cap = call(sock, "capture_grounding")
                                    payload = monitor.request(cap["image"])
                                    worker.stdin.write(json.dumps(payload) + "\n")
                                    worker.stdin.flush()
                                    response = read_json_line(
                                        worker, CONFIG.limit("worker_response_seconds")
                                    )
                                    event = monitor.update(step, response)
                                    checks.append(event)
                                    atomic_json(out / "milestone_checks.json", checks)
                                    journal.append("milestone_check", step=step, event=event)
                                    atomic_json(
                                        logs / "case_snapshot.json",
                                        {**row, "policy_steps": step, "last_check": event},
                                    )
                                    if monitor.done:
                                        break
                            final = call(sock, "state")
                            row["combined_official_success"] = bool(final["success"])
                            row["visual_milestones_complete"] = monitor.done
                            row["policy_steps"] = step
                            row["success"] = bool(row["combined_official_success"] and monitor.done)
                            row["combined_stop_reason"] = (
                                "qwen_milestones" if monitor.done else "step_limit"
                            )
                    except Exception as e:
                        row.update(error_record(e))
                        LOG.exception("Case failed: %s", tag)
                        journal.append("failed", **error_record(e))
                    finally:
                        if sock.exists():
                            try:
                                call(sock, "close_episode")
                            except Exception as exc:
                                row["close_error"] = error_record(exc)
                        for z in reversed(processes):
                            try:
                                stop(z)
                            except Exception as exc:
                                row.setdefault("cleanup_errors", []).append(error_record(exc))
                        row["process_exit_codes"] = [z.returncode for z in processes]
                        row["cleanup_ok"] = not any(
                            "Segmentation fault" in f.read_text(errors="replace")
                            for f in logs.glob("*.log")
                        )
                        for f in streams:
                            f.close()
                        sock.unlink(missing_ok=True)
                        journal.append("finished", success=row["success"])
                        journal.close()
                        atomic_json(logs / "case_snapshot.json", row)
                        if out.exists():
                            atomic_json(out / "case_result.json", row)
                        summary.append(row)
                        atomic_json(D / "results" / f"{a.prefix}_summary.json", summary)
                        print(json.dumps(row), flush=True)
    finally:
        if worker:
            close_json_worker(worker)
        if workerlog:
            workerlog.close()
    return int(any(not row["success"] for row in summary))


if __name__ == "__main__":
    raise SystemExit(main())
