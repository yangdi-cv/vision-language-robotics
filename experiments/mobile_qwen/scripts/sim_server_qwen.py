# ruff: noqa: E402
"""mobile_vla simulator server with Qwen mission planning and visual task stop.

Extends mobile_vla/scripts/sim_server.Engine without modifying it:
- before navigation, Qwen maps a free-form request to (station, skill) or refuses;
- pi0.5 is loaded once before navigation with the bounded chunk loader;
- manipulation ends when Qwen confirms the task (or the step limit); the official
  LIBERO predicate is recorded but never used to stop. --monitor milestone (default)
  confirms the skill's milestones in order on the current image with text memory of
  earlier confirmations; --monitor task is the earlier whole-task start/current pair check.
The Qwen worker is spawned before anything imports JAX.
"""

from pathlib import Path
import argparse
import json
import subprocess
import sys
import time

Q = Path(__file__).resolve().parents[1]
ROOT = Q.parents[1]
M = ROOT / "experiments/mobile_vla"
V = ROOT / "experiments/vla_libero"

sys.path.insert(0, str(Q / "scripts"))
from skills import STATIONS, SKILLS as SKILL_LIBRARY  # noqa: E402
from durable_log import Journal, atomic_json
from milestones import MilestoneMonitor  # noqa: E402

SKILLS = {k: v["prompt"] for k, v in SKILL_LIBRARY.items()}


class QwenWorker:
    def __init__(self, log_path, journal):
        self.journal = journal
        self.sequence = 0
        self.log = open(log_path, "w")
        self.proc = subprocess.Popen(
            [str(ROOT / ".venvs/qwen-scene/bin/python"), str(Q / "scripts/qwen_worker.py")],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            text=True,
            bufsize=1,
        )
        if not json.loads(self.proc.stdout.readline()).get("ready"):
            raise RuntimeError("Qwen worker failed to start")

    def request(self, payload):
        t0 = time.monotonic()
        self.sequence += 1
        self.journal.append("qwen_request", sequence=self.sequence, payload=payload)
        self.proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("Qwen worker exited")
        r = json.loads(line)
        r["wall_seconds"] = time.monotonic() - t0
        self.journal.append("qwen_response", sequence=self.sequence, payload=r)
        return r

    def close(self):
        if self.proc.poll() is None:
            try:
                self.proc.stdin.write(json.dumps({"stop": True}) + "\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=30)
            except Exception:
                self.proc.kill()
        self.log.close()


def load_policy_bounded():
    sys.path.insert(0, str(V / "skill_bootstrap/scripts"))
    from bounded_restore import restore
    from openpi.training import config, checkpoints
    from openpi.policies import policy as policy_module
    from openpi import transforms

    cfg = config.get_config("pi05_libero")
    data = cfg.data.create(cfg.assets_dirs, cfg.model)
    params, info = restore(V / "checkpoints/pi05_libero/params")
    model = cfg.model.load(params)
    norm = checkpoints.load_norm_stats(V / "checkpoints/pi05_libero/assets", data.asset_id)
    policy = policy_module.Policy(
        model,
        transforms=[
            transforms.InjectDefaultPrompt(None),
            *data.data_transforms.inputs,
            transforms.Normalize(norm, use_quantiles=data.use_quantile_norm),
            *data.model_transforms.inputs,
        ],
        output_transforms=[
            *data.model_transforms.outputs,
            transforms.Unnormalize(norm, use_quantiles=data.use_quantile_norm),
            *data.data_transforms.outputs,
        ],
        sample_kwargs={"num_steps": 10},
    )
    return policy, {k: v for k, v in info.items() if k != "chunk_sha256"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--socket", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--task", type=int, default=3)
    p.add_argument("--initial", type=int, default=0)
    p.add_argument("--request", required=True, help="free-form user instruction given to Qwen")
    p.add_argument("--check-every", type=int, default=20, help="policy steps between Qwen checks")
    p.add_argument("--min-steps", type=int, default=100, help="no Qwen check before this many policy steps")
    p.add_argument("--confirm", type=int, default=2, help="consecutive done votes needed to stop")
    p.add_argument("--max-steps", type=int, default=520)
    p.add_argument("--monitor", choices=["milestone", "task"], default="milestone")
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    journal = Journal(Q / "logs" / (a.out.name + "_journal"))
    journal.append("episode_start", arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()})
    qwen = QwenWorker(a.out.parent / (a.out.name + "_qwen_worker.log"), journal)

    sys.path.insert(0, str(M / "scripts"))
    import numpy as np
    from PIL import Image
    import sim_server as base

    class QEngine(base.Engine):
        def __init__(self, a):
            super().__init__(a)
            self.qwen = qwen
            self.events = []
            self.streak = 0
            self.first_official_success_step = None
            self.stop_reason = None
            self.mission = self.plan_mission()
            if self.mission.get("feasible"):
                self.instruction = SKILLS[self.mission["skill"]]
                self.monitor = MilestoneMonitor(SKILL_LIBRARY[self.mission["skill"]])
                t0 = time.monotonic()
                self.policy, info = load_policy_bounded()
                self.mission["policy_load_seconds"] = time.monotonic() - t0
                self.mission["policy_load"] = info
            else:
                self.phase = "rejected"
            (self.out / "mission.json").write_text(json.dumps(self.mission, indent=2, ensure_ascii=False))
            if self.phase == "rejected":
                self.finish()

        def render(self, camera, h, w, name):
            img = self.s.sim.render(camera_name=camera, height=h, width=w)[::-1]
            if camera == "agentview":
                img = img[:, ::-1]
            path = self.out / "qwen_inputs" / name
            path.parent.mkdir(exist_ok=True)
            Image.fromarray(np.ascontiguousarray(img)).save(path)
            return str(path)

        def plan_mission(self):
            images = [
                self.render("mobile_overview", 480, 640, "mission_overview.png"),
                self.render("agentview", 512, 512, "mission_table.png"),
            ]
            r = self.qwen.request(
                {"op": "mission", "images": images, "instruction": self.a.request, "stations": STATIONS, "skills": SKILLS}
            )
            return {
                "request": self.a.request,
                "feasible": bool(r.get("ok") and r.get("feasible")),
                "station": r.get("station"),
                "skill": r.get("skill"),
                "reason": r.get("reason"),
                "raw": r.get("raw"),
                "error": r.get("error"),
                "qwen_seconds": r.get("wall_seconds"),
            }

        def request(self, q):
            if q["op"] == "manipulate":
                if self.phase == "rejected":
                    raise ValueError("Mission was rejected by the planner")
                s = self.s
                error = float(np.linalg.norm(s.pose()[:2] - s.dock[:2]))
                angle = float(abs(s.pose()[2]))
                if self.phase != "navigation" or error > 0.045 or angle > 0.08:
                    raise ValueError(f"Not docked: {error}m {angle}rad")
                s.command[:] = 0
                self.phase = "unfold"
                s.engage_manipulation()
                self.ref_image = self.render("agentview", 512, 512, "manip_start.png")
                self.phase = "manipulation"
                self.predictions, self.applied = [], []
                self.manifest.update(
                    navigation_arrival_error_m=error,
                    navigation_arrival_yaw_rad=angle,
                    instruction=self.instruction,
                    stop_rule=f"Qwen {self.a.monitor} monitor; official predicate recorded only",
                )
                (self.out / "protocol.json").write_text(json.dumps(self.manifest, indent=2))
                return self.state()
            if q["op"] == "tick" and self.phase == "manipulation":
                self.manipulation_step()
                return self.state(scan=True)
            return super().request(q)

        def manipulation_step(self):
            import evaluate as ev

            s = self.s
            if not self.queue:
                inputs = ev.make_input(s.obs, self.instruction)
                actions = np.asarray(
                    self.policy.infer(inputs, noise=self.rng.standard_normal((10, 32)).astype(np.float32))["actions"]
                )
                if actions.shape != (10, 7) or not np.isfinite(actions).all():
                    raise ValueError("Invalid model action")
                self.queue.extend(actions[:5])
                self.predictions.append(actions)
            action = self.queue.popleft()
            s.step(action)
            self.policy_steps += 1
            self.applied.append(action.copy())
            official = bool(s.e.check_success())
            if official and self.first_official_success_step is None:
                self.first_official_success_step = self.policy_steps
            if self.policy_steps >= self.a.min_steps and self.policy_steps % self.a.check_every == 0:
                img = self.render("agentview", 512, 512, f"check_{self.policy_steps:04d}.png")
                if self.a.monitor == "milestone":
                    r = self.qwen.request(self.monitor.request(img))
                    event = self.monitor.update(self.policy_steps, r)
                    done = self.monitor.done
                else:
                    r = self.qwen.request({"op": "check", "images": [self.ref_image, img], "task": self.instruction})
                    vote = bool(r.get("ok") and r.get("done"))
                    self.streak = self.streak + 1 if vote else 0
                    event = {"policy_step": self.policy_steps, "done_vote": vote, "streak": self.streak,
                             "requirements": r.get("requirements"), "error": r.get("error")}
                    done = self.streak >= self.a.confirm
                event.update(qwen_seconds=r.get("wall_seconds"), shadow_official_success=official)
                self.events.append(event)
                journal.append("visual_check", **event)
                if done:
                    self.stop_reason = "qwen_confirmed"
            if self.stop_reason is None and self.policy_steps >= self.a.max_steps:
                self.stop_reason = "step_limit"
            journal.append("control_step", policy_step=self.policy_steps,
                           action=action.tolist(), joint_positions=np.asarray(s.obs["robot0_joint_pos"]).tolist(),
                           shadow_official_success=official)
            atomic_json(self.out / "progress.json", {
                "phase": self.phase, "policy_steps": self.policy_steps,
                "stop_reason": self.stop_reason,
                "first_official_success_step": self.first_official_success_step,
                "checks": len(self.events), "monitor": self.a.monitor,
                "milestone_memory": getattr(getattr(self, "monitor", None), "memory", None),
                "complete": False,
            })
            if self.stop_reason:
                # Terminal names are what mobile_vla/run_nav2.py waits for; the stop itself is Qwen's.
                self.phase = "success" if official else "timeout"
                self.finish()

        def finish(self):
            (self.out / "qwen_result.json").write_text(
                json.dumps(
                    {
                        "mission": self.mission,
                        "monitor": self.a.monitor,
                        "check_every": self.a.check_every,
                        "milestone_memory": getattr(getattr(self, "monitor", None), "memory", None),
                        "stop_reason": self.stop_reason,
                        "policy_steps": self.policy_steps,
                        "official_success_at_stop": bool(self.s.e.check_success()),
                        "first_official_success_step": self.first_official_success_step,
                        "checks": self.events,
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
            super().finish()
            journal.append("episode_finish", policy_steps=self.policy_steps, stop_reason=self.stop_reason)
            atomic_json(self.out / "progress.json", {"complete": True, "policy_steps": self.policy_steps, "stop_reason": self.stop_reason})

    base.Engine = QEngine
    sys.argv = [sys.argv[0], "--socket", str(a.socket), "--out", str(a.out), "--task", str(a.task), "--initial", str(a.initial)]
    original_parse = argparse.ArgumentParser.parse_args

    def parse_with_extras(self, *args, **kwargs):
        ns = original_parse(self, *args, **kwargs)
        for k in ("request", "check_every", "min_steps", "confirm", "max_steps", "monitor"):
            setattr(ns, k, getattr(a, k))
        return ns

    argparse.ArgumentParser.parse_args = parse_with_extras
    try:
        base.main()
    except BaseException as exc:
        journal.append("episode_error", error=repr(exc))
        raise
    finally:
        qwen.close()


if __name__ == "__main__":
    main()
