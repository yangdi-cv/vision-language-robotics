# ruff: noqa: E402
"""One MuJoCo world serves Nav2 and a policy; no reset between phases."""

from pathlib import Path
import argparse
import collections
import json
import socketserver
import sys
import traceback
import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "scripts"))
from scene import MobileScene


class Engine:
    def __init__(self, a):
        self.a = a
        self.out = a.out.resolve()
        self.out.mkdir(parents=True, exist_ok=False)
        self.s = MobileScene(task=a.task, initial=a.initial)
        self.phase = "navigation"
        self.policy = None
        self.queue = collections.deque()
        self.steps = 0
        self.policy_steps = 0
        self.collisions = []
        self.trace = []
        self.writer = imageio.get_writer(
            self.out / "rollout.mp4", fps=20, codec="libx264", quality=7
        )
        # Only collision geoms enter simulated laser; visual meshes retain their group.
        self.robot_geoms = set()
        for g in range(self.s.model.ngeom):
            b = int(self.s.model.geom_bodyid[g])
            while b and self.s.model.body_id2name(b) != "mobile_carrier":
                b = int(self.s.model.body_parentid[b])
            if b:
                self.robot_geoms.add(g)
                if self.s.model.geom_group[g] == 0:
                    self.s.model.geom_group[g] = 5
        self.rng = np.random.default_rng(930000 + a.task * 100 + a.initial)
        self.instruction = (
            "put the black bowl in the bottom drawer of the cabinet and close it"
            if a.task == 3
            else "put the yellow and white mug in the microwave and close it"
        )
        self.s.callback = self.record
        self.record()
        self.save_manifest()

    def save_manifest(self):
        self.manifest = {
            "task": self.a.task,
            "init": self.a.initial,
            "instruction": self.instruction,
            "dock": self.s.dock.tolist(),
            "model_identity": self.s.initial_model_id,
            "base_model": "planar velocity-actuated, finite-force Omron-inspired approximation",
            "localization": "ideal simulator pose; not SLAM/AMCL",
            "stage": "navigation",
            "no_reset_after_navigation": True,
        }
        (self.out / "protocol.json").write_text(json.dumps(self.manifest, indent=2))

    def record(self):
        s = self.s
        self.steps += 1
        frame = s.sim.render(camera_name="mobile_overview", height=480, width=640)[::-1]
        self.writer.append_data(frame)
        self.trace.append(
            {
                "time": float(s.data.time),
                "base": s.pose().tolist(),
                "arm_q": s.data.qpos[s.arm_indices].tolist(),
                "stage": self.phase,
                "success": bool(s.e.check_success()),
                "object_grasped": bool(
                    s.e.env._check_grasp(
                        s.robot.gripper,
                        s.e.env.get_object("akita_black_bowl_1").contact_geoms,
                    )
                )
                if self.a.task == 3
                else None,
                "object_z": float(
                    s.data.get_body_xpos(
                        s.e.env.get_object("akita_black_bowl_1").root_body
                    )[2]
                )
                if self.a.task == 3
                else None,
            }
        )
        if self.steps % 10 == 0:
            Image.fromarray(frame).save(self.out / "overview.jpg")

    def scan(self, height=0.20):
        s = self.s
        pose = s.pose()
        origin = np.array([*pose[:2], height])
        result = []
        gid = np.zeros(1, np.int32)
        for a in np.linspace(-np.pi, np.pi, 180, endpoint=False):
            direction = np.array([np.cos(a + pose[2]), np.sin(a + pose[2]), 0.0])
            d = mujoco.mj_ray(
                s.model._model,
                s.data._data,
                origin,
                direction,
                np.array([1, 0, 0, 0, 0, 0], np.uint8),
                1,
                -1,
                gid,
            )
            result.append(float(d) if 0 < d < 5 else 5.0)
        return result

    def state(self, scan=False):
        r = {
            "ok": True,
            "time": float(self.s.data.time),
            "pose": self.s.pose().tolist(),
            "velocity": self.s.data.qvel[self.s.dofs].tolist(),
            "dock": self.s.dock.tolist(),
            "phase": self.phase,
            "policy_steps": self.policy_steps,
            "success": bool(self.s.e.check_success()),
            "collision_count": len(self.collisions),
            "command": self.s.command.tolist(),
            "base_control": self.s.data.ctrl[self.s.act].tolist(),
        }
        if scan:
            r["scan"] = self.scan()
            r["scan_high"] = self.scan(0.875)
        return r

    def request(self, q):
        op = q["op"]
        s = self.s
        if op == "tick":
            if self.phase == "navigation":
                s.command[:] = np.clip(
                    q.get("velocity", [0, 0, 0]),
                    [-0.25, -0.25, -0.6],
                    [0.25, 0.25, 0.6],
                )
                s.step()
            elif self.phase == "manipulation":
                if not self.queue:
                    import evaluate as ev

                    inputs = ev.make_input(s.obs, self.instruction)
                    directory = self.out / "policy_inputs"
                    directory.mkdir(exist_ok=True)
                    np.savez_compressed(
                        directory / f"query{len(self.predictions):04d}.npz",
                        image=inputs["observation/image"],
                        wrist=inputs["observation/wrist_image"],
                        state=inputs["observation/state"],
                        instruction=inputs["prompt"],
                    )
                    actions = np.asarray(
                        self.policy.infer(
                            inputs,
                            noise=self.rng.standard_normal((10, 32)).astype(np.float32),
                        )["actions"]
                    )
                    if actions.shape != (10, 7) or not np.isfinite(actions).all():
                        raise ValueError("Invalid model action")
                    self.queue.extend(actions[:5])
                    self.predictions.append(actions)
                action = self.queue.popleft()
                s.step(action)
                self.policy_steps += 1
                self.applied.append(action.copy())
                if s.e.check_success() or self.policy_steps >= 520:
                    self.phase = "success" if s.e.check_success() else "timeout"
                    self.finish()
            elif self.phase not in ["success", "timeout", "collision_stop", "stopped"]:
                s.step()
            if self.phase == "navigation":
                for c in s.data.contact[: s.data.ncon]:
                    a, b = int(c.geom1), int(c.geom2)
                    if (a in self.robot_geoms) != (
                        b in self.robot_geoms
                    ) and c.dist < -0.001:
                        other = b if a in self.robot_geoms else a
                        if s.model.geom_id2name(other) == "floor":
                            continue
                        self.collisions.append(
                            {
                                "time": float(s.data.time),
                                "geom1": s.model.geom_id2name(a),
                                "geom2": s.model.geom_id2name(b),
                                "depth": float(c.dist),
                            }
                        )
                        self.phase = "collision_stop"
                        s.command[:] = 0
                        self.finish()
                        break
            return self.state(scan=True)
        if op == "state":
            return self.state()
        if op == "capture":
            path = self.out / f"camera_{self.steps:05d}.png"
            Image.fromarray(
                s.sim.render(camera_name="agentview", height=512, width=512)[::-1, ::-1]
            ).save(path)
            return {**self.state(), "image": str(path)}
        if op == "manipulate":
            error = float(np.linalg.norm(s.pose()[:2] - s.dock[:2]))
            angle = float(abs(s.pose()[2]))
            if self.phase != "navigation" or error > 0.045 or angle > 0.08:
                raise ValueError(f"Not docked: {error}m {angle}rad")
            s.command[:] = 0
            self.phase = "unfold"
            s.engage_manipulation()
            self.phase = "loading_policy"
            from openpi.training import config
            from openpi.policies import policy_config

            self.policy = policy_config.create_trained_policy(
                config.get_config("pi05_libero"),
                R.parent / "vla_libero/checkpoints/pi05_libero",
                sample_kwargs={"num_steps": 10},
            )
            self.phase = "manipulation"
            self.predictions = []
            self.applied = []
            self.manifest["navigation_arrival_error_m"] = error
            self.manifest["navigation_arrival_yaw_rad"] = angle
            (self.out / "protocol.json").write_text(json.dumps(self.manifest, indent=2))
            return self.state()
        if op == "stop":
            self.phase = "stopped"
            self.finish()
            return self.state()
        raise ValueError("Unknown operation")

    def finish(self):
        (self.out / "result.json").write_text(
            json.dumps(
                {**self.state(), "collisions": self.collisions, "steps": self.steps},
                indent=2,
            )
        )
        (self.out / "trace.json").write_text(json.dumps(self.trace))
        if hasattr(self, "applied"):
            np.savez_compressed(
                self.out / "policy_actions.npz",
                applied=self.applied,
                predictions=self.predictions,
            )
        self.writer.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--socket", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--task", type=int, default=3)
    p.add_argument("--initial", type=int, default=0)
    a = p.parse_args()
    if a.socket.exists():
        raise FileExistsError(a.socket)
    engine = Engine(a)

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            try:
                r = engine.request(json.loads(self.rfile.readline(8 * 1024 * 1024)))
            except Exception:
                r = {"ok": False, "error": traceback.format_exc()}
                print(r["error"], flush=True)
            self.wfile.write((json.dumps(r) + "\n").encode())

    try:
        with socketserver.UnixStreamServer(str(a.socket), Handler) as server:
            print("READY", flush=True)
            server.serve_forever()
    finally:
        engine.writer.close()
        engine.s.close()
        a.socket.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
