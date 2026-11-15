# ruff: noqa: E402
"""Mobile extension of a LIBERO scene; original environments are never modified.

The planar velocity-actuated base follows robosuite's MIT-licensed Omron model
(see third_party/sources.json). Primitive geometry replaces mesh visuals; no
wheel-slip or suspension claim is made. All runtime motion uses actuators.
"""

from pathlib import Path
import sys
import types
import xml.etree.ElementTree as ET
import numpy as np

R = Path(__file__).resolve().parents[1]
V = R.parent / "vla_libero"
sys.path.insert(0, str(V / "scripts"))
import evaluate as ev
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

STOW = np.array([0.0, -0.6, 0.0, -2.1, 0.0, 1.7, 0.78])


class MobileScene:
    def __init__(self, task=3, initial=0, mobile_start=(-0.7, -0.8, 0.0), stow=True):
        suite = benchmark.get_benchmark_dict()["libero_10"]()
        t = suite.get_task(task)
        self.e = OffScreenRenderEnv(
            bddl_file_name=str(
                Path(get_libero_path("bddl_files")) / t.problem_folder / t.bddl_file
            ),
            camera_heights=256,
            camera_widths=256,
            ignore_done=True,
        )
        self.e.seed(7)
        self.e.reset()
        self.e.set_init_state(np.asarray(ev.initial_states(suite, task)[initial]))
        for _ in range(10):
            self.e.step(ev.DUMMY.tolist())
        old = self.e.sim
        saved = {
            name: (
                np.array(old.data.get_joint_qpos(name)).copy(),
                np.array(old.data.get_joint_qvel(name)).copy(),
            )
            for name in old.model.joint_names
        }
        self.original_xml = old.model.get_xml()
        root = ET.fromstring(self.original_xml)
        w = root.find("worldbody")
        arm = w.find("body[@name='robot0_base']")
        pos = np.fromstring(arm.get("pos"), sep=" ")
        if pos[2] < 0.8:
            raise ValueError(
                "This fixture mounts Panda on a solid table; mobile extension rejected. Use kitchen tasks 2,3,9."
            )
        # Replace the fixed Rethink stand rather than carrying its 274kg geometry.
        # Keep its root frame for compatibility; new mobile support supplies mass/collision.
        for mount in arm.findall(".//body[@name='mount0_base']"):
            mount.clear()
            mount.set("name", "mount0_base")
        self.dock = np.array([pos[0] - 0.10, pos[1], 0.0])
        arm.set("pos", f"0 0 {pos[2]}")
        w.remove(arm)
        carrier = ET.SubElement(
            w, "body", name="mobile_carrier", pos=f"{self.dock[0]} {self.dock[1]} 0"
        )
        for name, axis, kind in [
            ("x", "1 0 0", "slide"),
            ("y", "0 1 0", "slide"),
            ("yaw", "0 0 1", "hinge"),
        ]:
            ET.SubElement(
                carrier,
                "joint",
                name="mobile_" + name,
                type=kind,
                axis=axis,
                limited="false",
                damping="5",
                armature="1",
            )
            ET.SubElement(
                root.find("actuator"),
                "velocity",
                name="mobile_" + name + "_act",
                joint="mobile_" + name,
                kv="150" if name == "yaw" else "1500",
                ctrllimited="true",
                ctrlrange="-0.5 0.5",
                forcelimited="true",
                forcerange="-600 600",
            )
        ET.SubElement(
            carrier,
            "geom",
            name="mobile_chassis",
            type="cylinder",
            size=".22 .15",
            pos="0 0 .155",
            mass="40",
            rgba=".13 .19 .26 1",
            friction=".8 .005 .0001",
        )
        ET.SubElement(
            carrier,
            "geom",
            name="mobile_column",
            type="cylinder",
            size=f".08 {(pos[2] - 0.32) / 2}",
            pos=f"0 0 {(pos[2] + 0.32) / 2}",
            mass="8",
            rgba=".35 .4 .46 1",
        )
        carrier.append(arm)
        # Fixed obstacles share the same collision world as the arm and objects.
        self.obstacles = [(-1.05, -0.4, 0.10, 0.10)]
        for j, (x, y, hx, hy) in enumerate(self.obstacles):
            ET.SubElement(
                w,
                "geom",
                name=f"nav_obstacle_{j}",
                type="box",
                pos=f"{x} {y} .30",
                size=f"{hx} {hy} .30",
                rgba=".55 .28 .18 1",
            )
        campos = np.array([-1.6, -1.6, 2.4])
        target = np.array([-0.35, 0.0, 0.4])
        z = campos - target
        z /= np.linalg.norm(z)
        right = np.cross([0, 0, 1], z)
        right /= np.linalg.norm(right)
        up = np.cross(z, right)
        ET.SubElement(
            w,
            "camera",
            name="mobile_overview",
            pos=" ".join(map(str, campos)),
            xyaxes=" ".join(map(str, np.r_[right, up])),
        )
        xml = ET.tostring(root, encoding="unicode")
        self.e.env.robots[0].robot_model.mount = None
        self.e.env.reset_from_xml_string(xml)
        self.sim = self.e.sim
        self.model = self.sim.model
        self.data = self.sim.data
        for name, (q, v) in saved.items():
            self.data.set_joint_qpos(name, q)
            self.data.set_joint_qvel(name, v)
        self.arm_q_initial = self.data.qpos[
            self.e.env.robots[0]._ref_joint_pos_indexes
        ].copy()
        self.joints = np.array(
            [self.model.get_joint_qpos_addr("mobile_" + x) for x in ["x", "y", "yaw"]]
        )
        self.dofs = np.array(
            [self.model.get_joint_qvel_addr("mobile_" + x) for x in ["x", "y", "yaw"]]
        )
        self.act = np.array(
            [
                self.model.actuator_name2id("mobile_" + x + "_act")
                for x in ["x", "y", "yaw"]
            ]
        )
        self.robot = self.e.env.robots[0]
        self.arm_indices = self.robot._ref_joint_pos_indexes
        self.arm_vel = self.robot._ref_joint_vel_indexes
        self.arm_act = self.robot._ref_joint_actuator_indexes
        self.data.qpos[self.joints] = mobile_start
        if stow:
            self.data.qpos[self.arm_indices] = STOW
        self.sim.forward()
        self.mode = "nav"
        self.command = np.zeros(3)
        self.hold = np.array(mobile_start)
        self.arm_target = STOW.copy() if stow else self.arm_q_initial.copy()
        original_pre = self.e.env._pre_action

        def pre(env, action, policy_step=False):
            original_pre(action, policy_step=policy_step)
            self.apply_base()
            if self.mode in ("nav", "unfold"):
                q = self.data.qpos[self.arm_indices]
                dq = self.data.qvel[self.arm_vel]
                tau = (
                    140 * (self.arm_target - q)
                    - 20 * dq
                    + self.data.qfrc_bias[self.arm_vel]
                )
                ranges = self.model.actuator_ctrlrange[self.arm_act]
                self.data.ctrl[self.arm_act] = np.clip(tau, ranges[:, 0], ranges[:, 1])

        self.e.env._pre_action = types.MethodType(pre, self.e.env)
        self.sim.forward()
        self.obs = self.e.env._get_observations(force_update=True)
        self.initial_model_id = id(self.sim)

        def reject_reset(*args, **kwargs):
            raise RuntimeError(
                "Runtime scene reset is forbidden across navigation and manipulation"
            )

        self.e.env.reset = reject_reset
        self.e.env.reset_from_xml_string = reject_reset
        self.history = []

    def apply_base(self):
        if self.mode in ("manip", "unfold"):
            error = self.hold - self.data.qpos[self.joints]
            error[2] = (error[2] + np.pi) % (2 * np.pi) - np.pi
            cmd = np.clip(8 * error, -0.35, 0.35)
        else:
            yaw = self.data.qpos[self.joints[2]]
            vx, vy, w = self.command
            c, s = np.cos(yaw), np.sin(yaw)
            cmd = np.array([c * vx - s * vy, s * vx + c * vy, w])
        self.data.ctrl[self.act] = cmd

    def pose(self):
        return np.r_[
            self.dock[:2] + self.data.qpos[self.joints[:2]],
            (self.data.qpos[self.joints[2]] + np.pi) % (2 * np.pi) - np.pi,
        ]

    def step(self, action=None):
        self.obs, _, done, info = self.e.step(
            ev.DUMMY.tolist() if action is None else np.asarray(action).tolist()
        )
        self.history.append(
            {
                "time": float(self.data.time),
                "base": self.pose().tolist(),
                "base_qvel": self.data.qvel[self.dofs].tolist(),
            }
        )
        assert id(self.sim) == self.initial_model_id
        if hasattr(self, "callback"):
            self.callback()
        return self.obs, done

    def park(self):
        self.command[:] = 0
        self.model.dof_frictionloss[self.dofs] = 250.0
        self.hold = self.data.qpos[self.joints].copy()

    def engage_manipulation(self, steps=60):
        self.park()
        self.mode = "unfold"
        start = self.data.qpos[self.arm_indices].copy()
        for k in range(steps):
            self.arm_target = start + (self.arm_q_initial - start) * (k + 1) / steps
            self.step()
        self.mode = "manip"
        self.hold = self.data.qpos[self.joints].copy()
        self.robot.controller.reset_goal()

    def close(self):
        self.e.close()
