"""Single-scene table sorting with physical contacts and one active ViperX arm."""

from pathlib import Path
import shutil, xml.etree.ElementTree as ET
import numpy as np
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from gym_aloha.constants import (
    ASSETS_DIR,
    START_ARM_POSE,
    unnormalize_puppet_gripper_position,
)

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
COLORS = {"red": [1, 0, 0, 1], "blue": [0, 0, 0.95, 1], "green": [0, 0.8, 0, 1]}
BINS = {"left": np.array([0.015, 0.68, 0.025]), "right": np.array([0.24, 0.68, 0.025])}


def build_assets():
    if not ASSETS.exists():
        shutil.copytree(ASSETS_DIR, ASSETS)
    tree = ET.parse(ASSETS / "bimanual_viperx_end_effector_transfer_cube.xml")
    root = tree.getroot()
    root.remove(root.find("keyframe"))
    world = root.find("worldbody")
    world.remove(world.find("body[@name='box']"))
    for i, (color, rgba) in enumerate(COLORS.items()):
        body = ET.SubElement(world, "body", name=color, pos=f"{.1+i*.08} .43 .05")
        ET.SubElement(body, "freejoint", name=color + "_joint")
        ET.SubElement(
            body,
            "inertial",
            pos="0 0 0",
            mass=".05",
            diaginertia=".000013333 .000013333 .000013333",
        )
        ET.SubElement(
            body,
            "geom",
            name=color + "_cube",
            type="box",
            size=".02 .02 .02",
            mass=".05",
            rgba=" ".join(map(str, rgba)),
            friction="1 .005 .0001",
            condim="4",
        )
    for name, pos in BINS.items():
        body = ET.SubElement(
            world,
            "body",
            name=name + "_bin",
            pos=" ".join(map(str, [pos[0], pos[1], 0])),
        )
        geoms = [
            ("floor", [0, 0, 0.003], [0.068, 0.058, 0.003]),
            ("west", [-0.07, 0, 0.026], [0.003, 0.064, 0.026]),
            ("east", [0.07, 0, 0.026], [0.003, 0.064, 0.026]),
            ("south", [0, -0.061, 0.026], [0.07, 0.003, 0.026]),
            ("north", [0, 0.061, 0.026], [0.07, 0.003, 0.026]),
        ]
        for part, p, s in geoms:
            ET.SubElement(
                body,
                "geom",
                name=name + "_" + part,
                type="box",
                pos=" ".join(map(str, p)),
                size=" ".join(map(str, s)),
                rgba=".5 .6 .65 1",
                friction="1 .005 .0001",
            )
    for body in world.findall("body"):
        if body.get("mocap") == "true":
            for site in body.findall("site"):
                site.set("rgba", "0 0 0 0")
    root.insert(0, ET.Element("option", timestep=".002", iterations="80"))
    import os

    temp = ASSETS / f"desktop_{os.getpid()}.xml"
    tree.write(temp)
    temp.replace(ASSETS / "desktop.xml")


class TableTask(base.Task):
    def __init__(self, seed=0, size=0.02, friction=1):
        super().__init__(random=seed)
        self.seed = seed
        self.size = size
        self.friction = friction

    def initialize_episode(self, p):
        p.data.qpos[:16] = START_ARM_POSE
        p.data.mocap_pos[:] = [
            [-0.31718881, 0.5, 0.29525084],
            [0.31718881, 0.49999888, 0.29525084],
        ]
        p.data.mocap_quat[:] = [[1, 0, 0, 0], [1, 0, 0, 0]]
        p.data.ctrl[:] = [0.021, -0.021, 0.021, -0.021]
        rng = np.random.default_rng(self.seed)
        slots = rng.permutation(3)
        for i, color in enumerate(COLORS):
            adr = p.model.jnt_qposadr[p.model.name2id(color + "_joint", "joint")]
            p.data.qpos[adr : adr + 7] = [
                0.075 + slots[i] * 0.085 + rng.uniform(-0.015, 0.015),
                0.44 + rng.uniform(-0.025, 0.025),
                self.size + 0.002,
                1,
                0,
                0,
                0,
            ]
            gid = p.model.name2id(color + "_cube", "geom")
            p.model.geom_size[gid] = self.size
            p.model.geom_rbound[gid] = np.sqrt(3) * self.size
            p.model.body_inertia[p.model.name2id(color, "body")] = (
                2 / 3 * 0.05 * self.size**2
            )
            p.model.geom_friction[gid, 0] = self.friction
        super().initialize_episode(p)

    def before_step(self, action, p):
        p.data.mocap_pos[1] = action[:3]
        p.data.mocap_quat[1] = action[3:7]
        g = unnormalize_puppet_gripper_position(action[7])
        p.data.ctrl[:] = [0.021, -0.021, g, -g]

    def get_observation(self, p):
        return {"qpos": p.data.qpos[:16].copy()}

    def get_reward(self, p):
        return 0.0


def make(seed, size=0.02, friction=1):
    build_assets()
    p = mujoco.Physics.from_xml_path(str(ASSETS / "desktop.xml"))
    env = control.Environment(
        p, TableTask(seed, size, friction), time_limit=30, control_timestep=0.02
    )
    env.reset()
    return env, p


def position(p, color):
    return p.named.data.xpos[color].copy()


def image(p):
    return p.render(height=480, width=640, camera_id="top").copy()


def waypoints(p, target, bin_name, offset=-0.015, estimated_position=None):
    start = p.data.mocap_pos[1].copy()
    cube = (
        position(p, target)
        if estimated_position is None
        else np.asarray(estimated_position)
    )
    dest = BINS[bin_name].copy()
    quat = np.array([np.cos(np.pi / 4), 0, -np.sin(np.pi / 4), 0])
    home = np.array([1.0, 0, 0, 0])
    return [
        (0, start, home, 1),
        (80, cube + [0, 0, 0.13], quat, 1),
        (150, cube + [0, 0, offset], quat, 1),
        (190, cube + [0, 0, offset], quat, 0),
        (280, cube + [0, 0, 0.2], quat, 0),
        (390, dest + [0, 0, 0.18], quat, 0),
        (470, dest + [0, 0, 0.06], quat, 0),
        (530, dest + [0, 0, 0.06], quat, 1),
        (610, dest + [0, 0, 0.2], quat, 1),
        (700, start, home, 1),
        (800, start, home, 1),
    ]


def action_at(points, t):
    for a, b in zip(points[:-1], points[1:]):
        if a[0] <= t <= b[0]:
            u = (t - a[0]) / (b[0] - a[0])
            u = u * u * (3 - 2 * u)
            pos = (1 - u) * a[1] + u * b[1]
            q = (1 - u) * a[2] + u * b[2]
            q /= np.linalg.norm(q)
            return np.r_[pos, q, (1 - u) * a[3] + u * b[3]]
    return np.r_[points[-1][1], points[-1][2], points[-1][3]]


def locate_rgbd(p, color):
    import cv2

    rgb = image(p).astype(float)
    channel = {"red": 0, "green": 1, "blue": 2}[color]
    others = [i for i in range(3) if i != channel]
    mask = (
        (rgb[:, :, channel] > 70)
        & (rgb[:, :, channel] > 1.8 * np.max(rgb[:, :, others], axis=2))
    ).astype("uint8")
    n, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return None
    index = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    u, v = centers[index]
    depth = p.render(height=480, width=640, camera_id="top", depth=True)
    z = float(np.median(depth[labels == index]))
    cam = p.model.name2id("top", "camera")
    f = 240 / np.tan(np.deg2rad(p.model.cam_fovy[cam]) / 2)
    xyz = p.data.cam_xpos[cam] + p.data.cam_xmat[cam].reshape(3, 3) @ np.array(
        [(u - 320) * z / f, -(v - 240) * z / f, -z]
    )
    xyz[2] -= 0.02
    return xyz
