import argparse, json, csv, time
from pathlib import Path
import numpy as np, av
from PIL import Image
from task import make, position, image, locate_rgbd, waypoints, action_at, BINS

p = argparse.ArgumentParser()
p.add_argument("--out", type=Path, required=True)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--target", default="blue")
p.add_argument("--bin", default="left")
p.add_argument("--offset", type=float, default=-0.015)
p.add_argument("--size", type=float, default=0.02)
p.add_argument("--friction", type=float, default=1)
p.add_argument("--absent-color", choices=["red", "blue", "green"])
p.add_argument("--perception", choices=["oracle", "rgbd"], default="oracle")
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=False)
env, physics = make(a.seed, a.size, a.friction)
if a.absent_color:
    adr = physics.model.jnt_qposadr[
        physics.model.name2id(a.absent_color + "_joint", "joint")
    ]
    physics.data.qpos[adr : adr + 3] = [0, 0, -2]
    physics.forward()
start = physics.data.mocap_pos[1].copy()
for _ in range(50):
    env.step(np.r_[start, [1, 0, 0, 0], 1])
initial_all = {c: position(physics, c).copy() for c in ("red", "blue", "green")}
initial = position(physics, a.target)
Image.fromarray(image(physics)).save(a.out / "initial.png")
estimate = locate_rgbd(physics, a.target) if a.perception == "rgbd" else initial.copy()
if estimate is None:
    (a.out / "summary.json").write_text(
        json.dumps(
            {
                "success": False,
                "blocked": True,
                "reason": "target_not_visible",
                "grasp_execution_started": False,
                "target": a.target,
                "bin": a.bin,
            },
            indent=2,
        )
    )
    physics.free()
    raise SystemExit(0)
points = waypoints(physics, a.target, a.bin, a.offset, estimate)
frames = []
positions = []
contacts = []
video = av.open(str(a.out / "rollout.mp4"), "w")
stream = video.add_stream("libx264", rate=25)
stream.width = 640
stream.height = 480
stream.pix_fmt = "yuv420p"
stream.options = {"crf": "23"}
with (a.out / "trajectory.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(
        ["step", "sim_time_s", "object_x", "object_y", "object_z", "right_gripper"]
        + [f"robot_q{i}" for i in range(16)]
    )
    for step in range(801):
        action = action_at(points, step)
        env.step(action)
        pos = position(physics, a.target)
        positions.append(pos.tolist())
        w.writerow([step, physics.data.time, *pos, action[7], *physics.data.qpos[:16]])
        touching = []
        for c in physics.data.contact[: physics.data.ncon]:
            names = [physics.model.id2name(int(g), "geom") for g in (c.geom1, c.geom2)]
            if a.target + "_cube" in names:
                touching.extend(names)
        contacts.append(touching)
        if step % 2 == 0:
            rgb = image(physics)
            for packet in stream.encode(
                av.VideoFrame.from_ndarray(rgb, format="rgb24")
            ):
                video.mux(packet)
        if step in (150, 190, 280, 390, 530, 700):
            Image.fromarray(image(physics)).save(a.out / f"step_{step}.png")
    for packet in stream.encode():
        video.mux(packet)
video.close()
final = position(physics, a.target)
dest = BINS[a.bin]
last = np.array(positions[-50:])
inside = np.all(abs(last[:, :2] - dest[:2]) < [0.043, 0.033], axis=1) & (
    last[:, 2] < 0.08
)
lifted = max(x[2] for x in positions) > initial[2] + 0.06
contact_gripper = any(any("vx300s_right/10_" in n for n in c) for c in contacts)
summary = {
    "perception": a.perception,
    "estimated_xyz": estimate.tolist(),
    "localization_error_m": float(np.linalg.norm(estimate - initial)),
    "distractor_displacement_m": {
        c: float(np.linalg.norm(position(physics, c) - initial_all[c]))
        for c in initial_all
        if c != a.target
    },
    "seed": a.seed,
    "target": a.target,
    "bin": a.bin,
    "size_half_m": a.size,
    "friction": a.friction,
    "offset_m": a.offset,
    "initial_xyz": initial.tolist(),
    "final_xyz": final.tolist(),
    "max_height_m": max(x[2] for x in positions),
    "gripper_contact": contact_gripper,
    "lifted": bool(lifted),
    "stable_inside_last_1s": bool(inside.all()),
    "success": bool(lifted and contact_gripper and inside.all()),
    "object_attached_or_teleported_during_execution": False,
    "controller": "Cartesian mocap weld end-effector tracking + physical gripper contacts; "
    + a.perception
    + " localization",
}
(a.out / "summary.json").write_text(json.dumps(summary, indent=2))
(a.out / "contacts.json").write_text(json.dumps(contacts))
print(json.dumps(summary, indent=2))
physics.free()
