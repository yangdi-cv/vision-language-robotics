"""Native Cartesian ACT execution in the current scene, with physical result checks."""

import json
from pathlib import Path
import av, numpy as np, torch
from PIL import Image
from task import image, position, BINS


def execute(env, physics, policy, target, box, estimate, out):
    out = Path(out)
    policy.reset()
    initial = position(physics, target)
    goal = np.array(
        [target == c for c in ("red", "blue", "green")]
        + [box == b for b in ("left", "right")],
        dtype=np.float32,
    )
    positions = []
    commands = []
    contacts = []
    clipped_xyz = 0
    clipped_gripper = 0
    video = av.open(str(out / "rollout.mp4"), "w")
    stream = video.add_stream("libx264", rate=25)
    stream.width = 640
    stream.height = 480
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "23"}
    try:
        for step in range(401):
            rgb = image(physics)
            state = np.r_[
                physics.data.qpos[8:14],
                (physics.data.qpos[14] - 0.021) / (0.057 - 0.021),
                goal,
                estimate,
            ].astype(np.float32)
            small = np.array(Image.fromarray(rgb).resize((320, 240)))
            batch = {
                "observation.state": torch.from_numpy(state)[None].to("cuda"),
                "observation.images.top": torch.from_numpy(small)
                .permute(2, 0, 1)[None]
                .to("cuda", dtype=torch.float32)
                / 255,
            }
            with torch.inference_mode():
                action = policy.select_action(batch)[0].cpu().numpy()
            if not np.isfinite(action).all() or np.linalg.norm(action[3:7]) < 0.01:
                raise ValueError("Invalid policy pose")
            action[3:7] /= np.linalg.norm(action[3:7])
            raw = action.copy()
            action[:3] = np.clip(action[:3], [-0.1, 0.25, -0.06], [0.4, 0.85, 0.55])
            action[7] = np.clip(action[7], 0, 1)
            clipped_xyz += int(not np.array_equal(raw[:3], action[:3]))
            clipped_gripper += int(raw[7] != action[7])
            commands.append(action.tolist())
            for _ in range(2):
                env.step(action)
            positions.append(position(physics, target).tolist())
            touching = []
            for c in physics.data.contact[: physics.data.ncon]:
                names = [
                    physics.model.id2name(int(g), "geom") for g in (c.geom1, c.geom2)
                ]
                if target + "_cube" in names:
                    touching.extend(names)
            contacts.append(touching)
            for packet in stream.encode(
                av.VideoFrame.from_ndarray(rgb, format="rgb24")
            ):
                video.mux(packet)
        for packet in stream.encode():
            video.mux(packet)
    finally:
        video.close()
    Image.fromarray(image(physics)).save(out / "final.png")
    last = np.array(positions[-25:])
    inside = (
        np.all(abs(last[:, :2] - BINS[box][:2]) < [0.043, 0.033], axis=1)
        & (last[:, 2] > 0)
        & (last[:, 2] < 0.08)
    )
    lift = max(x[2] for x in positions) > initial[2] + 0.06
    grasp = any(any("vx300s_right/10_" in n for n in c) for c in contacts)
    released = all(not any("vx300s_right/10_" in n for n in c) for c in contacts[-25:])
    floor = any(box + "_floor" in c for c in contacts[-25:])
    success = bool(lift and grasp and inside.all() and released and floor)
    summary = {
        "target": target,
        "destination": box,
        "success": success,
        "lifted": bool(lift),
        "stable_in_bin": bool(inside.all()),
        "released_contact_free": released,
        "floor_contact": floor,
        "xyz_clipping_steps": clipped_xyz,
        "gripper_clipping_steps": clipped_gripper,
        "final_xyz": positions[-1],
    }
    (out / "trajectory.json").write_text(
        json.dumps(
            {
                "summary": summary,
                "positions": positions,
                "commands": commands,
                "contacts": contacts,
            }
        )
    )
    return summary
