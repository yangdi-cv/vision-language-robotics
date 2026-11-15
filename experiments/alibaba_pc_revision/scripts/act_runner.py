"""Shared native rollout for training validation, final evaluation and demos."""

from pathlib import Path
import sys, json
import numpy as np, torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/desktop_pickplace/scripts"))
from task import make, image, position, locate_rgbd, BINS


def rollout(policy, seed, color, box, save=None):
    env, physics = make(seed)
    policy.reset()
    try:
        for _ in range(50):
            env.step(np.r_[physics.data.mocap_pos[1].copy(), [1, 0, 0, 0], 1])
        initial = position(physics, color).copy()
        estimate = locate_rgbd(physics, color)
        assert estimate is not None
        dim = policy.config.input_features["observation.state"].shape[0]
        goal = np.array(
            [color == c for c in ["red", "blue", "green"]]
            + [box == b for b in ["left", "right"]],
            np.float32,
        )
        positions = []
        contacts = []
        commands = []
        for step in range(401):
            rgb = image(physics)
            parts = [
                physics.data.qpos[8:14],
                [(physics.data.qpos[14] - 0.021) / 0.036],
                goal,
            ]
            if dim == 15:
                parts.append(estimate)
            state = np.concatenate(parts).astype(np.float32)
            rgb = np.array(Image.fromarray(rgb).resize((320, 240)))
            with torch.inference_mode():
                action = (
                    policy.select_action(
                        {
                            "observation.state": torch.from_numpy(state)[None].to(
                                "cuda"
                            ),
                            "observation.images.top": torch.from_numpy(rgb)
                            .permute(2, 0, 1)[None]
                            .to("cuda", dtype=torch.float32)
                            / 255,
                        }
                    )[0]
                    .cpu()
                    .numpy()
                )
            if not np.isfinite(action).all() or np.linalg.norm(action[3:7]) < 0.01:
                raise ValueError("Invalid pose")
            action[3:7] /= np.linalg.norm(action[3:7])
            action[:3] = np.clip(action[:3], [-0.1, 0.25, -0.06], [0.4, 0.85, 0.55])
            action[7] = np.clip(action[7], 0, 1)
            for _ in range(2):
                env.step(action)
            positions.append(position(physics, color).copy())
            commands.append(action.copy())
            names = []
            for c in physics.data.contact[: physics.data.ncon]:
                pair = [
                    physics.model.id2name(int(g), "geom") for g in [c.geom1, c.geom2]
                ]
                if color + "_cube" in pair:
                    names += pair
            contacts.append(names)
        last = np.array(positions[-25:])
        inside = np.all(abs(last[:, :2] - BINS[box][:2]) < [0.043, 0.033]) and np.all(
            (last[:, 2] > 0) & (last[:, 2] < 0.08)
        )
        lift = max(p[2] for p in positions) > initial[2] + 0.06
        grasp = any(any("vx300s_right/10_" in n for n in c) for c in contacts)
        released = all(
            not any("vx300s_right/10_" in n for n in c) for c in contacts[-25:]
        )
        floor = any(box + "_floor" in c for c in contacts[-25:])
        result = dict(
            seed=seed,
            target=color,
            destination=box,
            success=bool(inside and lift and grasp and released and floor),
            inside=bool(inside),
            lifted=bool(lift),
            grasp=grasp,
            released=released,
            floor=floor,
        )
        if save:
            save = Path(save)
            save.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                save / "trajectory.npz", positions=positions, commands=commands
            )
            (save / "result.json").write_text(json.dumps(result, indent=2))
        return result
    finally:
        physics.free()


def suite(policy, seeds, save=None):
    return [
        rollout(policy, s, c, b, Path(save) / f"{s}_{c}_{b}" if save else None)
        for s in seeds
        for c in ["red", "blue", "green"]
        for b in ["left", "right"]
    ]
