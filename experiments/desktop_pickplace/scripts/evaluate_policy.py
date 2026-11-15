import argparse, json, re
from pathlib import Path
import numpy as np, torch, av
from PIL import Image
from lerobot.policies.act.modeling_act import ACTPolicy
from task import make, image, position, locate_rgbd, BINS

p = argparse.ArgumentParser()
p.add_argument("--model", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--seed-start", type=int, default=700)
p.add_argument("--goals", type=Path)
p.add_argument("--target-position", action="store_true")
p.add_argument("--seed-count", type=int, default=2)
p.add_argument("--coordinate-noise-mm", type=float, default=0.0)
a = p.parse_args()
a.out.mkdir(exist_ok=False)
torch.set_num_threads(8)
policy = ACTPolicy.from_pretrained(a.model).to("cuda").eval()
rows = []
authorized = None
if a.goals:
    proposals = [json.loads(line) for line in a.goals.read_text().splitlines()]
    authorized = set()
    colors = {"红色": "red", "蓝色": "blue", "绿色": "green"}
    bins = {"左": "left", "右": "right"}
    for row in proposals:
        match = re.fullmatch(
            r"把(红色|蓝色|绿色)方块放进(左|右)边盒子。", row["case"]["instruction"]
        )
        goal = row.get("goal", {})
        seed = int(row["case"]["id"].split("_")[0])
        if (
            match
            and goal.get("should_execute") is True
            and goal.get("target_color") == colors[match[1]]
            and goal.get("destination") == bins[match[2]]
        ):
            authorized.add((seed, goal["target_color"], goal["destination"]))
for seed in range(a.seed_start, a.seed_start + a.seed_count):
    for color in ("red", "blue", "green"):
        for box in ("left", "right"):
            if authorized is not None and (seed, color, box) not in authorized:
                rows.append(
                    {
                        "seed": seed,
                        "target": color,
                        "bin": box,
                        "success": False,
                        "blocked": True,
                    }
                )
                continue
            env, physics = make(seed)
            policy.reset()
            for _ in range(50):
                env.step(np.r_[physics.data.mocap_pos[1].copy(), [1, 0, 0, 0], 1])
            initial = position(physics, color)
            estimate = locate_rgbd(physics, color) if a.target_position else None
            if a.target_position:
                assert estimate is not None
                if a.coordinate_noise_mm:
                    generator = np.random.default_rng(
                        seed * 10
                        + ("red", "blue", "green").index(color) * 2
                        + (box == "right")
                    )
                    estimate = estimate + generator.normal(
                        0, a.coordinate_noise_mm / 1000, 3
                    )
            goal = np.array(
                [color == c for c in ("red", "blue", "green")]
                + [box == b for b in ("left", "right")],
                dtype=np.float32,
            )
            positions = []
            contact = False
            commands = []
            clipped = 0
            out = a.out / f"{seed}_{color}_{box}"
            out.mkdir()
            video = av.open(str(out / "rollout.mp4"), "w")
            stream = video.add_stream("libx264", rate=25)
            stream.width = 640
            stream.height = 480
            stream.pix_fmt = "yuv420p"
            try:
                for step in range(401):
                    rgb = image(physics)
                    parts = [
                        physics.data.qpos[8:14],
                        np.array([(physics.data.qpos[14] - 0.021) / (0.057 - 0.021)]),
                        goal,
                    ]
                    if a.target_position:
                        parts.append(estimate)
                    state = np.concatenate(parts).astype(np.float32)
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
                    assert np.isfinite(action).all()
                    qnorm = np.linalg.norm(action[3:7])
                    assert qnorm > 0.01
                    action[3:7] /= qnorm
                    raw = action.copy()
                    action[:3] = np.clip(
                        action[:3], [-0.1, 0.25, -0.06], [0.4, 0.85, 0.55]
                    )
                    action[7] = np.clip(action[7], 0, 1)
                    clipped += int(not np.array_equal(raw, action))
                    commands.append(action.tolist())
                    for _ in range(2):
                        env.step(action)
                    positions.append(position(physics, color).tolist())
                    for c in physics.data.contact[: physics.data.ncon]:
                        names = [
                            physics.model.id2name(int(g), "geom")
                            for g in (c.geom1, c.geom2)
                        ]
                        if color + "_cube" in names and any(
                            n is not None and "vx300s_right/10_" in n for n in names
                        ):
                            contact = True
                    for packet in stream.encode(
                        av.VideoFrame.from_ndarray(rgb, format="rgb24")
                    ):
                        video.mux(packet)
                for packet in stream.encode():
                    video.mux(packet)
            finally:
                video.close()
            last = np.array(positions[-25:])
            inside = (
                np.all(abs(last[:, :2] - BINS[box][:2]) < [0.043, 0.033], axis=1)
                & (last[:, 2] < 0.08)
                & (last[:, 2] > 0)
            )
            lifted = max(x[2] for x in positions) > initial[2] + 0.06
            row = {
                "seed": seed,
                "target": color,
                "bin": box,
                "success": bool(lifted and contact and inside.all()),
                "lifted": bool(lifted),
                "stable_inside": bool(inside.all()),
                "clipped_commands": clipped,
                "final_xyz": positions[-1],
            }
            rows.append(row)
            (out / "trajectory.json").write_text(
                json.dumps(
                    {"summary": row, "positions": positions, "commands": commands}
                )
            )
            (a.out / "results.json").write_text(json.dumps(rows, indent=2))
            print(row, flush=True)
            physics.free()
(a.out / "metrics.json").write_text(
    json.dumps(
        {
            "cases": len(rows),
            "successes": sum(r["success"] for r in rows),
            "new_layout_seeds": list(range(a.seed_start, a.seed_start + a.seed_count)),
            "action_representation": "Cartesian targetxyz+normalized quaternion+gripper8",
            "target_position_input": a.target_position,
            "coordinate_noise_sigma_mm": a.coordinate_noise_mm,
            "goal_condition": "color/bin symbols, with optional RGB-D measured targetxyz and Qwen proposal source",
            "physics_timestep_s": 0.002,
            "policy_period_s": 0.04,
            "n_action_steps": policy.config.n_action_steps,
        },
        indent=2,
    )
)
