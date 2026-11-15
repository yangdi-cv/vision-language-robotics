"""CLI evaluation using the shared ALOHA rollout/video implementation."""

from pathlib import Path
import argparse, json, csv
import numpy as np, torch, gymnasium as gym, gym_aloha
from lerobot.policies.act.modeling_act import ACTPolicy
from aloha_runner import rollout_episode

p = argparse.ArgumentParser()
p.add_argument("--model", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--episodes", type=int, default=10)
p.add_argument("--seed-start", type=int, default=1000)
p.add_argument("--action-steps", type=int)
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=False)
torch.set_num_threads(8)
policy = ACTPolicy.from_pretrained(a.model).to("cuda").eval()
if a.action_steps:
    assert 1 <= a.action_steps <= policy.config.chunk_size
    policy.config.n_action_steps = a.action_steps
rows = []
timing = []
env = gym.make(
    "gym_aloha/AlohaTransferCube-v0", obs_type="pixels_agent_pos", max_episode_steps=400
)
try:
    for episode in range(a.episodes):
        seed = a.seed_start + episode
        obs, _ = env.reset(seed=seed)
        result = rollout_episode(policy, env, obs, a.out / f"rollout_{seed}.mp4")
        row = {k: result[k] for k in ["steps", "max_reward", "success", "wall_seconds"]}
        row.update(seed=seed, episode=episode)
        rows.append(row)
        timing.extend([[seed, *t] for t in result["timing"]])
        (a.out / f"rollout_{seed}.json").write_text(
            json.dumps(
                {
                    "summary": row,
                    "actions": result["actions"],
                    "rewards": result["rewards"],
                }
            )
        )
        print(row, flush=True)
finally:
    env.close()
full = [t[3] for t in timing if t[2]]
metrics = dict(
    episodes=len(rows),
    successes=sum(r["success"] for r in rows),
    success_rate=sum(r["success"] for r in rows) / len(rows),
    n_action_steps=policy.config.n_action_steps,
    model_path=str(a.model.resolve()),
    criterion="maximum reward>=4",
    full_chunk_inference_seconds_p50=float(np.median(full)),
    select_action_seconds_p50=float(np.median([t[3] for t in timing])),
)
metrics.update(
    environment="gym_aloha/AlohaTransferCube-v0",
    gym_aloha_version="0.1.1",
    mujoco_version="2.3.7",
    seed_start=a.seed_start,
    max_episode_steps=400,
    select_action_seconds_p95=float(np.percentile([t[3] for t in timing], 95)),
    select_action_seconds_max=float(max(t[3] for t in timing)),
    full_chunk_inference_seconds_p95=float(np.percentile(full, 95)),
)
(a.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
for name, data, header in [
    ("latencies.csv", timing, ["seed", "step", "full_chunk_inference", "seconds"])
]:
    with (a.out / name).open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(data)
with (a.out / "episodes.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0])
    w.writeheader()
    w.writerows(rows)
