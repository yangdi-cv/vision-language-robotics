"""Own ACT training on public demonstrations, with episode-level splits and train-only stats."""

import argparse, csv, json, time, random
from pathlib import Path
import numpy as np, torch
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.configs.types import FeatureType, PolicyFeature

ROOT = Path(__file__).resolve().parents[3]
TASK = ROOT / "experiments/desktop_pickplace"
parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--steps", type=int, default=5000)
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument(
    "--target-position",
    action="store_true",
    help="Append measured initial RGB-D target xyz to policy state",
)
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=False)
seed = 2026
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.set_num_threads(8)
torch.backends.cudnn.benchmark = True
cache = TASK / ("data/act_cache_rgbd" if args.target_position else "data/act_cache")
manifest = json.loads((cache / "manifest.json").read_text())
images = np.load(cache / "images.npy", mmap_mode="r")
states = np.load(cache / "states.npy", mmap_mode="r")
actions = np.load(cache / "actions.npy", mmap_mode="r")
episodes = np.load(cache / "episodes.npy", mmap_mode="r")
train_stats = json.loads((cache / "train_stats.json").read_text())
stats = {
    key: {name: torch.tensor(value) for name, value in fields.items()}
    for key, fields in train_stats.items()
}
config = ACTConfig(
    input_features={
        "observation.images.top": PolicyFeature(FeatureType.VISUAL, (3, 240, 320)),
        "observation.state": PolicyFeature(
            FeatureType.STATE, (15 if args.target_position else 12,)
        ),
    },
    output_features={"action": PolicyFeature(FeatureType.ACTION, (8,))},
    device="cuda",
    chunk_size=50,
    n_action_steps=25,
    dim_model=256,
    n_heads=4,
    n_encoder_layers=2,
    optimizer_lr=1e-4,
    optimizer_lr_backbone=1e-5,
)
policy = ACTPolicy(config, dataset_stats=stats).to("cuda")
optimizer = torch.optim.AdamW(
    policy.get_optim_params(),
    lr=config.optimizer_lr,
    weight_decay=config.optimizer_weight_decay,
)
train_indices = np.flatnonzero(np.isin(episodes, manifest["train_episodes"]))
val_indices = np.concatenate(
    [
        np.linspace(
            *[manifest["boundaries"][e][0], manifest["boundaries"][e][1] - 1],
            12,
            dtype=int
        )
        for e in manifest["validation_episodes"]
    ]
)
end_indices = np.array([manifest["boundaries"][int(e)][1] for e in episodes])
std = np.maximum(np.array(train_stats["action"]["std"]), 1e-6)
run_config = {
    "source_dataset": manifest["source"],
    "revision": manifest["revision"],
    "initialization": "ACT randomly initialized; ImageNet ResNet18 backbone, not official ACT weights",
    "steps": args.steps,
    "batch_size": args.batch_size,
    "seed": seed,
    "train_episodes": manifest["train_episodes"],
    "validation_episodes": manifest["validation_episodes"],
    "test_episodes": manifest["test_episodes"],
    "normalization": "training episodes only",
    "optimizer_lr": config.optimizer_lr,
    "backbone_lr": config.optimizer_lr_backbone,
    "chunk_size": 50,
    "target_position_input": args.target_position,
    "image_size": [240, 320],
    "torch_version": torch.__version__,
    "gpu": torch.cuda.get_device_name(0),
    "cudnn_benchmark": True,
}
(args.out / "training_config.json").write_text(json.dumps(run_config, indent=2))
(args.out / "normalization.json").write_text(json.dumps(train_stats, indent=2))


def batch(indices):
    future = indices[:, None] + np.arange(config.chunk_size)[None, :]
    padding = future >= end_indices[indices, None]
    future = np.minimum(future, end_indices[indices, None] - 1)
    return {
        "observation.images.top": torch.from_numpy(np.array(images[indices]))
        .permute(0, 3, 1, 2)
        .to("cuda", dtype=torch.float32)
        / 255,
        "observation.state": torch.from_numpy(np.array(states[indices])).to("cuda"),
        "action": torch.from_numpy(np.array(actions[future])).to("cuda"),
        "action_is_pad": torch.from_numpy(padding).to("cuda"),
    }


def validation(indices):
    policy.eval()
    losses = []
    with torch.inference_mode():
        for offset in range(0, len(indices), args.batch_size):
            sample = batch(indices[offset : offset + args.batch_size])
            pred = policy.predict_action_chunk(sample)
            difference = (pred - sample["action"]).abs() / torch.tensor(
                std, dtype=torch.float32, device="cuda"
            )
            valid = (~sample["action_is_pad"]).unsqueeze(-1)
            losses.append(float((difference * valid).sum() / (valid.sum() * 8)))
    return float(np.mean(losses))


initial_val = validation(val_indices)
started = time.perf_counter()
best = initial_val
with (args.out / "training.csv").open("w", newline="") as log:
    writer = csv.writer(log)
    writer.writerow(
        [
            "step",
            "wall_seconds",
            "loss",
            "l1_loss",
            "kld_loss",
            "validation_normalized_l1",
        ]
    )
    writer.writerow([0, 0, "", "", "", initial_val])
    log.flush()
    print("Initial validation normalized L1:", initial_val, flush=True)
    for step in range(1, args.steps + 1):
        policy.train()
        sample = batch(np.random.choice(train_indices, args.batch_size))
        optimizer.zero_grad(set_to_none=True)
        loss, details = policy(sample)
        if not torch.isfinite(loss):
            raise ValueError("Non-finite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
        optimizer.step()
        if step % 100 == 0:
            val = (
                validation(val_indices)
                if step % 500 == 0 or step == args.steps
                else None
            )
            writer.writerow(
                [
                    step,
                    time.perf_counter() - started,
                    float(loss.item()),
                    details["l1_loss"],
                    details.get("kld_loss"),
                    val,
                ]
            )
            log.flush()
            print(
                "step",
                step,
                "seconds",
                round(time.perf_counter() - started, 1),
                "L1",
                round(details["l1_loss"], 4),
                "val",
                val,
                flush=True,
            )
            if val is not None and val < best:
                best = val
                policy.save_pretrained(args.out / "best_checkpoint")
policy.eval()
final_val = validation(val_indices)
test_indices = np.concatenate(
    [
        np.linspace(
            manifest["boundaries"][e][0],
            manifest["boundaries"][e][1] - 1,
            12,
            dtype=int,
        )
        for e in manifest["test_episodes"]
    ]
)
test_loss = validation(test_indices)
policy.save_pretrained(args.out / "final_checkpoint")
summary = {
    "steps": args.steps,
    "wall_seconds": time.perf_counter() - started,
    "initial_validation_normalized_l1": initial_val,
    "final_validation_normalized_l1": final_val,
    "best_validation_normalized_l1": best,
    "test_normalized_l1_72_stratified_frames": test_loss,
    "parameters": sum(p.numel() for p in policy.parameters()),
    "checkpoint": "final_checkpoint",
    "notes": "Offline action prediction is not rollout success. Validation/test sample 72 stratified frames each from separate episodes.",
}
(args.out / "training_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
