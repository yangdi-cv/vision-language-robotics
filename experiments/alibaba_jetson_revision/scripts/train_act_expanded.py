"""Manifest-driven episode sampler and independent L1/rollout checkpoint selection."""

from pathlib import Path
import argparse, json, csv, time, random
import numpy as np, torch
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.configs.types import FeatureType, PolicyFeature
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "experiments/alibaba_pc_revision/scripts"))
from act_runner import suite

ROOT = Path(__file__).resolve().parents[3]
T = ROOT / "experiments/alibaba_jetson_revision"
parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, default=T / "runs/act_selection")
parser.add_argument(
    "--cache",
    type=Path,
    default=ROOT / "experiments/desktop_pickplace/data/act_cache_rgbd",
)
parser.add_argument("--steps", type=int, default=10000)
parser.add_argument("--interval", type=int, default=2000)
parser.add_argument("--seed", type=int, default=20261020)
parser.add_argument("--validation-seeds", type=int, nargs="+", default=[1100])
parser.add_argument(
    "--test-seeds", type=int, nargs="+", default=[1200, 1201, 1202, 1203]
)
args = parser.parse_args()
O = args.out.resolve()
O.mkdir(parents=True, exist_ok=False)
CACHE = args.cache.resolve()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.set_num_threads(8)
torch.backends.cudnn.benchmark = True
manifest = json.loads((CACHE / "manifest.json").read_text())
statistics = json.loads((CACHE / "train_stats.json").read_text())
stats = {k: {n: torch.tensor(v) for n, v in z.items()} for k, z in statistics.items()}
images = np.load(CACHE / "images.npy", mmap_mode="r")
states = np.load(CACHE / "states.npy", mmap_mode="r")
actions = np.load(CACHE / "actions.npy", mmap_mode="r")
episodes = np.load(CACHE / "episodes.npy", mmap_mode="r")
bound = np.array(manifest["boundaries"])
train = manifest["train_episodes"]
val = manifest["validation_episodes"]
test = manifest["test_episodes"]
assert not (set(train) & set(val) or set(train) & set(test) or set(val) & set(test))
config = ACTConfig(
    input_features={
        "observation.images.top": PolicyFeature(FeatureType.VISUAL, (3, 240, 320)),
        "observation.state": PolicyFeature(FeatureType.STATE, (15,)),
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
opt = torch.optim.AdamW(
    policy.get_optim_params(), lr=1e-4, weight_decay=config.optimizer_weight_decay
)
std = torch.tensor(
    np.maximum(statistics["action"]["std"], 1e-6), device="cuda", dtype=torch.float32
)


def indices(split):
    return np.concatenate(
        [np.linspace(*[bound[e, 0], bound[e, 1] - 1], 12, dtype=int) for e in split]
    )


def batch(idx):
    future = idx[:, None] + np.arange(50)[None]
    ends = bound[episodes[idx].astype(int), 1, None]
    pad = future >= ends
    future = np.minimum(future, ends - 1)
    return {
        "observation.images.top": torch.from_numpy(np.array(images[idx]))
        .permute(0, 3, 1, 2)
        .to("cuda", dtype=torch.float32)
        / 255,
        "observation.state": torch.from_numpy(np.array(states[idx])).to("cuda"),
        "action": torch.from_numpy(np.array(actions[future])).to("cuda"),
        "action_is_pad": torch.from_numpy(pad).to("cuda"),
    }


def loss_eval(model, idx):
    model.eval()
    total = 0
    count = 0
    with torch.inference_mode():
        for i in range(0, len(idx), 8):
            b = batch(idx[i : i + 8])
            pred = model.predict_action_chunk(b)
            mask = (~b["action_is_pad"]).unsqueeze(-1)
            total += float((((pred - b["action"]).abs() / std) * mask).sum())
            count += int(mask.sum()) * 8
    return total / count


(O / "experiment.json").write_text(
    json.dumps(
        {
            "seed": args.seed,
            "max_steps": args.steps,
            "validation_interval": args.interval,
            "sampler": "uniform episode then uniform frame, from manifest",
            "train_episodes": train,
            "validation_episodes": val,
            "test_episodes": test,
            "rollout_validation_seeds": args.validation_seeds,
            "final_test_seeds": args.test_seeds,
            "selection_rollout": "maximum success; lower validation L1 on ties; earliest exact tie",
            "normalization": "training only",
            "rollout_success": "lift+grasp+stable bin+release+floor contact",
            "initialization": "random ACT, ImageNet ResNet18 only",
        },
        indent=2,
    )
)
(O / "normalization.json").write_text(json.dumps(statistics))
start = time.perf_counter()
history = []
best_l1 = float("inf")
best_success = -1
best_rollout_l1 = float("inf")
selected = {}
for step in range(1, args.steps + 1):
    policy.train()
    chosen = np.random.choice(train, 8)
    idx = np.array([np.random.randint(bound[e, 0], bound[e, 1]) for e in chosen])
    opt.zero_grad(set_to_none=True)
    loss, details = policy(batch(idx))
    assert torch.isfinite(loss)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 10)
    opt.step()
    if step % args.interval == 0 or step == args.steps:
        vl = loss_eval(policy, indices(val))
        rows = suite(policy, args.validation_seeds)
        success = sum(r["success"] for r in rows)
        history.append(
            dict(
                step=step,
                validation_l1=vl,
                validation_success=success,
                validation_trials=len(rows),
                seconds=time.perf_counter() - start,
            )
        )
        (O / f"validation_{step}.json").write_text(json.dumps(rows, indent=2))
        (O / "history.json").write_text(json.dumps(history, indent=2))
        print(history[-1], flush=True)
        if vl < best_l1:
            best_l1 = vl
            policy.save_pretrained(O / "best_l1")
            selected["l1"] = step
        if success > best_success or (success == best_success and vl < best_rollout_l1):
            best_success = success
            best_rollout_l1 = vl
            policy.save_pretrained(O / "best_rollout")
            selected["rollout"] = step
policy.save_pretrained(O / "last_checkpoint")
results = {}
for name in ["l1", "rollout"]:
    model = ACTPolicy.from_pretrained(O / ("best_" + name)).to("cuda").eval()
    test_loss = loss_eval(model, indices(test))
    rows = suite(model, args.test_seeds, O / ("test_" + name))
    results[name] = {
        "selected_step": selected[name],
        "test_success": sum(r["success"] for r in rows),
        "test_trials": len(rows),
        "test_l1": test_loss,
    }
    (O / f"test_{name}.json").write_text(json.dumps(rows, indent=2))
    del model
(O / "summary.json").write_text(
    json.dumps(
        {
            "selection": results,
            "seconds": time.perf_counter() - start,
            "equal_episode_lengths": len(set(bound[e, 1] - bound[e, 0] for e in train))
            == 1,
            "test_used_for_selection": False,
        },
        indent=2,
    )
)
print(results, flush=True)
