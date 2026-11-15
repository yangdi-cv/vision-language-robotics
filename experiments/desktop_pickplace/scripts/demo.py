"""Current scene RGB -> Qwen goal -> RGB-D position -> ACT -> physical verification."""

import argparse, json, os, re, subprocess
from pathlib import Path
import numpy as np, torch
from PIL import Image
from lerobot.policies.act.modeling_act import ACTPolicy
from task import make, image, locate_rgbd
from policy_runtime import execute

ROOT = Path(__file__).resolve().parents[3]
TASK = ROOT / "experiments/desktop_pickplace"
parser = argparse.ArgumentParser()
parser.add_argument("--instruction", required=True)
parser.add_argument("--seed", type=int, default=900)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=False)
env, physics = make(args.seed)
try:
    for _ in range(50):
        env.step(np.r_[physics.data.mocap_pos[1].copy(), [1, 0, 0, 0], 1])
    frame = args.out / "initial.png"
    Image.fromarray(image(physics)).save(frame)
    worker_env = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH"):
        worker_env.pop(key, None)
    command = [
        str(ROOT / ".venvs/qwen-scene/bin/python"),
        str(TASK / "scripts/goal_worker.py"),
        "--image",
        str(frame.resolve()),
        "--instruction",
        args.instruction,
        "--adapter",
        str(TASK / "runs/goal_lora/adapter"),
    ]
    with (args.out / "qwen.log").open("w") as log:
        worker = subprocess.run(
            command,
            env=worker_env,
            stdout=subprocess.PIPE,
            stderr=log,
            text=True,
            timeout=120,
            check=True,
        )
    decision = json.loads(worker.stdout.strip())
    (args.out / "goal.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2)
    )
    goal = decision.get("goal", {})
    match = re.fullmatch(r"把(红色|蓝色|绿色)方块放进(左|右)边盒子。", args.instruction)
    colors = {"红色": "red", "蓝色": "blue", "绿色": "green"}
    bins = {"左": "left", "右": "right"}
    authorized = (
        decision["valid"]
        and goal.get("should_execute") is True
        and match is not None
        and goal.get("target_color") == colors[match[1]]
        and goal.get("destination") == bins[match[2]]
    )
    if not authorized:
        result = {
            "executed": False,
            "success": False,
            "status": "needs_clarification_or_unsupported_instruction",
        }
    else:
        estimate = locate_rgbd(physics, goal["target_color"])
        if estimate is None:
            result = {
                "executed": False,
                "success": False,
                "status": "target_not_visible",
            }
        else:
            torch.set_num_threads(8)
            policy = (
                ACTPolicy.from_pretrained(TASK / "runs/goal_act_rgbd/best_checkpoint")
                .to("cuda")
                .eval()
            )
            result = execute(
                env,
                physics,
                policy,
                goal["target_color"],
                goal["destination"],
                estimate,
                args.out,
            )
            result.update(
                executed=True,
                status="complete" if result["success"] else "failed_final_check",
            )
    result["instruction"] = args.instruction
    result["seed"] = args.seed
    (args.out / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
finally:
    physics.free()
