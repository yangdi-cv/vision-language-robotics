"""Run held-out table layouts using Qwen goal proposals and fresh RGB-D localization."""

import json, subprocess, re, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TASK = ROOT / "experiments/desktop_pickplace"
parser = argparse.ArgumentParser()
parser.add_argument(
    "--predictions", type=Path, default=TASK / "runs/qwen_goals/predictions.jsonl"
)
parser.add_argument("--out", type=Path, default=TASK / "runs/integrated")
parser.add_argument("--min-seed", type=int, default=400)
parser.add_argument("--max-seed", type=int, default=403)
args = parser.parse_args()
source = args.predictions
out = args.out
out.mkdir(parents=True, exist_ok=True)
names = {"红色": "red", "蓝色": "blue", "绿色": "green"}
boxes = {"左": "left", "右": "right"}
rows = []
for line in source.read_text().splitlines():
    row = json.loads(line)
    case = row["case"]
    seed = int(case["id"].split("_")[0])
    if not args.min_seed <= seed < args.max_seed:
        continue
    result = {
        "id": case["id"],
        "seed": seed,
        "instruction": case["instruction"],
        "executed": False,
        "success": False,
        "expected_execution": case["should_execute"],
    }
    goal = row.get("goal", {})
    match = re.fullmatch(
        r"把(红色|蓝色|绿色)方块放进(左|右)边盒子。", case["instruction"]
    )
    authorized = (
        row["valid"]
        and goal.get("should_execute") is True
        and match is not None
        and goal.get("target_color") == names[match[1]]
        and goal.get("destination") == boxes[match[2]]
    )
    if authorized:
        directory = out / case["id"]
        summary = directory / "summary.json"
        if not summary.exists():
            cmd = [
                str(ROOT / "experiments/desktop_pickplace/scripts/act-python"),
                str(TASK / "scripts/run_episode.py"),
                "--seed",
                str(seed),
                "--target",
                goal["target_color"],
                "--bin",
                goal["destination"],
                "--perception",
                "rgbd",
                "--out",
                str(directory),
            ]
            if case["id"].endswith("_no_red"):
                cmd += ["--absent-color", "red"]
            with (out / (case["id"] + ".log")).open("w") as log:
                process = subprocess.run(
                    cmd, stdout=log, stderr=subprocess.STDOUT, timeout=120
                )
            if process.returncode:
                result["process_error"] = True
        if summary.exists():
            result.update(
                executed=True, success=json.loads(summary.read_text())["success"]
            )
    rows.append(result)
    (out / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    print(result, flush=True)
positive = [r for r in rows if r["expected_execution"]]
negative = [r for r in rows if not r["expected_execution"]]
(out / "metrics.json").write_text(
    json.dumps(
        {
            "positive_cases": len(positive),
            "positive_successes": sum(r["success"] for r in positive),
            "negative_cases": len(negative),
            "negative_executions": sum(r["executed"] for r in negative),
            "scene_family": "same actual table scene and seed/layout as language images; not separate simulators",
            "controller": "geometry-derived scripted physical grasp; not ACT",
        },
        indent=2,
    )
)
