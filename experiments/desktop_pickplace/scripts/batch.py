"""Resumable serial batch; finished summaries are skipped, incomplete trials retained."""

from pathlib import Path
import subprocess, json, time, shutil, argparse

ROOT = Path(__file__).resolve().parents[3]
TASK = ROOT / "experiments/desktop_pickplace"
python = ROOT / "experiments/desktop_pickplace/scripts/act-python"
p = argparse.ArgumentParser()
p.add_argument("--out", type=Path, default=TASK / "runs/batch001")
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
cases = []
for mode in ("oracle", "rgbd"):
    for seed in range(100, 105):
        for color in ("red", "blue", "green"):
            for box in ("left", "right"):
                cases.append(
                    {
                        "group": mode,
                        "seed": seed,
                        "target": color,
                        "bin": box,
                        "size": 0.02,
                        "friction": 1.0,
                        "perception": mode,
                    }
                )
for field, values in [("size", (0.016, 0.024, 0.028)), ("friction", (0.4, 0.7, 1.3))]:
    for value in values:
        for color in ("red", "blue", "green"):
            for box in ("left", "right"):
                case = {
                    "group": field + "_" + str(value),
                    "seed": 200,
                    "target": color,
                    "bin": box,
                    "size": 0.02,
                    "friction": 1.0,
                    "perception": "rgbd",
                }
                case[field] = value
                cases.append(case)
(a.out / "config.json").write_text(
    json.dumps(
        {
            "cases": cases,
            "model": "one active ViperX arm; other arm idle",
            "object_positions": "color-to-position permutation plus jitter",
            "success": "gripper contact + lift>6cm + released cube stable inside target bin for1second",
            "train_policy": False,
        },
        indent=2,
    )
)
rows = []
for index, case in enumerate(cases):
    out = a.out / f"case_{index:03d}"
    summary = out / "summary.json"
    if not summary.exists():
        if out.exists():
            out.rename(out.with_name(out.name + "_incomplete_" + str(time.time_ns())))
        cmd = [str(python), str(TASK / "scripts/run_episode.py"), "--out", str(out)] + [
            s
            for key in ("seed", "target", "bin", "size", "friction", "perception")
            for s in ("--" + key, str(case[key]))
        ]
        with (a.out / f"case_{index:03d}.log").open("w") as f:
            result = subprocess.run(
                cmd, stdout=f, stderr=subprocess.STDOUT, timeout=120
            )
        if result.returncode:
            rows.append({**case, "process_error": True})
            continue
    rows.append({**case, **json.loads(summary.read_text())})
    (a.out / "results.json").write_text(json.dumps(rows, indent=2))
    print(
        index + 1,
        "/",
        len(cases),
        case["group"],
        "success",
        rows[-1].get("success"),
        flush=True,
    )
groups = {}
for row in rows:
    g = groups.setdefault(row["group"], {"cases": 0, "successes": 0, "errors": 0})
    g["cases"] += 1
    g["successes"] += int(row.get("success", False))
    g["errors"] += int(row.get("process_error", False))
(a.out / "metrics.json").write_text(
    json.dumps({"groups": groups, "total": len(rows)}, indent=2)
)
print(json.dumps(groups, indent=2))
