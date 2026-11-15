"""Three fixed integration cases with the official tf2 runtime overlay."""

from pathlib import Path
import subprocess, json, sys

R = Path(__file__).resolve().parents[1]
CASES = [(0, "tf22524_i00", 82), (1, "tf22524_i01", 83), (2, "tf22524_i02", 84)]
protocol = {
    "cases": CASES,
    "suite": "libero_10",
    "task": 3,
    "runtime": "official tf2/tf2_ros 0.25.24 scoped library overlay",
    "goal": "same-world navigation, parked unfolding, official pi0.5 task completion",
    "scope": "development regression, three public initial states; not population success estimate",
    "localization": "known map and ideal simulator odometry",
    "policy": "official pi05_libero, fixed English full-task instruction; Qwen modules not integrated yet",
}
(R / "REGRESSION_PROTOCOL.json").write_text(json.dumps(protocol, indent=2))
rows = []
for initial, tag, domain in CASES:
    target = R / "results" / tag
    if not (target / "navigation.json").exists():
        if target.exists():
            raise RuntimeError(
                f"Incomplete directory needs inspection, not overwrite: {target}"
            )
        subprocess.run(
            [
                sys.executable,
                str(R / "scripts/run_case.py"),
                "--initial",
                str(initial),
                "--tag",
                tag,
                "--domain",
                str(domain),
                "--wall-clock",
                "--tf-overlay",
                str(R / "runtime/geometry2_0_25_24/root"),
            ],
            check=True,
        )
    r = json.loads((target / "navigation.json").read_text())
    row = {
        "initial": initial,
        "tag": tag,
        "navigation_success": r["navigation_success"],
        "manipulation_success": r["manipulation_success"],
        "error": r.get("error"),
    }
    if r["navigation_success"] and r["manipulation_success"]:
        subprocess.run(
            [sys.executable, str(R / "scripts/verify_case.py"), str(target)], check=True
        )
        row["verification"] = json.loads((target / "verification.json").read_text())
    rows.append(row)
    (R / "results/regression_summary.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(row), flush=True)
