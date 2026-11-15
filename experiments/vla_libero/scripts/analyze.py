"""Task-level success, paired interventions, and observation/action checks."""

from pathlib import Path
import json, math
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[1]
LABEL = {
    "baseline": "原始输入",
    "blank_images": "图像置零",
    "frozen_images": "冻结初始画面",
    "wrong_language": "替换指令（原目标）",
    "state_noise": "状态位置噪声5mm",
    "action_delay": "动作延迟100ms",
    "replan_10": "每10步重规划",
    "denoising_5": "5步去噪",
}
plt.rcParams.update(
    {"font.family": "Noto Sans CJK JP", "axes.unicode_minus": False, "font.size": 10}
)


def wilson(k, n):
    if not n:
        return [None, None]
    z = 1.959963984540054
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [center - half, center + half]


def aggregate(rows):
    k = sum(r["success"] for r in rows)
    n = len(rows)
    return {
        "n": n,
        "successes": k,
        "rate": k / n if n else None,
        "wilson95": wilson(k, n),
        "errors": sum(r["status"] == "error" for r in rows),
    }


def main():
    rows = []
    for p in sorted((R / "results").rglob("result.json")):
        r = json.loads(p.read_text())
        r["directory"] = str(p.parent.relative_to(R))
        rows.append(r)
    summary = {
        "total": aggregate(rows),
        "baseline_by_suite": {},
        "interventions": {},
        "tasks": {},
        "timings": {},
        "rows": rows,
    }
    base = {
        (r["suite"], r["task_id"], r["initial_state_id"]): r
        for r in rows
        if r["mode"] == "baseline"
    }
    for suite in ["libero_spatial", "libero_object", "libero_goal", "libero_10"]:
        vals = [r for r in rows if r["mode"] == "baseline" and r["suite"] == suite]
        if vals:
            summary["baseline_by_suite"][suite] = aggregate(vals)
    for r in rows:
        if r["mode"] != "baseline":
            continue
        key = f"{r['suite']}/{r['task_id']:02d}"
        summary["tasks"][key] = aggregate(
            [
                v
                for v in rows
                if v["mode"] == "baseline"
                and v["suite"] == r["suite"]
                and v["task_id"] == r["task_id"]
            ]
        )
    for mode in list(LABEL)[1:]:
        vals = [r for r in rows if r["mode"] == mode]
        pairs = [
            (base[(r["suite"], r["task_id"], r["initial_state_id"])], r)
            for r in vals
            if (r["suite"], r["task_id"], r["initial_state_id"]) in base
        ]
        if not vals:
            continue
        compared = []
        for b, r in pairs:
            bp = R / b["directory"]
            rp = R / r["directory"]
            if (
                not (bp / "action_chunks.npz").exists()
                or not (rp / "action_chunks.npz").exists()
            ):
                continue
            bc = np.load(bp / "action_chunks.npz")["actions"][0]
            rc = np.load(rp / "action_chunks.npz")["actions"][0]
            bi = np.load(bp / "query_inputs.npz")
            ri = np.load(rp / "query_inputs.npz")
            compared.append(
                {
                    "suite": r["suite"],
                    "task_id": r["task_id"],
                    "initial_state_id": r["initial_state_id"],
                    "first_chunk_mae": float(np.abs(bc - rc).mean()),
                    "same_initial_state": b["initial_state_sha256"]
                    == r["initial_state_sha256"],
                    "same_first_state": bool(
                        np.array_equal(bi["state"][0], ri["state"][0])
                    ),
                    "same_first_base_image": bool(
                        np.array_equal(bi["base_rgb"][0], ri["base_rgb"][0])
                    ),
                    "same_first_wrist_image": bool(
                        np.array_equal(bi["wrist_rgb"][0], ri["wrist_rgb"][0])
                    ),
                }
            )
        summary["interventions"][mode] = {
            **aggregate(vals),
            "matched_baseline": aggregate([b for b, r in pairs]),
            "success_to_failure": sum(
                b["success"] and not r["success"] for b, r in pairs
            ),
            "failure_to_success": sum(
                not b["success"] and r["success"] for b, r in pairs
            ),
            "first_chunk_comparisons": compared,
            "alternate_goal_evaluable": sum(
                bool(r.get("alternate_goal_evaluable")) for r in vals
            ),
            "alternate_goal_ever_satisfied": sum(
                bool(r.get("alternate_goal_ever_satisfied")) for r in vals
            ),
        }
    times = []
    for r in rows:
        p = R / r["directory"] / "queries.json"
        if p.exists():
            times.extend(x["wall_s"] for x in json.loads(p.read_text())[1:])
    if times:
        summary["timings"] = {
            "queries_excluding_first_per_episode": len(times),
            "wall_median_s": float(np.median(times)),
            "wall_p95_s": float(np.percentile(times, 95)),
            "note": "Synchronous simulator pauses during policy inference; not a real-time deployment claim.",
        }
    (R / "summary.json").write_text(json.dumps(summary, indent=2))
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
    vals = list(summary["baseline_by_suite"].items())
    if vals:
        axs[0].bar(
            [k.removeprefix("libero_") for k, v in vals],
            [v["rate"] for k, v in vals],
            color="#407ab2",
        )
        for i, (k, v) in enumerate(vals):
            axs[0].text(i, v["rate"] + 0.03, f"{v['successes']}/{v['n']}", ha="center")
    axs[0].set(
        title="官方检查点：本机小样本复现", ylabel="任务成功比例", ylim=(0, 1.15)
    )
    for i, (mode, v) in enumerate(summary["interventions"].items()):
        axs[1].bar(
            i - 0.17,
            v["matched_baseline"]["rate"],
            0.34,
            color="#407ab2",
            label="配对基线" if i == 0 else None,
        )
        axs[1].bar(
            i + 0.17, v["rate"], 0.34, color="#ce8059", label="干预" if i == 0 else None
        )
    axs[1].set_xticks(
        range(len(summary["interventions"])),
        [LABEL[k] for k in summary["interventions"]],
        rotation=30,
        ha="right",
    )
    axs[1].set(title="相同初始状态的干预对照", ylim=(0, 1.1))
    if summary["interventions"]:
        axs[1].legend()
    fig.tight_layout()
    fig.savefig(R / "figures/success_summary.png", dpi=170)
    plt.close(fig)
    print(
        json.dumps(
            {
                "baseline": summary["baseline_by_suite"],
                "interventions": {
                    k: {a: b for a, b in v.items() if a != "first_chunk_comparisons"}
                    for k, v in summary["interventions"].items()
                },
                "timings": summary["timings"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
