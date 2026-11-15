"""Verify completed episode files, shadow timestamps and full video decoding."""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import subprocess
import numpy as np
import imageio_ffmpeg

R = Path(__file__).resolve().parents[1]
V = R.parent


def one(p):
    r = json.loads(p.read_text())
    tr = np.load(p.with_name("trace.npz"))
    assert r["status"] in ["success", "timeout"]
    assert r["steps"] == len(tr["applied"])
    assert r["success"] == bool(tr["success_predicate"][-1]) == all(r["final_atoms"])
    assert np.array_equal(tr["goal_atoms"][-1], r["final_atoms"])
    for e in r["events"]:
        if e["kind"] == "check":
            assert np.array_equal(
                tr["goal_atoms"][e["step"] - 1], e["shadow_long_task_atoms"]
            )
    count, duration = imageio_ffmpeg.count_frames_and_secs(
        str(p.with_name("rollout.mp4"))
    )
    assert count == r["steps"] + 1, (p, count, r["steps"])
    # count_frames_and_secs decodes all frames; also surface codec errors explicitly.
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-v",
            "error",
            "-threads",
            "1",
            "-i",
            str(p.with_name("rollout.mp4")),
            "-f",
            "null",
            "-",
        ],
        check=True,
        capture_output=True,
    )
    return {
        "source": str(p.relative_to(V)),
        "frames": count,
        "duration": duration,
        "steps": r["steps"],
        "check_count": r["qwen_checks"],
    }


def main():
    paths = sorted(
        (V / "hierarchical/checker_v2_exploratory_base/libero_10").glob("*/result.json")
    ) + sorted(
        (V / "hierarchical/checker_v2_exploratory_v2/libero_10").glob("*/result.json")
    )
    assert len(paths) == 60, "Complete all paired episodes before final verification"
    with ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(one, paths))

    def sha(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()

    frozen = json.loads((R / "EXPLORATORY_PROTOCOL.json").read_text())
    assert (
        sha(R / "results/model/best_adapter/adapter_model.safetensors")
        == frozen["adapter_sha256"]
    )
    result = {
        "episodes": len(rows),
        "all_video_frames_decoded": True,
        "shadow_trace_timestamp_alignment": True,
        "weights_unchanged": True,
        "steps": sum(x["steps"] for x in rows),
        "frames": sum(x["frames"] for x in rows),
        "rows": rows,
        "source_sha256": {
            str(p.relative_to(V)): sha(p)
            for p in [
                V / "scripts/run_hierarchical.py",
                V / "scripts/qwen_planner.py",
                V / "scripts/completion_checker_v2.py",
                R / "scripts/evaluate_loop.py",
                R / "scripts/summarize_loop.py",
                R / "scripts/verify_exploratory.py",
            ]
        },
    }
    (R / "EXPLORATORY_VERIFICATION.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ["rows", "source_sha256"]}
        )
    )


if __name__ == "__main__":
    main()
