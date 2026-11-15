"""Separate physical action replay and post-success release/hold check.

Does not change official rollout scores or count replays as new VLA trials.
"""

from pathlib import Path
import json, hashlib, argparse
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from evaluate import initial_states, DUMMY, sha

R = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    records = []
    paths = sorted((R / "results/baseline").rglob("result.json"))
    if args.pilot:
        paths = [R / "results/baseline/libero_spatial/task00_init00/result.json"]
    output_root = R / ("release_pilot" if args.pilot else "release_checks")
    for p in paths:
        r = json.loads(p.read_text())
        if not r["success"]:
            continue
        c = json.loads((p.parent / "config.json").read_text())
        dest = (
            output_root
            / r["suite"]
            / f"task{r['task_id']:02d}_init{r['initial_state_id']:02d}"
        )
        dest.mkdir(parents=True, exist_ok=True)
        if (dest / "result.json").exists():
            records.append(json.loads((dest / "result.json").read_text()))
            continue
        suite = benchmark.get_benchmark_dict()[r["suite"]]()
        task = suite.get_task(r["task_id"])
        env = OffScreenRenderEnv(
            bddl_file_name=str(
                Path(get_libero_path("bddl_files"))
                / task.problem_folder
                / task.bddl_file
            ),
            camera_heights=256,
            camera_widths=256,
        )
        env.seed(7)
        env.reset()
        obs = env.set_init_state(
            initial_states(suite, r["task_id"])[r["initial_state_id"]]
        )
        for _ in range(10):
            obs, _, _, _ = env.step(DUMMY.tolist())
        assert sha(env.sim.data.qpos) == c["post_wait_qpos_sha256"]
        trace = np.load(p.parent / "trace.npz")
        maxerr = 0.0
        for i, action in enumerate(trace["applied"]):
            obs, _, _, _ = env.step(action.tolist())
            maxerr = max(
                maxerr, float(np.max(np.abs(env.sim.data.qpos - trace["qpos"][i])))
            )
        assert maxerr < 1e-8, (r["suite"], r["task_id"], maxerr)
        assert env.check_success()
        frames = [np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])]
        rows = []
        for _ in range(20):
            obs, _, _, _ = env.step(DUMMY.tolist())
            frames.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
            rows.append(
                {
                    "predicate": bool(env.check_success()),
                    "qpos": env.sim.data.qpos.copy(),
                    "qvel": env.sim.data.qvel.copy(),
                    "gripper": obs["robot0_gripper_qpos"].copy(),
                    "ctrl": env.sim.data.ctrl.copy(),
                    "sim_time": float(env.sim.data.time),
                }
            )
        result = {
            k: r[k]
            for k in ["suite", "task_id", "initial_state_id", "initial_state_sha256"]
        }
        result.update(
            replay_max_qpos_error=maxerr,
            probe="1s zero Cartesian increment and gripper open; last 0.5s predicate sustained",
            official_success=True,
            final_goal=rows[-1]["predicate"],
            last_half_second_goal_maintained=bool(
                all(x["predicate"] for x in rows[-10:])
            ),
            gripper_qpos_after=rows[-1]["gripper"].tolist(),
            finger_joint_open_threshold_m=0.03,
            gripper_open_confirmed=bool(np.min(np.abs(rows[-1]["gripper"])) >= 0.03),
            open_and_stable=bool(
                np.min(np.abs(rows[-1]["gripper"])) >= 0.03
                and all(x["predicate"] for x in rows[-10:])
            ),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        )
        np.savez_compressed(
            dest / "trace.npz", **{k: np.asarray([x[k] for x in rows]) for k in rows[0]}
        )
        np.save(dest / "frames.npy", np.stack(frames))
        (dest / "result.json").write_text(json.dumps(result, indent=2))
        env.close()
        records.append(result)
        print(json.dumps(result), flush=True)
    summary = {
        "checked": len(records),
        "official_successes_replayed": len(records),
        "replay_max_qpos_error": max(
            (v["replay_max_qpos_error"] for v in records), default=0
        ),
        "stable_last_half_second": sum(
            v["last_half_second_goal_maintained"] for v in records
        ),
        "final_goal_true": sum(v["final_goal"] for v in records),
        "open_confirmed": sum(v["gripper_open_confirmed"] for v in records),
        "open_and_stable": sum(v["open_and_stable"] for v in records),
        "by_suite": {},
        "rows": records,
        "note": "Supplementary replay with explicit release/hold, not additional VLA trials and not the official benchmark definition.",
    }
    for suite in sorted({v["suite"] for v in records}):
        vals = [v for v in records if v["suite"] == suite]
        summary["by_suite"][suite] = {
            "n": len(vals),
            "stable": sum(v["last_half_second_goal_maintained"] for v in vals),
            "final_goal": sum(v["final_goal"] for v in vals),
            "open_and_stable": sum(v["open_and_stable"] for v in vals),
        }
    (
        R / ("release_pilot_summary.json" if args.pilot else "release_summary.json")
    ).write_text(json.dumps(summary, indent=2))
    print(
        "Release checks",
        summary["stable_last_half_second"],
        "/",
        summary["checked"],
        flush=True,
    )


if __name__ == "__main__":
    main()
