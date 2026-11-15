"""Official pi05/LIBERO interface with fixed paired trials and complete failure logs."""

from __future__ import annotations
import argparse, collections, hashlib, json, time, traceback, copy
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parents[1]
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
LIMITS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}
DUMMY = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def quat_axis(q):
    q = np.asarray(q).copy()
    w = float(np.clip(q[3], -1.0, 1.0))
    den = np.sqrt(1 - w * w)
    return np.zeros(3) if np.isclose(den, 0.0) else q[:3] * (2 * np.arccos(w) / den)


def make_input(obs, prompt):
    from openpi_client import image_tools

    def image(k):
        return image_tools.convert_to_uint8(
            image_tools.resize_with_pad(
                np.ascontiguousarray(obs[k][::-1, ::-1]), 224, 224
            )
        )

    return {
        "observation/image": image("agentview_image"),
        "observation/wrist_image": image("robot0_eye_in_hand_image"),
        "observation/state": np.concatenate(
            (
                obs["robot0_eef_pos"],
                quat_axis(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )
        ),
        "prompt": prompt,
    }


def initial_states(suite, task_id):
    # Official pinned repository assets contain a legacy torch serialization.
    # Load only this local, trusted upstream file with the explicit legacy option.
    import torch
    from libero.libero import get_libero_path

    task = suite.get_task(task_id)
    path = (
        Path(get_libero_path("init_states"))
        / task.problem_folder
        / task.init_states_file
    )
    return torch.load(path, map_location="cpu", weights_only=False)


def plans(phase):
    p = json.loads((R / "configs/protocol.json").read_text())
    if phase in ["smoke", "baseline", "all"]:
        suites = SUITES[:1] if phase == "smoke" else SUITES
        tids = [0] if phase == "smoke" else p["baseline"]["task_ids"]
        ids = [0] if phase == "smoke" else p["baseline"]["initial_state_ids"]
        for suite in suites:
            for tid in tids:
                for init in ids:
                    yield suite, tid, init, "baseline"
    if phase in ["interventions", "all"]:
        a = p["paired_interventions"]
        for mode in a["modes"]:
            for suite in a["suites"]:
                for tid in a["task_ids"]:
                    for init in a["initial_state_ids"]:
                        yield suite, tid, init, mode


def trial(policy, suite_name, task_id, init_id, mode, out, record_video=True):
    import imageio.v2 as imageio
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[suite_name]()
    task = suite.get_task(task_id)
    states = initial_states(suite, task_id)
    init = np.asarray(states[init_id])
    prompt = task.language
    if mode == "wrong_language":
        prompt = suite.get_task((task_id + 1) % suite.n_tasks).language
    replan_steps = 10 if mode == "replan_10" else 5
    denoising_steps = 5 if mode == "denoising_5" else 10
    if mode == "denoising_5":
        policy = copy.copy(policy)
        policy._sample_kwargs = {**policy._sample_kwargs, "num_steps": 5}
    seed = 700000 + SUITES.index(suite_name) * 10000 + task_id * 100 + init_id
    rng = np.random.default_rng(seed)
    sensor_rng = np.random.default_rng(seed + 900000)
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    config = {
        "runner_sha256": source_hash,
        "suite": suite_name,
        "task_id": task_id,
        "initial_state_id": init_id,
        "mode": mode,
        "environment_seed": 7,
        "noise_seed": seed,
        "initial_state_sha256": sha(init),
        "original_prompt": task.language,
        "input_prompt": prompt,
        "replan_steps": replan_steps,
        "denoising_steps": denoising_steps,
        "max_steps": LIMITS[suite_name],
    }
    result = {**config, "status": "running", "success": False, "steps": 0}
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(config, indent=2))
    start = time.monotonic()
    env = None
    writer = None
    traces = []
    queries = []
    chunks = []
    inputs = []
    error = None
    try:
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
        obs = env.set_init_state(init)
        for _ in range(10):
            obs, _, _, _ = env.step(DUMMY.tolist())
        initial_qpos = env.sim.data.qpos.copy()
        initial_eef = obs["robot0_eef_pos"].copy()
        config["post_wait_qpos_sha256"] = sha(initial_qpos)
        config["post_wait_state"] = make_input(obs, prompt)[
            "observation/state"
        ].tolist()
        config["physics_timestep_s"] = float(env.sim.model.opt.timestep)
        config["control_hz"] = int(env.env.control_freq)
        config["actuator_count"] = int(env.sim.model.nu)
        config["mocap_count"] = int(env.sim.model.nmocap)
        ctrl = env.robots[0].controller
        config["controller_class"] = type(ctrl).__name__
        config["controller_parameters"] = {}
        for key in [
            "input_min",
            "input_max",
            "output_min",
            "output_max",
            "kp",
            "damping_ratio",
            "control_delta",
            "impedance_mode",
        ]:
            if hasattr(ctrl, key):
                value = getattr(ctrl, key)
                config["controller_parameters"][key] = (
                    value.tolist() if hasattr(value, "tolist") else value
                )
        config["actuator_control_range"] = env.sim.model.actuator_ctrlrange.tolist()
        config["actuator_force_range"] = env.sim.model.actuator_forcerange.tolist()
        config["actuator_control_limited"] = env.sim.model.actuator_ctrllimited.tolist()
        config["actuator_force_limited"] = env.sim.model.actuator_forcelimited.tolist()
        config["initial_success_predicate"] = bool(env.check_success())
        if config["initial_success_predicate"]:
            raise ValueError("Initial state already satisfies goal")
        (out / "config.json").write_text(json.dumps(config, indent=2))
        alternate_goal = None
        alternate_hits = 0
        if mode == "wrong_language" and suite_name == "libero_goal":
            from libero.libero.envs.bddl_utils import robosuite_parse_problem

            alt = suite.get_task((task_id + 1) % suite.n_tasks)
            parsed = robosuite_parse_problem(
                str(
                    Path(get_libero_path("bddl_files"))
                    / alt.problem_folder
                    / alt.bddl_file
                )
            )
            needed = {name for pred in parsed["goal_state"] for name in pred[1:]}
            if needed.issubset(env.env.object_states_dict):
                alternate_goal = parsed["goal_state"]
        frozen = make_input(obs, prompt)
        action_plan = collections.deque()
        delay = collections.deque([DUMMY.copy(), DUMMY.copy()])
        if record_video:
            writer = imageio.get_writer(
                out / "rollout.mp4",
                fps=20,
                codec="libx264",
                quality=7,
                macro_block_size=16,
            )
        for step in range(LIMITS[suite_name]):
            frame = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            if writer:
                writer.append_data(frame)
            if not action_plan:
                element = make_input(obs, prompt)
                if mode == "blank_images":
                    for key in ["observation/image", "observation/wrist_image"]:
                        element[key] = np.zeros_like(element[key])
                elif mode == "frozen_images":
                    for key in ["observation/image", "observation/wrist_image"]:
                        element[key] = frozen[key].copy()
                elif mode == "state_noise":
                    element["observation/state"][:3] += sensor_rng.normal(0.0, 0.005, 3)
                noise = rng.standard_normal((10, 32)).astype(np.float32)
                before = time.monotonic()
                output = policy.infer(element, noise=noise)
                wall = time.monotonic() - before
                chunk = np.asarray(output["actions"], dtype=np.float64)
                if chunk.shape != (10, 7) or not np.isfinite(chunk).all():
                    raise ValueError(f"Invalid policy chunk {chunk.shape}")
                inputs.append(
                    {
                        k: v.copy() if isinstance(v, np.ndarray) else v
                        for k, v in element.items()
                    }
                )
                chunks.append(chunk.copy())
                queries.append(
                    {
                        "step": step,
                        "wall_s": wall,
                        "image_sha256": sha(element["observation/image"]),
                        "wrist_sha256": sha(element["observation/wrist_image"]),
                        "state": element["observation/state"].tolist(),
                        "noise_sha256": sha(noise),
                        "policy_timing": output.get("policy_timing", {}),
                    }
                )
                action_plan.extend(chunk[:replan_steps])
            command = np.asarray(action_plan.popleft()).copy()
            if mode == "action_delay":
                delay.append(command)
                applied = delay.popleft()
            else:
                applied = command.copy()
            previous_sim_time = float(env.sim.data.time)
            obs, reward, done, info = env.step(applied.tolist())
            if abs(float(env.sim.data.time) - previous_sim_time - 0.05) > 1e-8:
                raise RuntimeError("Unexpected simulator time jump or reset")
            # Only read state: no qpos assignment, no mocap targets, no applied virtual contact force.
            alternate_ok = (
                bool(all(env.env._eval_predicate(x) for x in alternate_goal))
                if alternate_goal
                else False
            )
            alternate_hits += int(alternate_ok)
            traces.append(
                {
                    "alternate_predicate": alternate_ok,
                    "command": command,
                    "applied": applied,
                    "qpos": env.sim.data.qpos.copy(),
                    "qvel": env.sim.data.qvel.copy(),
                    "ctrl": env.sim.data.ctrl.copy(),
                    "actuator_force": env.sim.data.actuator_force.copy(),
                    "eef": obs["robot0_eef_pos"].copy(),
                    "gripper": obs["robot0_gripper_qpos"].copy(),
                    "predicate": bool(env.check_success()),
                    "sim_time": float(env.sim.data.time),
                }
            )
            if done:
                result["success"] = bool(env.check_success())
                result["status"] = (
                    "success" if result["success"] else "environment_done"
                )
                break
        else:
            result["status"] = "timeout"
        if writer:
            writer.append_data(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
        result["alternate_goal_evaluable"] = alternate_goal is not None
        result["alternate_goal_ever_satisfied"] = (
            bool(alternate_hits) if alternate_goal else None
        )
        result.update(
            steps=len(traces),
            initial_eef=initial_eef.tolist(),
            final_eef=obs["robot0_eef_pos"].tolist(),
            qpos_displacement_norm=float(
                np.linalg.norm(env.sim.data.qpos - initial_qpos)
            ),
            simulation_elapsed_s=len(traces) / 20,
        )
        if inputs:
            np.savez_compressed(
                out / "query_inputs.npz",
                base_rgb=np.stack([x["observation/image"] for x in inputs]),
                wrist_rgb=np.stack([x["observation/wrist_image"] for x in inputs]),
                state=np.stack([x["observation/state"] for x in inputs]),
                noise_seed=seed,
            )
            np.savez_compressed(out / "action_chunks.npz", actions=np.stack(chunks))
        if traces:
            np.savez_compressed(
                out / "trace.npz",
                **{k: np.asarray([x[k] for x in traces]) for k in traces[0]},
            )
        (out / "queries.json").write_text(json.dumps(queries, indent=2))
    except Exception:
        error = traceback.format_exc()
        result["status"] = "error"
        result["error"] = error
        result["steps"] = len(traces)
        if traces:
            np.savez_compressed(
                out / "partial_trace.npz",
                **{k: np.asarray([x[k] for x in traces]) for k in traces[0]},
            )
        (out / "queries.json").write_text(json.dumps(queries, indent=2))
        print(error, flush=True)
    finally:
        if writer:
            writer.close()
        if env is not None:
            env.close()
    result["wall_s"] = time.monotonic() - start
    result["query_count"] = len(queries)
    result["warmup_query_wall_s"] = queries[0]["wall_s"] if queries else None
    result["query_median_excluding_first_s"] = (
        float(np.median([x["wall_s"] for x in queries[1:]]))
        if len(queries) > 1
        else None
    )
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                k: result[k]
                for k in [
                    "suite",
                    "task_id",
                    "initial_state_id",
                    "mode",
                    "status",
                    "steps",
                    "wall_s",
                ]
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if error:
        raise RuntimeError(error)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--phase",
        choices=["smoke", "baseline", "interventions", "all"],
        default="smoke",
    )
    p.add_argument("--limit", type=int)
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--suite", choices=SUITES)
    p.add_argument("--task-id", type=int, choices=range(10))
    p.add_argument(
        "--repeat-tag",
        help="Separate repeated demonstration; excluded from formal results",
    )
    a = p.parse_args()
    jobs = list(plans(a.phase))
    if a.suite:
        jobs = [j for j in jobs if j[0] == a.suite]
    if a.task_id is not None:
        jobs = [j for j in jobs if j[1] == a.task_id]
    if a.repeat_tag and (not a.repeat_tag.replace("_", "").replace("-", "").isalnum()):
        p.error("repeat-tag must contain letters, digits, underscores or hyphens")
    result_root = R / "replays" / a.repeat_tag if a.repeat_tag else R / "results"
    jobs = jobs[: a.limit] if a.limit else jobs
    pending = []
    for suite, tid, init, mode in jobs:
        out = result_root / mode / suite / f"task{tid:02d}_init{init:02d}"
        if (out / "result.json").exists():
            old = json.loads((out / "result.json").read_text())
            if old["status"] != "error":
                continue
            with (R / "logs/error_history.jsonl").open("a") as f:
                f.write(json.dumps(old) + "\n")
        pending.append((suite, tid, init, mode, out))
    if not pending:
        print("All requested trials already saved")
        return
    import jax
    from openpi.training import config
    from openpi.policies import policy_config

    print("Devices", jax.devices(), flush=True)
    policy = policy_config.create_trained_policy(
        config.get_config("pi05_libero"),
        R / "checkpoints/pi05_libero",
        sample_kwargs={"num_steps": 10},
    )
    for suite, tid, init, mode, out in pending:
        trial(policy, suite, tid, init, mode, out, not a.no_video)


if __name__ == "__main__":
    main()
