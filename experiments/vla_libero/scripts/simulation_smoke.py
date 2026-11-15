"""Check official tasks, physical actuators and no-op dynamics before policy loading."""

from pathlib import Path
import json
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from evaluate import make_input, DUMMY, sha, initial_states

R = Path(__file__).resolve().parents[1]
suite = benchmark.get_benchmark_dict()["libero_spatial"]()
task = suite.get_task(0)
env = OffScreenRenderEnv(
    bddl_file_name=str(
        Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    ),
    camera_heights=256,
    camera_widths=256,
)
env.seed(7)
obs = env.reset()
obs = env.set_init_state(initial_states(suite, 0)[0])
for _ in range(10):
    obs, _, _, _ = env.step(DUMMY.tolist())
x = make_input(obs, task.language)
model = env.sim.model
assert x["observation/image"].shape == (224, 224, 3)
assert x["observation/state"].shape == (8,)
assert model.nmocap == 0
assert model.nu > 0
info = {
    "task": task.language,
    "state": x["observation/state"].tolist(),
    "mocap_count": model.nmocap,
    "actuators": model.nu,
    "physics_dt_s": model.opt.timestep,
    "control_hz": env.env.control_freq,
    "controller_class": type(env.robots[0].controller).__name__,
    "initial_predicate": bool(env.check_success()),
    "base_image_sha256": sha(x["observation/image"]),
    "wrist_image_sha256": sha(x["observation/wrist_image"]),
}
from PIL import Image

Image.fromarray(x["observation/image"]).save(R / "figures/initial_scene.png")
Image.fromarray(x["observation/wrist_image"]).save(R / "figures/initial_wrist.png")
(R / "simulation_check.json").write_text(json.dumps(info, indent=2))
print(json.dumps(info, indent=2))
env.close()
