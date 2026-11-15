# ruff: noqa: E402
from pathlib import Path
import sys
import json
import numpy as np

R = Path(__file__).resolve().parents[1]
V = R.parent / "vla_libero"
sys.path.insert(0, str(V / "scripts"))
import evaluate as ev
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

s = benchmark.get_benchmark_dict()["libero_10"]()
t = s.get_task(0)
e = OffScreenRenderEnv(
    bddl_file_name=str(
        Path(get_libero_path("bddl_files")) / t.problem_folder / t.bddl_file
    ),
    camera_heights=256,
    camera_widths=256,
)
e.seed(7)
e.reset()
e.set_init_state(np.asarray(ev.initial_states(s, 0)[0]))
(R / "research/original_scene.xml").write_text(e.sim.model.get_xml())
m = e.sim.model
d = e.sim.data
info = {
    "joints": list(m.joint_names),
    "actuators": list(m.actuator_names),
    "robot_root": e.env.robots[0].robot_model.root_body,
    "bodies": [
        {"name": name, "position": d.get_body_xpos(name).tolist()}
        for name in m.body_names
        if name in ["robot0_base", "robot0_link0", "table", "floor", "mount0_base"]
    ],
    "cameras": list(m.camera_names),
    "goals": e.env.parsed_problem["goal_state"],
}
print(json.dumps(info, indent=2))
(R / "research/scene_probe.json").write_text(json.dumps(info, indent=2))
e.close()
