# ruff: noqa: E402
"""Offline diagnostic only: replay and inspect the official plate-contact predicate."""

from pathlib import Path
import sys
import json
import numpy as np

R = Path(__file__).resolve().parents[1]
V = R.parent
sys.path.insert(0, str(V / "scripts"))
import evaluate as ev
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

suite = benchmark.get_benchmark_dict()["libero_10"]()
task = suite.get_task(4)
env = OffScreenRenderEnv(
    bddl_file_name=str(
        Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    ),
    camera_heights=256,
    camera_widths=256,
)
source = V / "hierarchical/zh_eval_hier_v2/libero_10/task04_init02/trace.npz"
rows = {}
try:
    for mode in ["prefix", "continue", "full", "replan"]:
        env.seed(7)
        env.reset()
        env.set_init_state(np.asarray(ev.initial_states(suite, 4)[2]))
        for _ in range(10):
            env.step(ev.DUMMY.tolist())
        for a in np.load(source)["applied"][:150]:
            env.step(a.tolist())
        if mode != "prefix":
            for a in np.load(
                R / "results/task04_init02" / f"{mode}_seed31001/trace.npz"
            )["applied"]:
                env.step(a.tolist())
        e = env.env
        pos = {
            name: e.sim.data.body_xpos[e.obj_body_id[name]].copy()
            for name in ["porcelain_mug_1", "white_yellow_mug_1", "plate_1", "plate_2"]
        }
        detail = []
        for mug, plate in [
            ("porcelain_mug_1", "plate_1"),
            ("white_yellow_mug_1", "plate_2"),
        ]:
            detail.append(
                {
                    "mug": mug,
                    "plate": plate,
                    "xy_distance_m": float(
                        np.linalg.norm(pos[mug][:2] - pos[plate][:2])
                    ),
                    "contact": bool(
                        e.check_contact(e.get_object(mug), e.get_object(plate))
                    ),
                    "mug_above_plate": bool(pos[mug][2] >= pos[plate][2]),
                    "predicate": bool(e._eval_predicate(["on", mug, plate])),
                }
            )
        rows[mode] = {
            "positions": {k: v.tolist() for k, v in pos.items()},
            "predicate_details": detail,
        }
finally:
    env.close()
(R / "plate_contact_diagnostic.json").write_text(json.dumps(rows, indent=2))
for k, v in rows.items():
    print(k, v["predicate_details"])
