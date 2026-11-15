# ruff: noqa: E402
from pathlib import Path
import sys
import json
import numpy as np

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R.parent / "vla_libero/scripts"))
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

s = benchmark.get_benchmark_dict()["libero_10"]()
rows = []
for task in [2, 3, 4, 6, 9]:
    t = s.get_task(task)
    e = OffScreenRenderEnv(
        bddl_file_name=str(
            Path(get_libero_path("bddl_files")) / t.problem_folder / t.bddl_file
        ),
        camera_heights=64,
        camera_widths=64,
    )
    e.reset()
    m = e.sim.model
    d = e.sim.data
    root = d.get_body_xpos("robot0_base").copy()
    bounds = []
    for g in range(m.ngeom):
        body = m.geom_bodyid[g]
        name = m.body_id2name(body) or ""
        if (
            m.geom_type[g] != 6
            or not (m.geom_contype[g] or m.geom_conaffinity[g])
            or name.startswith(("robot", "gripper", "mount"))
        ):
            continue
        b = body
        while b and m.body_jntnum[b] == 0:
            b = m.body_parentid[b]
        if b:
            continue
        half = np.abs(d.geom_xmat[g].reshape(3, 3)) @ m.geom_size[g]
        lo = d.geom_xpos[g] - half
        hi = d.geom_xpos[g] + half
        if hi[2] > 0.1:
            bounds.append({"body": name, "lo": lo.tolist(), "hi": hi.tolist()})
    info = {"task": task, "root": root.tolist(), "fixtures": bounds}
    rows.append(info)
    print(json.dumps(info), flush=True)
    e.close()
(R / "research/station_bounds.json").write_text(json.dumps(rows, indent=2))
