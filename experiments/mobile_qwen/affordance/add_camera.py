"""Add agentview camera parameters to labels.jsonl and check the depth back-projection.

agentview must be a world-fixed camera (asserted). The check lifts the simulator handle-mask
centroid with the stored depth and reports its distance to the handle geom position, which
bounds the error of the lifting step itself. Run with vla_libero/scripts/python_shared.
Usage: add_camera.py <data_dir>
"""

from pathlib import Path
import json
import sys

import numpy as np

Q = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Q.parents[1] / "experiments/mobile_vla/scripts"))
sys.path.insert(0, str(Q / "affordance"))
from scene import MobileScene  # noqa: E402
from eval_qwen import lift  # noqa: E402

data = Path(sys.argv[1])
s = MobileScene(task=3, initial=0, mobile_start=(0.0, 0.0, 0.0), stow=False)
m, d = s.model, s.data
cid = m.camera_name2id("agentview")
assert m.cam_bodyid[cid] == 0, "agentview is not attached to the world body"
cam = {"R": d.cam_xmat[cid].reshape(3, 3).tolist(), "t": d.cam_xpos[cid].tolist(),
       "f": float(0.5 * 512 / np.tan(np.deg2rad(m.cam_fovy[cid]) / 2))}
rows = [json.loads(x) for x in (data / "labels.jsonl").read_text().splitlines()]
errs = []
for r in rows:
    r["camera"] = cam
    c = r["handle_mask"]["centroid"]
    if c and r["handle_mask"]["pixels"] >= 200:
        aux = np.load(data / "aux" / f"{r['id']}.npz")
        p = lift(r, c, aux["depth"].astype(np.float32))
        errs.append(float(np.linalg.norm(p - np.array(r["handle"]["world"])) * 100))
(data / "labels.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
print("frames", len(rows), "lift check n", len(errs), "median cm %.2f p90 cm %.2f" % (np.median(errs), np.percentile(errs, 90)))
s.close()
