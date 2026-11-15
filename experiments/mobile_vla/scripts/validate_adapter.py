# ruff: noqa: E402
from pathlib import Path
import sys
import json
import numpy as np
import imageio.v2 as imageio
from PIL import Image

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "scripts"))
from scene import MobileScene

rows = []
for task in [3, 9]:
    out = R / f"results/adapter_task{task:02d}"
    out.mkdir(exist_ok=True)
    s = MobileScene(task=task, mobile_start=(0, 0, 0), stow=False)
    tr = np.load(
        R.parent
        / f"vla_libero/results/baseline/libero_10/task{task:02d}_init00/trace.npz"
    )
    w = imageio.get_writer(out / "replay.mp4", fps=20, codec="libx264", quality=7)
    try:
        s.park()
        s.mode = "manip"
        s.hold[:] = 0
        ee = []
        base = []
        for action in tr["applied"]:
            obs, _ = s.step(action)
            ee.append(obs["robot0_eef_pos"].copy())
            base.append(s.pose().copy())
            w.append_data(
                s.sim.render(camera_name="mobile_overview", height=480, width=640)[::-1]
            )
        info = {
            "task": task,
            "replay_success": bool(s.e.check_success()),
            "steps": len(ee),
            "max_base_drift_m": float(
                np.max(np.linalg.norm(np.asarray(base)[:, :2] - s.dock[:2], axis=1))
            ),
            "final_base": s.pose().tolist(),
            "max_eef_difference_m": float(
                np.max(np.linalg.norm(np.asarray(ee) - tr["eef"], axis=1))
            )
            if "eef" in tr
            else None,
            "same_world": True,
        }
        rows.append(info)
        (out / "result.json").write_text(json.dumps(info, indent=2))
        print(json.dumps(info), flush=True)
        Image.fromarray(
            s.sim.render(camera_name="agentview", height=512, width=512)[::-1, ::-1]
        ).save(out / "agent.png")
    finally:
        w.close()
        s.close()
(R / "results/adapter_validation.json").write_text(json.dumps(rows, indent=2))
