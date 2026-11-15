# ruff: noqa: E402
from pathlib import Path
import json
import sys
import numpy as np
from PIL import Image

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "scripts"))
from scene import MobileScene

s = MobileScene(mobile_start=(0, 0, 0), stow=False)
try:
    (R / "results/mobile_scene.xml").write_text(s.model.get_xml())
    start = s.pose().copy()
    s.command[:] = [0, 0.15, 0]
    for _ in range(40):
        s.step()
    end = s.pose().copy()
    s.command[:] = 0
    for _ in range(20):
        s.step()
    s.engage_manipulation(20)
    Image.fromarray(
        s.sim.render(camera_name="mobile_overview", height=600, width=800)[::-1]
    ).save(R / "results/mobile_overview.png")
    info = {
        "start": start.tolist(),
        "after_2s": end.tolist(),
        "displacement_m": float(np.linalg.norm(end[:2] - start[:2])),
        "joint_initial_max_error": float(
            np.max(np.abs(s.data.qpos[s.arm_indices] - s.arm_q_initial))
        ),
        "single_sim": True,
        "nq": s.model.nq,
        "nu": s.model.nu,
        "base_actuator_forces": s.data.actuator_force[s.act].tolist(),
    }
    print(json.dumps(info, indent=2))
    (R / "results/scene_smoke.json").write_text(json.dumps(info, indent=2))
finally:
    s.close()
