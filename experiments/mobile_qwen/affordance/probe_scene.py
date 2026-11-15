"""Inspect cabinet geoms, drawer joint and render outputs (run with vla_libero/scripts/python_shared)."""

from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/mobile_vla/scripts"))
from scene import MobileScene  # noqa: E402

s = MobileScene(task=3, initial=0, mobile_start=(0.0, 0.0, 0.0), stow=False)
m, d = s.model, s.data
for name in ["white_cabinet_1_bottom_level", "white_cabinet_1_middle_level", "white_cabinet_1_top_level"]:
    j = m.joint_name2id(name)
    print(name, "qpos", d.get_joint_qpos(name), "range", m.jnt_range[j], "axis", m.jnt_axis[j])
for g in range(m.ngeom):
    n = m.geom_id2name(g)
    if n and n.startswith("white_cabinet"):
        print(n, m.body_id2name(m.geom_bodyid[g]), "type", m.geom_type[g], "size", np.round(m.geom_size[g], 3),
              "pos", np.round(d.geom_xpos[g], 3), "group", m.geom_group[g], "rgba", m.geom_rgba[g])
print("bowl", np.round(d.body_xpos[m.body_name2id("akita_black_bowl_1_main")], 3))
for site in ["white_cabinet_1_bottom_region", "white_cabinet_1_middle_region"]:
    i = m.site_name2id(site)
    print(site, np.round(d.site_xpos[i], 3), np.round(m.site_size[i], 3))
seg = s.sim.render(camera_name="agentview", height=128, width=128, segmentation=True)
print("seg", seg.shape, seg.dtype, np.unique(seg[..., 0])[:10], np.unique(seg[..., 1])[:20])
dep = s.sim.render(camera_name="agentview", height=128, width=128, depth=True)
print("depth", type(dep), [x.shape for x in dep] if isinstance(dep, tuple) else dep.shape, float(dep[1].min()), float(dep[1].max()))
print("success", s.e.check_success())
