"""Generate an auto-labelled affordance / state dataset from the mobile kitchen scene.

Frames come from (a) open-loop replay of recorded mobile_qwen episodes (realistic
arm occlusion and drawer motion) and (b) static frames at every initial state with
the bottom drawer opening randomised (bowl on the table). Labels are read from the
simulator only; nothing here is shown to Qwen except the RGB image.
Run with vla_libero/scripts/python_shared.
Usage: gen_dataset.py <out_dir> [--every 10] [--static-per-init 3]
"""

from pathlib import Path
import argparse
import json
import sys

import numpy as np
from PIL import Image

Q = Path(__file__).resolve().parents[1]
ROOT = Q.parents[1]
sys.path.insert(0, str(ROOT / "experiments/mobile_vla/scripts"))
from scene import MobileScene  # noqa: E402

H = W = 512
DRAWER_JOINT = "white_cabinet_1_bottom_level"
HANDLE_GEOM = "white_cabinet_1_g39"
DRAWER_BODY = "white_cabinet_1_cabinet_bottom"
BOWL_BODY = "akita_black_bowl_1_main"
REGION_SITE = "white_cabinet_1_bottom_region"
CLASSES = {"background": 0, "handle": 1, "drawer": 2, "bowl": 3, "cabinet": 4, "robot": 5}


class Labeller:
    def __init__(self, s):
        self.s, m = s, s.model
        self.cam = m.camera_name2id("agentview")
        self.f = 0.5 * H / np.tan(np.deg2rad(m.cam_fovy[self.cam]) / 2)
        self.near = m.vis.map.znear * m.stat.extent
        self.far = m.vis.map.zfar * m.stat.extent
        self.geom_class = np.zeros(m.ngeom, np.uint8)
        for g in range(m.ngeom):
            body = m.body_id2name(m.geom_bodyid[g]) or ""
            name = m.geom_id2name(g) or ""
            if name == HANDLE_GEOM:
                self.geom_class[g] = CLASSES["handle"]
            elif body == DRAWER_BODY:
                self.geom_class[g] = CLASSES["drawer"]
            elif body.startswith("akita_black_bowl"):
                self.geom_class[g] = CLASSES["bowl"]
            elif body.startswith("white_cabinet"):
                self.geom_class[g] = CLASSES["cabinet"]
            elif body.startswith(("robot0", "gripper0")):
                self.geom_class[g] = CLASSES["robot"]

    @staticmethod
    def flip(x):
        return np.ascontiguousarray(x[::-1][:, ::-1])

    def project(self, p):
        """World point -> (u, v) in the flipped 512px image used everywhere in mobile_qwen."""
        d = self.s.data
        R, t = d.cam_xmat[self.cam].reshape(3, 3), d.cam_xpos[self.cam]
        c = R.T @ (np.asarray(p) - t)
        u = W / 2 + self.f * c[0] / -c[2]
        v = H / 2 - self.f * c[1] / -c[2]
        return [float(W - 1 - u), float(v)], float(-c[2])

    def unproject(self, u, v, depth_m):
        d = self.s.data
        R, t = d.cam_xmat[self.cam].reshape(3, 3), d.cam_xpos[self.cam]
        x = (W - 1 - u - W / 2) / self.f * depth_m
        y = -(v - H / 2) / self.f * depth_m
        return (R @ np.array([x, y, -depth_m]) + t).tolist()

    def render(self):
        sim = self.s.sim
        rgb = self.flip(sim.render(camera_name="agentview", height=H, width=W))
        _, z = sim.render(camera_name="agentview", height=H, width=W, depth=True)
        z = self.flip(z)
        depth = self.near / (1 - z * (1 - self.near / self.far))
        seg = self.flip(sim.render(camera_name="agentview", height=H, width=W, segmentation=True))
        cls = np.where(seg[..., 0] == 5, self.geom_class[np.clip(seg[..., 1], 0, None)], 0).astype(np.uint8)
        return rgb, depth.astype(np.float32), cls

    def labels(self, cls):
        s, m, d = self.s, self.s.model, self.s.data
        handle = d.geom_xpos[m.geom_name2id(HANDLE_GEOM)].copy()
        bowl = d.body_xpos[m.body_name2id(BOWL_BODY)].copy()
        region = m.site_name2id(REGION_SITE)
        rc, rs = d.site_xpos[region].copy(), m.site_size[region]
        rR = d.site_xmat[region].reshape(3, 3)
        local = rR.T @ (bowl - rc)
        out = {
            "drawer_opening_m": float(-d.get_joint_qpos(DRAWER_JOINT)),
            "bowl_in_drawer_region": bool(np.all(np.abs(local) <= rs + 0.02)),
            "official_success": bool(s.e.check_success()),
        }
        for name, p in [("handle", handle), ("bowl", bowl), ("drawer_interior", rc)]:
            uv, depth = self.project(p)
            out[name] = {"world": p.tolist(), "uv": uv, "cam_depth_m": depth}
        for name in ("handle", "drawer", "bowl", "cabinet", "robot"):
            ys, xs = np.nonzero(cls == CLASSES[name])
            out[name + "_mask"] = {
                "pixels": int(xs.size),
                "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if xs.size else None,
                "centroid": [float(xs.mean()), float(ys.mean())] if xs.size else None,
            }
        cab = np.isin(cls, [CLASSES[k] for k in ("handle", "drawer", "cabinet")])
        ys, xs = np.nonzero(cab)
        out["cabinet_all_bbox"] = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if xs.size else None
        return out


def save(out, fid, lab, rgb, depth, cls, meta):
    (out / "rgb").mkdir(parents=True, exist_ok=True)
    (out / "aux").mkdir(exist_ok=True)
    Image.fromarray(rgb).save(out / "rgb" / f"{fid}.png")
    np.savez_compressed(out / "aux" / f"{fid}.npz", depth=depth.astype(np.float16), cls=cls)
    return {"id": fid, **meta, **lab}


def dock_scene(initial, base_qpos):
    s = MobileScene(task=3, initial=initial, mobile_start=tuple(base_qpos), stow=True)
    s.engage_manipulation()
    return s


def replay(out, case, every, initial=None):
    """Replay a recorded action sequence; with `initial` given, on another initial state (transfer replay)."""
    source = "replay" if initial is None else "transfer"
    if initial is None:
        initial = json.loads((case / "summary.json").read_text())["initial"]
    trace = json.loads((case / "trace.json").read_text())
    start = next(x for x in trace if x["stage"] == "manipulation")
    s = dock_scene(initial, (0.0, 0.0, 0.0))
    base = np.array(start["base"])
    s.data.qpos[s.joints] = np.r_[base[:2] - s.dock[:2], base[2]]
    s.hold = s.data.qpos[s.joints].copy()
    s.sim.forward()
    lab = Labeller(s)
    actions = np.load(case / "policy_actions.npz")["applied"]
    rows = []
    for k, a in enumerate(actions):
        s.step(a)
        if (k + 1) % every == 0:
            rgb, depth, cls = lab.render()
            fid = f"{case.name}_s{k + 1:04d}" if source == "replay" else f"transfer_i{initial:02d}_{case.name}_s{k + 1:04d}"
            rows.append(save(out, fid, lab.labels(cls), rgb, depth, cls,
                             {"source": source, "case": case.name, "initial": initial, "step": k + 1}))
    s.close()
    return rows


def static(out, initial, n, rng):
    s = dock_scene(initial, (0.0, 0.0, 0.0))
    lab = Labeller(s)
    rows = []
    openings = [0.0, *rng.uniform(0.0, 0.16, size=n - 1)]
    for j, o in enumerate(openings):
        s.data.set_joint_qpos(DRAWER_JOINT, -o)
        s.sim.forward()
        rgb, depth, cls = lab.render()
        rows.append(save(out, f"static_i{initial:02d}_{j}", lab.labels(cls), rgb, depth, cls,
                         {"source": "static", "case": None, "initial": initial, "step": 0}))
    s.close()
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("--every", type=int, default=10)
    p.add_argument("--static-per-init", type=int, default=3)
    p.add_argument("--inits", default="0-49")
    p.add_argument("--cases", default="all", help="comma list of result tags, 'all' or 'none'")
    p.add_argument("--transfer-inits", default=None, help="e.g. 13-49: replay recorded sequences (cycled) on these inits")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    rows = []
    cases = sorted(c for c in (Q / "results").iterdir() if (c / "policy_actions.npz").exists() and (c / "trace.json").exists())
    sources = list(cases)
    if a.cases != "all":
        keep = set(a.cases.split(",")) if a.cases != "none" else set()
        cases = [c for c in cases if c.name in keep]
    for c in cases:
        rows += replay(a.out, c, a.every)
        print(c.name, len(rows), flush=True)
    if a.transfer_inits:
        tlo, thi = map(int, a.transfer_inits.split("-"))
        for i in range(tlo, thi + 1):
            rows += replay(a.out, sources[i % len(sources)], a.every, initial=i)
            print("transfer", i, len(rows), flush=True)
    lo, hi = map(int, a.inits.split("-"))
    for i in range(lo, hi + 1):
        rows += static(a.out, i, a.static_per_init, rng)
        print("static", i, len(rows), flush=True)
    (a.out / "labels.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print("frames", len(rows))


if __name__ == "__main__":
    main()
