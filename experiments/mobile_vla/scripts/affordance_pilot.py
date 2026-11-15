# ruff: noqa: E402
"""Development-only functional-part labels, not generic object boxes.

Grounding pattern follows RoboPoint/RoboInter's explicit point/region interface.
Labels are visible handle mesh pixels; they do not prove full grasp feasibility.
"""

from pathlib import Path
import argparse
import json
import sys
import numpy as np
from PIL import Image, ImageDraw

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "scripts"))
from scene import MobileScene

PARTS = {
    "top": "white_cabinet_top_handle_vis",
    "middle": "white_cabinet_middle_handle_vis",
    "bottom": "white_cabinet_bottom_handle_vis",
}


def functional_part_masks(s, seg):
    masks = {}
    for part, suffix in PARTS.items():
        ids = []
        for gid in range(s.model.ngeom):
            mid = int(s.model.geom_dataid[gid])
            if (
                s.model.geom_type[gid] == 7
                and mid >= 0
                and suffix in s.model.mesh_names[mid]
            ):
                ids.append(gid)
        if not ids:
            raise ValueError(f"Missing named functional mesh {suffix}")
        masks[part] = np.isin(seg[..., 1], ids)
    return masks


def label(mask):
    y, x = np.where(mask)
    if len(x) < 20:
        return {"visible": False, "point": None, "bbox": None, "visible_pixels": len(x)}
    nearest = np.argmin((x - x.mean()) ** 2 + (y - y.mean()) ** 2)
    h, w = mask.shape
    return {
        "visible": True,
        "point": [
            round(float(x[nearest] / w * 1000)),
            round(float(y[nearest] / h * 1000)),
        ],
        "bbox": [
            round(float(x.min() / w * 1000)),
            round(float(y.min() / h * 1000)),
            round(float((x.max() + 1) / w * 1000)),
            round(float((y.max() + 1) / h * 1000)),
        ],
        "visible_pixels": len(x),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    s = MobileScene(initial=3, mobile_start=(0, 0, 0), stow=False)
    cam = s.model.camera_name2id("agentview")
    origin = s.model.cam_pos[cam].copy()
    fov = float(s.model.cam_fovy[cam])
    rng = np.random.default_rng(47013)
    rows = []
    try:
        for v in range(6):
            s.model.cam_pos[cam] = origin + rng.uniform(-0.035, 0.035, 3)
            s.model.cam_fovy[cam] = fov + rng.uniform(-3, 3)
            s.sim.forward()
            rgb = s.sim.render(camera_name="agentview", height=512, width=512)[
                ::-1, ::-1
            ]
            seg = s.sim.render(
                camera_name="agentview", height=512, width=512, segmentation=True
            )[::-1, ::-1]
            image = a.out / f"view{v:02d}.png"
            Image.fromarray(rgb).save(image)
            overlay = Image.fromarray(rgb)
            draw = ImageDraw.Draw(overlay)
            for j, (part, mask) in enumerate(functional_part_masks(s, seg).items()):
                target = label(mask)
                Image.fromarray((mask * 255).astype(np.uint8)).save(
                    a.out / f"view{v:02d}_{part}_mask.png"
                )
                rows.append(
                    {
                        "id": f"pilot_v{v}_{part}",
                        "development_only": True,
                        "image": image.name,
                        "functional_part": part,
                        "action": "pull_handle",
                        "label": target,
                        "label_source": "exact visible handle mesh segmentation; not full object segmentation or model self-label",
                    }
                )
                if target["visible"]:
                    x, y = [c / 1000 * 512 for c in target["point"]]
                    color = ["#e76653", "#58a9cf", "#d9b847"][j]
                    draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
                    draw.text((x + 6, y - 5), part, fill=color)
            overlay.save(a.out / f"view{v:02d}_labels.png")
        (a.out / "cases.json").write_text(json.dumps(rows, indent=2))
        print(
            json.dumps(
                {
                    "n": len(rows),
                    "visible": sum(x["label"]["visible"] for x in rows),
                    "parts": list(PARTS),
                    "development_only": True,
                }
            ),
            flush=True,
        )
    finally:
        s.close()


if __name__ == "__main__":
    main()
