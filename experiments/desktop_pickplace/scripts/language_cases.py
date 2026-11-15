from pathlib import Path
import json, numpy as np
from PIL import Image
from task import make, image

ROOT = Path(__file__).resolve().parents[1]
out = ROOT / "data/language"
out.mkdir(parents=True, exist_ok=True)
rows = []
names = {"red": "红色", "blue": "蓝色", "green": "绿色"}
for seed in range(400, 410):
    env, p = make(seed)
    for _ in range(50):
        env.step(np.r_[p.data.mocap_pos[1].copy(), [1, 0, 0, 0], 1])
    Image.fromarray(image(p)).save(out / f"{seed}.png")
    for c, name in names.items():
        for box, ch in [("left", "左"), ("right", "右")]:
            rows.append(
                {
                    "id": f"{seed}_{c}_{box}",
                    "image": f"{seed}.png",
                    "instruction": f"把{name}方块放进{ch}边盒子。",
                    "target_color": c,
                    "destination": box,
                    "should_execute": True,
                }
            )
    for label, text in [
        ("missing", "把黄色方块放进左边盒子。"),
        ("ambiguous", "把那个方块放进盒子。"),
        ("negation", "不要抓取蓝色方块。"),
    ]:
        rows.append(
            {
                "id": f"{seed}_{label}",
                "image": f"{seed}.png",
                "instruction": text,
                "target_color": None,
                "destination": None,
                "should_execute": False,
            }
        )
    # Extra scene where the requested red target is actually missing.
    adr = p.model.jnt_qposadr[p.model.name2id("red_joint", "joint")]
    p.data.qpos[adr : adr + 3] = [0, 0, -2]
    p.forward()
    Image.fromarray(image(p)).save(out / f"{seed}_no_red.png")
    rows.append(
        {
            "id": f"{seed}_no_red",
            "image": f"{seed}_no_red.png",
            "instruction": "把红色方块放进右边盒子。",
            "target_color": None,
            "destination": None,
            "should_execute": False,
        }
    )
    p.free()
(out / "cases.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
print("Language cases", len(rows))
