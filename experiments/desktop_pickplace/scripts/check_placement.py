"""Two-frame visual completion candidates, evaluated against physical task stages."""

import sys, json, time, argparse
from pathlib import Path
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[3]
TASK = ROOT / "experiments/desktop_pickplace"
sys.path.insert(0, str(ROOT / "perception/qwen_scene"))
from engine import SceneEngine

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, default=TASK / "runs/placement_observer")
parser.add_argument("--roi", action="store_true")
args = parser.parse_args()
out = args.out
out.mkdir(exist_ok=False)
engine = SceneEngine()
torch.set_num_threads(8)
rows = []
names = {"red": "红色", "blue": "蓝色", "green": "绿色"}
with (out / "predictions.jsonl").open("w") as log:
    for color in names:
        for box, side in [("left", "左"), ("right", "右")]:
            directory = TASK / "runs/integrated_lora_test" / f"408_{color}_{box}"
            for stage, filename in [
                ("before", "initial.png"),
                ("holding", "step_390.png"),
                ("released", "step_700.png"),
            ]:
                prompt = f"""任务是把{names[color]}方块放进{side}边盒子。第一张是操作前，第二张是当前画面。
输出视觉检查结果，而不是猜测机械臂已经成功。物体明显仍在桌面或夹爪里时，不可确认完成。
只有目标可见、位于指定盒子内且没有可见夹持时，标记complete_candidate；遮挡或位置/夹持不明确时uncertain；明显未放入则not_complete。
仅输出JSON：{{"placement_state":"complete_candidate"或"not_complete"或"uncertain"}}。"""
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": Image.open(directory / "initial.png").convert(
                                    "RGB"
                                ),
                            },
                            {
                                "type": "image",
                                "image": Image.open(directory / filename).convert(
                                    "RGB"
                                ),
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
                if args.roi:
                    current = Image.open(directory / filename).convert("RGB")
                    crops = [
                        current.crop((u - 45, 170, u + 45, 250)).resize((360, 320))
                        for u in (326, 411)
                    ]
                    messages[0]["content"].insert(
                        2, {"type": "image", "image": crops[0]}
                    )
                    messages[0]["content"].insert(
                        3, {"type": "image", "image": crops[1]}
                    )
                    messages[0]["content"][-1][
                        "text"
                    ] += "第三张是当前左盒局部放大，第四张是当前右盒局部放大。请结合局部图识别目标颜色；只判断是否可作为放置完成候选，不要求从照片证明长期稳定性。"
                inputs = engine.processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to("cuda")
                with torch.inference_mode():
                    answer = engine.model.generate(
                        **inputs, max_new_tokens=80, do_sample=False
                    )
                raw = engine.processor.decode(
                    answer[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
                )
                row = {
                    "target": color,
                    "bin": box,
                    "stage": stage,
                    "input_frames": ["initial.png", filename],
                    "raw": raw,
                    "valid": False,
                }
                try:
                    text = raw.strip()
                    if text.startswith("```"):
                        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                    state = json.loads(text)["placement_state"]
                    assert state in ("complete_candidate", "not_complete", "uncertain")
                    row.update(
                        valid=True,
                        state=state,
                        correct=(
                            (state == "complete_candidate")
                            if stage == "released"
                            else (state != "complete_candidate")
                        ),
                    )
                except (ValueError, AssertionError, KeyError, TypeError):
                    row.update(correct=False)
                rows.append(row)
                log.write(json.dumps(row, ensure_ascii=False) + "\n")
                log.flush()
metrics = {
    "cases": len(rows),
    "correct": sum(r["correct"] for r in rows),
    "complete_candidate_on_released": sum(
        r.get("state") == "complete_candidate" for r in rows if r["stage"] == "released"
    ),
    "released_cases": 6,
    "false_complete_on_before_or_holding": sum(
        r.get("state") == "complete_candidate" for r in rows if r["stage"] != "released"
    ),
    "notes": "18stage pairs from6 scripted trials, single layout408; holding is not a completed placement; uncertain accepted as conservative on incomplete stages; no actuator access",
}
(out / "metrics.json").write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2))
