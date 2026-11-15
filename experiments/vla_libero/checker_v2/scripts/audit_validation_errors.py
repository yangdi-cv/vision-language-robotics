"""Validation-only residual audit; does not read or change independent test scores."""

from pathlib import Path
import json
from collections import Counter
from PIL import Image, ImageDraw, ImageFont

R = Path(__file__).resolve().parents[1]


def main():
    metadata = {x["id"]: x for x in json.loads((R / "data/cases.json").read_text())}
    scores = json.loads(
        (R / "results/model/adapter_validation_stable.json").read_text()
    )
    t = json.loads((R / "results/model/summary.json").read_text())["thresholds"][
        "adapter"
    ]
    fps = []
    fns = []
    for x in scores:
        assert x["split"] == "validation"
        s = x["scores"]
        confirmed = s["yes_log_odds"] >= t and s["yes"] >= max(s["no"], s["unknown"])
        m = metadata[x["id"]]
        if confirmed and not x["done"]:
            fps.append(m)
        if not confirmed and x["done"]:
            fns.append(m)
    data = {
        "threshold_frozen": t,
        "split": "validation",
        "false_positive_n": len(fps),
        "fp_by_kind": dict(Counter(x["kind"] for x in fps)),
        "false_negative_n": len(fns),
        "fn_by_kind": dict(Counter(x["kind"] for x in fns)),
        "fp_rows": fps,
        "used_for_new_training_or_threshold": False,
    }
    (R / "results/validation_residuals.json").write_text(json.dumps(data, indent=2))
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    canvas = Image.new("RGB", (1024, 320 * len(fps)), "#e9edf2")
    d = ImageDraw.Draw(canvas)
    for j, x in enumerate(fps):
        a = Image.open(R / "data" / x["images"][0]).resize((280, 280))
        b = Image.open(R / "data" / x["images"][1]).resize((280, 280))
        canvas.paste(a, (0, j * 320))
        canvas.paste(b, (280, j * 320))
        text = f"{x['id']}\n{x['kind']} / label {x['status']}\nrelation={x.get('relation_predicate')}\nsupport={x.get('support_contact')}\ngrasp={x.get('grasp')}\nvisible pixels={x.get('visible_pixels')}"
        d.multiline_text(
            (570, j * 320 + 10), text, font=font, fill="#243448", spacing=8
        )
    canvas.save(R / "figures/validation_false_positives.png")
    print(json.dumps({k: v for k, v in data.items() if k != "fp_rows"}, indent=2))


if __name__ == "__main__":
    main()
