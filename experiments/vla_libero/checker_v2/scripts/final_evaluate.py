# ruff: noqa: E402
"""Fresh-model, stable log-odds calibration. Test only after a validation go/no-go."""

from pathlib import Path
import contextlib
import json
import sys
from functools import lru_cache
import torch
from PIL import Image
from peft import PeftModel

R = Path(__file__).resolve().parents[1]
ROOT = R.parents[2]
sys.path[:0] = [str(ROOT / "perception/qwen_scene"), str(R / "scripts")]
from engine import SceneEngine
from protocol import (
    score_input,
    probabilities,
    evaluate,
    threshold_on_validation,
    macro_recall,
)


def main():
    out = R / "results/model"
    target = out / "summary.json"
    if target.exists():
        raise FileExistsError(target)
    torch.set_num_threads(4)
    e = SceneEngine()
    model = PeftModel.from_pretrained(
        e.model, out / "best_adapter", is_trainable=False
    ).eval()
    allrows = json.loads((R / "data/cases.json").read_text())
    rows = {r["id"]: r for r in allrows}

    @lru_cache(maxsize=24)
    def prepared(uid):
        r = rows[uid]
        return score_input(
            e.processor,
            [Image.open(R / "data" / p).convert("RGB") for p in r["images"]],
            r["instruction"],
        )

    def run(split, base=False):
        outputs = []
        context = model.disable_adapter() if base else contextlib.nullcontext()
        with context:
            for r in allrows:
                if r["split"] != split:
                    continue
                p = probabilities(model, prepared(r["id"]))
                outputs.append(
                    {
                        k: r[k]
                        for k in [
                            "id",
                            "group",
                            "split",
                            "status",
                            "done",
                            "source",
                            "kind",
                        ]
                    }
                    | {"scores": p}
                )
        return outputs

    bv = run("validation", True)
    av = run("validation")
    bt = threshold_on_validation(bv)
    at = threshold_on_validation(av)
    bvm = evaluate(bv, bt)
    avm = evaluate(av, at)
    (out / "base_validation_stable.json").write_text(json.dumps(bv, indent=2))
    (out / "adapter_validation_stable.json").write_text(json.dumps(av, indent=2))
    (out / "thresholds.json").write_text(
        json.dumps(
            {
                "base": bt,
                "adapter": at,
                "score_type": "yes log-odds against no+unknown; float64 reduction",
            },
            indent=2,
        )
    )
    common = {
        "thresholds": {"base": bt, "adapter": at},
        "validation_base": bvm,
        "validation_adapter": avm,
        "score_type": "log_odds",
        "fresh_reload": True,
        "validation_macro_recall": macro_recall(av),
    }
    if avm["recall"] < 0.20 or avm["recall"] - bvm["recall"] < 0.15:
        common.update(
            offline_pass=False,
            test_not_scored=True,
            reason="Validation recall/gain criterion failed; test remains unopened",
        )
        target.write_text(json.dumps(common, indent=2))
        print(json.dumps(common), flush=True)
        return
    # These are the first and only held-out test scores for V2.
    btest = run("test", True)
    atest = run("test")
    (out / "base_test.json").write_text(json.dumps(btest, indent=2))
    (out / "adapter_test.json").write_text(json.dumps(atest, indent=2))
    bm = evaluate(btest, bt)
    am = evaluate(atest, at)
    accepted = (
        am["fpr"] <= 0.05
        and am["recall"] - bm["recall"] >= 0.15
        and am["recall"] >= 0.20
    )
    common.update(
        base=bm,
        adapter=am,
        recall_gain=am["recall"] - bm["recall"],
        offline_pass=accepted,
        test_not_scored=False,
    )
    target.write_text(json.dumps(common, indent=2))
    print(json.dumps(common), flush=True)


if __name__ == "__main__":
    main()
