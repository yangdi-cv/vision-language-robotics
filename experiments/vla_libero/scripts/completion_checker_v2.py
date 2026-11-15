"""V2 checker adapter. Planning and unsupported checks retain the base model."""

from pathlib import Path
import importlib.util
import time
import hashlib
from PIL import Image
from qwen_planner import Planner
from engine import MODEL_ID, REVISION

R = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "completion_v2_protocol", R / "checker_v2/scripts/protocol.py"
)
protocol = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(protocol)


def supported(subgoal):
    s = subgoal.lower().strip()
    if s.startswith(("open", "close", "turn", "switch")):
        return False
    if ("pick up" in s or "grasp" in s) and not any(x in s for x in ["place", "put"]):
        return False
    if any(x in s for x in ["book", "caddy", "first moka", "second moka"]):
        return False
    return s.startswith(("put", "place", "pick up"))


class V2Planner(Planner):
    def __init__(self, adapter, threshold):
        super().__init__()
        from peft import PeftModel

        self.engine.model = PeftModel.from_pretrained(
            self.engine.model, adapter, is_trainable=False
        ).eval()
        self.adapter = Path(adapter)
        self.adapter_hashes = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in self.adapter.iterdir()
            if p.is_file()
        }
        self.threshold = threshold

    def generate(self, *args, **kwargs):
        with self.engine.model.disable_adapter():
            return super().generate(*args, **kwargs)

    def check(self, image, subgoal, mode="v2"):
        if (
            mode == "base"
            or not supported(subgoal)
            or not isinstance(image, list)
            or len(image) != 2
        ):
            r = super().check(image, subgoal)
            r["checker"] = "base" if mode == "base" else "base_unsupported"
            return r
        started = time.monotonic()
        images = [Image.open(p).convert("RGB").resize((512, 512)) for p in image]
        scores = protocol.probabilities(
            self.engine.model,
            protocol.score_input(self.engine.processor, images, subgoal),
        )
        status = max(protocol.CLASSES, key=scores.get)
        done = status == "yes" and scores["yes_log_odds"] >= self.threshold
        status = (
            "yes" if done else ("unknown" if status in ("yes", "unknown") else "no")
        )
        return {
            "inference": {
                "model": MODEL_ID,
                "revision": REVISION,
                "prompt": protocol.prompt(subgoal),
                "prompt_sha256": hashlib.sha256(
                    protocol.prompt(subgoal).encode()
                ).hexdigest(),
                "scoring": "three_class_log_likelihood",
                "threshold": self.threshold,
                "adapter": str(self.adapter),
                "adapter_files_sha256": self.adapter_hashes,
                "protocol_sha256": hashlib.sha256(
                    Path(protocol.__file__).read_bytes()
                ).hexdigest(),
                "images": [
                    {
                        "size": list(im.size),
                        "pixel_sha256": hashlib.sha256(im.tobytes()).hexdigest(),
                    }
                    for im in images
                ],
            },
            "ok": True,
            "done": done,
            "status": status,
            "scores": scores,
            "threshold": self.threshold,
            "checker": "v2",
            "seconds": time.monotonic() - started,
            "reason": f"V2 {status}; p_yes={scores['yes']:.4f}, threshold={self.threshold:.4f}",
        }
