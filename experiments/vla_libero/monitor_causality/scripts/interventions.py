"""Image-only interventions and predeclared controls; never receives simulator truth."""

import hashlib
import json
import re
import time
from pathlib import Path

MODES = ("live_base", "frozen_base", "timer150", "live_lora", "hold_first")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Monitor:
    def __init__(self, worker, plan, mode, log):
        if mode not in MODES:
            raise ValueError(mode)
        if len(plan) != 2:
            raise ValueError("This protocol requires exactly two frozen subgoals")
        self.worker, self.plan, self.mode, self.log = (
            worker,
            list(plan),
            mode,
            Path(log),
        )
        self.actual_calls = 0
        self.opportunities = 0

    def request(self, payload, timeout=120):
        if payload.get("op") == "plan":
            return dict(
                ok=True,
                subgoals=[dict(instruction=x) for x in self.plan],
                raw=json.dumps(self.plan),
            )
        if payload.get("op") != "check":
            raise ValueError("No replanning/translation allowed")
        original = list(payload["image"])
        if len(original) != 2:
            raise ValueError("Expected reference/current pair")
        match = re.fullmatch(r"check_step(\d+)\.png", Path(original[1]).name)
        if match is None:
            raise ValueError("Missing frame step")
        step = int(match.group(1))
        self.opportunities += 1
        started = time.monotonic()
        actual = original.copy()
        if self.mode in ("live_base", "frozen_base", "live_lora"):
            if self.mode == "frozen_base":
                actual = [original[0], original[0]]
            actual_payload = {
                **payload,
                "image": actual,
                "checker_mode": "v2" if self.mode == "live_lora" else "base",
            }
            reply = self.worker.request(actual_payload, timeout)
            self.actual_calls += 1
        else:
            reply = dict(
                ok=True,
                done=self.mode == "timer150" and step >= 150,
                checker=self.mode,
                reason="predeclared step threshold"
                if self.mode == "timer150"
                else "no high-level transition",
            )
        entry = dict(
            mode=self.mode,
            step=step,
            subgoal=payload["subgoal"],
            original_images=original,
            actual_images=actual if self.mode not in ("timer150", "hold_first") else [],
            original_pixel_files_sha256=[sha(p) for p in original],
            actual_pixel_files_sha256=[sha(p) for p in actual]
            if self.mode not in ("timer150", "hold_first")
            else [],
            actual_model_call=self.mode in ("live_base", "frozen_base", "live_lora"),
            seconds=time.monotonic() - started,
            reply=reply,
        )
        self.log.parent.mkdir(parents=True, exist_ok=True)
        with self.log.open("a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()
            __import__("os").fsync(f.fileno())
        return reply
