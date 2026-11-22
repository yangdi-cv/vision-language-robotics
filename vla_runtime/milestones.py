"""Stateful milestone monitor shared by the online server and offline probes."""

from typing import Any


class MilestoneMonitor:
    def __init__(self, skill: dict[str, Any]) -> None:
        self.task = skill["prompt"]
        self.milestones = skill["milestones"]
        self.stage = 0
        self.streak = 0
        self.memory: list[str] = []

    @property
    def done(self) -> bool:
        return self.stage >= len(self.milestones)

    def request(self, image: str) -> dict[str, Any]:
        return {
            "op": "milestone",
            "image": image,
            "task": self.task,
            "memory": list(self.memory),
            "milestone": self.milestones[self.stage],
        }

    def update(self, step: int, result: dict[str, Any]) -> dict[str, Any]:
        """Apply one Qwen result; returns the event record."""
        if self.done:
            raise ValueError("All milestones already completed")
        m = self.milestones[self.stage]
        satisfied = result.get("ok") is True and result.get("satisfied") is True
        self.streak = self.streak + 1 if satisfied else 0
        event = {
            "policy_step": step,
            "milestone": m["name"],
            "satisfied": satisfied,
            "streak": self.streak,
            "answers": result.get("answers"),
            "description": result.get("description"),
            "error": result.get("error"),
        }
        if self.streak >= m.get("confirm", 2):
            self.memory.append(f"at step {step}: {m['fact']}")
            self.stage += 1
            self.streak = 0
            event["confirmed"] = m["name"]
        return event
