"""Stateful milestone monitor shared by the online server and offline probes."""


class MilestoneMonitor:
    def __init__(self, skill):
        self.task = skill["prompt"]
        self.milestones = skill["milestones"]
        self.stage = 0
        self.streak = 0
        self.memory = []

    @property
    def done(self):
        return self.stage >= len(self.milestones)

    def request(self, image):
        return {"op": "milestone", "image": image, "task": self.task,
                "memory": list(self.memory), "milestone": self.milestones[self.stage]}

    def update(self, step, result):
        """Apply one Qwen result; returns the event record."""
        m = self.milestones[self.stage]
        satisfied = bool(result.get("ok") and result.get("satisfied"))
        self.streak = self.streak + 1 if satisfied else 0
        event = {"policy_step": step, "milestone": m["name"], "satisfied": satisfied,
                 "streak": self.streak, "answers": result.get("answers"),
                 "description": result.get("description"), "error": result.get("error")}
        if self.streak >= m.get("confirm", 2):
            self.memory.append(f"at step {step}: {m['fact']}")
            self.stage += 1
            self.streak = 0
            event["confirmed"] = m["name"]
        return event
