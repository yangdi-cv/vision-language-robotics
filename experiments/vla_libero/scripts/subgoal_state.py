"""Execution bookkeeping. Only observed confirmations advance a plan; timeouts do not."""

from dataclasses import dataclass, field
import re


@dataclass
class SubgoalState:
    instructions: list[str]
    confirm: int = 2
    index: int = 0
    streak: int = 0
    confirmed: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.instructions = self.validate(self.instructions)
        if self.confirm < 1:
            raise ValueError("confirm must be positive")

    @staticmethod
    def validate(items):
        if (
            not isinstance(items, list)
            or not items
            or any(not isinstance(x, str) or not x.strip() for x in items)
        ):
            raise ValueError("nonempty instruction list required")
        return list(items)

    @property
    def current(self):
        return self.instructions[self.index]

    @property
    def last(self):
        return self.index == len(self.instructions) - 1

    def observe(self, valid, done):
        self.streak = self.streak + 1 if valid and done else 0
        return self.streak >= self.confirm

    def advance(self):
        if self.last or self.streak < self.confirm:
            raise ValueError("only confirmed non-final goals can advance")
        self.confirmed.append(self.current)
        self.index += 1
        self.streak = 0

    def replace_pending(self, pending):
        pending = self.validate(pending)  # validate before mutation
        self.instructions = self.instructions[: self.index] + pending
        self.streak = 0
        # The current goal was attempted, not completed. It is never added to confirmed.


def requires_release(instruction):
    """Release is required for transport, not manipulation of an articulated fixture.

    Unknown phrasing keeps the conservative transport gate. This recognizes the
    supported imperative forms, rather than inferring task success from state.
    """
    return not bool(
        re.match(
            r"^(turn (on|off)|switch (on|off)|open|close|shut)\b",
            instruction.strip(),
            re.I,
        )
    )
