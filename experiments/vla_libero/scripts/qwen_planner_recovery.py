"""End-state subgoals and explicit coverage for recovery; original worker is unchanged."""

import json
import re
import sys
from qwen_planner import Planner, parse_plan


def decode(raw):
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(text)


def validate_end_state(instruction):
    if re.match(r"^(pick\s+up|grasp|grab|lift)\b", instruction.strip(), re.I):
        if not re.search(r"\b(place|put)\b.+\b(in|into|on|onto)\b", instruction, re.I):
            raise ValueError("Standalone pickup has no release-observable end state")


def parse_repair(raw, remaining):
    value = decode(raw)
    steps = value.get("subgoals")
    if not isinstance(steps, list) or not 1 <= len(steps) <= 6:
        raise ValueError("Expected 1..6 end-state subgoals")
    expected = {f"g{i}" for i in range(len(remaining))}
    seen = []
    for s in steps:
        if (
            not isinstance(s, dict)
            or not isinstance(s.get("instruction"), str)
            or not s["instruction"].strip()
        ):
            raise ValueError("Invalid step")
        if s.get("goal_id") not in expected:
            raise ValueError("Unknown or missing goal_id")
        seen.append(s["goal_id"])
        validate_end_state(s["instruction"])
    if set(seen) != expected or len(seen) != len(set(seen)):
        raise ValueError("Every unconfirmed goal must occur exactly once")
    return {
        "subgoals": [
            {
                "instruction": s["instruction"],
                "completion_hint": str(s.get("completion_hint", "")),
                "goal_id": s["goal_id"],
            }
            for s in steps
        ]
    }


class RecoveryPlanner(Planner):
    def plan(self, image, instruction, context=""):
        prompt = f"""{context}Plan a tabletop robot task from the image and instruction.
Task: {instruction}
Each step must be an independently executable END-STATE goal for one object.
For transport, combine picking and placing into ONE instruction: "pick up X and place it on/in Y".
NEVER output standalone pick-up, grasp, hold or lift steps: the monitor checks placement after release.
State-change goals such as open/close a drawer or turn on a stove are allowed.
Preserve every requested object and relation. Do not invent coordinates or claim execution succeeded.
Return only JSON: {{"subgoals":[{{"instruction":"English end-state instruction","completion_hint":"visible destination/state change"}}]}}"""
        r = self.generate(image, prompt, max_new_tokens=384)
        try:
            plan = parse_plan(r["raw"])
            for s in plan["subgoals"]:
                validate_end_state(s["instruction"])
            r.update(plan, ok=True)
        except (ValueError, TypeError, AttributeError) as e:
            r.update(ok=False, error=str(e))
        return r

    def repair(self, image, instruction, remaining, completed, observation):
        goals = [
            {"goal_id": f"g{i}", "required_instruction": s}
            for i, s in enumerate(remaining)
        ]
        prompt = f"""You are repairing a tabletop robot plan from its current image.
Full task: {instruction}
Previously visually confirmed goals (not simulator truth): {json.dumps(completed)}
UNCONFIRMED required goals: {json.dumps(goals)}
Observation: {observation}
A budget timeout does NOT prove completion. You MUST retain every unconfirmed goal exactly once, with its original goal_id. Do not infer a goal is completed from a plausible-looking image. You may reorder or rephrase it into a corrective instruction, preserving its required final relation.
Each step must end in a visible destination/state change. Combine pick and place in ONE step; NO standalone pick-up/grasp/hold/lift steps. Turn-on/open/close are valid end states. No coordinates or extra unsupported skills.
Return only JSON: {{"subgoals":[{{"goal_id":"g0","instruction":"English end-state goal","completion_hint":"visible final condition"}}]}}"""
        r = self.generate(image, prompt, max_new_tokens=512)
        try:
            r.update(parse_repair(r["raw"], remaining), ok=True)
        except (ValueError, TypeError, AttributeError) as e:
            r.update(ok=False, error=str(e))
        return r


def main():
    from qwen_planner import main as unified_main

    unified_main(["--worker", "--profile", "contract", *sys.argv[1:]])


if __name__ == "__main__":
    main()
