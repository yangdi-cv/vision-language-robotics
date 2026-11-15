"""Qwen-VL mission planner and task-completion checker for the mobile scene.

Runs in .venvs/qwen-scene as a JSON-lines worker. Reuses the frozen
vla_libero Planner (same base model and generation path); adds two ops:
  mission: overview + table image, free-form instruction -> station/skill or refusal
  check:   manipulation-start image + current image -> whole-task done?
Never receives simulator state.
"""

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/vla_libero/scripts"))
from qwen_planner import Planner  # noqa: E402


def parse_json(raw):
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(text)


class MissionPlanner(Planner):
    def mission(self, images, instruction, stations, skills):
        station_lines = "\n".join(f'- "{k}": {v}' for k, v in stations.items())
        skill_lines = "\n".join(f'- "{k}": "{v}"' for k, v in skills.items())
        prompt = f"""You are the task planner of a mobile manipulator. Image 1 is an overview of the room; image 2 is a close view of the work table.
User request (may be Chinese, English, or colloquial): {instruction}

Navigation stations:
{station_lines}

Verified manipulation skills (the low-level policy only executes these exact skills):
{skill_lines}

Decide whether the request can be fulfilled in this scene with one station and one skill. Refuse if a requested object or appliance is not visible in the scene, or if no listed skill achieves the request. Do not invent stations or skills.
Return only JSON: {{"feasible": true or false, "station": station key or null, "skill": skill key or null, "reason": "short explanation"}}"""
        result = self.generate(images, prompt, max_new_tokens=160)
        try:
            value = parse_json(result["raw"])
            if not isinstance(value.get("feasible"), bool):
                raise ValueError("Expected boolean feasible")
            if value["feasible"]:
                if value.get("station") not in stations or value.get("skill") not in skills:
                    raise ValueError("Unknown station or skill")
            result.update(
                feasible=value["feasible"],
                station=value.get("station") if value["feasible"] else None,
                skill=value.get("skill") if value["feasible"] else None,
                reason=str(value.get("reason", "")),
            )
            result["ok"] = True
        except (ValueError, TypeError, AttributeError) as exc:
            result["error"] = str(exc)
        return result

    def task_check(self, images, task):
        prompt = f"""Image 1 was taken when the robot arm started working; image 2 is the scene now.
Task: "{task}".
Split the task into its requirements and judge each one from image 2, using image 1 for comparison (an object that disappeared into a closed container counts as inside it). The task is done only if every requirement is satisfied and the robot is no longer holding the object.
Return only JSON: {{"requirements": [{{"text": "...", "satisfied": true or false}}], "done": true or false}}"""
        result = self.generate(images, prompt, max_new_tokens=200)
        try:
            value = parse_json(result["raw"])
            if not isinstance(value.get("done"), bool):
                raise ValueError("Expected boolean done")
            result.update(done=value["done"], requirements=value.get("requirements", []))
            result["ok"] = True
        except (ValueError, TypeError, AttributeError) as exc:
            result["error"] = str(exc)
        return result


    def question_check(self, images, questions):
        """Answer the skill's observable yes/no completion questions; done iff all match."""
        lines = "\n".join(f"{i + 1}. {q['question']}" for i, q in enumerate(questions))
        prompt = f"""Image 1 was taken when the robot arm started working; image 2 is the scene now. Answer each question about image 2. Use image 1 only to know where things were. Ignore the robot arm itself.
{lines}
Return only JSON: {{"answers": [true or false for each question, in order]}}"""
        result = self.generate(images, prompt, max_new_tokens=48)
        try:
            answers = parse_json(result["raw"])["answers"]
            if len(answers) != len(questions) or not all(isinstance(x, bool) for x in answers):
                raise ValueError("Expected one boolean per question")
            result.update(
                answers=answers,
                done=all(x == q["expected"] for x, q in zip(answers, questions)),
            )
            result["ok"] = True
        except (ValueError, TypeError, AttributeError, KeyError) as exc:
            result["error"] = str(exc)
        return result


    def milestone_check(self, image, task, memory, milestone):
        """Judge one milestone on the current image; earlier confirmations arrive as text memory."""
        memory_text = "\n".join(f"- {m}" for m in memory) or "- nothing confirmed yet"
        answer_lines = "\n".join(f'- "{k}": {q}' for k, q in milestone["answers"].items())
        keys = ", ".join(f'"{k}": true or false' for k in milestone["answers"])
        prompt = f"""You are monitoring a robot arm doing the task "{task}". The image is the current view of a fixed camera.
What you already confirmed earlier in this episode:
{memory_text}
{milestone["question"]}
Then answer:
{answer_lines}
Return only JSON: {{"description": "...", {keys}}}"""
        result = self.generate(image, prompt, max_new_tokens=120)
        try:
            value = parse_json(result["raw"])
            answers = {k: value[k] for k in milestone["answers"]}
            if not all(isinstance(x, bool) for x in answers.values()):
                raise ValueError("Expected boolean answers")
            result.update(
                description=str(value.get("description", "")),
                answers=answers,
                satisfied=all(answers[k] == v for k, v in milestone["expect"].items()),
            )
            result["ok"] = True
        except (ValueError, TypeError, AttributeError, KeyError) as exc:
            result["error"] = str(exc)
        return result


def main():
    planner = MissionPlanner()
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        try:
            q = json.loads(line)
            if q.get("stop"):
                break
            if q["op"] == "mission":
                r = planner.mission(q["images"], q["instruction"], q["stations"], q["skills"])
            elif q["op"] == "check":
                r = planner.task_check(q["images"], q["task"])
            elif q["op"] == "question_check":
                r = planner.question_check(q["images"], q["questions"])
            elif q["op"] == "milestone":
                r = planner.milestone_check(q["image"], q["task"], q["memory"], q["milestone"])
            else:
                raise ValueError("Unknown op")
        except Exception as exc:
            r = {"ok": False, "error": repr(exc)}
        print(json.dumps(r, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
