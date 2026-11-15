"""Qwen-VL subgoal planner and completion checker. No environment or robot execution interface."""

from pathlib import Path
import argparse, contextlib, json, sys, time

R = Path(__file__).resolve().parents[1]
ROOT = R.parents[1]
sys.path.insert(0, str(ROOT / "perception/qwen_scene"))


def parse_plan(raw):
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    value = json.loads(text)
    if not isinstance(value, dict) or set(value) != {"subgoals"}:
        raise ValueError("Expected subgoals object")
    if not isinstance(value["subgoals"], list) or not 1 <= len(value["subgoals"]) <= 6:
        raise ValueError("Expected 1..6 subgoals")
    for step in value["subgoals"]:
        if not isinstance(step, dict) or "instruction" not in step:
            raise ValueError("Invalid subgoal fields")
        step.setdefault("completion_hint", "")
        if not all(
            isinstance(v, str) and len(v) < 1200 for v in step.values()
        ) or not step["instruction"].strip():
            raise ValueError("Invalid subgoal text")
    value["subgoals"] = [
        {k: step[k] for k in ("instruction", "completion_hint")} for step in value["subgoals"]
    ]
    return value


class Planner:
    def __init__(self):
        from engine import SceneEngine

        with contextlib.redirect_stdout(sys.stderr):
            self.engine = SceneEngine()

    def generate(self, image, prompt, max_new_tokens=256, scale=1):
        from PIL import Image

        e = self.engine
        torch = e.torch
        paths = image if isinstance(image, list) else [image]
        images = [Image.open(x).convert("RGB") for x in paths]
        images = [
            x.resize((x.width * scale, x.height * scale), Image.BICUBIC) for x in images
        ]
        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": x} for x in images]
                + [{"type": "text", "text": prompt}],
            }
        ]
        inputs = e.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to("cuda:0")
        torch.cuda.synchronize()
        start = time.monotonic()
        with torch.inference_mode():
            output = e.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
        torch.cuda.synchronize()
        elapsed = time.monotonic() - start
        raw = e.processor.decode(
            output[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
        )
        return {
            "ok": False,
            "raw": raw,
            "seconds": elapsed,
            "device": str(next(e.model.parameters()).device),
            "allocated_mib": torch.cuda.memory_allocated() / 2**20,
            "reserved_mib": torch.cuda.memory_reserved() / 2**20,
            "tokens": int(output.shape[-1] - inputs["input_ids"].shape[-1]),
        }

    def plan(self, image, instruction, context=""):
        prompt = f"""{context}You are a high-level planner for a tabletop robot. Inspect the image and split the requested task into a short ordered list of executable English subgoal instructions. Each subgoal must move or operate exactly one object. Preserve all requested objects and relations. Include opening/closing steps only when required. Do not invent coordinates, robot joint values, unseen objects, or claim execution has happened. Completion hints are descriptions, not executable predicate code.
The low-level executor only understands short imperative phrases such as:
- "pick up the alphabet soup and place it in the basket"
- "put the bowl on the stove"
- "put the black bowl in the bottom drawer of the cabinet"
- "open the top drawer of the cabinet"
- "turn on the stove"
Requested task (may be in any language): {instruction}
Return only JSON with 1 to 6 steps, exactly this schema: {{"subgoals":[{{"instruction":"English instruction","completion_hint":"observable completion description"}}]}}"""
        result = self.generate(image, prompt)
        try:
            result.update(parse_plan(result["raw"]))
            result["ok"] = True
        except (ValueError, TypeError) as exc:
            result["error"] = str(exc)
        return result

    def replan(self, image, instruction, completed, failed, observation):
        done = "; ".join(completed) if completed else "nothing yet"
        context = f"""The robot was executing a plan for this task but got stuck.
Already completed: {done}.
Stuck on subgoal: "{failed}".
Last visual observation: {observation or "none"}.
The image shows the scene now. Plan only the remaining subgoals needed from this state, including corrective steps if an object was put in the wrong place. You may rephrase the stuck subgoal.

"""
        return self.plan(image, instruction, context)

    def translate(self, image, instruction):
        prompt = f"""Translate this robot task instruction into one short English imperative sentence, keeping every object and relation. Use the image only to name the objects correctly. Do not split it into steps.
Instruction: {instruction}
Return only JSON: {{"instruction": "English instruction"}}"""
        result = self.generate(image, prompt, max_new_tokens=64)
        try:
            text = result["raw"].strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            value = json.loads(text)
            if not isinstance(value.get("instruction"), str) or not value["instruction"].strip():
                raise ValueError("Expected instruction string")
            result["subgoals"] = [{"instruction": value["instruction"].strip(), "completion_hint": ""}]
            result["ok"] = True
        except (ValueError, TypeError, AttributeError) as exc:
            result["error"] = str(exc)
        return result

    def check(self, image, subgoal):
        if isinstance(image, list) and len(image) == 2:
            prompt = f"""Image 1 was taken when the robot started this subgoal; image 2 is the scene now.
Subgoal: "{subgoal}".
Look at the destination named in the subgoal. Is there a new object at the destination in image 2 that was not there in image 1, and is it the target object? For a state change (open, close, turn on), compare that object between the two images.
Return only JSON: {{"change": "short description", "done": true or false}}"""
        else:
            prompt = f"""You are monitoring a tabletop robot from a third-person camera.
Subgoal: "{subgoal}".
First describe where the target object is right now, then decide whether the subgoal is completed.
Return only JSON: {{"change": "short description", "done": true or false}}"""
        result = self.generate(image, prompt, max_new_tokens=96)
        try:
            text = result["raw"].strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            value = json.loads(text)
            if not isinstance(value.get("done"), bool):
                raise ValueError("Expected boolean done")
            result.update(done=value["done"], reason=str(value.get("change", "")))
            result["ok"] = True
        except (ValueError, TypeError, AttributeError) as exc:
            result["error"] = str(exc)
        return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--image")
    ap.add_argument("--instruction")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    planner = Planner()
    if a.worker:
        print(
            json.dumps(
                {
                    "ready": True,
                    "device": str(next(planner.engine.model.parameters()).device),
                    "load_seconds": planner.engine.load_seconds,
                }
            ),
            flush=True,
        )
        for line in sys.stdin:
            try:
                request = json.loads(line)
                if request.get("stop"):
                    break
                if request.get("op") == "replan":
                    result = planner.replan(
                        request["image"], request["instruction"], request["completed"],
                        request["failed"], request.get("observation", ""),
                    )
                elif request.get("op") == "translate":
                    result = planner.translate(request["image"], request["instruction"])
                elif request.get("op", "plan") == "check":
                    result = planner.check(request["image"], request["subgoal"])
                else:
                    result = planner.plan(request["image"], request["instruction"])
            except Exception as exc:
                result = {"ok": False, "error": repr(exc)}
            print(json.dumps(result, ensure_ascii=False), flush=True)
    else:
        if not a.image or not a.instruction:
            ap.error("--image and --instruction required")
        result = planner.plan(a.image, a.instruction)
        if a.out:
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
