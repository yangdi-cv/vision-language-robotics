"""Offline test of a stateful (memory) completion monitor on saved check frames.

Each check asks only about the next unconfirmed milestone. Confirmed milestones are
kept as text memory plus the keyframe at which they were confirmed, and that keyframe
is shown in later checks. Official predicate labels are used only for scoring.
Modes: keyframes (all images in time order), current (current image + text memory),
current_first (current image, then start image as reference). Suffix "+describe" uses
describe-then-answer milestone questions that name the gripper.
Usage (qwen-scene venv): probe_memory.py <out_json> <mode[,mode]> <case_dir> [<case_dir> ...]
"""

from pathlib import Path
import json
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen_worker import MissionPlanner, parse_json  # noqa: E402

GRIPPER = "The white object with two fingers is the robot gripper, not a drawer or a bowl."
MILESTONES_DESCRIBE = [
    {
        "name": "bowl_in_drawer",
        "fact": "the black bowl was resting inside the open bottom drawer of the cabinet",
        "question": f"Look at the dark cabinet on the left and its pulled-out bottom drawer. {GRIPPER} First describe in one sentence what is inside the drawer and where the bowl is, then answer: is the bowl inside the drawer?",
    },
    {
        "name": "drawer_closed",
        "fact": "the bottom drawer was closed",
        "question": f"Look at the dark cabinet on the left side of the image. {GRIPPER} First describe the front of the cabinet in one sentence, then answer: is the bottom drawer pulled out?",
        "satisfied_when": False,
    },
]

MILESTONES = [
    {
        "name": "bowl_in_drawer",
        "fact": "the black bowl was resting inside the open bottom drawer of the cabinet",
        "question": "Is the black bowl now resting inside the bottom drawer of the cabinet (inside the drawer box, not on the table and not still lifted above it)?",
    },
    {
        "name": "drawer_closed",
        "fact": "the bottom drawer was closed",
        "question": "Is the bottom drawer of the cabinet now closed? In the earlier images the white drawer box sticks out of the cabinet. If the drawer box and its inside are no longer visible and only the handle remains on the cabinet front, it is closed. The robot gripper may be in front of the cabinet; judge by the drawer box, not by the gripper.",
    },
]


def ask(planner, images, labels, memory, question, target):
    image_lines = "\n".join(f"Image {i + 1}: {label}" for i, label in enumerate(labels))
    memory_text = "\n".join(f"- {m}" for m in memory) or "- nothing confirmed yet"
    prompt = f"""You are monitoring a robot arm doing the task "put the black bowl in the bottom drawer of the cabinet and close it". The images are from a fixed camera:
{image_lines}
What you already confirmed earlier in this episode:
{memory_text}
Question about {target}: {question}
Return only JSON: {{"evidence": "short description of what you see", "answer": true or false}}"""
    result = planner.generate(images, prompt, max_new_tokens=120)
    try:
        value = parse_json(result["raw"])
        if not isinstance(value.get("answer"), bool):
            raise ValueError("Expected boolean answer")
        return value["answer"], str(value.get("evidence", "")), None, result["raw"]
    except (ValueError, TypeError, AttributeError) as exc:
        return False, "", str(exc), result["raw"]


def build_inputs(mode, start, keyframes, prev_path, prev, now_path, step):
    """Return (images, labels, target phrase) for one check."""
    if mode == "current":
        return [now_path], [f"step {step}, now"], "this image"
    if mode == "current_first":
        return ([now_path, start], [f"step {step}, NOW", "reference: when the robot started this task"],
                "image 1 (now)")
    images, labels = [start], ["when the robot started this task"]
    for name, kstep, path in keyframes:
        images.append(path)
        labels.append(f"step {kstep}, when you confirmed {name}")
    if prev_path is not None and (not keyframes or keyframes[-1][1] != prev):
        images.append(prev_path)
        labels.append(f"step {prev}, previous check")
    images.append(now_path)
    labels.append(f"step {step}, now")
    return images, labels, "the LAST image"


def run_case(planner, case, mode, confirm=2):
    milestones = MILESTONES
    if mode.endswith("+describe"):
        mode, milestones = mode.removesuffix("+describe"), MILESTONES_DESCRIBE
    r = json.loads((case / "qwen_result.json").read_text())
    frame = lambda s: str(case / f"qwen_inputs/check_{s:04d}.png")  # noqa: E731
    start = str(case / "qwen_inputs/manip_start.png")
    stage, streak, memory, keyframes, prev = 0, 0, [], [], None
    rows, stop_step = [], None
    for e in r["checks"]:
        step = e["policy_step"]
        images, labels, target = build_inputs(
            mode, start, keyframes, frame(prev) if prev else None, prev, frame(step), step)
        m = milestones[stage]
        t = time.perf_counter()
        answer, evidence, error, raw = ask(planner, images, labels, memory, m["question"], target)
        seconds = time.perf_counter() - t
        if error is None:
            answer = answer == m.get("satisfied_when", True)
        streak = streak + 1 if answer else 0
        rows.append(dict(step=step, stage=m["name"], answer=answer, streak=streak,
                         evidence=evidence, error=error, seconds=round(seconds, 2),
                         n_images=len(images), official=e["shadow_official_success"]))
        if streak >= confirm:
            memory.append(f"at step {step}: {m['fact']}")
            keyframes.append((m["name"], step, frame(step)))
            stage, streak = stage + 1, 0
            if stage == len(milestones):
                stop_step = step
                break
        prev = step
    first = next((e["policy_step"] for e in r["checks"] if e["shadow_official_success"]), None)
    official_at_stop = next((x["official"] for x in rows if x["step"] == stop_step), None)
    return dict(case=case.name, stop_step=stop_step, first_official_check_step=first,
                official_at_stop=official_at_stop,
                premature=stop_step is not None and not official_at_stop,
                memory=memory, rows=rows)


def main():
    out, modes = Path(sys.argv[1]), sys.argv[2].split(",")
    planner = MissionPlanner()
    results = []
    for mode in modes:
        for c in sys.argv[3:]:
            res = run_case(planner, Path(c), mode)
            res["mode"] = mode
            results.append(res)
            print(json.dumps({k: res[k] for k in ("mode", "case", "stop_step", "first_official_check_step", "official_at_stop", "premature", "memory")}, ensure_ascii=False), flush=True)
            out.write_text(json.dumps(results, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
