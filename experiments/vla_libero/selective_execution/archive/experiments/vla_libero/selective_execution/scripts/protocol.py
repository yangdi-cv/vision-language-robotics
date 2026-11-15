"""Declared selective-task semantics; simulator labels are for scoring only."""
import numpy as np

MODES = ("flat_zh", "translated_en", "skill_continue", "timer150", "timer_calibrated", "visual", "frozen")
OBJECTS = {
    0: [("alphabet_soup_1", "alphabet soup can", "字母汤罐头"), ("tomato_sauce_1", "tomato sauce can", "番茄酱罐头")],
    1: [("cream_cheese_1", "cream cheese box", "奶油奶酪盒"), ("butter_1", "butter", "黄油")],
}
MAX_STEPS = 300
SETTLE_STEPS = 25
DISPLACEMENT_LIMIT = 0.03
CHECK_INTERVAL = 10
VOTE_TTL = 20


def case(task, target, initial):
    objects = OBJECTS[task]
    wanted, avoid = objects[target], objects[1-target]
    return dict(task=task, target_index=target, initial=initial,
                target=wanted[0], protected=avoid[0], objects=objects,
                instruction=f"只把{wanted[2]}放进篮子里，{avoid[2]}保持原位。放好后停止，不要继续搬其他物体。",
                reference_skill=skill(wanted[1]),
                key=f"task{task:02d}_target{target}_init{initial:02d}")


def skill(name):
    return f"pick up the {name} and place it in the basket"


def released(gripper, last_action):
    return bool(last_action[6] < 0 and gripper[0] - gripper[1] > 0.07)


def calibrate(rows):
    """One development initial state per direction; no held-out state labels."""
    thresholds = {}
    for row in rows:
        step = row["first_valid_step"]
        thresholds[row["case"]["target"]] = (
            int(np.ceil(step / CHECK_INTERVAL) * CHECK_INTERVAL) if step is not None else 150
        )
    return thresholds


def score(trace, decision_step, stop_reason):
    """Full-path protected displacement; last ten settling steps must stay placed."""
    target = np.asarray(trace["target_in"], bool)
    grip = np.asarray(trace["target_free"], bool)
    displacement = np.asarray(trace["protected_displacement"], float)
    stable = len(target) >= 10 and bool(target[-10:].all())
    free = len(grip) >= 10 and bool(grip[-10:].all())
    safe = bool(displacement.max(initial=0) <= DISPLACEMENT_LIMIT)
    final = stable and free and safe
    running_safe = np.maximum.accumulate(displacement) <= DISPLACEMENT_LIMIT
    valid = target & grip & running_safe
    valid_policy = valid[:decision_step]
    before = np.flatnonzero(np.convolve(valid_policy.astype(int), np.ones(10, int), mode='valid') == 10) + 9 if len(valid_policy) >= 10 else np.array([], int)
    violation = np.flatnonzero(displacement > DISPLACEMENT_LIMIT)
    return dict(
        selective_success=final,
        target_final_stable=stable,
        final_released=free,
        target_ever_reached=bool(target[:decision_step].any()),
        protected_untouched=safe,
        max_protected_displacement_m=float(displacement.max(initial=0)),
        first_valid_step=int(before[0] + 1) if len(before) else None,
        first_violation_step=int(violation[0] + 1) if len(violation) else None,
        self_stop_success=final and stop_reason != "budget",
        premature_stop=stop_reason != "budget" and not (stable and free),
    )
