"""Shared V2 prompt, constrained three-class scores and operational metrics."""

import json
import numpy as np

CLASSES = ["yes", "no", "unknown"]


def prompt(instruction):
    return f"""Image 1 is the scene when this SUBGOAL started; image 2 is the current scene.
Subgoal: {instruction}
Decide whether the target is now at the requested destination, supported there and released from the robot. Merely touching an edge, hovering, approaching or still holding the target does not complete a placement. Use the reference to identify what changed and which object is intended. In robot-task left/right instructions, left/right can differ from screen-left/right; do not swap the task frame with the viewer frame. If the visual evidence is occluded or too close to a placement/contact boundary to tell, answer unknown rather than assume completion.
Return only JSON: {{"status":"yes"}} for visibly completed, {{"status":"no"}} for visibly incomplete, or {{"status":"unknown"}} when the images do not support a reliable decision."""


def prepare(processor, images, instruction, status=None):
    msg = [
        {
            "role": "user",
            "content": [{"type": "image", "image": im} for im in images]
            + [{"type": "text", "text": prompt(instruction)}],
        }
    ]
    prefix = processor.apply_chat_template(
        msg,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    if status is None:
        return dict(prefix)
    msg.append(
        {
            "role": "assistant",
            "content": [{"type": "text", "text": json.dumps({"status": status})}],
        }
    )
    full = processor.apply_chat_template(
        msg,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt",
    )
    n = prefix["input_ids"].shape[-1]
    import torch

    assert torch.equal(prefix["input_ids"], full["input_ids"][:, :n])
    labels = full["input_ids"].clone()
    labels[:, :n] = -100
    full["labels"] = labels
    return dict(full)


def score_input(processor, images, instruction):
    versions = [prepare(processor, images, instruction, c) for c in CLASSES]
    tokens = [x["input_ids"][0].tolist() for x in versions]
    idx = next(i for i, x in enumerate(zip(*tokens)) if len(set(x)) > 1)
    ids = [t[idx] for t in tokens]
    assert len(set(ids)) == 3
    data = {k: v for k, v in versions[0].items() if k != "labels"}
    data["input_ids"] = data["input_ids"][:, :idx]
    data["attention_mask"] = data["attention_mask"][:, :idx]
    return data, ids


def probabilities(model, prepared):
    import torch

    data, ids = prepared
    with torch.inference_mode():
        logits = model(
            **{k: v.to("cuda") for k, v in data.items()},
            use_cache=False,
            logits_to_keep=1,
        ).logits[0, -1]
        selected = logits[ids].double()
        p = torch.softmax(selected, dim=0).cpu().tolist()
        margin = float(selected[0] - torch.logsumexp(selected[1:], dim=0))
    assert np.isfinite(p).all()
    return dict(zip(CLASSES, p)) | {"yes_log_odds": margin}


def evaluate(rows, threshold):
    positive = [r for r in rows if r["done"]]
    negative = [r for r in rows if not r["done"]]

    def yes(r):
        return r["scores"].get("yes_log_odds", r["scores"]["yes"]) >= threshold and r[
            "scores"
        ]["yes"] >= max(r["scores"]["no"], r["scores"]["unknown"])

    tp = sum(yes(r) for r in positive)
    fp = sum(yes(r) for r in negative)
    unknown = sum(
        not yes(r) and r["scores"]["unknown"] >= r["scores"]["no"] for r in rows
    )
    rejected = sum(not yes(r) for r in rows)
    return {
        "n": len(rows),
        "positive": len(positive),
        "negative": len(negative),
        "tp": tp,
        "fp": fp,
        "recall": tp / len(positive) if positive else None,
        "fpr": fp / len(negative) if negative else None,
        "unknown_class": unknown,
        "rejected": rejected,
        "confirmation_coverage": sum(yes(r) for r in rows) / len(rows),
    }


def threshold_on_validation(rows, fpr_limit=0.05):
    values = {r["scores"].get("yes_log_odds", r["scores"]["yes"]) for r in rows}
    candidates = sorted(values | {max(values) + 1.0})
    best = None
    selected = None
    for t in candidates:
        m = evaluate(rows, t)
        if m["fpr"] > fpr_limit:
            continue
        rank = (m["recall"], -m["fpr"], t)
        if best is None or rank > best:
            best = rank
            selected = t
    return selected


def macro_recall(rows):
    recalls = []
    for c in CLASSES:
        group = [r for r in rows if r["status"] == c]
        if group:
            recalls.append(
                sum(max(CLASSES, key=r["scores"].get) == c for r in group) / len(group)
            )
    return float(np.mean(recalls))
