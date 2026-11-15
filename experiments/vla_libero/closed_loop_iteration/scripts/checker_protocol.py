"""One shared prompt and strict answer parser for base/adapter comparisons."""

import json


def prompt(instruction):
    return f"""Image 1 is the robot scene at the start; image 2 is the scene now.
Task or goal: {instruction}
Are ALL requested final object relations and state changes satisfied in image 2? A grasp, approach, or partial completion is not enough. Compare the images, but judge the requested final state, not merely whether something moved.
Return only JSON: {{"done": true or false}}"""


def parse(raw):
    t = raw.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    x = json.loads(t)
    if not isinstance(x, dict) or set(x) != {"done"} or type(x["done"]) is not bool:
        raise ValueError("expected done boolean")
    return x["done"]


def metrics(rows):
    pos = [x for x in rows if x["expected"]]
    neg = [x for x in rows if not x["expected"]]
    tp = sum(x["valid"] and x["predicted"] is True for x in pos)
    tn = sum(x["valid"] and x["predicted"] is False for x in neg)
    fp = sum(x["valid"] and x["predicted"] is True for x in neg)
    fn = sum(x["valid"] and x["predicted"] is False for x in pos)
    return {
        "n": len(rows),
        "positive": len(pos),
        "negative": len(neg),
        "valid": sum(x["valid"] for x in rows),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": (tp + tn) / len(rows),
        "balanced_accuracy": 0.5 * (tp / len(pos) + tn / len(neg))
        if pos and neg
        else None,
        "false_positive_rate": fp / len(neg) if neg else None,
        "false_negative_rate": fn / len(pos) if pos else None,
        "invalid": sum(not x["valid"] for x in rows),
    }


def boolean_probability(engine, model, images, instruction):
    """The same restricted binary probability used in validation calibration."""
    import torch

    msg = [
        {
            "role": "user",
            "content": [{"type": "image", "image": im} for im in images]
            + [{"type": "text", "text": prompt(instruction)}],
        }
    ]
    versions = []
    for value in [False, True]:
        messages = msg + [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": json.dumps({"done": value})}],
            }
        ]
        versions.append(
            engine.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_tensors="pt",
            )
        )
    false, true = versions
    a = false["input_ids"][0]
    b = true["input_ids"][0]
    assert len(a) == len(b)
    idx = int(torch.nonzero(a != b)[0])
    assert torch.equal(a[:idx], b[:idx])
    inputs = dict(true)
    inputs["input_ids"] = inputs["input_ids"][:, :idx]
    inputs["attention_mask"] = inputs["attention_mask"][:, :idx]
    with torch.inference_mode():
        logits = model(
            **{k: v.to("cuda") for k, v in inputs.items()}, use_cache=False
        ).logits[0, -1]
        value = float(
            torch.softmax(logits[[int(a[idx]), int(b[idx])]].float(), dim=0)[1]
        )
    assert 0 <= value <= 1
    return value
