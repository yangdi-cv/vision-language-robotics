"""Base Qwen3-VL affordance grounding on the auto-labelled dataset, with depth lifting to 3D.

Queries per frame (RGB only goes to Qwen): bottom-drawer handle point, bowl grasp point,
place point inside the pulled-out bottom drawer, cabinet bounding box. Scores use the
simulator masks / poses. Coordinates are scored under both conventions (absolute pixels
and 0-1000 relative) so the model's convention is decided from data.
Usage (qwen-scene venv): eval_qwen.py <data_dir> <out_json> [--replay-every 20] [--adapter DIR]
"""

from pathlib import Path
import argparse
import json
import sys
import time

import numpy as np

Q = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Q / "scripts"))
from qwen_worker import MissionPlanner, parse_json  # noqa: E402

H = W = 512
QUERIES = {
    "handle": ('Point to the handle of the bottom drawer of the dark cabinet on the left, i.e. where a robot gripper '
               'should grasp to pull or push that drawer. Return only JSON: {"point_2d": [x, y]}'),
    "bowl": ('Point to the rim of the bowl, where a robot gripper should grasp it. '
             'Return only JSON: {"point_2d": [x, y]}'),
    "place": ('Point to the spot inside the pulled-out bottom drawer of the dark cabinet where the bowl should be put down. '
              'Return only JSON: {"point_2d": [x, y]}'),
    "cabinet": ('Give the bounding box of the dark cabinet with drawers on the left side of the table. '
                'Return only JSON: {"bbox_2d": [x1, y1, x2, y2]}'),
}
CAB = "the small white cabinet with three drawers standing on the table at the left edge of the image"
QUERIES_V2 = {
    "handle": (f'Point to the handle of the lowest drawer of {CAB}, i.e. where a robot gripper should grasp to pull '
               'or push that drawer. Return only JSON: {"point_2d": [x, y]}'),
    "bowl": QUERIES["bowl"],
    "place": (f'Point to the spot inside the pulled-out lowest drawer of {CAB} where the bowl should be put down. '
              'Return only JSON: {"point_2d": [x, y]}'),
    "cabinet": f'Give the bounding box of {CAB}. Return only JSON: {{"bbox_2d": [x1, y1, x2, y2]}}',
}
CLS = {"handle": 1, "drawer": 2, "bowl": 3}


def scales(values):
    v = np.asarray(values, float)
    return {"abs": v, "rel1000": v / 1000.0 * W}


def near(mask, u, v, r=4):
    return bool(mask[max(0, v - r):v + r + 1, max(0, u - r):u + r + 1].any())


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def lift(r, uv, depth):
    """Back-project an image point with the depth map into world coordinates (same camera model as gen_dataset)."""
    u, v = int(np.clip(round(uv[0]), 0, W - 1)), int(np.clip(round(uv[1]), 0, H - 1))
    z = float(np.median(depth[max(0, v - 2):v + 3, max(0, u - 2):u + 3]))
    cam = r["camera"]
    R, t, f = np.array(cam["R"]), np.array(cam["t"]), cam["f"]
    x = (W - 1 - uv[0] - W / 2) / f * z
    y = -(uv[1] - H / 2) / f * z
    return R @ np.array([x, y, -z]) + t


def applicable(name, r):
    if name == "handle":
        return r["handle_mask"]["pixels"] >= 50
    if name == "bowl":
        return r["bowl_mask"]["pixels"] >= 300 and not r["bowl_in_drawer_region"]
    if name == "place":
        return r["drawer_opening_m"] >= 0.08 and r["drawer_mask"]["pixels"] >= 1500 and not r["bowl_in_drawer_region"]
    return r["cabinet_all_bbox"] is not None


def score(name, r, parsed, aux):
    out = {}
    for conv, val in scales(parsed).items():
        if name == "cabinet":
            out[conv] = {"iou": iou(val.tolist(), r["cabinet_all_bbox"])}
            continue
        u, v = val
        inside = 0 <= u < W and 0 <= v < H
        key = {"handle": "handle", "bowl": "bowl", "place": "drawer"}[name]
        mask = aux["cls"] == CLS[key]
        hit = bool(inside and near(mask, int(u), int(v)))
        c = r[key + "_mask"]["centroid"]
        s = {"hit": hit, "px_err": float(np.hypot(u - c[0], v - c[1]))}
        if name == "handle" and inside:
            p = lift(r, (u, v), aux["depth"].astype(np.float32))
            s["err3d_cm"] = float(np.linalg.norm(p - np.array(r["handle"]["world"])) * 100)
        if name == "place":
            iu, iv = r["drawer_interior"]["uv"]
            s["px_err_interior"] = float(np.hypot(u - iu, v - iv))
        out[conv] = s
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--replay-every", type=int, default=20)
    p.add_argument("--inits", default="0-49")
    p.add_argument("--adapter", type=Path)
    p.add_argument("--prompts", choices=["v1", "v2"], default="v2")
    p.add_argument("--rescore", action="store_true", help="re-score saved raw answers in <out_json> without running Qwen")
    a = p.parse_args()
    if a.rescore:
        labels = {json.loads(x)["id"]: json.loads(x) for x in (a.data / "labels.jsonl").read_text().splitlines()}
        results = json.loads(a.out.read_text())
        for item in results:
            r = labels[item["id"]]
            aux = np.load(a.data / "aux" / f"{r['id']}.npz")
            item.pop("error", None)
            try:
                val = parse_json(item["raw"])
                val = val[0] if isinstance(val, list) else val
                item["parsed"] = val["bbox_2d" if item["query"] == "cabinet" else "point_2d"]
                item["score"] = score(item["query"], r, item["parsed"], aux)
            except Exception as exc:  # noqa: BLE001
                item["error"] = repr(exc)
        a.out.write_text(json.dumps(results))
        summarize(results)
        return
    lo, hi = map(int, a.inits.split("-"))
    rows = [json.loads(x) for x in (a.data / "labels.jsonl").read_text().splitlines()]
    rows = [r for r in rows if lo <= r["initial"] <= hi and (r["source"] == "static" or r["step"] % a.replay_every == 0)]
    planner = MissionPlanner()
    if a.adapter:
        from peft import PeftModel
        planner.engine.model = PeftModel.from_pretrained(planner.engine.model, str(a.adapter))
    results = []
    queries = QUERIES_V2 if a.prompts == "v2" else QUERIES
    for k, r in enumerate(rows):
        aux = np.load(a.data / "aux" / f"{r['id']}.npz")
        img = str(a.data / "rgb" / f"{r['id']}.png")
        for name, prompt in queries.items():
            if not applicable(name, r):
                continue
            t = time.perf_counter()
            res = planner.generate(img, prompt, max_new_tokens=40)
            sec = time.perf_counter() - t
            item = {"id": r["id"], "source": r["source"], "initial": r["initial"], "query": name,
                    "raw": res["raw"], "seconds": round(sec, 3), "drawer_opening_m": r["drawer_opening_m"]}
            try:
                val = parse_json(res["raw"])
                val = val[0] if isinstance(val, list) else val
                parsed = val["bbox_2d" if name == "cabinet" else "point_2d"]
                item["parsed"] = parsed
                item["score"] = score(name, r, parsed, aux)
            except Exception as exc:  # noqa: BLE001
                item["error"] = repr(exc)
            results.append(item)
        if k % 25 == 0:
            print(k, len(rows), flush=True)
            a.out.write_text(json.dumps(results))
    a.out.write_text(json.dumps(results))
    summarize(results)


def summarize(results):
    for conv in ("abs", "rel1000"):
        print("== convention", conv)
        for name in QUERIES:
            sub = [x for x in results if x["query"] == name]
            ok = [x["score"][conv] for x in sub if "score" in x]
            if not sub:
                continue
            line = {"query": name, "n": len(sub), "parse_fail": len(sub) - len(ok)}
            if name == "cabinet":
                line["mean_iou"] = round(float(np.mean([s["iou"] for s in ok])), 3) if ok else None
            else:
                line["hit_rate"] = round(float(np.mean([s["hit"] for s in ok])), 3) if ok else None
                line["median_px_err"] = round(float(np.median([s["px_err"] for s in ok])), 1) if ok else None
                e3 = [s["err3d_cm"] for s in ok if "err3d_cm" in s]
                line["within_15px"] = round(float(np.mean([s["px_err"] <= 15 for s in ok])), 3) if ok else None
                if e3:
                    line["median_3d_cm"] = round(float(np.median(e3)), 1)
                    line["within_3cm"] = round(float(np.mean(np.array(e3) <= 3)), 3)
                    line["within_5cm"] = round(float(np.mean(np.array(e3) <= 5)), 3)
            print(json.dumps(line))


if __name__ == "__main__":
    main()
