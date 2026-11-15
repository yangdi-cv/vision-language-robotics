"""Freeze audited data before any model scoring or checkpoint selection."""

from pathlib import Path
import collections
import hashlib
import json
import numpy as np
from PIL import Image

R = Path(__file__).resolve().parents[1]


def main():
    p = R / "data/cases.json"
    rows = json.loads(p.read_text())
    changes = 0
    for r in rows:
        if r.get("task") == 8:
            r["instruction"] = (
                "put "
                + ("another" if r.get("companion_already_placed") else "a")
                + " moka pot on the stove"
            )
            changes += 1
        if r["source"] == "synthetic":
            assert r["done"] == (
                r["relation_predicate"] and not r["grasp"] and r["support_contact"]
            )
            if r["status"] == "yes":
                assert r["done"] and r["visible_pixels"] >= 48
            if r["status"] == "no":
                assert not r["done"]
        for image in r["images"]:
            im = Image.open(R / "data" / image)
            assert im.size == (512, 512)
            assert np.asarray(im).std() > 5
    p.write_text(json.dumps(rows, indent=2))
    groups = {
        s: {r["group"] for r in rows if r["split"] == s}
        for s in ["train", "validation", "test"]
    }
    hashes = {s: set() for s in groups}
    labels = {}
    for r in rows:
        pair = (
            r["instruction"],
            tuple(
                hashlib.sha256((R / "data" / im).read_bytes()).hexdigest()
                for im in r["images"]
            ),
        )
        hashes[r["split"]].add(pair)
        labels.setdefault(pair, set()).add(r["done"])
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        assert not groups[a] & groups[b] and not hashes[a] & hashes[b]
    conflicting = {k for k, v in labels.items() if len(v) > 1}
    for r in rows:
        pair = (
            r["instruction"],
            tuple(
                hashlib.sha256((R / "data" / im).read_bytes()).hexdigest()
                for im in r["images"]
            ),
        )
        if pair in conflicting:
            r["status"] = "unknown"
            r["ambiguity_reason"] = "identical images, different physical truth"
    p.write_text(json.dumps(rows, indent=2))
    audit = {
        "n": len(rows),
        "splits": {
            s: {
                "n": sum(r["split"] == s for r in rows),
                "groups": len(groups[s]),
                "labels": dict(
                    collections.Counter(r["status"] for r in rows if r["split"] == s)
                ),
                "done_positive": sum(r["split"] == s and r["done"] for r in rows),
            }
            for s in groups
        },
        "moka_reference_count_instruction_adjustments": changes,
        "data_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        "group_overlap": False,
        "exact_pair_overlap": False,
        "pixel_truth_conflict_groups": len(conflicting),
        "support_release_checked": True,
        "images_512": True,
    }
    (R / "data/audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
