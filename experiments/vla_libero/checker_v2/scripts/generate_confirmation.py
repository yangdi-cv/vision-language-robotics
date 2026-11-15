# ruff: noqa: E402
"""Matched static counterfactuals plus genuine grasp/release frames. No policy inference."""

from pathlib import Path
import json
import sys
import hashlib
import math
import numpy as np
from PIL import Image

BASE = Path(__file__).resolve().parents[1]
R = BASE / "confirmation"
V = BASE.parent
sys.path.insert(0, str(V / "scripts"))
import evaluate as ev
from run_hierarchical import monitor_view
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

TASKS = json.loads((V / "configs/libero_long_tasks.json").read_text())
NAMES = {
    "alphabet_soup": "alphabet soup can",
    "tomato_sauce": "tomato sauce can",
    "cream_cheese": "cream cheese box",
    "butter": "butter",
    "porcelain_mug": "white mug",
    "white_yellow_mug": "yellow and white mug",
    "moka_pot": "moka pot",
    "black_book": "black book",
    "chocolate_pudding": "chocolate pudding",
}
TARGETS = {
    "basket_1_contain_region": "in the basket",
    "plate_1": "on the plate",
    "plate_2": "on the plate",
    "flat_stove_1_cook_region": "on the stove",
    "desk_caddy_1_back_contain_region": "in the back compartment of the caddy",
    "microwave_1_heating_region": "in the microwave",
    "living_room_table_plate_right_region": "to the right of the plate",
}


def env_for(suite, tid):
    t = suite.get_task(tid)
    e = OffScreenRenderEnv(
        bddl_file_name=str(
            Path(get_libero_path("bddl_files")) / t.problem_folder / t.bddl_file
        ),
        camera_heights=256,
        camera_widths=256,
    )
    e.seed(7)
    return e


def reset(e, suite, tid, init):
    e.reset()
    o = e.set_init_state(np.asarray(ev.initial_states(suite, tid)[init]))
    for _ in range(10):
        o, _, _, _ = e.step(ev.DUMMY.tolist())
    return o


def save(im, name):
    p = R / "data/images" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(im).save(p)
    return str(p.relative_to(R / "data"))


def regen(e):
    return e.regenerate_obs_from_state(e.sim.get_state().flatten())


def grasp(e, name):
    return bool(
        e.env._check_grasp(
            e.env.robots[0].gripper, e.env.get_object(name).contact_geoms
        )
    )


def target_pos(e, target):
    return np.array(e.env.object_states_dict[target].get_geom_state()["pos"]).copy()


def support(e, name):
    ids = {e.sim.model.geom_name2id(g) for g in e.env.get_object(name).contact_geoms}
    for c in e.sim.data.contact[: e.sim.data.ncon]:
        a, b = int(c.geom1), int(c.geom2)
        if (a in ids) == (b in ids):
            continue
        other = b if a in ids else a
        label = e.sim.model.geom_id2name(other) or ""
        if not label.startswith(("robot", "gripper")):
            return True
    return False


def visible(e, name):
    obj = e.env.get_object(name)
    ids = [
        e.sim.model.geom_name2id(g) for g in set(obj.contact_geoms + obj.visual_geoms)
    ]
    seg = e.sim.render(
        camera_name="agentview", height=512, width=512, segmentation=True
    )
    return int(np.isin(seg[..., 1], ids).sum())


def side_in_reference(e, name):
    names = ["moka_pot_1", "moka_pot_2"]
    seg = e.sim.render(
        camera_name="agentview", height=512, width=512, segmentation=True
    )
    columns = []
    for n in names:
        obj = e.env.get_object(n)
        ids = [
            e.sim.model.geom_name2id(g)
            for g in set(obj.contact_geoms + obj.visual_geoms)
        ]
        yy, xx = np.where(np.isin(seg[..., 1], ids))
        if len(xx) < 5:
            return None, None
        columns.append(511 - float(xx.mean()))
    return (
        "left"
        if columns[names.index(name)] < columns[1 - names.index(name)]
        else "right"
    ), columns


def main():
    out = R / "data/cases.json"
    if out.exists():
        raise FileExistsError(out)
    rows = []
    rejections = []
    suite = benchmark.get_benchmark_dict()["libero_10"]()
    initials = {"confirmation": [11, 12]}
    for tid in [0, 1, 4, 5, 6, 7, 8, 9]:
        e = env_for(suite, tid)
        try:
            reset(e, suite, tid, 1)
            base = V / f"results/baseline/libero_10/task{tid:02d}_init01"
            trace = np.load(base / "trace.npz")
            for a in trace["applied"]:
                e.step(a.tolist())
            templates = []
            for atom in TASKS[tid]["goal_predicates"]:
                if atom[0] not in ["on", "in"] or atom[1] not in e.env.objects_dict:
                    continue
                name, target = atom[1:]
                joint = e.env.get_object(name).joints[0]
                q = np.array(e.sim.data.get_joint_qpos(joint)).copy()
                templates.append((atom, q, q[:3] - target_pos(e, target)))
            for split, ids in initials.items():
                for init in ids:
                    for goal, (atom, q, offset) in enumerate(templates):
                        name, target = atom[1:]
                        rng = np.random.default_rng(
                            420000 + tid * 1000 + init * 10 + goal
                        )
                        reset(e, suite, tid, init)
                        cam = e.sim.model.camera_name2id("agentview")
                        e.sim.model.cam_pos[cam] += rng.uniform(-0.025, 0.025, 3)
                        e.sim.model.cam_fovy[cam] += float(rng.uniform(-2, 2))
                        # Both robot-pose options occur for every label in each matched scene group.
                        if rng.random() < 0.5:
                            e.sim.data.qpos[:9] = trace["qpos"][-1, :9]
                        e.sim.forward()
                        regen(e)
                        group = f"synth_t{tid:02d}_i{init:02d}_g{goal}"
                        companion = False
                        if len(templates) > 1 and rng.random() < 0.5:
                            other = templates[(goal + 1) % len(templates)]
                            oa, oq, od = other
                            oj = e.env.get_object(oa[1]).joints[0]
                            saved = np.array(e.sim.data.get_joint_qpos(oj)).copy()
                            oq = oq.copy()
                            oq[:3] = target_pos(e, oa[2]) + od
                            if oa[2].startswith("plate_"):
                                oq[:2] = target_pos(e, oa[2])[:2]
                            e.sim.data.set_joint_qpos(oj, oq)
                            e.sim.data.set_joint_qvel(oj, np.zeros(6))
                            e.sim.forward()
                            regen(e)
                            for _ in range(20):
                                e.step(ev.DUMMY.tolist())
                            companion = bool(e.env._eval_predicate(oa))
                            if not companion:
                                e.sim.data.set_joint_qpos(oj, saved)
                                e.sim.forward()
                                regen(e)
                        reference_side, pixels = (
                            side_in_reference(e, name) if tid == 8 else (None, None)
                        )
                        ref = save(monitor_view(e, 512), group + "_ref.png")
                        joint = e.env.get_object(name).joints[0]
                        q = q.copy()
                        q[:3] = target_pos(e, target) + offset
                        # Prefer the center for direct plate targets; keep validated offsets for narrow cavities.
                        if target.startswith("plate_"):
                            q[:2] = target_pos(e, target)[:2]
                        e.sim.data.set_joint_qpos(joint, q)
                        e.sim.data.set_joint_qvel(joint, np.zeros(6))
                        e.sim.forward()
                        regen(e)
                        for _ in range(20):
                            e.step(ev.DUMMY.tolist())
                        if (
                            not e.env._eval_predicate(atom)
                            or grasp(e, name)
                            or not support(e, name)
                        ):
                            rejections.append(
                                {
                                    "group": group,
                                    "reason": "no supported released positive anchor",
                                }
                            )
                            continue
                        anchor = np.asarray(e.sim.get_state().flatten()).copy()
                        center = np.array(e.sim.data.get_joint_qpos(joint)).copy()
                        angle = rng.uniform(0, 2 * math.pi)
                        direction = np.array([math.cos(angle), math.sin(angle), 0.0])

                        def place(delta):
                            e.set_init_state(anchor)
                            qq = center.copy()
                            qq[:3] += delta
                            e.sim.data.set_joint_qpos(joint, qq)
                            e.sim.data.set_joint_qvel(joint, np.zeros(6))
                            e.sim.forward()
                            regen(e)
                            return bool(e.env._eval_predicate(atom))

                        hi = 0.02
                        while hi < 0.65 and place(direction * hi):
                            hi *= 1.6
                        if hi >= 0.65:
                            rejections.append(
                                {
                                    "group": group,
                                    "reason": "no local predicate boundary; unsupported region",
                                }
                            )
                            continue
                        lo = 0.0
                        for _ in range(12):
                            m = (lo + hi) / 2
                            if place(direction * m):
                                lo = m
                            else:
                                hi = m
                        variants = [
                            ("center", "yes", np.zeros(3)),
                            ("center_jitter", "yes", direction * min(0.004, lo / 4)),
                            ("edge_inside", "unknown", direction * max(0, lo - 0.001)),
                            ("edge_outside", "unknown", direction * (hi + 0.002)),
                            ("offset", "no", direction * (hi + 0.04)),
                            ("far", "no", direction * (hi + 0.09)),
                            ("floating", "no", np.array([0, 0, 0.14])),
                            ("near_contact", "unknown", np.array([0, 0, 0.004])),
                        ]
                        instruction = (
                            "put the "
                            + NAMES.get(
                                name.rsplit("_", 1)[0],
                                name.rsplit("_", 1)[0].replace("_", " "),
                            )
                            + " "
                            + TARGETS.get(target, target.replace("_", " "))
                        )
                        # Name two identical pots by their reference-frame side, rather than hidden object IDs.
                        if tid == 8:
                            instruction = (
                                "put "
                                + ("another" if companion else "a")
                                + " moka pot on the stove"
                            )
                        if tid == 4:
                            instruction = (
                                "put the "
                                + NAMES[name.rsplit("_", 1)[0]]
                                + " on the "
                                + ("left" if target == "plate_1" else "right")
                                + " plate"
                            )
                        for kind, status, delta in variants:
                            relation = place(delta)
                            held = grasp(e, name)
                            supported = support(e, name)
                            done = relation and not held and supported
                            if status == "yes" and not done or status == "no" and done:
                                rejections.append(
                                    {
                                        "group": group,
                                        "kind": kind,
                                        "reason": "desired label not established",
                                    }
                                )
                                continue
                            pixels_visible = visible(e, name)
                            if pixels_visible < 48:
                                status = "unknown"
                            cur = save(
                                monitor_view(e, 512), group + "_" + kind + ".png"
                            )
                            state_dir = R / "data/states"
                            state_dir.mkdir(exist_ok=True)
                            np.savez_compressed(
                                state_dir / (group + "_" + kind + ".npz"),
                                state=np.asarray(e.sim.get_state().flatten()),
                                cam_pos=e.sim.model.cam_pos[cam].copy(),
                                cam_fovy=float(e.sim.model.cam_fovy[cam]),
                            )
                            rows.append(
                                {
                                    "id": group + "_" + kind,
                                    "group": group,
                                    "split": split,
                                    "source": "synthetic",
                                    "task": tid,
                                    "init": init,
                                    "goal": goal,
                                    "instruction": instruction,
                                    "images": [ref, cur],
                                    "status": status,
                                    "done": done,
                                    "relation_predicate": relation,
                                    "grasp": held,
                                    "kind": kind,
                                    "atom": atom,
                                    "offset_m": delta.tolist(),
                                    "boundary_radius_m": hi,
                                    "support_contact": supported,
                                    "visible_pixels": pixels_visible,
                                    "reference_side": reference_side,
                                    "reference_pixels": pixels,
                                    "companion_already_placed": companion,
                                }
                            )
            print(
                json.dumps(
                    {"task": tid, "rows": len(rows), "rejections": len(rejections)}
                ),
                flush=True,
            )
        finally:
            e.close()
    # Physical held/released samples were already replayed and audited in the pilot.

    splits = {s: {x["group"] for x in rows if x["split"] == s} for s in initials}

    out.write_text(json.dumps(rows, indent=2))
    (R / "data/rejections.json").write_text(json.dumps(rejections, indent=2))
    audit = {
        "n": len(rows),
        "splits": {
            s: {
                "n": sum(x["split"] == s for x in rows),
                "groups": len(splits[s]),
                "classes": {
                    c: sum(x["split"] == s and x["status"] == c for x in rows)
                    for c in ["yes", "no", "unknown"]
                },
            }
            for s in initials
        },
        "source_counts": {
            s: sum(x["source"] == s for x in rows) for s in ["synthetic", "natural"]
        },
        "rejected": len(rejections),
        "protocol_sha256": hashlib.sha256(
            (R / "PROTOCOL.json").read_bytes()
        ).hexdigest(),
    }
    (R / "data/audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit), flush=True)


if __name__ == "__main__":
    main()
