from scene import MobileScene

for stow in [True, False]:
    s = MobileScene(mobile_start=(0.0, 0.0, 0.0), stow=stow)
    try:
        print(
            "STOW",
            stow,
            "max_body_masses",
            sorted(zip(s.model.body_mass, s.model.body_names), reverse=True)[:8],
            flush=True,
        )
        for c in s.data.contact[: s.data.ncon]:
            a, b = c.geom1, c.geom2
            if c.dist < -0.0005:
                print(
                    "CONTACT",
                    s.model.geom_id2name(a),
                    s.model.geom_id2name(b),
                    s.model.body_id2name(s.model.geom_bodyid[a]),
                    s.model.body_id2name(s.model.geom_bodyid[b]),
                    float(c.dist),
                    flush=True,
                )
    finally:
        s.close()
