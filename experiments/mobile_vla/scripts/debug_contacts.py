from scene import MobileScene

s = MobileScene(mobile_start=(0, 0, 0), stow=False)
try:
    for c in s.data.contact[: s.data.ncon]:
        if "mobile_" in (s.model.geom_id2name(c.geom1) or "") or "mobile_" in (
            s.model.geom_id2name(c.geom2) or ""
        ):
            print(
                s.model.geom_id2name(c.geom1),
                s.model.geom_id2name(c.geom2),
                c.dist,
                c.pos,
            )
    print("geom masks")
    for name in ["mobile_chassis", "mobile_column"]:
        g = s.model.geom_name2id(name)
        print(
            name,
            s.model.geom_contype[g],
            s.model.geom_conaffinity[g],
            s.data.geom_xpos[g],
        )
finally:
    s.close()
