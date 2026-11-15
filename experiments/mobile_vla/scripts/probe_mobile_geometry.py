from scene import MobileScene

s = MobileScene(mobile_start=(0, 0, 0), stow=True)
try:
    for c in s.data.contact[: s.data.ncon]:
        a, b = c.geom1, c.geom2
        if "mobile" in (s.model.geom_id2name(a) or "") or "mobile" in (
            s.model.geom_id2name(b) or ""
        ):
            print(
                "CONTACT",
                s.model.geom_id2name(a),
                s.model.geom_id2name(b),
                "bodies",
                s.model.body_id2name(s.model.geom_bodyid[a]),
                s.model.body_id2name(s.model.geom_bodyid[b]),
                "pos",
                c.pos,
                "dist",
                c.dist,
                "sizes",
                s.model.geom_size[a],
                s.model.geom_size[b],
            )
    for n in ["mobile_overview", "agentview", "galleryview"]:
        i = s.model.camera_name2id(n)
        print("CAM", n, s.data.cam_xpos[i], s.data.cam_xmat[i].reshape(3, 3))
finally:
    s.close()
