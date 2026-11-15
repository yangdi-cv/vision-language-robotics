from scene import MobileScene
import json
import numpy as np
import mujoco

s = MobileScene(mobile_start=(0.0, 0.0, 0.0), stow=True)
try:
    print(
        "CONFIG",
        s.joints,
        s.dofs,
        s.act,
        s.arm_indices,
        s.arm_vel,
        s.arm_act,
        s.model.opt.timestep,
        s.model.opt.integrator,
        flush=True,
    )
    mass = np.zeros((s.model.nv, s.model.nv))
    mujoco.mj_fullM(s.model._model, mass, s.data.qM)
    print("MASS", mass[s.dofs][:, s.dofs], flush=True)
    print(
        "PARAM",
        s.model.actuator_gainprm[s.act],
        s.model.actuator_biasprm[s.act],
        flush=True,
    )
    for i in range(30):
        s.step()
        if i < 10 or i % 5 == 0:
            print(
                i,
                json.dumps(
                    {
                        "q": s.data.qpos[s.joints].tolist(),
                        "v": s.data.qvel[s.dofs].tolist(),
                        "ctrl": s.data.ctrl[s.act].tolist(),
                        "force": s.data.actuator_force[s.act].tolist(),
                        "qfrc_act": s.data.qfrc_actuator[s.dofs].tolist(),
                        "contact": s.data.qfrc_constraint[s.dofs].tolist(),
                    }
                ),
                flush=True,
            )
finally:
    s.close()
