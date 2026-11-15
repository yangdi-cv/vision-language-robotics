from scene import MobileScene
import math
import json

s = MobileScene(mobile_start=(-0.7, -0.8, 0), stow=True)
try:
    for command in [[0, 0, 0.3], [0.15, 0, 0], [0, 0, -0.3]]:
        s.command[:] = command
        for _ in range(30):
            s.step()
        pos = s.data.get_body_xpos("mobile_carrier")
        mat = s.data.get_body_xmat("mobile_carrier").reshape(3, 3)
        print(
            json.dumps(
                {
                    "qpos_pose": s.pose().tolist(),
                    "actual_pose": [
                        float(pos[0]),
                        float(pos[1]),
                        math.atan2(mat[1, 0], mat[0, 0]),
                    ],
                    "command": command,
                }
            ),
            flush=True,
        )
finally:
    s.close()
