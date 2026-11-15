"""Read-only preflight for the supplied lerobot.zip environment."""

from pathlib import Path

from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig

ROOT = Path(r"G:\VLM\weights\lerobot\calibration")
devices = (
    (
        "white",
        SO101Follower(
            SO101FollowerConfig(
                port="COM12",
                id="my_awesome_follower_arm",
                calibration_dir=ROOT / "robots" / "so_follower",
                use_degrees=True,
            )
        ),
    ),
    (
        "black",
        SO101Leader(
            SO101LeaderConfig(
                port="COM13",
                id="my_awesome_leader_arm",
                calibration_dir=ROOT / "teleoperators" / "so_leader",
                use_degrees=True,
            )
        ),
    ),
)

for label, device in devices:
    bus = device.bus
    try:
        bus.connect()
        print(label, "calibration_file", device.calibration_fpath, flush=True)
        print(label, "calibration_matches", bus.is_calibrated, flush=True)
        print(label, "positions", bus.sync_read("Present_Position"), flush=True)
        print(label, "shoulder_P", bus.read("P_Coefficient", "shoulder_lift"), flush=True)
        print(label, "shoulder_torque", bus.read("Torque_Enable", "shoulder_lift"), flush=True)
    finally:
        if bus.is_connected:
            bus.disconnect(disable_torque=False)
