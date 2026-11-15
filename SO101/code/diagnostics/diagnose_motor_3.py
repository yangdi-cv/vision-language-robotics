"""Read-only COM12 motor presence check. Does not command movement or torque."""

import time

from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig


robot = SOFollower(SOFollowerRobotConfig(port="COM12", id="my_awesome_follower_arm"))
bus = robot.bus
try:
    bus._connect(handshake=False)
    bus.set_timeout()
    for attempt in range(3):
        print(f"scan {attempt + 1}:", {motor_id: bus.ping(motor_id) for motor_id in range(1, 7)}, flush=True)
        time.sleep(0.3)
    print(
        "motor 3 status:",
        {
            register: bus.read(register, "elbow_flex", normalize=False)
            for register in (
                "Torque_Enable",
                "Status",
                "Present_Position",
                "Min_Position_Limit",
                "Max_Position_Limit",
                "Present_Voltage",
                "Present_Temperature",
            )
        },
        flush=True,
    )
finally:
    if bus.is_connected:
        bus.disconnect(disable_torque=False)
