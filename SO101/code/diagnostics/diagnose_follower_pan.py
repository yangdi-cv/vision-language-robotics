"""One small, measured movement of a SO-ARM101 follower motor."""

import argparse
import time

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.hiwonder import HiwonderMotorsBus


parser = argparse.ArgumentParser()
parser.add_argument("motor", choices=["shoulder_pan", "shoulder_lift"])
parser.add_argument("--reverse", action="store_true")
args = parser.parse_args()
MOTOR = args.motor
MOTOR_ID = {"shoulder_pan": 1, "shoulder_lift": 2}[MOTOR]
bus = HiwonderMotorsBus(
    port="COM12",
    motors={MOTOR: Motor(MOTOR_ID, "hx30hm", MotorNormMode.DEGREES)},
)

try:
    bus.connect()
    start = int(bus.read("Present_Position", MOTOR, normalize=False))
    low = int(bus.read("Min_Position_Limit", MOTOR))
    high = int(bus.read("Max_Position_Limit", MOTOR))
    direction = -1 if args.reverse else (1 if start + 40 <= high - 20 else -1)
    target = start + direction * 40
    if not low + 20 <= target <= high - 20:
        raise RuntimeError(f"No safe 40-tick movement from {start} within {low}..{high}")

    bus.write("Goal_Position", MOTOR, start, normalize=False)
    bus.enable_torque()
    time.sleep(0.3)
    bus.write("Goal_Position", MOTOR, target, normalize=False)
    time.sleep(1.0)
    after = int(bus.read("Present_Position", MOTOR, normalize=False))
    voltage = bus.read("Present_Voltage", MOTOR)
    status = bus.read("Status", MOTOR)
    current = bus.read("Present_Current", MOTOR)
    print(
        f"start={start} target={target} actual={after} "
        f"moved={after - start} voltage_raw={voltage} "
        f"status={status} current_raw={current}",
        flush=True,
    )
finally:
    if bus.is_connected:
        bus.disable_torque()
        bus.disconnect()
