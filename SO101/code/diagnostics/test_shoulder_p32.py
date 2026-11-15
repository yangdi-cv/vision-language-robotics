"""One small white shoulder lift test; restore original P afterward."""

import time

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.hiwonder import HiwonderMotorsBus

MOTOR = "shoulder_lift"
bus = HiwonderMotorsBus(
    port="COM12",
    motors={MOTOR: Motor(2, "hx30hm", MotorNormMode.DEGREES)},
)
original_p = None
try:
    bus.connect()
    start = int(bus.read("Present_Position", MOTOR, normalize=False))
    low = int(bus.read("Min_Position_Limit", MOTOR))
    target = start - 40
    if target < low + 20:
        raise RuntimeError(f"Target {target} is too close to the lower limit {low}")
    original_p = int(bus.read("P_Coefficient", MOTOR))
    bus.write("Goal_Position", MOTOR, start, normalize=False)
    bus.write("P_Coefficient", MOTOR, 32)
    bus.enable_torque()
    time.sleep(0.3)
    bus.write("Goal_Position", MOTOR, target, normalize=False)
    time.sleep(1.0)
    after = int(bus.read("Present_Position", MOTOR, normalize=False))
    print(
        f"P={original_p}->32 start={start} target={target} actual={after} "
        f"moved={after-start} current_raw={bus.read('Present_Current', MOTOR)} "
        f"voltage_raw={bus.read('Present_Voltage', MOTOR)} "
        f"status={bus.read('Status', MOTOR)}",
        flush=True,
    )
finally:
    if bus.is_connected:
        try:
            bus.disable_torque()
        finally:
            try:
                if original_p is not None:
                    bus.write("P_Coefficient", MOTOR, original_p)
            finally:
                bus.disconnect(disable_torque=False)
