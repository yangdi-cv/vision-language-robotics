"""Read-only check of the supplied lerobot.zip Feetech driver on COM12."""

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

bus = FeetechMotorsBus(
    port="COM12",
    motors={"shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES)},
)
try:
    bus._connect(handshake=False)
    bus.set_timeout()
    print("motor 2 model number:", bus.ping(2), flush=True)
    print("motor 2 position:", bus.read("Present_Position", "shoulder_lift", normalize=False), flush=True)
    print("motor 2 status:", bus.read("Status", "shoulder_lift"), flush=True)
finally:
    if bus.is_connected:
        bus.disconnect(disable_torque=False)
