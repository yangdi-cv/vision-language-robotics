"""Read only the white arm shoulder (motor 2) on COM12."""

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.hiwonder import HiwonderMotorsBus

bus = HiwonderMotorsBus(
    port="COM12",
    motors={"shoulder_lift": Motor(2, "hx30hm", MotorNormMode.DEGREES)},
)
try:
    bus.connect()
    print(
        {
            register: bus.read(register, "shoulder_lift", normalize=False)
            for register in (
                "Torque_Enable",
                "Status",
                "Present_Position",
                "Goal_Position",
                "Min_Position_Limit",
                "Max_Position_Limit",
                "P_Coefficient",
                "Max_Torque_Limit",
                "Torque_Limit",
                "Protection_Current",
                "Present_Current",
                "Present_Voltage",
                "Present_Temperature",
            )
        }
    )
finally:
    if bus.is_connected:
        bus.disconnect(disable_torque=False)
