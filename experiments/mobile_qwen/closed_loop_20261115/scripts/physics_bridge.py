"""FollowJointTrajectory server backed by the one MuJoCo process.
Publishes measured state; never echoes commanded joints as feedback.
"""

from pathlib import Path
import argparse
import time
import threading
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor, ExternalShutdownException
from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from scipy.spatial.transform import Rotation
from vla_runtime.transport import call
from vla_runtime.records import Journal, atomic_json, error_record, logger
from vla_runtime.schemas import ArmState, trajectory, vector
from vla_runtime.geometry import rotation


D = Path(__file__).resolve().parents[1]
ROOT = D.parents[2]

LOG = logger(__name__)


class Bridge(Node):
    def __init__(self, a):
        super().__init__("mujoco_joint_driver")
        self.a = a
        self.names = [f"panda_joint{i}" for i in range(1, 8)]
        self.busy = threading.Lock()
        self.pub = self.create_publisher(JointState, "joint_states", 20)
        self.tf = TransformBroadcaster(self)
        self.group = ReentrantCallbackGroup()
        self.server = ActionServer(
            self,
            FollowJointTrajectory,
            "panda_arm_controller/follow_joint_trajectory",
            execute_callback=self.execute,
            goal_callback=self.accept,
            cancel_callback=lambda _: CancelResponse.ACCEPT,
            callback_group=self.group,
        )
        self.timer = self.create_timer(0.05, self.publish, callback_group=self.group)
        self.latest = call(a.socket, "arm_state")
        self.publish()

    def publish(self):
        try:
            s = call(self.a.socket, "arm_state", timeout=5)
        except Exception:
            LOG.exception("Measured state unavailable; no new joint feedback published")
            return
        ArmState.parse(s)
        vector(s["joint_velocities"], 7, "joint velocities")
        rotation(s["base_R"])
        self.latest = s
        m = JointState()
        m.header.stamp = self.get_clock().now().to_msg()
        m.name = self.names + ["panda_finger_joint1", "panda_finger_joint2"]
        m.position = s["joint_positions"] + s["finger_positions"]
        m.velocity = s["joint_velocities"] + [0.0, 0.0]
        self.pub.publish(m)
        t = TransformStamped()
        t.header.stamp = m.header.stamp
        t.header.frame_id = "world"
        t.child_frame_id = "panda_link0"
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = map(
            float, s["base_p"]
        )
        quat = Rotation.from_matrix(s["base_R"]).as_quat()
        (
            t.transform.rotation.x,
            t.transform.rotation.y,
            t.transform.rotation.z,
            t.transform.rotation.w,
        ) = map(float, quat)
        self.tf.sendTransform(t)

    def accept(self, request):
        t = request.trajectory
        if list(t.joint_names) != self.names or not t.points or self.busy.locked():
            return GoalResponse.REJECT
        try:
            trajectory(
                list(t.joint_names),
                [
                    {
                        "positions": list(p.positions),
                        "time_from_start": p.time_from_start.sec + p.time_from_start.nanosec / 1e9,
                    }
                    for p in t.points
                ],
            )
        except ValueError:
            LOG.exception("Rejected malformed trajectory")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def execute(self, handle):
        result = FollowJointTrajectory.Result()
        if not self.busy.acquire(False):
            handle.abort()
            result.error_code = -1
            return result
        rows = []
        failure = {}
        journal = None
        try:
            journal = Journal(self.a.out / f"controller_{time.time_ns()}.jsonl")
            tr = handle.request.trajectory
            points = np.asarray([p.positions for p in tr.points])
            times = np.array(
                [p.time_from_start.sec + p.time_from_start.nanosec / 1e9 for p in tr.points]
            )
            duration = float(times[-1])
            start = call(self.a.socket, "arm_state")
            ArmState.parse(start)
            start_error = float(np.max(np.abs(np.array(start["joint_positions"]) - points[0])))
            if start_error > 0.06:
                raise ValueError(f"Actual start-state error {start_error}")
            # Fixed 20Hz policy-control steps; each contains MuJoCo's physical substeps.
            for k in range(int(np.ceil(duration / 0.05)) + 31):
                if handle.is_cancel_requested:
                    handle.canceled()
                    result.error_code = -1
                    return result
                tick = time.monotonic()
                tau = min(k * 0.05, duration)
                target = np.array([np.interp(tau, times, points[:, j]) for j in range(7)])
                state = call(self.a.socket, "joint_step", positions=target.tolist(), timeout=10)
                ArmState.parse(state)
                err = target - np.array(state["joint_positions"])
                rows.append(
                    {
                        "trajectory_time": tau,
                        "command": target.tolist(),
                        "error": err.tolist(),
                        **state,
                    }
                )
                journal.append("joint_feedback", **rows[-1])
                feedback = FollowJointTrajectory.Feedback()
                feedback.header.stamp = self.get_clock().now().to_msg()
                feedback.joint_names = self.names
                feedback.desired.positions = target.tolist()
                feedback.actual.positions = state["joint_positions"]
                feedback.error.positions = err.tolist()
                handle.publish_feedback(feedback)
                if state["contacts"] or state.get("self_contacts"):
                    raise RuntimeError(f"Physical arm-environment contact: {state['contacts']}")
                time.sleep(max(0, 0.05 - (time.monotonic() - tick)))
            error = float(np.max(np.abs(np.array(rows[-1]["joint_positions"]) - points[-1])))
            if error > 0.015:
                raise RuntimeError(f"Physical final joint error {error}")
            result.error_code = 0
            result.error_string = f"measured joint error {error:.6f}rad"
            handle.succeed()
        except Exception as e:
            failure = error_record(e)
            LOG.exception("Simulated trajectory execution failed")
            result.error_code = -5
            result.error_string = str(e)
            handle.abort()
        finally:
            try:
                if journal:
                    journal.close()
                path = self.a.out / f"controller_{time.time_ns()}.json"
                atomic_json(
                    path,
                    {
                        "joint_feedback_source": "MuJoCo measured qpos/qvel",
                        "trajectory": rows,
                        "error_code": result.error_code,
                        "error_string": result.error_string,
                        **failure,
                    },
                )
            finally:
                self.busy.release()
        return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--socket", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    rclpy.init()
    n = Bridge(a)
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(n)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.shutdown()
        n.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
