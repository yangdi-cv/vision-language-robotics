# ruff: noqa: E402
"""Nav2 action client and ROS bridge to the same MuJoCo world.

Ideal simulator localization is intentional and reported. LiDAR comes from
MuJoCo ray intersections, not the navigation map. Map edges are virtual keepout.
"""

from pathlib import Path
import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from nav2_msgs.action import NavigateToPose
from lifecycle_msgs.srv import GetState
from geometry_msgs.msg import Twist, TransformStamped, PointStamped
from nav_msgs.msg import Odometry, OccupancyGrid, Path as NavPath
from sensor_msgs.msg import LaserScan, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from rosgraph_msgs.msg import Clock
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "scripts"))
from rpc import call


class Bridge(Node):
    def __init__(self, a):
        super().__init__("mobile_vla_bridge")
        self.a = a
        self.cmd = [0.0, 0.0, 0.0]
        self.last_cmd = 0.0
        self.latest = call(a.socket, "state")
        self.rows = []
        self.paths = []
        self.lookaheads = []
        self.failure = None
        self.odom = self.create_publisher(Odometry, "odom", 10)
        self.scan = self.create_publisher(LaserScan, "scan", 10)
        self.cloud = self.create_publisher(PointCloud2, "obstacle_cloud", 10)
        self.clockpub = self.create_publisher(Clock, "clock", 10)
        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.map = self.create_publisher(OccupancyGrid, "map", qos)
        self.tf = TransformBroadcaster(self)
        self.static = StaticTransformBroadcaster(self)
        self.create_subscription(Twist, "cmd_vel", self.command, 10)
        self.create_subscription(NavPath, "plan", self.plan, 10)
        self.create_subscription(
            PointStamped,
            "lookahead_point",
            lambda m: self.lookaheads.append(
                {
                    "frame": m.header.frame_id,
                    "x": m.point.x,
                    "y": m.point.y,
                    "time": self.latest["time"],
                    "pose": self.latest["pose"],
                }
            ),
            10,
        )
        transforms = []
        for parent, child in [
            ("base_footprint", "base_link"),
            ("base_footprint", "laser"),
        ]:
            t = TransformStamped()
            t.header.frame_id = parent
            t.child_frame_id = child
            t.transform.rotation.w = 1.0
            if child == "laser":
                t.transform.translation.z = 0.2
            transforms.append(t)
        self.static.sendTransform(transforms)
        self.publish_map()
        self.timer = self.create_timer(0.05, self.tick)
        self.map_timer = None  # Map is transient-local and static for this diagnostic.

    def command(self, m):
        self.cmd = [m.linear.x, m.linear.y, m.angular.z]
        self.last_cmd = time.monotonic()

    def plan(self, m):
        self.paths.append([[p.pose.position.x, p.pose.position.y] for p in m.poses])

    def publish_map(self):
        m = OccupancyGrid()
        m.header.frame_id = "map"
        m.info.resolution = 0.05
        m.info.width = 80
        m.info.height = 80
        m.info.origin.position.x = -2.0
        m.info.origin.position.y = -2.0
        m.info.origin.orientation.w = 1.0
        data = []
        for y in range(80):
            for x in range(80):
                wx = -2 + (x + 0.5) * 0.05
                wy = -2 + (y + 0.5) * 0.05
                blocked = (
                    x < 2
                    or y < 2
                    or x >= 78
                    or y >= 78
                    or (abs(wx + 1.05) < 0.1 and abs(wy + 0.4) < 0.1)
                    or (-0.5 <= wx <= 0.5 and -0.6 <= wy <= 0.6)
                )
                data.append(100 if blocked else 0)
        m.data = data
        self.map.publish(m)

    def tick(self):
        try:
            velocity = (
                self.cmd if time.monotonic() - self.last_cmd < 0.5 else [0.0, 0.0, 0.0]
            )
            r = call(self.a.socket, "tick", velocity=velocity)
            self.latest = r
            self.rows.append(r)
            seconds = r["time"]
            clock = Clock()
            clock.clock.sec = int(seconds)
            clock.clock.nanosec = int((seconds % 1) * 1e9)
            self.clockpub.publish(clock)
            t = TransformStamped()
            stamp = (
                self.get_clock().now().to_msg() if self.a.wall_clock else clock.clock
            )
            t.header.stamp = stamp
            t.header.frame_id = "odom"
            t.child_frame_id = "base_footprint"
            x, y, yaw = r["pose"]
            t.transform.translation.x = x
            t.transform.translation.y = y
            t.transform.rotation.z = math.sin(yaw / 2)
            t.transform.rotation.w = math.cos(yaw / 2)
            localization = TransformStamped()
            localization.header.stamp = stamp
            localization.header.frame_id = "map"
            localization.child_frame_id = "odom"
            localization.transform.rotation.w = 1.0
            self.tf.sendTransform([localization, t])
            od = Odometry()
            od.header = t.header
            od.child_frame_id = t.child_frame_id
            od.pose.pose.position.x = x
            od.pose.pose.position.y = y
            od.pose.pose.orientation = t.transform.rotation
            vx, vy, w = r["velocity"]
            od.twist.twist.linear.x = math.cos(yaw) * vx + math.sin(yaw) * vy
            od.twist.twist.linear.y = -math.sin(yaw) * vx + math.cos(yaw) * vy
            od.twist.twist.angular.z = w
            self.odom.publish(od)
            laser = LaserScan()
            laser.header.stamp = stamp
            laser.header.frame_id = "laser"
            laser.angle_min = -math.pi
            laser.angle_increment = 2 * math.pi / 180
            laser.angle_max = laser.angle_min + 179 * laser.angle_increment
            laser.range_min = 0.02
            laser.range_max = 5.0
            laser.scan_time = 0.05
            laser.ranges = r["scan"]
            self.scan.publish(laser)
            header = Header()
            header.stamp = stamp
            header.frame_id = "base_footprint"
            points = [
                [
                    d * math.cos(-math.pi + i * 2 * math.pi / 180),
                    d * math.sin(-math.pi + i * 2 * math.pi / 180),
                    0.875,
                ]
                for i, d in enumerate(r["scan_high"])
                if d < 4.99
            ]
            self.cloud.publish(point_cloud2.create_cloud_xyz32(header, points))
            if r["phase"] == "collision_stop":
                raise RuntimeError("Physics contact watchdog stopped navigation")
        except Exception as e:
            self.failure = str(e)
            self.timer.cancel()


def spin_until(node, predicate, seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.05)
        if node.failure:
            raise RuntimeError(node.failure)
        if predicate():
            return
    raise TimeoutError("ROS deadline exceeded")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--socket", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--nav-only", action="store_true")
    p.add_argument("--wall-clock", action="store_true")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = Bridge(a)
    f = (a.out / "nav2.log").open("w")
    proc = subprocess.Popen(
        [
            "ros2",
            "launch",
            "nav2_bringup",
            "navigation_launch.py",
            "use_sim_time:=" + ("False" if a.wall_clock else "True"),
            "autostart:=True",
            "use_composition:=False",
            "params_file:="
            + str(
                R / ("configs/nav2_wall.yaml" if a.wall_clock else "configs/nav2.yaml")
            ),
        ],
        stdout=f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    record = {
        "ros_clock": "wall" if a.wall_clock else "simulation",
        "navigation_success": False,
        "manipulation_success": False,
        "localization": "ideal simulator odometry, map->odom identity",
        "costmap": "chassis footprint; no SLAM/AMCL; virtual keepout boundary",
    }
    try:
        client = node.create_client(GetState, "bt_navigator/get_state")
        spin_until(node, client.service_is_ready, 60)
        future = client.call_async(GetState.Request())
        spin_until(node, future.done, 30)
        # Lifecycle activation may still be in progress; query without blocking sensor timers.
        deadline = time.monotonic() + 60
        while future.result().current_state.id != 3:
            if time.monotonic() > deadline:
                raise TimeoutError("Navigator not active")
            future = client.call_async(GetState.Request())
            spin_until(node, future.done, 5)
        action = ActionClient(node, NavigateToPose, "navigate_to_pose")
        spin_until(node, action.server_is_ready, 30)
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = "map"
        goal.pose.pose.position.x = node.latest["dock"][0]
        goal.pose.pose.position.y = node.latest["dock"][1]
        goal.pose.pose.orientation.w = 1.0
        start = time.monotonic()
        future = action.send_goal_async(goal)
        spin_until(node, future.done, 10)
        handle = future.result()
        if not handle.accepted:
            raise RuntimeError("Nav2 rejected goal")
        result = handle.get_result_async()
        spin_until(node, result.done, 180)
        record.update(
            navigation_status=result.result().status,
            navigation_success=result.result().status == 4,
            navigation_wall_seconds=time.monotonic() - start,
            arrival=node.latest,
        )
        if not record["navigation_success"]:
            raise RuntimeError("Nav2 did not reach dock")
        node.cmd = [0.0, 0.0, 0.0]
        node.last_cmd = time.monotonic()
        if not a.nav_only:
            node.latest = call(a.socket, "manipulate", timeout=180)
            spin_until(
                node,
                lambda: node.latest["phase"]
                in ["success", "timeout", "collision_stop"],
                240,
            )
            record["manipulation_success"] = node.latest["success"]
            record["final"] = node.latest
        else:
            call(a.socket, "stop")
    except Exception as e:
        record["error"] = str(e)
        print("ERROR", e, flush=True)
    finally:
        node.timer.cancel()
        (a.out / "navigation.json").write_text(json.dumps(record, indent=2))
        (a.out / "odometry.json").write_text(json.dumps(node.rows))
        (a.out / "global_paths.json").write_text(json.dumps(node.paths))
        (a.out / "lookaheads.json").write_text(json.dumps(node.lookaheads))
        print(json.dumps(record), flush=True)
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGINT)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        f.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
