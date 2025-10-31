import math
from typing import Any, Dict, List

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Twist, Point
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor


class GoToRegion(Node):
    def __init__(self):
        super().__init__("go_to_region")

        self.main_group = ReentrantCallbackGroup()
        self.cmd_group = MutuallyExclusiveCallbackGroup()

        # Store all named regions by name
        self.areas: Dict[str, Marker] = {}

        # Publisher for cmd_vel
        self.cmd_publishers: List[Any] = [
            self.create_publisher(Twist, f"/agent_{i}/cmd_vel", 10)
            for i in range(self.num_agents)
        ]

        # Subscriptions
        self.create_subscription(MarkerArray, "/named_areas", self.named_areas_callback, 10, callback_group=self.main_group)
        self.create_subscription(String, "/go_to_region", self.region_name_callback, 10, callback_group=self.cmd_group)
        self.create_subscription(Odometry, "/agent_0/odometry", self.odometry_callback, 10, callback_group=self.main_group)

        self.cmd_publishers: List[Any] = [
            self.create_subscription(Odometry, f"/agent_{i}/go_to_region", self.region_name_callback, 10, callback_group=self.cmd_group)
            for i in range(self.num_agents)
        ]
        # subscription to odometry
        self.cmd_publishers: List[Any] = [
            self.create_subscription(Odometry, f"/agent_{i}/odometry", lambda msg, agent_num=i: self.odometry_callback(msg, agent_num), 10, callback_group=self.main_group)
            for i in range(self.num_agents)
        ]

        # Initialize robot position
        self.robot_positions = [Point(x=0.0, y=0.0, z=0.0) for i in range(self.num_agents)]
        self.target_radius = 0.5

    def odometry_callback(self, msg: Odometry, agent_num: int):
        """Update robot position from odometry data."""
        self.robot_positions[agent_num] = msg.pose.pose.position

    def named_areas_callback(self, msg: MarkerArray):
        """Store named areas by their text label."""
        for marker in msg.markers:
            if marker.text:
                self.areas[marker.text] = marker
                # self.get_logger().info(f"Stored region: {marker.text}")

    def region_name_callback(self, region_name: String, agent_num: int):
        """Move the robot toward the named region."""
        if region_name.data not in self.areas:
            self.get_logger().warn(f"Region '{region_name.data}' not found!")
            return

        target_marker = self.areas[region_name.data]
        target = target_marker.pose.position

        twist = Twist()
        tolerance = 0.5  # stop if within 0.5m of both x and y

        while True:
            dx = target.x - self.robot_positions[agent_num].x
            dy = target.y - self.robot_positions[agent_num].y
            distance = math.sqrt(dx**2 + dy**2)

            # Stop condition: both x and y within 0.5
            if abs(dx) <= tolerance and abs(dy) <= tolerance:
                self.get_logger().info(
                    f"Reached '{region_name.data}' within tolerance "
                    f"(dx={dx:.2f}, dy={dy:.2f}, distance={distance:.2f})"
                )
                twist.linear.x = 0.0
                twist.linear.y = 0.0
                self.cmd_publishers[agent_num].publish(twist)
                break

            twist.linear.x = 0.5 * (dx / distance)
            twist.linear.y = 0.5 * (dy / distance)

            self.cmd_publishers[agent_num].publish(twist)
            self.get_logger().info(
                f"Moving toward '{region_name.data}' "
                f"(dx={dx:.2f}, dy={dy:.2f}, distance={distance:.2f})"
            )
            rclpy.spin_once(self, timeout_sec=0.1)


            



def main(args=None):
    rclpy.init(args=args)
    node = GoToRegion()

    # Multi-threaded executor allows parallel callback execution
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()



if __name__ == '__main__':
    main()