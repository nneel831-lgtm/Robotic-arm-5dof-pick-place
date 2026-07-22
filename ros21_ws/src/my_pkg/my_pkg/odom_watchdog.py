#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
import time

class OdomWatchdog(Node):
    def __init__(self):
        super().__init__('odom_watchdog')

        self.declare_parameter('inlier_thresh', 3)
        self.declare_parameter('tf_timeout', 0.15)
        self.declare_parameter('cov_baseline', 0.05)
        self.declare_parameter('cov_spike_mult', 5.0)

        self.inlier_thresh = self.get_parameter('inlier_thresh').value
        self.tf_timeout = self.get_parameter('tf_timeout').value
        self.cov_baseline = self.get_parameter('cov_baseline').value
        self.cov_spike_mult = self.get_parameter('cov_spike_mult').value

        self.last_odom_time = time.time()
        self.confidence = 1.0
        self.cov_history = []

        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_cb, 10)

        self.odom_pub = self.create_publisher(
            Odometry, '/odom_scaled', 10)
        self.quality_pub = self.create_publisher(
            Float32, '/odom_quality', 10)

        self.timer = self.create_timer(0.05, self.watchdog_tick)

    def odom_cb(self, msg: Odometry):
        self.last_odom_time = time.time()

        cov_trace = sum(msg.pose.covariance[i] for i in (0, 7, 14))
        self.cov_history.append(cov_trace)
        if len(self.cov_history) > 50:
            self.cov_history.pop(0)
        median_cov = sorted(self.cov_history)[len(self.cov_history)//2] \
            if self.cov_history else self.cov_baseline

        if cov_trace > median_cov * self.cov_spike_mult:
            target_conf = 0.1
        elif cov_trace > median_cov * 2.0:
            target_conf = 0.4
        else:
            target_conf = 1.0

        # smooth ramp (no hard switching)
        self.confidence += (target_conf - self.confidence) * 0.2
        self.confidence = max(0.0, min(1.0, self.confidence))

        scale = 1.0 / max(self.confidence, 0.01)
        out = msg
        out.pose.covariance = [c * scale if i in (0,7,14,21,28,35) else c
                                for i, c in enumerate(msg.pose.covariance)]
        self.odom_pub.publish(out)
        self.quality_pub.publish(Float32(data=self.confidence))

    def watchdog_tick(self):
        gap = time.time() - self.last_odom_time
        if gap > self.tf_timeout:
            self.confidence = 0.0
            self.quality_pub.publish(Float32(data=self.confidence))
            self.get_logger().warn(f'Odom stale ({gap:.2f}s) — forcing IMU fallback')

def main():
    rclpy.init()
    node = OdomWatchdog()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
