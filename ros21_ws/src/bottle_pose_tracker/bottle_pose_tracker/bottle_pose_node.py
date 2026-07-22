#!/usr/bin/env python3
"""
bottle_pose_node.py

Real-time single-bottle detection + stable tracking + 3D pose estimation
in camera optical frame, for Intel RealSense D435i on ROS2 Humble.

Pipeline:
  1. YOLOv8 (ultralytics) object detector, class filtered to COCO "bottle" (id 39)
  2. Ultralytics built-in ByteTrack (model.track(..., tracker="bytetrack.yaml"))
     -> proper Kalman + IoU + appearance-free association, persistent track IDs,
        robust to short occlusions / fast motion (no naive "lock-on bbox").
  3. Track selection: among all "bottle" tracks, lock onto the track ID that was
     selected first (or highest confidence on first sight) and STICK to that ID
     across frames (re-acquire by ID, not by re-detecting closest box every frame).
     If the locked ID disappears for > LOST_ID_TIMEOUT seconds, release the lock
     and re-acquire the next best bottle track. This prevents flicker / random
     reselection while still recovering from real track loss.
  4. Depth fusion: ROI median depth (robust to holes/noise), invalid (0) depth
     rejected, in millimeters from the aligned depth image (16UC1).
  5. 3D back-projection using CameraInfo intrinsics (fx, fy, cx, cy) from
     /camera/camera/color/camera_info (pinhole model, already depth-aligned).
  6. Temporal smoothing of the 3D point via an exponential moving average (EMA)
     decoupled from frame rate (uses dt), to remove residual jitter from depth
     quantization without adding perceptible lag.

Output: geometry_msgs/PointStamped on /bottle/position (camera optical frame),
        plus an annotated debug image on /bottle/debug_image.
"""

import time
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Bool
from cv_bridge import CvBridge

from ultralytics import YOLO

COCO_BOTTLE_CLASS_ID = 39  # COCO class index for "bottle"


class EMA3D:
    """Frame-rate independent exponential moving average for a 3D point."""

    def __init__(self, time_constant: float = 0.15):
        self.tau = time_constant
        self.value = None
        self.last_t = None

    def reset(self):
        self.value = None
        self.last_t = None

    def update(self, measurement: np.ndarray, t: float) -> np.ndarray:
        if self.value is None:
            self.value = measurement.copy()
            self.last_t = t
            return self.value

        dt = max(t - self.last_t, 1e-3)
        alpha = 1.0 - np.exp(-dt / self.tau)
        self.value = self.value + alpha * (measurement - self.value)
        self.last_t = t
        return self.value


class BottlePoseNode(Node):

    def __init__(self):
        super().__init__('bottle_pose_node')

        # ---------------- Parameters ----------------
        self.declare_parameter('model_path', 'yolov8n.pt')
        self.declare_parameter('confidence_threshold', 0.45)
        self.declare_parameter('iou_threshold', 0.5)
        self.declare_parameter('tracker_config', 'bytetrack.yaml')
        self.declare_parameter('roi_shrink_ratio', 0.6)  # use central % of bbox for depth
        self.declare_parameter('min_valid_depth_m', 0.1)
        self.declare_parameter('max_valid_depth_m', 8.0)
        self.declare_parameter('ema_time_constant', 0.15)
        self.declare_parameter('lost_id_timeout', 1.0)  # seconds before releasing locked track
        self.declare_parameter('device', '0')  # '0' for first GPU, 'cpu' for CPU
        self.declare_parameter('image_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera/color/camera_info')
        self.declare_parameter('publish_debug_image', True)

        self.model_path = self.get_parameter('model_path').value
        self.conf_thr = float(self.get_parameter('confidence_threshold').value)
        self.iou_thr = float(self.get_parameter('iou_threshold').value)
        self.tracker_cfg = self.get_parameter('tracker_config').value
        self.roi_shrink = float(self.get_parameter('roi_shrink_ratio').value)
        self.min_depth = float(self.get_parameter('min_valid_depth_m').value)
        self.max_depth = float(self.get_parameter('max_valid_depth_m').value)
        self.lost_id_timeout = float(self.get_parameter('lost_id_timeout').value)
        self.device = self.get_parameter('device').value
        self.publish_debug = bool(self.get_parameter('publish_debug_image').value)

        image_topic = self.get_parameter('image_topic').value
        depth_topic = self.get_parameter('depth_topic').value
        info_topic = self.get_parameter('camera_info_topic').value

        # ---------------- Model ----------------
        self.get_logger().info(f'Loading YOLOv8 model: {self.model_path}')
        self.model = YOLO(self.model_path)
        self.get_logger().info('Model loaded.')

        # ---------------- State ----------------
        self.bridge = CvBridge()
        self.fx = self.fy = self.cx = self.cy = None
        self.intrinsics_ready = False

        self.locked_track_id = None
        self.last_seen_locked_t = None

        self.ema = EMA3D(time_constant=float(self.get_parameter('ema_time_constant').value))

        self.latest_depth = None
        self.latest_depth_stamp = None

        # ---------------- QoS ----------------
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # ---------------- Subscribers ----------------
        self.create_subscription(CameraInfo, info_topic, self.camera_info_cb, 10)
        self.create_subscription(Image, depth_topic, self.depth_cb, sensor_qos)
        self.create_subscription(Image, image_topic, self.image_cb, sensor_qos)

        # ---------------- Publishers ----------------
        self.pose_pub = self.create_publisher(PointStamped, '/bottle/position', 10)
        self.tracked_pub = self.create_publisher(Bool, '/bottle/tracked', 10)
        if self.publish_debug:
            self.debug_pub = self.create_publisher(Image, '/bottle/debug_image', 10)

        self.get_logger().info('bottle_pose_node initialized. Waiting for camera_info + frames...')

    # ------------------------------------------------------------------
    def camera_info_cb(self, msg: CameraInfo):
        if not self.intrinsics_ready:
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]
            self.intrinsics_ready = True
            self.get_logger().info(
                f'Camera intrinsics received: fx={self.fx:.2f} fy={self.fy:.2f} '
                f'cx={self.cx:.2f} cy={self.cy:.2f}'
            )

    # ------------------------------------------------------------------
    def depth_cb(self, msg: Image):
        # Aligned depth is typically 16UC1 in millimeters from RealSense.
        try:
            depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().warn(f'Depth conversion failed: {e}')
            return
        self.latest_depth = depth_img
        self.latest_depth_stamp = msg.header.stamp

    # ------------------------------------------------------------------
    def get_roi_median_depth_m(self, x1, y1, x2, y2) -> float:
        """Robust ROI median depth in meters. Returns np.nan if invalid."""
        if self.latest_depth is None:
            return float('nan')

        h, w = self.latest_depth.shape[:2]
        x1 = max(0, min(int(x1), w - 1))
        x2 = max(0, min(int(x2), w - 1))
        y1 = max(0, min(int(y1), h - 1))
        y2 = max(0, min(int(y2), h - 1))
        if x2 <= x1 or y2 <= y1:
            return float('nan')

        # Shrink ROI toward center to avoid edge/background bleed.
        bw = x2 - x1
        bh = y2 - y1
        cx0 = x1 + bw / 2.0
        cy0 = y1 + bh / 2.0
        sw = bw * self.roi_shrink / 2.0
        sh = bh * self.roi_shrink / 2.0
        rx1 = int(max(0, cx0 - sw))
        rx2 = int(min(w - 1, cx0 + sw))
        ry1 = int(max(0, cy0 - sh))
        ry2 = int(min(h - 1, cy0 + sh))
        if rx2 <= rx1 or ry2 <= ry1:
            rx1, ry1, rx2, ry2 = x1, y1, x2, y2

        roi = self.latest_depth[ry1:ry2, rx1:rx2].astype(np.float32)

        # Determine units: RealSense aligned depth is normally 16UC1 mm.
        if self.latest_depth.dtype == np.uint16:
            roi_m = roi / 1000.0
        else:
            # Already float (meters) in some configurations.
            roi_m = roi

        valid = roi_m[(roi_m > self.min_depth) & (roi_m < self.max_depth)]
        if valid.size == 0:
            return float('nan')

        return float(np.median(valid))

    # ------------------------------------------------------------------
    def select_bottle_track(self, boxes):
        """
        boxes: ultralytics Boxes object (already filtered to class 'bottle').
        Returns (chosen_index, track_id) or (None, None).

        Strategy: stick to self.locked_track_id if still present. Otherwise pick
        the highest-confidence bottle track and lock onto it. This avoids
        flicker/reselection while remaining robust to true track loss.
        """
        if boxes is None or boxes.id is None or len(boxes) == 0:
            return None, None

        ids = boxes.id.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()

        now = time.time()

        # If we have a locked ID, prefer it if still present this frame.
        if self.locked_track_id is not None:
            matches = np.where(ids == self.locked_track_id)[0]
            if matches.size > 0:
                self.last_seen_locked_t = now
                return int(matches[0]), self.locked_track_id
            else:
                # Not seen this frame -- check timeout before releasing.
                if self.last_seen_locked_t is not None and \
                        (now - self.last_seen_locked_t) < self.lost_id_timeout:
                    return None, self.locked_track_id  # still "locked", just not visible
                else:
                    self.get_logger().info(
                        f'Lost track id {self.locked_track_id} (timeout). Releasing lock.'
                    )
                    self.locked_track_id = None
                    self.last_seen_locked_t = None
                    self.ema.reset()

        # No active lock -> acquire highest-confidence bottle track.
        best_idx = int(np.argmax(confs))
        self.locked_track_id = int(ids[best_idx])
        self.last_seen_locked_t = now
        self.get_logger().info(f'Locked onto new bottle track id {self.locked_track_id}')
        return best_idx, self.locked_track_id

    # ------------------------------------------------------------------
    def image_cb(self, msg: Image):
        if not self.intrinsics_ready:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'Image conversion failed: {e}')
            return

        t_now = time.time()

        # Ultralytics native tracking call: YOLOv8 detector + ByteTrack association.
        results = self.model.track(
            source=frame,
            persist=True,
            classes=[COCO_BOTTLE_CLASS_ID],
            conf=self.conf_thr,
            iou=self.iou_thr,
            tracker=self.tracker_cfg,
            device=self.device,
            verbose=False,
        )

        result = results[0]
        boxes = result.boxes

        tracked_flag = Bool()
        debug_frame = frame if self.publish_debug else None

        idx, track_id = self.select_bottle_track(boxes)

        if idx is not None and track_id is not None:
            xyxy = boxes.xyxy[idx].cpu().numpy()
            conf = float(boxes.conf[idx].cpu().numpy())
            x1, y1, x2, y2 = xyxy

            z = self.get_roi_median_depth_m(x1, y1, x2, y2)

            if not np.isnan(z):
                u = (x1 + x2) / 2.0
                v = (y1 + y2) / 2.0
                X = (u - self.cx) * z / self.fx
                Y = (v - self.cy) * z / self.fy
                Z = z

                raw_point = np.array([X, Y, Z], dtype=np.float64)
                smoothed = self.ema.update(raw_point, t_now)

                pt_msg = PointStamped()
                pt_msg.header = msg.header  # camera optical frame, same stamp as image
                pt_msg.point.x = float(smoothed[0])
                pt_msg.point.y = float(smoothed[1])
                pt_msg.point.z = float(smoothed[2])
                self.pose_pub.publish(pt_msg)

                tracked_flag.data = True

                if self.publish_debug:
                    cv2.rectangle(debug_frame, (int(x1), int(y1)), (int(x2), int(y2)),
                                  (0, 255, 0), 2)
                    label = f'bottle id={track_id} conf={conf:.2f}'
                    cv2.putText(debug_frame, label, (int(x1), max(0, int(y1) - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    coord_txt = f'X={smoothed[0]:.2f} Y={smoothed[1]:.2f} Z={smoothed[2]:.2f}m'
                    cv2.putText(debug_frame, coord_txt, (int(x1), int(y2) + 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            else:
                # Box present but no valid depth this frame -- keep last EMA value,
                # do not publish a corrupted reading.
                tracked_flag.data = True
                if self.publish_debug:
                    cv2.rectangle(debug_frame, (int(x1), int(y1)), (int(x2), int(y2)),
                                  (0, 165, 255), 2)
                    cv2.putText(debug_frame, f'bottle id={track_id} depth invalid',
                                (int(x1), max(0, int(y1) - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        else:
            tracked_flag.data = False

        self.tracked_pub.publish(tracked_flag)

        if self.publish_debug:
            try:
                out_msg = self.bridge.cv2_to_imgmsg(debug_frame, encoding='bgr8')
                out_msg.header = msg.header
                self.debug_pub.publish(out_msg)
            except Exception as e:
                self.get_logger().warn(f'Debug image publish failed: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = BottlePoseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
