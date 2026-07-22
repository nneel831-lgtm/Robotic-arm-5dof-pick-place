#!/usr/bin/env python3

"""
Production-Grade 3D Bottle Perception Node (Explicit Quaternions & GPU)
--------------------------------------------------
Environment: ROS2 Humble, Ubuntu 22.04
Hardware: Intel RealSense D435i | NVIDIA RTX 3050

Adds appearance-based re-identification so the lock survives heavy occlusion
(e.g. the bottle is gripped and only the cap is visible, and YOLO no longer
fires the 'bottle' class at all). See the "Appearance-based Re-ID" block
below for the mechanism.
--------------------------------------------------
"""

import time
import threading
import queue
import numpy as np
import cv2
import traceback

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, TransformStamped
from std_msgs.msg import Bool

import message_filters
from cv_bridge import CvBridge, CvBridgeError

import tf2_ros
from tf2_ros import Buffer, TransformListener, TransformBroadcaster, LookupException, ConnectivityException, ExtrapolationException

from ultralytics import YOLO


def quat_to_rot_matrix(qx, qy, qz, qw):
    R = np.zeros((3, 3))
    R[0, 0] = 1 - 2 * (qy**2 + qz**2)
    R[0, 1] = 2 * (qx*qy - qz*qw)
    R[0, 2] = 2 * (qx*qz + qy*qw)
    R[1, 0] = 2 * (qx*qy + qz*qw)
    R[1, 1] = 1 - 2 * (qx**2 + qz**2)
    R[1, 2] = 2 * (qy*qz - qx*qw)
    R[2, 0] = 2 * (qx*qz - qy*qw)
    R[2, 1] = 2 * (qy*qz + qx*qw)
    R[2, 2] = 1 - 2 * (qx**2 + qy**2)
    return R

def transform_point_to_map(pt_cam, trans_msg):
    t = trans_msg.transform.translation
    q = trans_msg.transform.rotation
    R = quat_to_rot_matrix(q.x, q.y, q.z, q.w)
    translation = np.array([t.x, t.y, t.z])
    return R.dot(pt_cam) + translation

def project_3d_to_pixel(pt_cam, fx, fy, cx, cy):
    if pt_cam[2] <= 0.0: return -1, -1
    return int((pt_cam[0] * fx / pt_cam[2]) + cx), int((pt_cam[1] * fy / pt_cam[2]) + cy)


class BottleDetectorNode(Node):
    def __init__(self):
        super().__init__('bottle_detector')

        self.target_class = 39          
        self.conf_thresh = 0.50         
        self.ema_alpha = 0.20           
        self.tf_coast_duration = 0.5    
        
        self.map_frame = 'map'
        self.camera_frame = 'camera_color_frame'
        self.bottle_frame = 'bottle'

        self.state_lock = threading.Lock()
        self.bridge = CvBridge()
        self.intrinsics = None          
        
        self.is_locked = False
        self.locked_cam_pose = None     
        self.locked_map_pose = None     
        self.last_valid_detection_time = 0.0

        # --- Appearance-based Re-ID state ---
        # Carries the lock through heavy occlusion (bottle gripped, only the cap
        # visible) where YOLO no longer fires the 'bottle' class at all, so
        # ByteTrack has nothing to associate against. Only touched inside
        # inference_loop (single thread) — no lock needed for these.
        self.template = None               # last confidently-captured BGR patch
        self.template_hist = None          # HSV histogram signature of that patch
        self.last_known_bbox = None        # seeds the local search window
        self.using_fallback = False        # True when current lock is from template match, not YOLO
        self.fallback_start_time = 0.0
        self.template_update_conf = 0.65   # only refresh template on strong, trustworthy detections
        self.hist_ema_alpha = 0.3          # smooths histogram across frames, avoids single-frame noise
        self.search_pad_px = 60            # local search window padding (px) — raise if the object
                                            # moves fast between frames relative to your FPS
        self.match_scales = [0.85, 1.0, 1.15]  # handles moderate scale change as distance to cam changes
        self.match_score_thresh = 0.55     # min normalized template-match correlation to accept
        self.hist_score_thresh = 0.45      # min histogram correlation — guards against drifting onto the hand
        self.max_fallback_duration = 4.0   # seconds; beyond this, require a real YOLO re-detection

        self.frame_queue = queue.Queue(maxsize=1)
        self.render_queue = queue.Queue(maxsize=1)
        
        self.get_logger().info("Initializing YOLO Engine on CUDA...")
        # Force YOLO to utilize your RTX 3050 VRAM to drop CPU load
        self.model = YOLO('yolov8n.pt').to('cuda')
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        self.model.track(dummy_img, persist=True, classes=[self.target_class], verbose=False)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.pose_map_pub = self.create_publisher(PoseStamped, '/bottle_pose_map', 10)
        self.pose_cam_pub = self.create_publisher(PoseStamped, '/bottle_pose_camera', 10)
        self.tracked_pub = self.create_publisher(Bool, '/bottle_tracked', 10)

        self.info_sub = self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.camera_info_cb, 10)
            
        self.color_sub = message_filters.Subscriber(self, Image, '/camera/camera/color/image_raw')
        self.depth_sub = message_filters.Subscriber(self, Image, '/camera/camera/aligned_depth_to_color/image_raw')
        self.ts = message_filters.ApproximateTimeSynchronizer([self.color_sub, self.depth_sub], queue_size=5, slop=0.05)
        self.ts.registerCallback(self.sync_callback)

        self.inference_thread = threading.Thread(target=self.inference_loop, daemon=True)
        self.render_thread = threading.Thread(target=self.render_loop, daemon=True)
        self.inference_thread.start()
        self.render_thread.start()
        
        self.get_logger().info("Full XYZ Engine Online.")

    def camera_info_cb(self, msg):
        if self.intrinsics is None:
            self.intrinsics = [msg.k[0], msg.k[4], msg.k[2], msg.k[5]]
            self.get_logger().info(f"Intrinsics Locked: fx={msg.k[0]:.1f}, fy={msg.k[4]:.1f}")

    def sync_callback(self, color_msg, depth_msg):
        try:
            self.frame_queue.put_nowait((color_msg, depth_msg))
        except queue.Full:
            try:
                self.frame_queue.get_nowait()
                self.frame_queue.put_nowait((color_msg, depth_msg))
            except queue.Empty: pass

    def extract_robust_depth(self, depth_img, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2
        
        roi_w, roi_h = max(1, int(w * 0.25)), max(1, int(h * 0.25))
        rx1, ry1 = max(0, cx - roi_w // 2), max(0, cy - roi_h // 2)
        rx2, ry2 = min(depth_img.shape[1], cx + roi_w // 2), min(depth_img.shape[0], cy + roi_h // 2)
        
        depth_roi = depth_img[ry1:ry2, rx1:rx2]
        valid_depths = depth_roi[(depth_roi > 100) & (depth_roi < 5000)]
        if len(valid_depths) < 5: return 0.0
            
        q75, q25 = np.percentile(valid_depths, [75, 25])
        iqr = q75 - q25
        filtered_depths = valid_depths[(valid_depths >= q25 - 1.5 * iqr) & (valid_depths <= q75 + 1.5 * iqr)]
        return np.median(filtered_depths) / 1000.0 if len(filtered_depths) > 0 else 0.0

    def compute_3d_camera_point(self, bbox, depth_m):
        u = (bbox[0] + bbox[2]) / 2.0
        v = (bbox[1] + bbox[3]) / 2.0
        fx, fy, cx, cy = self.intrinsics
        
        x_m = (u - cx) * depth_m / fx
        y_m = (v - cy) * depth_m / fy
        z_m = depth_m
        return np.array([x_m, y_m, z_m])

    def get_camera_to_map_transform(self, stamp):
        try:
            return self.tf_buffer.lookup_transform(self.map_frame, self.camera_frame, stamp, timeout=Duration(seconds=0.02))
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None

    # ------------------------------------------------------------------
    # Appearance-based Re-ID
    # ------------------------------------------------------------------
    def compute_hsv_hist(self, roi_bgr):
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist

    def update_appearance_template(self, frame, bbox):
        """Refresh the stored appearance signature. Only call this on high-confidence
        YOLO hits — never on fallback matches — or the template will slowly drift
        onto the gripping hand instead of the bottle."""
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        if x2 - x1 < 8 or y2 - y1 < 8:
            return

        roi = frame[y1:y2, x1:x2]
        new_hist = self.compute_hsv_hist(roi)

        self.template = roi.copy()
        if self.template_hist is None:
            self.template_hist = new_hist
        else:
            # EMA blend so the signature reflects recent appearance (including the
            # transition toward "mostly hand, cap visible") without one noisy
            # frame wiping it out.
            self.template_hist = cv2.addWeighted(
                self.template_hist, 1 - self.hist_ema_alpha, new_hist, self.hist_ema_alpha, 0
            )

    def template_match_fallback(self, frame):
        """Search a local window around the last known position for the stored
        appearance signature (multi-scale template match + histogram cross-check).
        This is what keeps the lock alive through occlusion, since it never
        depends on YOLO firing the 'bottle' class."""
        if self.template is None or self.last_known_bbox is None:
            return None, 0.0

        x1, y1, x2, y2 = map(int, self.last_known_bbox)
        if x2 - x1 < 8 or y2 - y1 < 8:
            return None, 0.0

        pad = self.search_pad_px
        sx1, sy1 = max(0, x1 - pad), max(0, y1 - pad)
        sx2, sy2 = min(frame.shape[1], x2 + pad), min(frame.shape[0], y2 + pad)
        search_roi = frame[sy1:sy2, sx1:sx2]
        if search_roi.size == 0:
            return None, 0.0

        th0, tw0 = self.template.shape[:2]
        best_val, best_loc, best_size = -1.0, None, None

        for scale in self.match_scales:
            tw, th = max(4, int(tw0 * scale)), max(4, int(th0 * scale))
            if th > search_roi.shape[0] or tw > search_roi.shape[1]:
                continue
            template_scaled = cv2.resize(self.template, (tw, th))
            result = cv2.matchTemplate(search_roi, template_scaled, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if max_val > best_val:
                best_val, best_loc, best_size = max_val, max_loc, (tw, th)

        if best_loc is None or best_val < self.match_score_thresh:
            return None, 0.0

        mx1, my1 = sx1 + best_loc[0], sy1 + best_loc[1]
        mx2, my2 = mx1 + best_size[0], my1 + best_size[1]
        candidate_bbox = np.array([mx1, my1, mx2, my2], dtype=np.float32)

        candidate_roi = frame[my1:my2, mx1:mx2]
        if candidate_roi.size == 0:
            return None, 0.0

        hist_score = cv2.compareHist(self.template_hist, self.compute_hsv_hist(candidate_roi), cv2.HISTCMP_CORREL)
        if hist_score < self.hist_score_thresh:
            return None, 0.0

        return candidate_bbox, 0.5 * best_val + 0.5 * hist_score

    def inference_loop(self):
        prev_time = time.time()
        
        while rclpy.ok():
            try:
                try:
                    color_msg, depth_msg = self.frame_queue.get(timeout=0.5)
                except queue.Empty: continue

                if self.intrinsics is None: continue

                fps = 1.0 / (time.time() - prev_time)
                prev_time = time.time()

                try:
                    cv_color = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
                    cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='16UC1')
                except CvBridgeError: continue

                results = self.model.track(cv_color, persist=True, classes=[self.target_class], conf=self.conf_thresh, tracker="bytetrack.yaml", verbose=False)
                
                best_bbox, best_conf, new_cam_pose = None, 0.0, None
                
                if results and len(results[0].boxes) > 0:
                    for bbox, conf in zip(results[0].boxes.xyxy.cpu().numpy(), results[0].boxes.conf.cpu().numpy()):
                        depth_m = self.extract_robust_depth(cv_depth, bbox)
                        if depth_m > 0.1 and conf > best_conf:
                            best_conf = conf
                            best_bbox = bbox
                            new_cam_pose = self.compute_3d_camera_point(bbox, depth_m)

                # --- Appearance Re-ID: keep the lock through occlusion ---
                self.using_fallback = False
                if best_bbox is not None:
                    # Genuine YOLO hit — ground truth. Only refresh the template on
                    # strong detections so hand/occlusion pixels never get baked in.
                    if best_conf > self.template_update_conf:
                        self.update_appearance_template(cv_color, best_bbox)
                    self.last_known_bbox = best_bbox
                    self.fallback_start_time = 0.0
                elif self.last_known_bbox is not None:
                    # YOLO found nothing this frame (e.g. gripped, mostly occluded,
                    # only the cap visible — no longer looks like class 'bottle').
                    if self.fallback_start_time == 0.0:
                        self.fallback_start_time = time.time()

                    if time.time() - self.fallback_start_time < self.max_fallback_duration:
                        fb_bbox, fb_score = self.template_match_fallback(cv_color)
                        if fb_bbox is not None:
                            depth_m = self.extract_robust_depth(cv_depth, fb_bbox)
                            if depth_m > 0.1:
                                best_bbox = fb_bbox
                                best_conf = fb_score
                                new_cam_pose = self.compute_3d_camera_point(fb_bbox, depth_m)
                                self.last_known_bbox = fb_bbox
                                self.using_fallback = True
                    # else: fallback window expired without a real re-detection —
                    # let the lock drop below rather than trusting a stale template.

                trans_cam_to_map = self.get_camera_to_map_transform(color_msg.header.stamp)

                with self.state_lock:
                    if new_cam_pose is not None:
                        self.is_locked = True
                        self.last_valid_detection_time = time.time()
                        
                        # FIX: Pass RAW unfiltered data. 
                        # Do not smooth data inside a moving reference frame!
                        self.locked_cam_pose = new_cam_pose
                    else:
                        self.is_locked = False
                    
                    tracked_msg = Bool()
                    tracked_msg.data = self.is_locked
                    self.tracked_pub.publish(tracked_msg)

                    tf_status_ok = trans_cam_to_map is not None
                    should_broadcast = self.is_locked or (self.locked_cam_pose is not None and (time.time() - self.last_valid_detection_time) <= self.tf_coast_duration)

                    if should_broadcast and self.locked_cam_pose is not None:
                        # --- 1. PUBLISH CAMERA POSE ---
                        cam_pose_msg = PoseStamped()
                        cam_pose_msg.header.stamp = color_msg.header.stamp
                        cam_pose_msg.header.frame_id = self.camera_frame
                        cam_pose_msg.pose.position.x = float(self.locked_cam_pose[0])
                        cam_pose_msg.pose.position.y = float(self.locked_cam_pose[1])
                        cam_pose_msg.pose.position.z = float(self.locked_cam_pose[2])
                        # EXPLICIT QUATERNION
                        cam_pose_msg.pose.orientation.x = 0.0
                        cam_pose_msg.pose.orientation.y = 0.0
                        cam_pose_msg.pose.orientation.z = 0.0
                        cam_pose_msg.pose.orientation.w = 1.0
                        self.pose_cam_pub.publish(cam_pose_msg)

                        if tf_status_ok:
                            self.locked_map_pose = transform_point_to_map(self.locked_cam_pose, trans_cam_to_map)
                            
                            # --- 2. PUBLISH MAP POSE ---
                            map_pose_msg = PoseStamped()
                            map_pose_msg.header.stamp = color_msg.header.stamp
                            map_pose_msg.header.frame_id = self.map_frame
                            map_pose_msg.pose.position.x = float(self.locked_map_pose[0])
                            map_pose_msg.pose.position.y = float(self.locked_map_pose[1])
                            map_pose_msg.pose.position.z = float(self.locked_map_pose[2])
                            # EXPLICIT QUATERNION
                            map_pose_msg.pose.orientation.x = 0.0
                            map_pose_msg.pose.orientation.y = 0.0
                            map_pose_msg.pose.orientation.z = 0.0
                            map_pose_msg.pose.orientation.w = 1.0
                            self.pose_map_pub.publish(map_pose_msg)

                            # --- 3. BROADCAST DYNAMIC TF ---
                            tf_msg = TransformStamped()
                            tf_msg.header.stamp = color_msg.header.stamp
                            tf_msg.header.frame_id = self.map_frame
                            tf_msg.child_frame_id = self.bottle_frame
                            tf_msg.transform.translation.x = float(self.locked_map_pose[0])
                            tf_msg.transform.translation.y = float(self.locked_map_pose[1])
                            tf_msg.transform.translation.z = float(self.locked_map_pose[2])
                            # EXPLICIT QUATERNION
                            tf_msg.transform.rotation.x = 0.0
                            tf_msg.transform.rotation.y = 0.0
                            tf_msg.transform.rotation.z = 0.0
                            tf_msg.transform.rotation.w = 1.0
                            self.tf_broadcaster.sendTransform(tf_msg)
                    else:
                        if not self.is_locked:
                            self.locked_cam_pose = None
                            self.locked_map_pose = None
                            # Truly lost (beyond coast + fallback window) — clear the
                            # appearance signature so it can't cause a false
                            # re-acquisition on some unrelated object later.
                            self.last_known_bbox = None
                            self.template = None
                            self.template_hist = None
                            self.fallback_start_time = 0.0

                # Push to Render Queue
                try:
                    self.render_queue.put_nowait({
                        'image': cv_color, 'is_locked': self.is_locked, 'bbox': best_bbox, 'conf': best_conf,
                        'cam_pose': self.locked_cam_pose, 'map_pose': self.locked_map_pose, 'tf_ok': tf_status_ok, 'fps': fps,
                        'fallback': self.using_fallback
                    })
                except queue.Full: pass

            except Exception as e:
                self.get_logger().error(f"Inference Thread Crashed: {str(e)}")
                self.get_logger().error(traceback.format_exc())
                time.sleep(1.0) # Prevent log spam if it enters a death loop

    def render_loop(self):
        cv2.namedWindow("Bottle Tracking", cv2.WINDOW_AUTOSIZE)
        while rclpy.ok():
            try:
                data = self.render_queue.get(timeout=0.1)
            except queue.Empty: continue

            img = data['image']
            fallback = data.get('fallback', False)
            box_color = (0, 165, 255) if fallback else (0, 255, 0)

            if data['bbox'] is not None:
                x1, y1, x2, y2 = map(int, data['bbox'])
                cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)
                
                if data['cam_pose'] is not None:
                    u, v = project_3d_to_pixel(data['cam_pose'], self.intrinsics[0], self.intrinsics[1], self.intrinsics[2], self.intrinsics[3])
                    if u != -1:
                        cv2.circle(img, (u, v), 5, (0, 255, 255), -1)
                        
                        c_x, c_y, c_z = data['cam_pose']
                        cv2.putText(img, f"CAMERA XYZ: [{c_x:.2f}, {c_y:.2f}, {c_z:.2f}] m", (10, img.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                        
                        if data['tf_ok'] and data['map_pose'] is not None:
                            m_x, m_y, m_z = data['map_pose']
                            cv2.putText(img, f"MAP XYZ:    [{m_x:.2f}, {m_y:.2f}, {m_z:.2f}] m", (10, img.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 2)

            if not data['tf_ok']:
                cv2.putText(img, "MAP STATUS: TF Tree Missing", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                cv2.putText(img, "MAP STATUS: Synchronized", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)

            cv2.putText(img, f"FPS: {data['fps']:.1f}", (img.shape[1] - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

            if data['is_locked']:
                status_text = "TRACKER: LOCKED (VISUAL FALLBACK)" if fallback else "TRACKER: LOCKED"
                status_color = (0, 165, 255) if fallback else (0, 255, 0)
            else:
                status_text = "TRACKER: SEARCHING"
                status_color = (0, 0, 255)
            cv2.putText(img, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

            cv2.imshow("Bottle Tracking", img)
            cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = BottleDetectorNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()
