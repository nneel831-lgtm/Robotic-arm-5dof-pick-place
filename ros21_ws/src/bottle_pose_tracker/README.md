# bottle_pose_tracker

ROS2 Humble package: real-time single-bottle detection + stable tracking +
3D pose estimation in the camera optical frame, using a RealSense D435i.

## Pipeline
- **Detection**: YOLOv8 (`ultralytics`), filtered to COCO class `bottle` (id 39).
- **Tracking**: ultralytics' built-in ByteTrack via `model.track(..., tracker="bytetrack.yaml")`
  (official ultralytics tracking API — Kalman filter + IoU association, persists
  track IDs across frames, robust to fast motion and brief occlusion).
- **Identity stability**: the node locks onto a single track ID (first acquired /
  highest confidence) and sticks to it by ID every frame, instead of re-picking
  the closest box each frame. A short timeout (`lost_id_timeout`) tolerates
  momentary detector dropouts before releasing the lock and re-acquiring.
- **Depth**: median over a shrunk, centered ROI inside the bbox, computed from
  `/camera/camera/aligned_depth_to_color/image_raw` (16UC1, mm). Zero/invalid
  depth values and out-of-range depth are rejected before the median.
- **3D back-projection**: pinhole model using intrinsics from
  `/camera/camera/color/camera_info`:
  ```
  X = (u - cx) * Z / fx
  Y = (v - cy) * Z / fy
  Z = median(ROI depth)
  ```
- **Smoothing**: frame-rate-independent EMA (time-constant based) on the 3D
  point to remove residual depth-quantization jitter without adding lag.

## Build

```bash
cd ~/ros2_ws/src
# copy this bottle_pose_tracker/ folder here
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --packages-select bottle_pose_tracker
source install/setup.bash
```

## Run

```bash
ros2 launch bottle_pose_tracker bottle_pose.launch.py
```

Override params, e.g. to force CPU or use a larger model:

```bash
ros2 launch bottle_pose_tracker bottle_pose.launch.py device:=cpu model_path:=yolov8s.pt
```

## Topics

| Topic | Type | Description |
|---|---|---|
| `/bottle/position` | `geometry_msgs/PointStamped` | Smoothed 3D position in camera optical frame (meters) |
| `/bottle/tracked` | `std_msgs/Bool` | True while a bottle track is locked/visible |
| `/bottle/debug_image` | `sensor_msgs/Image` | Annotated RGB image (bbox, ID, XYZ) |

## Notes
- `model_path` defaults to `yolov8n.pt` (auto-downloaded by ultralytics on
  first run if not cached) for best real-time FPS. Use `yolov8s.pt` for higher
  accuracy at lower FPS.
- `device:='0'` uses the first CUDA GPU; set `device:=cpu` if no GPU available.
- To use a custom-tuned ByteTrack config, copy `config/bytetrack.yaml` next to
  your launch invocation path or pass an absolute path via `tracker_config`.
- The aligned depth topic must already be in the same optical frame as the
  color image (RealSense `align_depth` is enabled in your camera launch), which
  matches the topics listed in the task.
