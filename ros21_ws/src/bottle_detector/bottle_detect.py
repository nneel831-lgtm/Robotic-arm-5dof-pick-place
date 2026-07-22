"""
bottle_detect.py
=================
Persistent, identity-preserving bottle tracker (YOLO11 + OpenCV).
"""

from __future__ import annotations

import argparse
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


# =====================================================================
# Camera abstraction
# =====================================================================

class CameraSource(ABC):
    @abstractmethod
    def read(self) -> Optional[np.ndarray]:
        ...

    @abstractmethod
    def release(self) -> None:
        ...

    def __enter__(self) -> "CameraSource":
        return self

    def __exit__(self, *exc) -> None:
        self.release()


class WebcamSource(CameraSource):
    def __init__(self, index: int = 0, width: int = 1280, height: int = 720) -> None:
        self._cap = cv2.VideoCapture(index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open webcam index {index} (expected /dev/video{index})")

    def read(self) -> Optional[np.ndarray]:
        ok, frame = self._cap.read()
        return frame if ok else None

    def release(self) -> None:
        self._cap.release()


class RealSenseSource(CameraSource):
    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30) -> None:
        raise NotImplementedError("Install pyrealsense2 and implement pipeline.start() here.")

    def read(self) -> Optional[np.ndarray]:
        raise NotImplementedError

    def release(self) -> None:
        raise NotImplementedError


class ROS2ImageSource(CameraSource):
    def __init__(self, topic: str = "/camera/image_raw") -> None:
        raise NotImplementedError("Wire to an rclpy Image subscriber + cv_bridge.")

    def read(self) -> Optional[np.ndarray]:
        raise NotImplementedError

    def release(self) -> None:
        raise NotImplementedError


# =====================================================================
# YOLO detector
# =====================================================================

COCO_BOTTLE_CLASS_ID = 39


@dataclass
class Detection:
    bbox: np.ndarray
    confidence: float


class YOLODetector:
    def __init__(
        self,
        model_path: str = "/home/neel/ros21_ws/yolo11n.pt",
        class_id: int = COCO_BOTTLE_CLASS_ID,
        conf_threshold: float = 0.4,
        device: str = "cpu",
    ) -> None:
        self.model = YOLO(model_path)
        self.class_id = class_id
        self.conf_threshold = conf_threshold
        self.device = device

    def detect(self, frame: np.ndarray) -> List[Detection]:
        results = self.model.predict(
            frame, classes=[self.class_id], conf=self.conf_threshold,
            device=self.device, verbose=False,
        )
        detections: List[Detection] = []
        if not results:
            return detections
        boxes = results[0].boxes
        if boxes is None:
            return detections
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(np.float32)
            conf = float(box.conf[0].cpu().numpy())
            detections.append(Detection(bbox=xyxy, confidence=conf))
        return detections


# =====================================================================
# Feature extraction
# =====================================================================

class FeatureExtractor:
    def __init__(
        self, max_corners: int = 150, quality_level: float = 0.01,
        min_distance: int = 7, orb_features: int = 500,
    ) -> None:
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.orb = cv2.ORB_create(nfeatures=orb_features)

    def shi_tomasi_corners(self, gray: np.ndarray, bbox: Optional[np.ndarray] = None) -> np.ndarray:
        mask = None
        if bbox is not None:
            mask = np.zeros(gray.shape[:2], dtype=np.uint8)
            x1, y1, x2, y2 = self._clamp_bbox(bbox, gray.shape)
            mask[y1:y2, x1:x2] = 255

        corners = cv2.goodFeaturesToTrack(
            gray, maxCorners=self.max_corners, qualityLevel=self.quality_level,
            minDistance=self.min_distance, mask=mask,
        )
        if corners is None:
            return np.empty((0, 1, 2), dtype=np.float32)
        return corners.astype(np.float32)

    def orb_descriptors_at(self, gray: np.ndarray, points: np.ndarray, patch_size: int = 31):
        if points is None or len(points) == 0:
            return np.empty((0, 1, 2), dtype=np.float32), None
        keypoints = [cv2.KeyPoint(float(p[0][0]), float(p[0][1]), patch_size) for p in points]
        keypoints, descriptors = self.orb.compute(gray, keypoints)
        if not keypoints:
            return np.empty((0, 1, 2), dtype=np.float32), None
        valid_points = np.array([[[kp.pt[0], kp.pt[1]]] for kp in keypoints], dtype=np.float32)
        return valid_points, descriptors

    def orb_full(self, gray: np.ndarray, bbox: np.ndarray):
        x1, y1, x2, y2 = self._clamp_bbox(bbox, gray.shape)
        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            return np.empty((0, 1, 2), dtype=np.float32), None
        keypoints, descriptors = self.orb.detectAndCompute(roi, None)
        if not keypoints:
            return np.empty((0, 1, 2), dtype=np.float32), None
        points = np.array([[[kp.pt[0] + x1, kp.pt[1] + y1]] for kp in keypoints], dtype=np.float32)
        return points, descriptors

    @staticmethod
    def color_histogram(frame_bgr: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = FeatureExtractor._clamp_bbox(bbox, frame_bgr.shape)
        roi = frame_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            return np.zeros((50, 60), dtype=np.float32)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist

    @staticmethod
    def _clamp_bbox(bbox: np.ndarray, shape) -> Tuple[int, int, int, int]:
        h, w = shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = int(max(0, min(x1, w - 1)))
        y1 = int(max(0, min(y1, h - 1)))
        x2 = int(max(x1 + 1, min(x2, w)))
        y2 = int(max(y1 + 1, min(y2, h)))
        return x1, y1, x2, y2


# =====================================================================
# Frame-to-frame feature tracking
# =====================================================================

LK_PARAMS = dict(
    winSize=(21, 21), maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)


@dataclass
class TrackUpdateResult:
    success: bool
    bbox: Optional[np.ndarray]
    points: np.ndarray
    confidence: float
    flow_vectors: np.ndarray


class FeatureTracker:
    def __init__(self, extractor: FeatureExtractor, min_points: int = 8,
                 replenish_below: int = 40, ransac_thresh: float = 3.0) -> None:
        self.extractor = extractor
        self.min_points = min_points
        self.replenish_below = replenish_below
        self.ransac_thresh = ransac_thresh

    def update(self, prev_gray, curr_gray, points, bbox) -> TrackUpdateResult:
        if points is None or len(points) < self.min_points:
            return TrackUpdateResult(False, None, np.empty((0, 1, 2), np.float32), 0.0,
                                      np.empty((0, 2, 2), np.float32))

        next_points, status, _err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, points, None, **LK_PARAMS)
        if next_points is None:
            return TrackUpdateResult(False, None, np.empty((0, 1, 2), np.float32), 0.0,
                                      np.empty((0, 2, 2), np.float32))

        status = status.reshape(-1).astype(bool)
        good_old = points[status]
        good_new = next_points[status]

        if len(good_new) < self.min_points:
            return TrackUpdateResult(False, None, good_new, 0.0, np.empty((0, 2, 2), np.float32))

        transform, inlier_mask = cv2.estimateAffinePartial2D(
            good_old, good_new, method=cv2.RANSAC, ransacReprojThreshold=self.ransac_thresh
        )
        if transform is None:
            return TrackUpdateResult(False, None, good_new, 0.0, np.empty((0, 2, 2), np.float32))

        inlier_mask = inlier_mask.reshape(-1).astype(bool)
        inliers_new = good_new[inlier_mask]
        inliers_old = good_old[inlier_mask]

        new_bbox = self._transform_bbox(bbox, transform)

        lk_retention = len(good_new) / max(len(points), 1)
        inlier_ratio = inlier_mask.sum() / max(len(inlier_mask), 1)
        confidence = float(np.clip(0.5 * lk_retention + 0.5 * inlier_ratio, 0.0, 1.0))

        flow_vectors = np.stack([inliers_old.reshape(-1, 2), inliers_new.reshape(-1, 2)], axis=1)
        return TrackUpdateResult(True, new_bbox, inliers_new, confidence, flow_vectors)

    def replenish(self, gray, points, bbox) -> np.ndarray:
        if len(points) >= self.replenish_below:
            return points
        fresh = self.extractor.shi_tomasi_corners(gray, bbox)
        if len(fresh) == 0:
            return points
        if len(points) > 0:
            existing = points.reshape(-1, 2)
            keep = []
            for p in fresh.reshape(-1, 2):
                d = np.linalg.norm(existing - p, axis=1)
                if d.min() > 4.0:
                    keep.append(p)
            fresh = (np.array(keep, dtype=np.float32).reshape(-1, 1, 2)
                     if keep else np.empty((0, 1, 2), np.float32))
        return np.concatenate([points, fresh], axis=0) if len(fresh) else points

    @staticmethod
    def _transform_bbox(bbox: np.ndarray, transform: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        ones = np.ones((4, 1), dtype=np.float32)
        homog = np.hstack([corners, ones])
        transformed = (transform @ homog.T).T
        new_x1 = transformed[:, 0].min()
        new_y1 = transformed[:, 1].min()
        new_x2 = transformed[:, 0].max()
        new_y2 = transformed[:, 1].max()
        return np.array([new_x1, new_y1, new_x2, new_y2], dtype=np.float32)


# =====================================================================
# Per-track Kalman filter
# (FIX: statePost is now built as an explicit (6,1) column vector.
#  OpenCV 5.x's stricter Mat handling was rejecting the previous flat
#  (6,) array during the transitionMatrix @ statePost multiply inside
#  predict() -> that was the exact crash you hit.)
# =====================================================================

class MotionPredictor:
    def __init__(self) -> None:
        self.kf = cv2.KalmanFilter(6, 4)
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0],
             [0, 0, 0, 1, 0, 0]], dtype=np.float32,
        )
        self._base_transition = np.eye(6, dtype=np.float32)
        self.kf.transitionMatrix = self._base_transition.copy()
        self.kf.processNoiseCov = np.diag([1e-2, 1e-2, 1e-2, 1e-2, 1e-1, 1e-1]).astype(np.float32)
        self.kf.measurementNoiseCov = np.diag([1e-1, 1e-1, 1e-1, 1e-1]).astype(np.float32)
        self.kf.errorCovPost = np.eye(6, dtype=np.float32)
        self._initialized = False

    def init(self, bbox: np.ndarray) -> None:
        cx, cy, w, h = self._bbox_to_cxcywh(bbox)
        # Explicit column vector (6,1) -- required by OpenCV 5.x's stricter gemm checks.
        self.kf.statePost = np.array(
            [[cx], [cy], [w], [h], [0.0], [0.0]], dtype=np.float32
        )
        self._initialized = True

    def predict(self, dt: float = 1.0) -> np.ndarray:
        transition = self._base_transition.copy()
        transition[0, 4] = dt
        transition[1, 5] = dt
        self.kf.transitionMatrix = transition
        state = self.kf.predict()
        return self._cxcywh_to_bbox(state[0], state[1], state[2], state[3])

    def correct(self, bbox: np.ndarray) -> np.ndarray:
        cx, cy, w, h = self._bbox_to_cxcywh(bbox)
        measurement = np.array([[cx], [cy], [w], [h]], dtype=np.float32)
        state = self.kf.correct(measurement)
        return self._cxcywh_to_bbox(state[0], state[1], state[2], state[3])

    @property
    def initialized(self) -> bool:
        return self._initialized

    @staticmethod
    def _bbox_to_cxcywh(bbox: np.ndarray):
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0, (x2 - x1), (y2 - y1)

    @staticmethod
    def _cxcywh_to_bbox(cx, cy, w, h) -> np.ndarray:
        cx, cy, w, h = (
            float(np.asarray(cx).reshape(-1)[0]),
            float(np.asarray(cy).reshape(-1)[0]),
            float(np.asarray(w).reshape(-1)[0]),
            float(np.asarray(h).reshape(-1)[0]),
        )
        return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], dtype=np.float32)


# =====================================================================
# Track data model
# =====================================================================

class TrackState(Enum):
    INITIALIZING = auto()
    TRACKING = auto()
    COASTING = auto()
    LOST = auto()


@dataclass
class Signature:
    orb_descriptors: Optional[np.ndarray]
    color_hist: np.ndarray


@dataclass
class BottleTrack:
    track_id: int
    bbox: np.ndarray
    points: np.ndarray
    signature: Signature
    state: TrackState = TrackState.INITIALIZING
    predictor: MotionPredictor = field(default_factory=MotionPredictor)
    confidence: float = 0.0
    frames_since_yolo: int = 0
    frames_since_seen: int = 0
    age: int = 0

    def center(self) -> np.ndarray:
        x1, y1, x2, y2 = self.bbox
        return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)

    def size(self) -> np.ndarray:
        x1, y1, x2, y2 = self.bbox
        return np.array([x2 - x1, y2 - y1], dtype=np.float32)

    def is_active(self) -> bool:
        return self.state in (TrackState.INITIALIZING, TrackState.TRACKING, TrackState.COASTING)


# =====================================================================
# Re-identification
# =====================================================================

_BF_MATCHER = cv2.BFMatcher(cv2.NORM_HAMMING)


class ReIdentifier:
    def __init__(self, hist_threshold: float = 0.5, match_ratio: float = 0.75, min_good_matches: int = 8) -> None:
        self.hist_threshold = hist_threshold
        self.match_ratio = match_ratio
        self.min_good_matches = min_good_matches

    @staticmethod
    def compute_signature(extractor: FeatureExtractor, frame_bgr, gray, bbox) -> Signature:
        _points, descriptors = extractor.orb_full(gray, bbox)
        hist = FeatureExtractor.color_histogram(frame_bgr, bbox)
        return Signature(orb_descriptors=descriptors, color_hist=hist)

    def similarity(self, sig_a: Signature, sig_b: Signature) -> float:
        hist_score = cv2.compareHist(sig_a.color_hist, sig_b.color_hist, cv2.HISTCMP_CORREL)
        if hist_score < self.hist_threshold:
            return 0.0
        if sig_a.orb_descriptors is None or sig_b.orb_descriptors is None:
            return 0.0
        if len(sig_a.orb_descriptors) < 2 or len(sig_b.orb_descriptors) < 2:
            return 0.0
        matches = _BF_MATCHER.knnMatch(sig_a.orb_descriptors, sig_b.orb_descriptors, k=2)
        good = [m for m, n in matches if m.distance < self.match_ratio * n.distance]
        match_score = len(good) / max(min(len(sig_a.orb_descriptors), len(sig_b.orb_descriptors)), 1)
        if len(good) < self.min_good_matches:
            return 0.0
        return float(np.clip(0.4 * hist_score + 0.6 * match_score, 0.0, 1.0))

    def find_match(self, candidate: Signature, lost_tracks: dict, accept_threshold: float = 0.35) -> Optional[int]:
        best_id, best_score = None, 0.0
        for track_id, track in lost_tracks.items():
            score = self.similarity(candidate, track.signature)
            if score > best_score:
                best_id, best_score = track_id, score
        return best_id if best_score >= accept_threshold else None


# =====================================================================
# Orchestration
# =====================================================================

@dataclass
class FrameOutput:
    frame_number: int
    fps: float
    track_id: int
    bbox: np.ndarray
    center: np.ndarray
    points: np.ndarray
    feature_count: int
    tracking_confidence: float
    yolo_confidence: Optional[float]
    state: str
    lost: bool
    flow_vectors: np.ndarray


def iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    xa1, ya1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    xa2, ya2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    inter = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class TrackingManager:
    def __init__(self, detector: YOLODetector, yolo_interval: int = 30,
                 low_confidence_threshold: float = 0.35, tracking_confidence_threshold: float = 0.3,
                 coasting_timeout: int = 15, lost_ttl: int = 150, iou_match_threshold: float = 0.3) -> None:
        self.detector = detector
        self.extractor = FeatureExtractor()
        self.feature_tracker = FeatureTracker(self.extractor)
        self.reidentifier = ReIdentifier()
        self.yolo_interval = yolo_interval
        self.low_confidence_threshold = low_confidence_threshold
        self.tracking_confidence_threshold = tracking_confidence_threshold
        self.coasting_timeout = coasting_timeout
        self.lost_ttl = lost_ttl
        self.iou_match_threshold = iou_match_threshold
        self.active_tracks: Dict[int, BottleTrack] = {}
        self.lost_tracks: Dict[int, BottleTrack] = {}
        self._next_id = 1
        self._prev_gray: Optional[np.ndarray] = None
        self.frame_count = 0
        self._last_time = time.time()
        self.fps = 0.0

    def process_frame(self, frame: np.ndarray) -> List[FrameOutput]:
        self.frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._update_fps()
        yolo_confidences: Dict[int, float] = {}
        self._propagate_active_tracks(gray)
        if self._should_run_yolo():
            detections = self.detector.detect(frame)
            yolo_confidences.update(self._match_and_update(frame, gray, detections))
        self._expire_tracks()
        self._prev_gray = gray
        return self._build_outputs(yolo_confidences)

    def _propagate_active_tracks(self, gray: np.ndarray) -> None:
        for track in self.active_tracks.values():
            predicted_bbox = track.predictor.predict()
            track.age += 1
            track.frames_since_yolo += 1

            if self._prev_gray is None or len(track.points) < self.feature_tracker.min_points:
                track.bbox = predicted_bbox
                track.state = TrackState.COASTING
                track.frames_since_seen += 1
                continue

            result = self.feature_tracker.update(self._prev_gray, gray, track.points, track.bbox)

            if result.success and result.confidence >= self.tracking_confidence_threshold:
                track.bbox = track.predictor.correct(result.bbox)
                track.points = self.feature_tracker.replenish(gray, result.points, track.bbox)
                track.confidence = result.confidence
                track.state = TrackState.TRACKING
                track.frames_since_seen = 0
            else:
                track.bbox = predicted_bbox
                track.points = result.points
                track.confidence = result.confidence
                track.state = TrackState.COASTING
                track.frames_since_seen += 1

    def _should_run_yolo(self) -> bool:
        if not self.active_tracks:
            return True
        if self.frame_count % self.yolo_interval == 0:
            return True
        return any(t.confidence < self.low_confidence_threshold or t.state == TrackState.COASTING
                   for t in self.active_tracks.values())

    def _match_and_update(self, frame, gray, detections) -> Dict[int, float]:
        yolo_confidences: Dict[int, float] = {}
        matched_ids = set()
        unmatched_detections = []

        for det in detections:
            best_id, best_iou = None, 0.0
            for track_id, track in self.active_tracks.items():
                if track_id in matched_ids:
                    continue
                score = iou(det.bbox, track.bbox)
                if score > best_iou:
                    best_id, best_iou = track_id, score

            if best_id is not None and best_iou >= self.iou_match_threshold:
                track = self.active_tracks[best_id]
                track.bbox = track.predictor.correct(det.bbox)
                track.points = self.extractor.shi_tomasi_corners(gray, track.bbox)
                track.frames_since_yolo = 0
                track.frames_since_seen = 0
                track.confidence = max(track.confidence, det.confidence)
                track.state = TrackState.TRACKING
                matched_ids.add(best_id)
                yolo_confidences[best_id] = det.confidence
            else:
                unmatched_detections.append(det)

        for det in unmatched_detections:
            self._resolve_unmatched_detection(frame, gray, det, yolo_confidences)
        return yolo_confidences

    def _resolve_unmatched_detection(self, frame, gray, det, yolo_confidences: Dict[int, float]) -> None:
        candidate_sig = self.reidentifier.compute_signature(self.extractor, frame, gray, det.bbox)
        match_id = self.reidentifier.find_match(candidate_sig, self.lost_tracks)

        if match_id is not None:
            track = self.lost_tracks.pop(match_id)
            track.bbox = det.bbox
            track.points = self.extractor.shi_tomasi_corners(gray, det.bbox)
            track.signature = candidate_sig
            track.predictor.init(det.bbox)
            track.state = TrackState.TRACKING
            track.frames_since_seen = 0
            track.frames_since_yolo = 0
            track.confidence = det.confidence
            self.active_tracks[track.track_id] = track
            yolo_confidences[track.track_id] = det.confidence
        else:
            new_id = self._next_id
            self._next_id += 1
            points = self.extractor.shi_tomasi_corners(gray, det.bbox)
            track = BottleTrack(track_id=new_id, bbox=det.bbox, points=points, signature=candidate_sig)
            track.predictor.init(det.bbox)
            track.state = TrackState.INITIALIZING
            track.confidence = det.confidence
            self.active_tracks[new_id] = track
            yolo_confidences[new_id] = det.confidence

    def _expire_tracks(self) -> None:
        for track_id in list(self.active_tracks.keys()):
            track = self.active_tracks[track_id]
            if track.state == TrackState.COASTING and track.frames_since_seen > self.coasting_timeout:
                track.state = TrackState.LOST
                track.frames_since_seen = 0
                self.lost_tracks[track_id] = track
                del self.active_tracks[track_id]

        for track_id in list(self.lost_tracks.keys()):
            track = self.lost_tracks[track_id]
            track.frames_since_seen += 1
            if track.frames_since_seen > self.lost_ttl:
                del self.lost_tracks[track_id]

    def _build_outputs(self, yolo_confidences: Dict[int, float]) -> List[FrameOutput]:
        outputs = []
        for track_id, track in self.active_tracks.items():
            outputs.append(FrameOutput(
                frame_number=self.frame_count, fps=self.fps, track_id=track_id,
                bbox=track.bbox.copy(), center=track.center(), points=track.points.copy(),
                feature_count=len(track.points), tracking_confidence=track.confidence,
                yolo_confidence=yolo_confidences.get(track_id), state=track.state.name,
                lost=False, flow_vectors=np.empty((0, 2, 2), dtype=np.float32),
            ))
        return outputs

    def _update_fps(self) -> None:
        now = time.time()
        dt = now - self._last_time
        self._last_time = now
        if dt > 0:
            instant_fps = 1.0 / dt
            self.fps = instant_fps if self.fps == 0 else 0.9 * self.fps + 0.1 * instant_fps


# =====================================================================
# Visualization
# =====================================================================

_COLORS = [(66, 135, 245), (66, 245, 129), (245, 66, 66), (245, 209, 66), (176, 66, 245), (66, 245, 236)]


def _color_for(track_id: int):
    return _COLORS[track_id % len(_COLORS)]


def draw(frame: np.ndarray, outputs: List[FrameOutput]) -> np.ndarray:
    canvas = frame.copy()
    for out in outputs:
        color = _color_for(out.track_id)
        x1, y1, x2, y2 = out.bbox.astype(int)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        for p in out.points.reshape(-1, 2):
            cv2.circle(canvas, (int(p[0]), int(p[1])), 2, color, -1)
        for start, end in out.flow_vectors:
            cv2.arrowedLine(canvas, tuple(start.astype(int)), tuple(end.astype(int)), color, 1, tipLength=0.3)
        cx, cy = out.center.astype(int)
        cv2.drawMarker(canvas, (cx, cy), color, cv2.MARKER_CROSS, 12, 2)
        label = f"ID {out.track_id} | {out.state} | conf {out.tracking_confidence:.2f}"
        if out.yolo_confidence is not None:
            label += f" | yolo {out.yolo_confidence:.2f}"
        cv2.putText(canvas, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(canvas, f"feats {out.feature_count}", (x1, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    fps = outputs[0].fps if outputs else 0.0
    cv2.putText(canvas, f"FPS: {fps:.1f}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return canvas


# =====================================================================
# Entry point
# =====================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent bottle tracker")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--model", type=str, default="/home/neel/ros21_ws/yolo11n.pt")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--yolo-interval", type=int, default=30)
    parser.add_argument("--print-json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detector = YOLODetector(model_path=args.model, device=args.device)
    manager = TrackingManager(detector=detector, yolo_interval=args.yolo_interval)

    with WebcamSource(index=args.camera_index) as camera:
        while True:
            frame = camera.read()
            if frame is None:
                break

            outputs = manager.process_frame(frame)
            annotated = draw(frame, outputs)
            cv2.imshow("Bottle Tracker", annotated)

            if args.print_json:
                for out in outputs:
                    print(json.dumps({
                        "frame": out.frame_number, "fps": round(out.fps, 1),
                        "track_id": out.track_id, "bbox": out.bbox.tolist(),
                        "center": out.center.tolist(), "feature_count": out.feature_count,
                        "tracking_confidence": round(out.tracking_confidence, 3),
                        "yolo_confidence": out.yolo_confidence, "state": out.state,
                    }))

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
