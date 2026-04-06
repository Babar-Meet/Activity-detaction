"""
MediaPipe-based pose detector using the new Tasks API (0.10.33+).
Extracts 33 body landmarks for each person crop.
"""

import os
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)
import config

# Mapping of landmark indices to names (MediaPipe Pose 33 landmarks)
LANDMARK_NAMES = [
    "NOSE",
    "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER",
    "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER",
    "LEFT_EAR", "RIGHT_EAR",
    "MOUTH_LEFT", "MOUTH_RIGHT",
    "LEFT_SHOULDER", "RIGHT_SHOULDER",
    "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_WRIST", "RIGHT_WRIST",
    "LEFT_PINKY", "RIGHT_PINKY",
    "LEFT_INDEX", "RIGHT_INDEX",
    "LEFT_THUMB", "RIGHT_THUMB",
    "LEFT_HIP", "RIGHT_HIP",
    "LEFT_KNEE", "RIGHT_KNEE",
    "LEFT_ANKLE", "RIGHT_ANKLE",
    "LEFT_HEEL", "RIGHT_HEEL",
    "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX",
]

# Pose skeleton connections for drawing (pairs of landmark indices)
POSE_CONNECTIONS = [
    # Face
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 9), (0, 10),

    # Upper body
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),

    # Hand details
    (15, 17), (15, 19), (15, 21),
    (16, 18), (16, 20), (16, 22),

    # Lower body
    (23, 25), (25, 27),
    (24, 26), (26, 28),

    # Feet
    (27, 29), (29, 31),
    (28, 30), (30, 32),
]

FACE_IDX = set(range(0, 11))
HAND_IDX = {15, 16, 17, 18, 19, 20, 21, 22}
LOWER_IDX = {23, 24, 25, 26, 27, 28, 29, 30, 31, 32}
FOOT_IDX = {29, 30, 31, 32}

HAND_LANDMARK_NAMES = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17),
    (17, 18), (18, 19), (19, 20),
]

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "pose_landmarker_full.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
HAND_MODEL_PATH = os.path.join(MODEL_DIR, "hand_landmarker.task")
HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"


def _ensure_models():
    """Download required MediaPipe task models if missing."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        print(f"[PoseDetector] Downloading pose model to {MODEL_PATH}...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    if not os.path.exists(HAND_MODEL_PATH):
        print(f"[PoseDetector] Downloading hand model to {HAND_MODEL_PATH}...")
        urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL_PATH)

    print("[PoseDetector] Task models are ready.")


class PoseDetector:
    """Wraps MediaPipe PoseLandmarker (Tasks API) for body keypoint extraction."""

    def __init__(self):
        _ensure_models()

        # Create PoseLandmarker in IMAGE mode (we process crops individually)
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=config.POSE_MIN_DETECTION_CONF,
            min_pose_presence_confidence=config.POSE_MIN_DETECTION_CONF,
            min_tracking_confidence=config.POSE_MIN_TRACKING_CONF,
        )
        self.landmarker = PoseLandmarker.create_from_options(options)
        hand_options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
            running_mode=RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=config.HAND_MIN_DETECTION_CONF,
            min_hand_presence_confidence=config.HAND_MIN_DETECTION_CONF,
            min_tracking_confidence=config.HAND_MIN_TRACKING_CONF,
        )
        self.hand_landmarker = HandLandmarker.create_from_options(hand_options)
        print("[PoseDetector] MediaPipe PoseLandmarker initialized (Tasks API).")

    def _detect_hand_landmarks(self, crop_rgb, x1, y1, crop_w, crop_h):
        """Detect 21-point hand landmarks and map them to frame coordinates."""
        hand_landmarks = {}
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)
        result = self.hand_landmarker.detect(mp_image)

        if not result.hand_landmarks:
            return hand_landmarks

        handedness = result.handedness or []

        for idx, hand in enumerate(result.hand_landmarks):
            side = "LEFT"
            if idx < len(handedness) and handedness[idx]:
                label = str(handedness[idx][0].category_name).upper()
                side = "LEFT" if label == "LEFT" else "RIGHT"

            for lm_idx, lm in enumerate(hand):
                if lm_idx >= len(HAND_LANDMARK_NAMES):
                    continue

                name = f"{side}_{HAND_LANDMARK_NAMES[lm_idx]}"
                px = int(lm.x * crop_w) + x1
                py = int(lm.y * crop_h) + y1
                hand_landmarks[name] = (px, py, 1.0)

        return hand_landmarks

    @staticmethod
    def _scale_color(color, visibility):
        """Scale color intensity using visibility score."""
        intensity = max(0.35, min(1.0, visibility))
        return tuple(int(c * intensity) for c in color)

    @staticmethod
    def _lerp_int(a, b, t):
        return int(a + (b - a) * t)

    def _landmark_color(self, idx):
        """Color-map landmarks by body region for a clearer human figure."""
        if idx in HAND_IDX:
            return config.SKELETON_COLOR_HAND
        if idx in FOOT_IDX:
            return config.SKELETON_COLOR_FOOT
        if idx in FACE_IDX:
            return config.SKELETON_COLOR_FACE
        if idx in LOWER_IDX:
            return config.SKELETON_COLOR_LOWER
        return config.SKELETON_COLOR_UPPER

    def _connection_color(self, idx_a, idx_b):
        """Select line color by the strongest body region of the connection."""
        if idx_a in HAND_IDX or idx_b in HAND_IDX:
            return config.SKELETON_COLOR_HAND
        if idx_a in FOOT_IDX or idx_b in FOOT_IDX:
            return config.SKELETON_COLOR_FOOT
        if idx_a in FACE_IDX and idx_b in FACE_IDX:
            return config.SKELETON_COLOR_FACE
        if idx_a in LOWER_IDX or idx_b in LOWER_IDX:
            return config.SKELETON_COLOR_LOWER
        return config.SKELETON_COLOR_UPPER

    def detect(self, frame, bbox):
        """
        Run pose detection on a person crop.

        Args:
            frame: Full BGR frame.
            bbox: (x1, y1, x2, y2) bounding box of the person.

        Returns:
            dict of landmark_name -> (x_pixel, y_pixel, visibility)
            or None if no pose detected.
        """
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]

        # Clamp bbox to frame boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        # Validate crop size
        crop_w = x2 - x1
        crop_h = y2 - y1
        if crop_w < 30 or crop_h < 50:
            return None

        # Crop person region and convert to RGB
        crop = frame[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)

        # Detect pose
        result = self.landmarker.detect(mp_image)

        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return None

        # Convert landmarks to absolute pixel coordinates in the full frame
        pose_landmarks = result.pose_landmarks[0]  # First (only) pose
        landmarks = {}

        for idx, lm in enumerate(pose_landmarks):
            if idx < len(LANDMARK_NAMES):
                name = LANDMARK_NAMES[idx]
                px = int(lm.x * crop_w) + x1
                py = int(lm.y * crop_h) + y1
                vis = lm.visibility if hasattr(lm, 'visibility') else 1.0
                landmarks[name] = (px, py, vis)

        # Add richer hand landmarks for gesture classification and display.
        hand_landmarks = self._detect_hand_landmarks(crop_rgb, x1, y1, crop_w, crop_h)
        for name, value in hand_landmarks.items():
            if name not in landmarks:
                landmarks[name] = value

        return landmarks

    def draw_skeleton(self, frame, landmarks, alpha=None, show_confidence=False):
        """
        Draw pose skeleton overlay using detected landmarks.

        Args:
            frame: Full BGR frame (modified in-place).
            landmarks: dict from detect() method.
            alpha: Transparency of the overlay.
            show_confidence: Draw visibility/confidence text near landmarks.
        """
        if not landmarks:
            return

        if alpha is None:
            alpha = config.SKELETON_ALPHA

        overlay = frame.copy()

        # Draw connections
        for (i, j) in POSE_CONNECTIONS:
            name_i = LANDMARK_NAMES[i] if i < len(LANDMARK_NAMES) else None
            name_j = LANDMARK_NAMES[j] if j < len(LANDMARK_NAMES) else None
            if name_i in landmarks and name_j in landmarks:
                vis_i = landmarks[name_i][2]
                vis_j = landmarks[name_j][2]
                edge_vis = min(vis_i, vis_j)
                if edge_vis < config.LANDMARK_VISIBILITY_MIN:
                    continue

                pt1 = landmarks[name_i][:2]
                pt2 = landmarks[name_j][:2]

                base_color = self._connection_color(i, j)
                color = self._scale_color(base_color, edge_vis)
                t = max(0.0, min(1.0, edge_vis))
                thickness = self._lerp_int(
                    config.SKELETON_LINE_THICKNESS_MIN,
                    config.SKELETON_LINE_THICKNESS_MAX,
                    t,
                )
                cv2.line(overlay, pt1, pt2, color, thickness, cv2.LINE_AA)

        # Draw landmark points
        for idx, name in enumerate(LANDMARK_NAMES):
            if name not in landmarks:
                continue

            px, py, vis = landmarks[name]
            if vis < config.LANDMARK_VISIBILITY_MIN:
                continue

            base_color = self._landmark_color(idx)
            color = self._scale_color(base_color, vis)
            t = max(0.0, min(1.0, vis))
            radius = self._lerp_int(
                config.SKELETON_POINT_RADIUS_MIN,
                config.SKELETON_POINT_RADIUS_MAX,
                t,
            )

            cv2.circle(overlay, (px, py), radius, color, -1, cv2.LINE_AA)

            if show_confidence and vis >= config.SKELETON_CONF_TEXT_MIN_VIS:
                cv2.putText(
                    overlay,
                    f"{vis:.2f}",
                    (px + 5, py - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.30,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        # Draw detailed hand skeletons when available.
        for side in ["LEFT", "RIGHT"]:
            for (i, j) in HAND_CONNECTIONS:
                name_i = f"{side}_{HAND_LANDMARK_NAMES[i]}"
                name_j = f"{side}_{HAND_LANDMARK_NAMES[j]}"
                if name_i not in landmarks or name_j not in landmarks:
                    continue

                vis_i = landmarks[name_i][2]
                vis_j = landmarks[name_j][2]
                edge_vis = min(vis_i, vis_j)

                pt1 = landmarks[name_i][:2]
                pt2 = landmarks[name_j][:2]
                color = self._scale_color(config.SKELETON_COLOR_HAND, edge_vis)
                thickness = self._lerp_int(
                    max(config.SKELETON_LINE_THICKNESS_MIN, 2),
                    max(config.SKELETON_LINE_THICKNESS_MAX, 4),
                    max(0.0, min(1.0, edge_vis)),
                )
                cv2.line(overlay, pt1, pt2, color, thickness, cv2.LINE_AA)

            for idx, name in enumerate(HAND_LANDMARK_NAMES):
                key = f"{side}_{name}"
                if key not in landmarks:
                    continue    

                px, py, vis = landmarks[key]
                color = self._scale_color(config.SKELETON_COLOR_HAND, vis)
                radius = self._lerp_int(
                    max(config.SKELETON_POINT_RADIUS_MIN, 2),
                    max(config.SKELETON_POINT_RADIUS_MAX, 5),
                    max(0.0, min(1.0, vis)),
                )
                cv2.circle(overlay, (px, py), radius, color, -1, cv2.LINE_AA)

        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    def release(self):
        """Release MediaPipe resources."""
        if self.landmarker:
            self.landmarker.close()
            self.landmarker = None
        if self.hand_landmarker:
            self.hand_landmarker.close()
            self.hand_landmarker = None
