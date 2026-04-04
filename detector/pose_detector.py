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
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # Arms
    (11, 23), (12, 24), (23, 24),                        # Torso
    (23, 25), (25, 27), (24, 26), (26, 28),              # Legs
    (27, 29), (28, 30), (29, 31), (30, 32),              # Feet
    (0, 1), (0, 4),                                       # Face to eyes
]

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "pose_landmarker_full.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"


def _ensure_model():
    """Download the pose landmarker model if it doesn't exist."""
    if os.path.exists(MODEL_PATH):
        return
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"[PoseDetector] Downloading pose model to {MODEL_PATH}...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("[PoseDetector] Model downloaded.")


class PoseDetector:
    """Wraps MediaPipe PoseLandmarker (Tasks API) for body keypoint extraction."""

    def __init__(self):
        _ensure_model()

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
        print("[PoseDetector] MediaPipe PoseLandmarker initialized (Tasks API).")

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

        return landmarks

    def draw_skeleton(self, frame, landmarks, alpha=0.4):
        """
        Draw pose skeleton overlay using detected landmarks.

        Args:
            frame: Full BGR frame (modified in-place).
            landmarks: dict from detect() method.
            alpha: Transparency of the overlay.
        """
        if not landmarks:
            return

        overlay = frame.copy()

        # Draw connections
        for (i, j) in POSE_CONNECTIONS:
            name_i = LANDMARK_NAMES[i] if i < len(LANDMARK_NAMES) else None
            name_j = LANDMARK_NAMES[j] if j < len(LANDMARK_NAMES) else None
            if name_i in landmarks and name_j in landmarks:
                pt1 = landmarks[name_i][:2]
                pt2 = landmarks[name_j][:2]
                cv2.line(overlay, pt1, pt2, (0, 255, 128), 2, cv2.LINE_AA)

        # Draw landmark points
        for name, (px, py, vis) in landmarks.items():
            if vis > 0.3:
                cv2.circle(overlay, (px, py), 3, (0, 200, 255), -1, cv2.LINE_AA)

        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    def release(self):
        """Release MediaPipe resources."""
        if self.landmarker:
            self.landmarker.close()
            self.landmarker = None
