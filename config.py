"""
Configuration constants for the Activity Detection system.
All tunable parameters are centralized here.
"""

# ─── Camera ──────────────────────────────────────────────────────────────────
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# ─── YOLO Object Detection ──────────────────────────────────────────────────
YOLO_MODEL = "yolov8n.pt"          # nano model for real-time speed
YOLO_CONFIDENCE = 0.45
YOLO_IOU_THRESHOLD = 0.50

# ─── MediaPipe Pose ─────────────────────────────────────────────────────────
POSE_MIN_DETECTION_CONF = 0.5
POSE_MIN_TRACKING_CONF = 0.5
POSE_MODEL_COMPLEXITY = 1          # 0=lite, 1=full, 2=heavy

# ─── Centroid Tracker ───────────────────────────────────────────────────────
TRACKER_MAX_DISAPPEARED = 30       # frames before dropping an ID
TRACKER_MAX_DISTANCE = 80          # max pixel distance to associate

# ─── Action Detection Thresholds ────────────────────────────────────────────
# Angles (degrees)
STANDING_KNEE_ANGLE_MIN = 155      # nearly straight leg
SITTING_CHAIR_KNEE_ANGLE_MAX = 130
SITTING_CHAIR_KNEE_ANGLE_MIN = 60
SITTING_GROUND_HIP_RATIO = 0.80   # hip_y / frame_h threshold

# Waving
WAVE_HAND_ABOVE_SHOULDER = True
WAVE_MOVEMENT_THRESHOLD = 15      # pixels of hand movement between frames
WAVE_HISTORY_FRAMES = 8           # frames to track hand movement

# Using Phone
PHONE_HAND_FACE_DISTANCE = 80     # max pixels between wrist and nose

# Using Laptop
LAPTOP_PERSON_DISTANCE = 200      # max pixels between person center and laptop center

# Talking (head movement)
TALK_MOVEMENT_THRESHOLD = 4       # pixels of nose movement
TALK_HISTORY_FRAMES = 15          # frames to analyze
TALK_MIN_MOVEMENTS = 5            # min direction changes to classify as talking

# ─── UI Rendering ───────────────────────────────────────────────────────────
# Colors (BGR format for OpenCV)
COLOR_STANDING = (0, 200, 0)       # Green
COLOR_SITTING = (200, 150, 0)      # Blue-ish
COLOR_GROUND = (0, 220, 220)       # Yellow
COLOR_DEFAULT = (200, 200, 200)    # Gray
COLOR_OBJECT = (180, 120, 60)      # Teal for non-person objects

COLOR_TOP_BAR = (40, 40, 40)       # Dark gray
COLOR_BOTTOM_BAR = (30, 30, 30)    # Darker gray
COLOR_TEXT_WHITE = (255, 255, 255)
COLOR_TEXT_ACCENT = (0, 220, 255)  # Amber/Gold
COLOR_TEXT_GREEN = (0, 255, 100)
COLOR_TEXT_RED = (80, 80, 255)

# Font
FONT_SCALE_LABEL = 0.55
FONT_SCALE_BAR = 0.55
FONT_THICKNESS = 2
BAR_HEIGHT = 45

# ─── Window ─────────────────────────────────────────────────────────────────
WINDOW_NAME = "Activity Detection - Live Demo"
