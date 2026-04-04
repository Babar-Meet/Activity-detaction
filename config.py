"""Configuration constants for the Activity Detection system."""

# Camera
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# YOLO object detection
YOLO_MODEL = "yolov8n.pt"  # nano model for real-time speed
YOLO_CONFIDENCE = 0.35
YOLO_IOU_THRESHOLD = 0.55

# AI lab showcase filtering
YOLO_ALLOWED_CLASSES = {
	"person",
	"laptop",
	"tv",
	"keyboard",
	"mouse",
	"chair",
	"cell phone",
}

YOLO_CLASS_CONFIDENCE = {
	"person": 0.40,
	"laptop": 0.55,
	"tv": 0.60,
	"keyboard": 0.50,
	"mouse": 0.55,
	"chair": 0.68,
	"cell phone": 0.60,
}

YOLO_CLASS_ALIASES = {
	"tv": "monitor",
	"cell phone": "phone",
}

# Class-specific geometric heuristics for lab scenes
CHAIR_MAX_ASPECT_RATIO = 1.20   # chairs are usually taller than wide
CHAIR_MIN_BOTTOM_RATIO = 0.45   # chair bottom should be near lower half of frame
MONITOR_MIN_ASPECT_RATIO = 1.10  # monitor/tv is usually wider than tall

# Filter tiny person detections that often become ghost IDs
YOLO_MIN_PERSON_HEIGHT = 100
YOLO_MIN_PERSON_AREA = 10000
PERSON_DUPLICATE_IOU = 0.65

# MediaPipe pose
POSE_MIN_DETECTION_CONF = 0.55
POSE_MIN_TRACKING_CONF = 0.55
POSE_MODEL_COMPLEXITY = 1  # 0=lite, 1=full, 2=heavy
LANDMARK_VISIBILITY_MIN = 0.45

# Centroid tracker
TRACKER_MAX_DISAPPEARED = 40  # frames before dropping an ID
TRACKER_MAX_DISTANCE = 120  # max pixel distance to associate
TRACKER_MIN_IOU_MATCH = 0.06
TRACKER_CONFIRM_FRAMES = 5  # must be seen this many frames before counted
TRACKER_SMOOTHING_ALPHA = 0.45

# Action detection thresholds
# Angles (degrees)
STANDING_KNEE_ANGLE_MIN = 148  # lowered from strict 155 to reduce false sitting
SITTING_CHAIR_KNEE_ANGLE_MAX = 125
SITTING_CHAIR_KNEE_ANGLE_MIN = 65
SITTING_GROUND_HIP_RATIO = 0.82  # hip_y / frame_h threshold
STANDING_HIP_ABOVE_KNEE_MARGIN = 6
SITTING_GROUND_HIP_ANKLE_DIFF_MAX = 55

# Temporal stabilization
POSTURE_SWITCH_DELAY_SEC = 1.0
ACTION_SWITCH_DELAY_SEC = 1.0
POSTURE_HISTORY_FRAMES = 12

# Waving
WAVE_HAND_ABOVE_SHOULDER = True
WAVE_MOVEMENT_THRESHOLD = 14  # pixels of hand movement between frames
WAVE_HISTORY_FRAMES = 10

# Using phone
PHONE_HAND_FACE_DISTANCE = 80

# Using laptop
LAPTOP_PERSON_DISTANCE = 185

# Talking (head movement)
TALK_MOVEMENT_THRESHOLD = 4
TALK_HISTORY_FRAMES = 15
TALK_MIN_MOVEMENTS = 5

# Skeleton and debug visualization
SKELETON_ENABLED = True
SKELETON_ALPHA = 0.55
SKELETON_LINE_THICKNESS_MIN = 1
SKELETON_LINE_THICKNESS_MAX = 4
SKELETON_POINT_RADIUS_MIN = 2
SKELETON_POINT_RADIUS_MAX = 6
SKELETON_SHOW_CONFIDENCE_TEXT = True
SKELETON_CONF_TEXT_MIN_VIS = 0.55

SKELETON_COLOR_UPPER = (50, 220, 90)  # green
SKELETON_COLOR_LOWER = (255, 120, 40)  # blue/orange contrast in BGR space
SKELETON_COLOR_FACE = (220, 220, 0)
SKELETON_COLOR_HAND = (255, 70, 180)
SKELETON_COLOR_FOOT = (255, 200, 0)

# UI rendering
# Colors (BGR format for OpenCV)
COLOR_STANDING = (0, 200, 0)
COLOR_SITTING = (200, 150, 0)
COLOR_GROUND = (0, 220, 220)
COLOR_DEFAULT = (200, 200, 200)
COLOR_OBJECT = (180, 120, 60)
COLOR_UNCONFIRMED_TRACK = (120, 120, 120)

COLOR_TOP_BAR = (40, 40, 40)
COLOR_BOTTOM_BAR = (30, 30, 30)
COLOR_TEXT_WHITE = (255, 255, 255)
COLOR_TEXT_ACCENT = (0, 220, 255)
COLOR_TEXT_GREEN = (0, 255, 100)
COLOR_TEXT_RED = (80, 80, 255)

# Debug toggles for AI lab demo
DEBUG_SHOW_POSTURE_DETAILS = True
DEBUG_SHOW_TRACK_STATUS = True

# Font
FONT_SCALE_LABEL = 0.55
FONT_SCALE_BAR = 0.55
FONT_THICKNESS = 2
BAR_HEIGHT = 45

# Window
WINDOW_NAME = "Activity Detection - Live Demo"
