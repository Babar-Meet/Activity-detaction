# Activity Detection System

Real-time webcam-based activity detection for lab demos. Detects objects, human posture, and actions using YOLOv8 + MediaPipe, with a clean OpenCV-based live UI.

---

## Features

- **Object Detection (Lab Focused)** — YOLOv8 with class whitelist and per-class confidence gates for cleaner AI lab output
- **Pose Estimation** — MediaPipe Pose with 33 body keypoints
- **Dense Face Features** — MediaPipe Face Mesh with eyes, pupils/iris approximation, lips, brows, and face contour overlays
- **Action Recognition** — Rule-based classification:
  - Standing, Sitting (Chair), Sitting (Ground)
  - Walking, Saying Hello, V Sign, Using Laptop, Using Phone, Talking
- **Stabilized Person Tracking** — confirmation-based tracking to reduce one-person-to-many-ID spikes
- **Temporal Delay/Hysteresis** — 1.0s posture/action transition delay to reduce flicker and wrong instant labels
- **Enhanced Skeleton View** — blue/green human skeleton with hand-foot emphasis and confidence text for debugging/showcase
- **Live UI** — Bounding boxes, color-coded labels, stats bar, FPS counter
- **GPU Acceleration** — Auto-detects CUDA, falls back to CPU
  - If CUDA GPU is found, startup prompts whether to run on GPU or CPU
  - If CUDA is unavailable, app runs directly on CPU without prompting

---

## Requirements

- **Python 3.10+**
- **Windows 10/11**
- **Webcam**
- **NVIDIA GPU with CUDA** (recommended, not required)

---

## Setup

1. **Run the setup script** (creates venv and installs all dependencies):

```
setup.bat
```

Setup now also validates required model files:
- yolov8n.pt
- models/pose_landmarker_full.task
- models/hand_landmarker.task

2. **Run the application**:

```
run.bat
```

Run script behavior is now strict: it fails fast with clear messages if setup or required files are missing.

3. **Build standalone executable**:

```
build_exe.bat
```

Build script now validates setup/model files before packaging.

That's it. The webcam window will open and detection starts immediately.

---

## Controls

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `ESC` | Quit |

---

## Project Structure

```
Activity detaction/
├── main.py                  # Application entry point
├── config.py                # All configuration constants
├── requirements.txt         # Python dependencies
├── setup.bat                # One-time setup script
├── run.bat                  # Run the application
├── detector/
│   ├── object_detector.py   # YOLOv8 wrapper
│   ├── pose_detector.py     # MediaPipe Pose wrapper
│   └── action_classifier.py # Rule-based action logic
├── tracker/
│   └── centroid_tracker.py  # Persistent person tracking
├── ui/
│   └── renderer.py          # OpenCV UI rendering
└── utils/
    └── helpers.py           # Geometry & utility functions
```

---

## UI Layout

```
┌──────────────────────────────────────────────────────┐
│ ACTIVITY DETECTION    Humans: 2 · Standing: 1 · ... │  ← Top Bar
├──────────────────────────────────────────────────────┤
│                                                      │
│   ┌─Person 1 | Standing──┐                          │
│   │                       │                          │
│   │      (webcam feed)    │                          │
│   └───────────────────────┘                          │
│                                                      │
├──────────────────────────────────────────────────────┤
│ YOLOv8 | MediaPipe | Mode: GPU (CUDA)      FPS: 24  │  ← Bottom Bar
└──────────────────────────────────────────────────────┘
```

---

## Color Coding

| Posture | Color |
|---------|-------|
| Standing | 🟢 Green |
| Sitting (Chair) | 🔵 Blue |
| Sitting (Ground) | 🟡 Yellow |

---

## Configuration

All parameters are in `config.py`. Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CAMERA_INDEX` | 0 | Webcam device index |
| `CAMERA_ASK_ON_STARTUP` | True | Ask which detected camera index to use at launch |
| `CAMERA_SCAN_MAX_INDEX` | 6 | Maximum index scanned when listing available cameras |
| `YOLO_MODEL` | yolov8n.pt | YOLO model size (n/s/m/l/x) |
| `YOLO_CONFIDENCE` | 0.35 | Base detection confidence threshold |
| `YOLO_ALLOWED_CLASSES` | lab set | Restricts labels to lab-relevant objects |
| `YOLO_CLASS_CONFIDENCE` | per-class dict | Higher thresholds for confusing classes (chair, monitor, etc.) |
| `TRACKER_CONFIRM_FRAMES` | 5 | Frames required before counting a person |
| `POSTURE_SWITCH_DELAY_SEC` | 1.0 | Delay before posture label changes |
| `ACTION_SWITCH_DELAY_SEC` | 1.0 | Delay before action label changes |
| `ASK_GPU_ON_STARTUP` | True | Ask user to choose GPU/CPU when CUDA GPU exists |
| `GPU_DEFAULT_USE_CUDA` | True | Default startup choice when GPU prompt is shown |
| `SKELETON_SHOW_CONFIDENCE_TEXT` | True | Show landmark confidence text in debug view |
| `POSE_MODEL_COMPLEXITY` | 1 | MediaPipe complexity (0/1/2) |

---

## Credits

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Google MediaPipe](https://mediapipe.dev/)
- [OpenCV](https://opencv.org/)
