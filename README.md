# Activity Detection System

Real-time webcam-based activity detection for lab demos. Detects objects, human posture, and actions using YOLOv8 + MediaPipe, with a clean OpenCV-based live UI.

---

## Features

- **Object Detection** — YOLOv8 (Ultralytics) detecting all COCO classes
- **Pose Estimation** — MediaPipe Pose with 33 body keypoints
- **Action Recognition** — Rule-based classification:
  - Standing, Sitting (Chair), Sitting (Ground)
  - Waving, Using Laptop, Using Phone, Talking
- **Person Tracking** — Centroid-based persistent ID assignment
- **Live UI** — Bounding boxes, color-coded labels, stats bar, FPS counter
- **GPU Acceleration** — Auto-detects CUDA, falls back to CPU

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

2. **Run the application**:

```
run.bat
```

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
│ YOLOv8 | MediaPipe | GPU: ON               FPS: 24  │  ← Bottom Bar
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
| `YOLO_MODEL` | yolov8n.pt | YOLO model size (n/s/m/l/x) |
| `YOLO_CONFIDENCE` | 0.45 | Detection confidence threshold |
| `POSE_MODEL_COMPLEXITY` | 1 | MediaPipe complexity (0/1/2) |

---

## Credits

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Google MediaPipe](https://mediapipe.dev/)
- [OpenCV](https://opencv.org/)
