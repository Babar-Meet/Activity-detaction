"""
YOLO-based object detector using Ultralytics.
Supports GPU (CUDA) with automatic CPU fallback.
"""

import torch
from ultralytics import YOLO
import config


class ObjectDetector:
    """Wraps Ultralytics YOLO for real-time object detection."""

    def __init__(self):
        # Determine device
        if torch.cuda.is_available():
            self.device = "cuda"
            self.gpu_active = True
            print(f"[ObjectDetector] Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = "cpu"
            self.gpu_active = False
            print("[ObjectDetector] GPU not available — using CPU")

        # Load YOLO model (auto-downloads if not present)
        print(f"[ObjectDetector] Loading model: {config.YOLO_MODEL}")
        self.model = YOLO(config.YOLO_MODEL)

        # Pre-warm the model
        print("[ObjectDetector] Model loaded successfully.")

    def detect(self, frame):
        """
        Run detection on a frame.

        Args:
            frame: BGR numpy array from OpenCV.

        Returns:
            List of dicts: {
                'bbox': (x1, y1, x2, y2),
                'confidence': float,
                'class_id': int,
                'class_name': str
            }
        """
        results = self.model.predict(
            source=frame,
            conf=config.YOLO_CONFIDENCE,
            iou=config.YOLO_IOU_THRESHOLD,
            device=self.device,
            verbose=False,
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = self.model.names[cls_id]

                detections.append({
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "confidence": conf,
                    "class_id": cls_id,
                    "class_name": cls_name,
                })

        return detections

    def is_gpu(self):
        """Return True if running on GPU."""
        return self.gpu_active
