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

    @staticmethod
    def _bbox_iou(box_a, box_b):
        """Compute IoU between two bounding boxes."""
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter_area

        if union <= 0:
            return 0.0
        return inter_area / union

    def _suppress_person_duplicates(self, detections):
        """Remove highly-overlapping person boxes to avoid duplicate IDs."""
        person_dets = [d for d in detections if d["class_name"] == "person"]
        other_dets = [d for d in detections if d["class_name"] != "person"]

        person_dets.sort(key=lambda d: d["confidence"], reverse=True)
        kept = []

        for det in person_dets:
            keep = True
            for prev in kept:
                if self._bbox_iou(det["bbox"], prev["bbox"]) >= config.PERSON_DUPLICATE_IOU:
                    keep = False
                    break
            if keep:
                kept.append(det)

        return kept + other_dets

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
        frame_h = frame.shape[0]

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
                raw_class_name = self.model.names[cls_id]

                if config.YOLO_ALLOWED_CLASSES and raw_class_name not in config.YOLO_ALLOWED_CLASSES:
                    continue

                class_conf = config.YOLO_CLASS_CONFIDENCE.get(raw_class_name, config.YOLO_CONFIDENCE)
                if conf < class_conf:
                    continue

                if raw_class_name == "person":
                    width = max(0, int(x2) - int(x1))
                    height = max(0, int(y2) - int(y1))
                    area = width * height
                    if height < config.YOLO_MIN_PERSON_HEIGHT:
                        continue
                    if area < config.YOLO_MIN_PERSON_AREA:
                        continue

                # Scene heuristics for lab environments to reduce chair/monitor confusion.
                width = max(0, int(x2) - int(x1))
                height = max(1, int(y2) - int(y1))
                aspect_ratio = width / float(height)
                bottom_ratio = int(y2) / float(max(1, frame_h))

                if raw_class_name == "chair":
                    if aspect_ratio > config.CHAIR_MAX_ASPECT_RATIO and bottom_ratio < config.CHAIR_MIN_BOTTOM_RATIO:
                        continue

                if raw_class_name == "tv":
                    if aspect_ratio < config.MONITOR_MIN_ASPECT_RATIO:
                        continue

                cls_name = config.YOLO_CLASS_ALIASES.get(raw_class_name, raw_class_name)

                detections.append({
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "confidence": conf,
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "raw_class_name": raw_class_name,
                })

        return self._suppress_person_duplicates(detections)

    def is_gpu(self):
        """Return True if running on GPU."""
        return self.gpu_active
