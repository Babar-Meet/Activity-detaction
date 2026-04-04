"""
Activity Detection — Main Application
======================================
Real-time webcam-based activity detection system combining:
  - YOLOv8 object detection (Ultralytics)
  - MediaPipe Pose estimation
  - Rule-based action classification
  - Centroid-based person tracking
  - OpenCV-only UI rendering

Pipeline:
  Webcam → Frame Capture → YOLO Detection → Pose Detection →
  Action Logic → Tracking → UI Rendering → Display

Author: Activity Detection Lab Demo
"""

import sys
import time
import cv2
import numpy as np

import config
from detector.object_detector import ObjectDetector
from detector.pose_detector import PoseDetector
from detector.action_classifier import ActionClassifier
from tracker.centroid_tracker import CentroidTracker
from ui.renderer import Renderer


def main():
    """Main application entry point."""
    print("=" * 60)
    print("  ACTIVITY DETECTION SYSTEM")
    print("  Real-time Lab Demo")
    print("=" * 60)
    print()

    # ─── Initialize components ───────────────────────────────────────
    print("[Main] Initializing components...")
    object_detector = ObjectDetector()
    pose_detector = PoseDetector()
    action_classifier = ActionClassifier()
    person_tracker = CentroidTracker()
    renderer = Renderer(gpu_active=object_detector.is_gpu())

    # ─── Open webcam ─────────────────────────────────────────────────
    print(f"[Main] Opening camera (index {config.CAMERA_INDEX})...")
    cap = cv2.VideoCapture(config.CAMERA_INDEX)

    if not cap.isOpened():
        print("[ERROR] Could not open webcam. Check camera connection.")
        sys.exit(1)

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[Main] Camera opened: {actual_w}x{actual_h}")

    # Set up display window
    cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(config.WINDOW_NAME, actual_w, actual_h)

    print()
    print("[Main] System ready. Press 'Q' to quit.")
    print("-" * 60)

    # ─── FPS tracking ────────────────────────────────────────────────
    fps = 0.0
    frame_times = []
    fps_update_interval = 0.5  # Update FPS display every 0.5 seconds
    last_fps_update = time.time()

    # ─── Track previous person IDs for cleanup ───────────────────────
    prev_person_ids = set()

    # ─── Main loop ───────────────────────────────────────────────────
    try:
        while True:
            frame_start = time.time()

            ret, frame = cap.read()
            if not ret:
                print("[WARNING] Failed to read frame. Retrying...")
                time.sleep(0.1)
                continue

            # Ensure frame is writable
            frame = frame.copy()
            frame_h, frame_w = frame.shape[:2]

            # ── Step 1: YOLO Object Detection ────────────────────────
            all_detections = object_detector.detect(frame)

            # Separate person detections from other objects
            person_detections = [d for d in all_detections if d["class_name"] == "person"]
            other_detections = [d for d in all_detections if d["class_name"] != "person"]

            # ── Step 2: Update person tracker ────────────────────────
            person_bboxes = [d["bbox"] for d in person_detections]
            tracked_persons = person_tracker.update(person_bboxes)
            confirmed_tracked_persons = person_tracker.get_confirmed_tracks()

            # Cleanup action classifier for disappeared persons
            current_ids = set(tracked_persons.keys())
            disappeared_ids = prev_person_ids - current_ids
            for pid in disappeared_ids:
                action_classifier.cleanup_person(pid)
            prev_person_ids = current_ids

            # ── Step 3: Pose + Action for each tracked person ────────
            person_results = {}  # pid -> { posture, actions, color, bbox }

            for person_id, bbox in confirmed_tracked_persons.items():
                # Run pose detection
                landmarks = pose_detector.detect(frame, bbox)
                track_status = person_tracker.get_track_status(person_id)

                if landmarks:
                    # Classify action
                    result = action_classifier.classify(
                        person_id, landmarks, frame.shape, all_detections
                    )
                    person_results[person_id] = {
                        **result,
                        "bbox": bbox,
                        "landmarks": landmarks,
                        "track_status": track_status,
                    }
                else:
                    # No pose detected — show as Unknown
                    person_results[person_id] = {
                        "posture": "Detected",
                        "raw_posture": "Unknown",
                        "posture_confidence": 0.0,
                        "actions": [],
                        "raw_actions": [],
                        "color": config.COLOR_DEFAULT,
                        "bbox": bbox,
                        "track_status": track_status,
                    }

            # ── Step 4: Compute stats ────────────────────────────────
            stats = {
                "humans": len(confirmed_tracked_persons),
                "standing": sum(1 for r in person_results.values() if r["posture"] == "Standing"),
                "sitting": sum(1 for r in person_results.values() if r["posture"] == "Sitting (Chair)"),
                "ground": sum(1 for r in person_results.values() if r["posture"] == "Sitting (Ground)"),
            }

            # ── Step 5: Render UI ────────────────────────────────────

            # Draw non-person object boxes (behind person boxes)
            for det in other_detections:
                renderer.draw_object_box(
                    frame, det["bbox"], det["class_name"], det["confidence"]
                )

            # Draw person bounding boxes and labels
            for person_id, result in person_results.items():
                renderer.draw_person_box(
                    frame,
                    result["bbox"],
                    person_id,
                    result["posture"],
                    result["actions"],
                    result["color"],
                )

                if config.DEBUG_SHOW_POSTURE_DETAILS or config.DEBUG_SHOW_TRACK_STATUS:
                    renderer.draw_person_debug(frame, result["bbox"], person_id, result)

            # Draw enhanced skeletons for confirmed persons
            if config.SKELETON_ENABLED:
                for result in person_results.values():
                    if "landmarks" in result:
                        pose_detector.draw_skeleton(
                            frame,
                            result["landmarks"],
                            alpha=config.SKELETON_ALPHA,
                            show_confidence=config.SKELETON_SHOW_CONFIDENCE_TEXT,
                        )

            # Show message if no persons detected
            if len(confirmed_tracked_persons) == 0:
                renderer.draw_no_detection_message(frame)

            # Draw overlay bars (on top of everything)
            renderer.draw_top_bar(frame, stats)
            renderer.draw_bottom_bar(frame, fps)

            # ── Step 6: Calculate FPS ────────────────────────────────
            frame_end = time.time()
            frame_times.append(frame_end - frame_start)

            if frame_end - last_fps_update >= fps_update_interval:
                if frame_times:
                    avg_time = sum(frame_times) / len(frame_times)
                    fps = 1.0 / avg_time if avg_time > 0 else 0
                frame_times.clear()
                last_fps_update = frame_end

            # ── Step 7: Display ──────────────────────────────────────
            cv2.imshow(config.WINDOW_NAME, frame)

            # Handle key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == ord("Q") or key == 27:  # Q or ESC
                print("\n[Main] Exiting...")
                break

    except KeyboardInterrupt:
        print("\n[Main] Interrupted by user.")

    finally:
        # ─── Cleanup ─────────────────────────────────────────────────
        print("[Main] Releasing resources...")
        cap.release()
        pose_detector.release()
        cv2.destroyAllWindows()
        print("[Main] Done.")


if __name__ == "__main__":
    main()
