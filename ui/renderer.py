"""
OpenCV-based UI renderer for the Activity Detection system.
Draws bounding boxes, labels, overlay bars, and stats — all using cv2 only.
"""

import cv2
import numpy as np
import config


class Renderer:
    """Renders all visual elements on the frame using OpenCV."""

    def __init__(self, gpu_active=False):
        self.gpu_active = gpu_active
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_alt = cv2.FONT_HERSHEY_DUPLEX
        print("[Renderer] UI renderer initialized.")

    def draw_person_box(self, frame, bbox, person_id, posture, actions, color):
        """
        Draw bounding box and label for a detected person.

        Args:
            frame: BGR image (modified in-place).
            bbox: (x1, y1, x2, y2).
            person_id: int tracker ID.
            posture: str like "Standing".
            actions: list of additional action strings.
            color: BGR tuple.
        """
        x1, y1, x2, y2 = bbox

        # Draw rounded-corner-style bounding box with corner accents
        thickness = 2
        corner_len = 20

        # Full semi-transparent rectangle
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        # Corner accents (thicker, colored)
        # Top-left
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, thickness + 1)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, thickness + 1)
        # Top-right
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, thickness + 1)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, thickness + 1)
        # Bottom-left
        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, thickness + 1)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, thickness + 1)
        # Bottom-right
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, thickness + 1)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, thickness + 1)

        # Build label text
        label_parts = [f"Person {person_id}", posture]
        if actions:
            label_parts.extend(actions)
        label = " | ".join(label_parts)

        # Draw label background
        font_scale = config.FONT_SCALE_LABEL
        thickness_text = config.FONT_THICKNESS
        (tw, th), baseline = cv2.getTextSize(label, self.font, font_scale, thickness_text)

        label_y = y1 - 10
        if label_y - th - 8 < 0:
            label_y = y2 + th + 14

        # Background pill
        bg_x1 = x1
        bg_y1 = label_y - th - 8
        bg_x2 = x1 + tw + 16
        bg_y2 = label_y + 4

        # Draw background with slight transparency
        overlay = frame.copy()
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Draw text
        cv2.putText(
            frame, label, (x1 + 8, label_y - 2),
            self.font, font_scale, (255, 255, 255), thickness_text, cv2.LINE_AA
        )

    def draw_object_box(self, frame, bbox, class_name, confidence):
        """Draw a simple bounding box for non-person objects."""
        x1, y1, x2, y2 = bbox
        color = config.COLOR_OBJECT

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

        label = f"{class_name} {confidence:.0%}"
        font_scale = 0.45
        (tw, th), _ = cv2.getTextSize(label, self.font, font_scale, 1)

        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 8, y1), color, -1)
        cv2.putText(
            frame, label, (x1 + 4, y1 - 4),
            self.font, font_scale, (255, 255, 255), 1, cv2.LINE_AA
        )

    def draw_top_bar(self, frame, stats):
        """
        Draw the top summary bar.

        Args:
            frame: BGR image.
            stats: dict with keys 'humans', 'standing', 'sitting', 'ground'.
        """
        h, w = frame.shape[:2]
        bar_h = config.BAR_HEIGHT

        # Semi-transparent dark bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), config.COLOR_TOP_BAR, -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Accent line at bottom of bar
        cv2.line(frame, (0, bar_h), (w, bar_h), config.COLOR_TEXT_ACCENT, 1)

        # Title on the left
        title = "ACTIVITY DETECTION"
        cv2.putText(
            frame, title, (15, bar_h - 14),
            self.font_alt, 0.6, config.COLOR_TEXT_ACCENT, 1, cv2.LINE_AA
        )

        # Stats on the right
        humans = stats.get("humans", 0)
        standing = stats.get("standing", 0)
        sitting = stats.get("sitting", 0)
        ground = stats.get("ground", 0)

        stat_parts = [
            (f"Humans: {humans}", config.COLOR_TEXT_WHITE),
            (f"Standing: {standing}", config.COLOR_STANDING),
            (f"Sitting: {sitting}", config.COLOR_SITTING),
            (f"Ground: {ground}", config.COLOR_GROUND),
        ]

        # Calculate total width for right-alignment
        x_offset = w - 15
        for text, color in reversed(stat_parts):
            (tw, th), _ = cv2.getTextSize(text, self.font, config.FONT_SCALE_BAR, 1)
            x_offset -= tw
            cv2.putText(
                frame, text, (x_offset, bar_h - 14),
                self.font, config.FONT_SCALE_BAR, color, 1, cv2.LINE_AA
            )
            # Separator dot
            x_offset -= 25
            if x_offset > 300:
                cv2.circle(frame, (x_offset + 10, bar_h - 18), 2, (120, 120, 120), -1)

    def draw_bottom_bar(self, frame, fps):
        """
        Draw the bottom credits bar.
        Shows: YOLOv8 | MediaPipe | GPU status | FPS
        """
        h, w = frame.shape[:2]
        bar_h = config.BAR_HEIGHT
        y_top = h - bar_h

        # Semi-transparent dark bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, y_top), (w, h), config.COLOR_BOTTOM_BAR, -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Accent line at top of bar
        cv2.line(frame, (0, y_top), (w, y_top), (60, 60, 60), 1)

        text_y = h - 14

        # Left side: model credits
        credits = [
            ("YOLOv8", config.COLOR_TEXT_ACCENT),
            ("MediaPipe", config.COLOR_TEXT_WHITE),
        ]

        x = 15
        for text, color in credits:
            cv2.putText(
                frame, text, (x, text_y),
                self.font, config.FONT_SCALE_BAR, color, 1, cv2.LINE_AA
            )
            (tw, _), _ = cv2.getTextSize(text, self.font, config.FONT_SCALE_BAR, 1)
            x += tw + 10
            # Separator
            cv2.putText(
                frame, "|", (x, text_y),
                self.font, config.FONT_SCALE_BAR, (80, 80, 80), 1, cv2.LINE_AA
            )
            x += 18

        # GPU status
        if self.gpu_active:
            gpu_text = "GPU: ON"
            gpu_color = config.COLOR_TEXT_GREEN
        else:
            gpu_text = "GPU: OFF (CPU)"
            gpu_color = config.COLOR_TEXT_RED

        cv2.putText(
            frame, gpu_text, (x, text_y),
            self.font, config.FONT_SCALE_BAR, gpu_color, 1, cv2.LINE_AA
        )

        # Right side: FPS
        fps_text = f"FPS: {fps:.0f}"
        fps_color = config.COLOR_TEXT_GREEN if fps >= 15 else config.COLOR_TEXT_ACCENT
        (tw, _), _ = cv2.getTextSize(fps_text, self.font, config.FONT_SCALE_BAR, 1)
        cv2.putText(
            frame, fps_text, (w - tw - 15, text_y),
            self.font, config.FONT_SCALE_BAR, fps_color, 1, cv2.LINE_AA
        )

    def draw_no_detection_message(self, frame):
        """Draw a message when no persons are detected."""
        h, w = frame.shape[:2]
        text = "No persons detected"
        font_scale = 0.7
        (tw, th), _ = cv2.getTextSize(text, self.font_alt, font_scale, 1)
        x = (w - tw) // 2
        y = (h + th) // 2

        # Background pill
        pad = 15
        overlay = frame.copy()
        cv2.rectangle(
            overlay, (x - pad, y - th - pad), (x + tw + pad, y + pad),
            (40, 40, 40), -1
        )
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        cv2.putText(
            frame, text, (x, y),
            self.font_alt, font_scale, (180, 180, 180), 1, cv2.LINE_AA
        )
