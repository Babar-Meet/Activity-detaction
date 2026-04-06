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

    @staticmethod
    def _format_posture(posture):
        if posture == "Sitting (Chair)":
            return "Sitting on Chair"
        if posture == "Sitting (Ground)":
            return "Sitting on Ground"
        return posture

    @staticmethod
    def _select_gesture(actions):
        action_set = set(actions or [])
        if "V Sign" in action_set:
            return "V Shape"
        if "Saying Hello" in action_set:
            return "Waving Hi"
        if "Walking" in action_set:
            return "Walking"
        return "Neutral"

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

        posture_text = self._format_posture(posture)
        gesture_text = self._select_gesture(actions)

        lines = [
            f"ID: {person_id}",
            f"Gesture: {gesture_text}",
            f"Posture: {posture_text}",
        ]

        font_scale = config.FONT_SCALE_LABEL
        thickness_text = config.FONT_THICKNESS
        line_height = int(26 * max(0.75, font_scale))
        pad_x = 12
        pad_y = 10

        text_width = 0
        for line in lines:
            (tw, _), _ = cv2.getTextSize(line, self.font, font_scale, thickness_text)
            text_width = max(text_width, tw)

        card_w = text_width + (pad_x * 2)
        card_h = (line_height * len(lines)) + (pad_y * 2) - 6

        h, w = frame.shape[:2]
        card_x1 = max(4, min(x1, w - card_w - 4))
        card_y1 = y1 - card_h - 8
        if card_y1 < 4:
            card_y1 = min(h - card_h - 4, y2 + 8)

        card_x2 = min(w - 4, card_x1 + card_w)
        card_y2 = min(h - 4, card_y1 + card_h)

        overlay = frame.copy()
        cv2.rectangle(overlay, (card_x1, card_y1), (card_x2, card_y2), color, -1)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
        cv2.rectangle(frame, (card_x1, card_y1), (card_x2, card_y2), (255, 255, 255), 1)

        text_y = card_y1 + pad_y + line_height - 8
        for line in lines:
            cv2.putText(
                frame,
                line,
                (card_x1 + pad_x, text_y),
                self.font,
                font_scale,
                (255, 255, 255),
                thickness_text,
                cv2.LINE_AA,
            )
            text_y += line_height

    def draw_person_debug(self, frame, bbox, person_id, result):
        """Draw low-profile debug details for posture confidence and tracking status."""
        x1, y1, x2, y2 = bbox
        lines = []

        if config.DEBUG_SHOW_POSTURE_DETAILS:
            raw_posture = result.get("raw_posture", "-")
            stable_posture = result.get("posture", "-")
            confidence = result.get("posture_confidence", 0.0)
            lines.append(f"Raw:{raw_posture}  Stable:{stable_posture}  C:{confidence:.2f}")

            raw_actions = ",".join(result.get("raw_actions", [])) or "-"
            stable_actions = ",".join(result.get("actions", [])) or "-"
            lines.append(f"Araw:{raw_actions}  Astable:{stable_actions}")

        if config.DEBUG_SHOW_TRACK_STATUS:
            status = result.get("track_status", {})
            if status:
                lines.append(
                    "ID:{0} age:{1} hits:{2} conf:{3} miss:{4}".format(
                        person_id,
                        status.get("age", 0),
                        status.get("hits", 0),
                        "Y" if status.get("confirmed", False) else "N",
                        status.get("disappeared", 0),
                    )
                )

        if not lines:
            return

        font_scale = 0.42
        line_height = 16
        pad = 6
        width = 0
        for line in lines:
            (tw, _), _ = cv2.getTextSize(line, self.font, font_scale, 1)
            width = max(width, tw)

        h, w = frame.shape[:2]
        block_h = (line_height * len(lines)) + (pad * 2)
        debug_x1 = x1
        debug_x2 = min(w - 4, debug_x1 + width + (pad * 2))
        debug_y1 = y2 + 4
        debug_y2 = min(h - 4, debug_y1 + block_h)

        # If there is not enough room below, place debug block above the bbox.
        if debug_y2 - debug_y1 < block_h:
            debug_y2 = max(4, y1 - 4)
            debug_y1 = max(0, debug_y2 - block_h)

        if debug_x2 <= debug_x1 or debug_y2 <= debug_y1:
            return

        overlay = frame.copy()
        cv2.rectangle(overlay, (debug_x1, debug_y1), (debug_x2, debug_y2), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

        text_y = debug_y1 + 14
        for line in lines:
            cv2.putText(
                frame,
                line,
                (debug_x1 + pad, text_y),
                self.font,
                font_scale,
                (210, 210, 210),
                1,
                cv2.LINE_AA,
            )
            text_y += line_height

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
            stats: dict with key 'objects'.
        """
        h, w = frame.shape[:2]
        bar_h = config.BAR_HEIGHT

        # Semi-transparent dark bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), config.COLOR_TOP_BAR, -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Accent line at bottom of bar
        cv2.line(frame, (0, bar_h), (w, bar_h), config.COLOR_TEXT_ACCENT, 1)

        object_count = stats.get("objects", 0)
        text = f"Objects Detected: {object_count}"
        text_scale = max(0.65, config.FONT_SCALE_BAR + 0.10)
        (tw, _), _ = cv2.getTextSize(text, self.font_alt, text_scale, 1)
        tx = max(15, (w - tw) // 2)

        cv2.putText(
            frame,
            text,
            (tx, bar_h - 14),
            self.font_alt,
            text_scale,
            config.COLOR_TEXT_WHITE,
            1,
            cv2.LINE_AA,
        )

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
            gpu_text = "Mode: GPU (CUDA)"
            gpu_color = config.COLOR_TEXT_GREEN
        else:
            gpu_text = "Mode: CPU"
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
