"""
Utility helpers for geometry, angle calculations, and frame processing.
"""

import math
import numpy as np


def calculate_angle(a, b, c):
    """
    Calculate the angle (in degrees) at point B formed by points A-B-C.
    Each point is (x, y).
    Returns angle in degrees [0, 360).
    """
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)

    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cosine = np.clip(cosine, -1.0, 1.0)
    angle = math.degrees(math.acos(cosine))
    return angle


def distance(p1, p2):
    """Euclidean distance between two (x,y) points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def midpoint(p1, p2):
    """Midpoint between two (x,y) points."""
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)


def clamp(value, lo, hi):
    """Clamp value between lo and hi."""
    return max(lo, min(hi, value))


def bbox_center(bbox):
    """Return center (cx, cy) of a bounding box (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def bbox_area(bbox):
    """Return area of a bounding box (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def smooth_value(history, new_value, max_len=5):
    """
    Append new_value to history list, keep max_len items,
    and return the most common value (mode) for label stability.
    """
    history.append(new_value)
    if len(history) > max_len:
        history.pop(0)
    # Return mode (most frequent)
    from collections import Counter
    counts = Counter(history)
    return counts.most_common(1)[0][0]
