"""
Lightweight centroid-based object tracker.
Assigns persistent IDs to detected persons across frames.
"""

from collections import OrderedDict
import numpy as np
from scipy.spatial.distance import cdist
import config


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


class CentroidTracker:
    """
    Tracks objects across frames by matching centroids.
    Assigns unique IDs that persist across frames.
    """

    def __init__(self):
        self.next_id = 1
        self.objects = OrderedDict()      # id -> centroid (cx, cy)
        self.bboxes = OrderedDict()       # id -> bbox (x1,y1,x2,y2)
        self.disappeared = OrderedDict()  # id -> count of frames disappeared
        self.hits = OrderedDict()         # id -> matched frames count
        self.ages = OrderedDict()         # id -> total age in frames
        self.confirmed = OrderedDict()    # id -> bool (stable enough to count)

        self.max_disappeared = config.TRACKER_MAX_DISAPPEARED
        self.max_distance = config.TRACKER_MAX_DISTANCE
        self.min_iou_match = config.TRACKER_MIN_IOU_MATCH
        self.confirm_frames = config.TRACKER_CONFIRM_FRAMES
        self.smoothing_alpha = config.TRACKER_SMOOTHING_ALPHA

        print("[CentroidTracker] Initialized.")

    def _smooth_bbox(self, previous_bbox, new_bbox):
        """Apply light EMA smoothing to reduce jitter in bbox motion."""
        if previous_bbox is None:
            return tuple(int(v) for v in new_bbox)

        a = self.smoothing_alpha
        return (
            int((1.0 - a) * previous_bbox[0] + a * new_bbox[0]),
            int((1.0 - a) * previous_bbox[1] + a * new_bbox[1]),
            int((1.0 - a) * previous_bbox[2] + a * new_bbox[2]),
            int((1.0 - a) * previous_bbox[3] + a * new_bbox[3]),
        )

    def update(self, detections):
        """
        Update tracker with new detections.

        Args:
            detections: list of (x1, y1, x2, y2) bounding boxes.

        Returns:
            OrderedDict of { person_id: bbox } for currently tracked persons.
        """
        # If no detections, mark all existing objects as disappeared
        if len(detections) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                self.ages[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self._deregister(obj_id)
            return self._get_tracked(active_only=True)

        # Compute centroids for new detections
        input_centroids = []
        for (x1, y1, x2, y2) in detections:
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            input_centroids.append((cx, cy))
        input_centroids = np.array(input_centroids)

        # If no existing objects, register all detections
        if len(self.objects) == 0:
            for i, bbox in enumerate(detections):
                self._register(input_centroids[i], bbox)
            return self._get_tracked(active_only=True)

        # Match existing objects to new detections
        object_ids = list(self.objects.keys())
        object_centroids = np.array(list(self.objects.values()))

        # Compute pairwise distances and IoU scores
        D = cdist(object_centroids, input_centroids)
        I = np.zeros((len(object_ids), len(detections)), dtype=float)
        for row, obj_id in enumerate(object_ids):
            old_bbox = self.bboxes[obj_id]
            for col, det_bbox in enumerate(detections):
                I[row, col] = _bbox_iou(old_bbox, det_bbox)

        # Build candidate pairs using distance and IoU gating.
        # A pair is allowed if distance is close OR overlap is reasonable.
        pairs = []
        distance_norm = max(float(self.max_distance), 1.0)
        for row in range(D.shape[0]):
            for col in range(D.shape[1]):
                dist_ok = D[row, col] <= self.max_distance
                iou_ok = I[row, col] >= self.min_iou_match
                if not (dist_ok or iou_ok):
                    continue
                score = (D[row, col] / distance_norm) - (0.35 * I[row, col])
                pairs.append((score, row, col))

        pairs.sort(key=lambda item: item[0])

        used_rows = set()
        used_cols = set()

        for _, row, col in pairs:
            if row in used_rows or col in used_cols:
                continue

            obj_id = object_ids[row]

            prev_cx, prev_cy = self.objects[obj_id]
            new_cx, new_cy = input_centroids[col]
            a = self.smoothing_alpha
            smoothed_centroid = (
                (1.0 - a) * prev_cx + a * new_cx,
                (1.0 - a) * prev_cy + a * new_cy,
            )

            self.objects[obj_id] = smoothed_centroid
            self.bboxes[obj_id] = self._smooth_bbox(self.bboxes.get(obj_id), detections[col])
            self.disappeared[obj_id] = 0
            self.hits[obj_id] += 1
            self.ages[obj_id] += 1
            if self.hits[obj_id] >= self.confirm_frames:
                self.confirmed[obj_id] = True

            used_rows.add(row)
            used_cols.add(col)

        # Handle unmatched existing objects (disappeared)
        unused_rows = set(range(len(object_ids))) - used_rows
        for row in unused_rows:
            obj_id = object_ids[row]
            self.disappeared[obj_id] += 1
            self.ages[obj_id] += 1
            if self.disappeared[obj_id] > self.max_disappeared:
                self._deregister(obj_id)

        # Handle unmatched new detections (register as new)
        unused_cols = set(range(len(input_centroids))) - used_cols
        for col in unused_cols:
            self._register(input_centroids[col], detections[col])

        return self._get_tracked(active_only=True)

    def _register(self, centroid, bbox):
        """Register a new object."""
        self.objects[self.next_id] = (float(centroid[0]), float(centroid[1]))
        self.bboxes[self.next_id] = bbox
        self.disappeared[self.next_id] = 0
        self.hits[self.next_id] = 1
        self.ages[self.next_id] = 1
        self.confirmed[self.next_id] = self.hits[self.next_id] >= self.confirm_frames
        self.next_id += 1

    def _deregister(self, obj_id):
        """Remove a tracked object."""
        del self.objects[obj_id]
        del self.bboxes[obj_id]
        del self.disappeared[obj_id]
        del self.hits[obj_id]
        del self.ages[obj_id]
        del self.confirmed[obj_id]

    def _get_tracked(self, confirmed_only=False, active_only=True):
        """Return tracked objects as {id: bbox} with optional filters."""
        return OrderedDict(
            (obj_id, self.bboxes[obj_id])
            for obj_id in self.objects
            if (not confirmed_only or self.confirmed.get(obj_id, False))
            and (not active_only or self.disappeared.get(obj_id, 0) == 0)
        )

    def get_confirmed_tracks(self):
        """Return only active tracks that passed confirmation frames."""
        return self._get_tracked(confirmed_only=True, active_only=True)

    def is_confirmed(self, obj_id):
        """Return whether a track is confirmed."""
        return bool(self.confirmed.get(obj_id, False))

    def get_track_status(self, obj_id):
        """Return metadata for debug overlay."""
        return {
            "age": self.ages.get(obj_id, 0),
            "hits": self.hits.get(obj_id, 0),
            "confirmed": bool(self.confirmed.get(obj_id, False)),
            "disappeared": self.disappeared.get(obj_id, 0),
        }

    def get_disappeared_ids(self, since_last_call=False):
        """Return IDs that have recently been deregistered."""
        # In this simple implementation, cleanup is handled externally
        return []
