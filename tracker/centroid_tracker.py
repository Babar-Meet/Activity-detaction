"""
Lightweight centroid-based object tracker.
Assigns persistent IDs to detected persons across frames.
"""

from collections import OrderedDict
import numpy as np
from scipy.spatial.distance import cdist
import config


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

        self.max_disappeared = config.TRACKER_MAX_DISAPPEARED
        self.max_distance = config.TRACKER_MAX_DISTANCE

        print("[CentroidTracker] Initialized.")

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
                if self.disappeared[obj_id] > self.max_disappeared:
                    self._deregister(obj_id)
            return self._get_tracked()

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
            return self._get_tracked()

        # Match existing objects to new detections
        object_ids = list(self.objects.keys())
        object_centroids = np.array(list(self.objects.values()))

        # Compute pairwise distances
        D = cdist(object_centroids, input_centroids)

        # Find best matches (Hungarian-like greedy approach)
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for (row, col) in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue

            # Only match if distance is within threshold
            if D[row, col] > self.max_distance:
                continue

            obj_id = object_ids[row]
            self.objects[obj_id] = input_centroids[col]
            self.bboxes[obj_id] = detections[col]
            self.disappeared[obj_id] = 0

            used_rows.add(row)
            used_cols.add(col)

        # Handle unmatched existing objects (disappeared)
        unused_rows = set(range(len(object_ids))) - used_rows
        for row in unused_rows:
            obj_id = object_ids[row]
            self.disappeared[obj_id] += 1
            if self.disappeared[obj_id] > self.max_disappeared:
                self._deregister(obj_id)

        # Handle unmatched new detections (register as new)
        unused_cols = set(range(len(input_centroids))) - used_cols
        for col in unused_cols:
            self._register(input_centroids[col], detections[col])

        return self._get_tracked()

    def _register(self, centroid, bbox):
        """Register a new object."""
        self.objects[self.next_id] = centroid
        self.bboxes[self.next_id] = bbox
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def _deregister(self, obj_id):
        """Remove a tracked object."""
        del self.objects[obj_id]
        del self.bboxes[obj_id]
        del self.disappeared[obj_id]

    def _get_tracked(self):
        """Return current tracked objects as {id: bbox}."""
        return OrderedDict(
            (obj_id, self.bboxes[obj_id])
            for obj_id in self.objects
        )

    def get_disappeared_ids(self, since_last_call=False):
        """Return IDs that have recently been deregistered."""
        # In this simple implementation, cleanup is handled externally
        return []
