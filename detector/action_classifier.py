"""
Rule-based action classifier using pose keypoints.
Detects: Standing, Sitting (Chair), Sitting (Ground), Waving, Using Laptop,
         Using Phone, Talking.
"""

import time
from collections import Counter, defaultdict, deque
from utils.helpers import calculate_angle, distance
import config


ACTION_ORDER = ["Waving", "Using Phone", "Using Laptop", "Talking"]


class ActionClassifier:
    """
    Classifies human actions based on MediaPipe pose landmarks
    and contextual object detections.
    """

    def __init__(self):
        # Per-person histories for temporal analysis
        # key = person_id, value = deque of (x,y) positions
        self._hand_history = defaultdict(
            lambda: {
                "LEFT": deque(maxlen=config.WAVE_HISTORY_FRAMES),
                "RIGHT": deque(maxlen=config.WAVE_HISTORY_FRAMES),
            }
        )
        self._nose_history = defaultdict(lambda: deque(maxlen=config.TALK_HISTORY_FRAMES))

        # Per-person label smoothing
        self._label_history = defaultdict(lambda: deque(maxlen=config.POSTURE_HISTORY_FRAMES))

        # Stable output state + delayed transition state
        self._stable_posture = {}
        self._pending_posture = {}
        self._stable_actions = defaultdict(set)
        self._pending_actions = defaultdict(dict)

        print("[ActionClassifier] Initialized.")

    @staticmethod
    def _clamp01(value):
        return max(0.0, min(1.0, value))

    def _is_visible(self, kp, name, threshold=None):
        """Check if a keypoint exists and exceeds visibility threshold."""
        if name not in kp:
            return False
        vis_threshold = config.LANDMARK_VISIBILITY_MIN if threshold is None else threshold
        return kp[name][2] >= vis_threshold

    def _stabilize_posture(self, person_id, raw_posture, now_ts):
        """Apply time-based hysteresis to avoid fast posture flicker."""
        stable = self._stable_posture.get(person_id)
        if stable is None:
            self._stable_posture[person_id] = raw_posture
            return raw_posture

        if raw_posture == stable:
            self._pending_posture.pop(person_id, None)
            return stable

        pending = self._pending_posture.get(person_id)
        if pending is None or pending[0] != raw_posture:
            self._pending_posture[person_id] = (raw_posture, now_ts)
            return stable

        if now_ts - pending[1] >= config.POSTURE_SWITCH_DELAY_SEC:
            self._stable_posture[person_id] = raw_posture
            self._pending_posture.pop(person_id, None)
            return raw_posture

        return stable

    def _stabilize_actions(self, person_id, raw_actions, now_ts):
        """Apply time-based hysteresis to action labels."""
        raw_set = set(raw_actions)
        stable_set = self._stable_actions[person_id]
        pending = self._pending_actions[person_id]

        for action_name in ACTION_ORDER:
            raw_state = action_name in raw_set
            stable_state = action_name in stable_set

            if raw_state == stable_state:
                pending.pop(action_name, None)
                continue

            transition = pending.get(action_name)
            if transition is None or transition[0] != raw_state:
                pending[action_name] = (raw_state, now_ts)
                continue

            if now_ts - transition[1] >= config.ACTION_SWITCH_DELAY_SEC:
                if raw_state:
                    stable_set.add(action_name)
                else:
                    stable_set.discard(action_name)
                pending.pop(action_name, None)

        return [action_name for action_name in ACTION_ORDER if action_name in stable_set]

    def classify(self, person_id, landmarks, frame_shape, object_detections):
        """
        Classify the action of a detected person.

        Args:
            person_id: Unique tracker ID for this person.
            landmarks: dict of landmark_name -> (x, y, visibility).
            frame_shape: (height, width, channels) of the frame.
            object_detections: list of detected objects (for laptop proximity).

        Returns:
            dict with keys:
                'posture': str ("Standing", "Sitting (Chair)", "Sitting (Ground)")
                'actions': list of str (additional actions like "Waving", etc.)
                'color': BGR tuple for the posture
        """
        now_ts = time.time()
        frame_h, frame_w = frame_shape[:2]

        # Extract key landmarks with visibility check
        kp = {}
        required = [
            "NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER",
            "LEFT_HIP", "RIGHT_HIP",
            "LEFT_KNEE", "RIGHT_KNEE",
            "LEFT_ANKLE", "RIGHT_ANKLE",
            "LEFT_WRIST", "RIGHT_WRIST",
            "LEFT_ELBOW", "RIGHT_ELBOW",
        ]
        for name in required:
            if name in landmarks:
                x, y, vis = landmarks[name]
                kp[name] = (x, y, vis)

        # --- Posture Classification ---
        raw_posture, posture_confidence = self._classify_posture(kp, frame_h)

        # Smooth raw posture by short history mode before time delay gating.
        self._label_history[person_id].append(raw_posture)
        counts = Counter(self._label_history[person_id])
        smoothed_raw_posture = counts.most_common(1)[0][0]
        posture = self._stabilize_posture(person_id, smoothed_raw_posture, now_ts)

        # --- Additional Actions ---
        raw_actions = []

        if self._is_waving(person_id, kp):
            raw_actions.append("Waving")

        if self._is_using_phone(kp):
            raw_actions.append("Using Phone")

        if self._is_using_laptop(kp, object_detections):
            raw_actions.append("Using Laptop")

        if self._is_talking(person_id, kp):
            raw_actions.append("Talking")

        actions = self._stabilize_actions(person_id, raw_actions, now_ts)

        # Color based on posture
        if posture == "Standing":
            color = config.COLOR_STANDING
        elif posture == "Sitting (Chair)":
            color = config.COLOR_SITTING
        elif posture == "Sitting (Ground)":
            color = config.COLOR_GROUND
        else:
            color = config.COLOR_DEFAULT

        return {
            "posture": posture,
            "raw_posture": smoothed_raw_posture,
            "posture_confidence": posture_confidence,
            "actions": actions,
            "raw_actions": raw_actions,
            "color": color,
        }

    def _classify_posture(self, kp, frame_h):
        """Determine Standing / Sitting (Chair) / Sitting (Ground)."""
        # Need hip, knee, ankle landmarks
        has_left_leg = all(
            self._is_visible(kp, key)
            for key in ["LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"]
        )
        has_right_leg = all(
            self._is_visible(kp, key)
            for key in ["RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"]
        )

        if not has_left_leg and not has_right_leg:
            return "Unknown", 0.0

        # Calculate knee angles for available legs
        knee_angles = []
        hip_ys = []
        knee_ys = []
        ankle_ys = []
        leg_vis = []

        if has_left_leg:
            l_angle = calculate_angle(
                kp["LEFT_HIP"][:2], kp["LEFT_KNEE"][:2], kp["LEFT_ANKLE"][:2]
            )
            knee_angles.append(l_angle)
            hip_ys.append(kp["LEFT_HIP"][1])
            knee_ys.append(kp["LEFT_KNEE"][1])
            ankle_ys.append(kp["LEFT_ANKLE"][1])
            leg_vis.append((kp["LEFT_HIP"][2] + kp["LEFT_KNEE"][2] + kp["LEFT_ANKLE"][2]) / 3.0)

        if has_right_leg:
            r_angle = calculate_angle(
                kp["RIGHT_HIP"][:2], kp["RIGHT_KNEE"][:2], kp["RIGHT_ANKLE"][:2]
            )
            knee_angles.append(r_angle)
            hip_ys.append(kp["RIGHT_HIP"][1])
            knee_ys.append(kp["RIGHT_KNEE"][1])
            ankle_ys.append(kp["RIGHT_ANKLE"][1])
            leg_vis.append((kp["RIGHT_HIP"][2] + kp["RIGHT_KNEE"][2] + kp["RIGHT_ANKLE"][2]) / 3.0)

        avg_knee_angle = sum(knee_angles) / len(knee_angles)
        avg_hip_y = sum(hip_ys) / len(hip_ys)
        avg_knee_y = sum(knee_ys) / len(knee_ys)
        avg_vis = sum(leg_vis) / len(leg_vis) if leg_vis else 0.0

        # Hip position ratio (how low in the frame)
        hip_ratio = avg_hip_y / frame_h
        avg_ankle_y = sum(ankle_ys) / len(ankle_ys) if ankle_ys else frame_h
        hip_ankle_diff = abs(avg_hip_y - avg_ankle_y)
        hip_above_knee = (avg_hip_y + config.STANDING_HIP_ABOVE_KNEE_MARGIN) < avg_knee_y

        # Ground sitting is the strongest check first.
        if (
            hip_ratio >= config.SITTING_GROUND_HIP_RATIO
            or hip_ankle_diff <= config.SITTING_GROUND_HIP_ANKLE_DIFF_MAX
        ):
            confidence = self._clamp01(0.65 + 0.35 * avg_vis)
            return "Sitting (Ground)", confidence

        # Standing: straight legs and hip above knees.
        if avg_knee_angle >= config.STANDING_KNEE_ANGLE_MIN and hip_above_knee:
            straightness = self._clamp01(
                (avg_knee_angle - config.STANDING_KNEE_ANGLE_MIN) / 20.0 + 0.5
            )
            confidence = self._clamp01(0.55 * straightness + 0.45 * avg_vis)
            return "Standing", confidence

        # Chair sitting: bent knee range with hip clearly above ankle level.
        if (config.SITTING_CHAIR_KNEE_ANGLE_MIN <= avg_knee_angle
                <= config.SITTING_CHAIR_KNEE_ANGLE_MAX):
            mid = (config.SITTING_CHAIR_KNEE_ANGLE_MIN + config.SITTING_CHAIR_KNEE_ANGLE_MAX) / 2.0
            half_range = max(
                1.0,
                (config.SITTING_CHAIR_KNEE_ANGLE_MAX - config.SITTING_CHAIR_KNEE_ANGLE_MIN) / 2.0,
            )
            center_score = self._clamp01(1.0 - (abs(avg_knee_angle - mid) / half_range))
            confidence = self._clamp01(0.5 * center_score + 0.5 * avg_vis)
            return "Sitting (Chair)", confidence

        # Soft fallback to standing when geometry is almost standing.
        if avg_knee_angle >= (config.STANDING_KNEE_ANGLE_MIN - 10) and hip_above_knee:
            confidence = self._clamp01(0.4 + 0.4 * avg_vis)
            return "Standing", confidence

        return "Unknown", self._clamp01(0.25 + 0.5 * avg_vis)

    def _is_waving(self, person_id, kp):
        """Detect waving: hand above shoulder with movement."""
        for side in ["LEFT", "RIGHT"]:
            wrist_key = f"{side}_WRIST"
            shoulder_key = f"{side}_SHOULDER"
            if not self._is_visible(kp, wrist_key):
                continue
            if not self._is_visible(kp, shoulder_key):
                continue

            wrist = kp[wrist_key]
            shoulder = kp[shoulder_key]

            # Hand must be above shoulder (lower y value)
            if config.WAVE_HAND_ABOVE_SHOULDER and wrist[1] >= shoulder[1]:
                continue

            hist = self._hand_history[person_id][side]
            hist.append(wrist[:2])

            if len(hist) >= 4:
                # Check for lateral movement (oscillation)
                x_positions = [p[0] for p in hist]
                movements = 0
                for i in range(1, len(x_positions)):
                    if abs(x_positions[i] - x_positions[i - 1]) > config.WAVE_MOVEMENT_THRESHOLD:
                        movements += 1
                if movements >= 2:
                    return True
        return False

    def _is_using_phone(self, kp):
        """Detect using phone: wrist close to nose/face."""
        if not self._is_visible(kp, "NOSE", threshold=0.35):
            return False
        nose = kp["NOSE"][:2]

        for side in ["LEFT", "RIGHT"]:
            wrist_key = f"{side}_WRIST"
            if self._is_visible(kp, wrist_key, threshold=0.35):
                wrist = kp[wrist_key][:2]
                dist = distance(nose, wrist)
                if dist < config.PHONE_HAND_FACE_DISTANCE:
                    return True
        return False

    def _is_using_laptop(self, kp, object_detections):
        """Detect using laptop: person near a detected laptop with forward hand position."""
        # Find laptop detections
        laptops = [
            d for d in object_detections
            if d.get("raw_class_name", d["class_name"]) == "laptop"
        ]
        if not laptops:
            return False

        # Get person center from available landmarks
        pts = []
        for name in ["LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP"]:
            if self._is_visible(kp, name):
                pts.append(kp[name][:2])
        if not pts:
            return False

        person_cx = sum(p[0] for p in pts) / len(pts)
        person_cy = sum(p[1] for p in pts) / len(pts)

        for laptop in laptops:
            lx1, ly1, lx2, ly2 = laptop["bbox"]
            laptop_cx = (lx1 + lx2) / 2
            laptop_cy = (ly1 + ly2) / 2
            dist = distance((person_cx, person_cy), (laptop_cx, laptop_cy))
            if dist < config.LAPTOP_PERSON_DISTANCE:
                # Check if hands are roughly in front (wrists below shoulders)
                for side in ["LEFT", "RIGHT"]:
                    wrist_key = f"{side}_WRIST"
                    shoulder_key = f"{side}_SHOULDER"
                    if self._is_visible(kp, wrist_key) and self._is_visible(kp, shoulder_key):
                        if kp[wrist_key][1] > kp[shoulder_key][1]:
                            return True
        return False

    def _is_talking(self, person_id, kp):
        """
        Approximate talking detection via repeated small head movements.
        Tracks nose position across frames and counts direction changes.
        """
        if not self._is_visible(kp, "NOSE", threshold=0.35):
            return False

        nose = kp["NOSE"][:2]
        self._nose_history[person_id].append(nose)
        hist = self._nose_history[person_id]

        if len(hist) < config.TALK_HISTORY_FRAMES // 2:
            return False

        # Count direction changes in X and Y
        direction_changes = 0
        for i in range(2, len(hist)):
            dx_prev = hist[i - 1][0] - hist[i - 2][0]
            dx_curr = hist[i][0] - hist[i - 1][0]
            dy_prev = hist[i - 1][1] - hist[i - 2][1]
            dy_curr = hist[i][1] - hist[i - 1][1]

            # Check for movement reversal
            if (dx_prev * dx_curr < 0 and
                    abs(dx_curr) > config.TALK_MOVEMENT_THRESHOLD):
                direction_changes += 1
            if (dy_prev * dy_curr < 0 and
                    abs(dy_curr) > config.TALK_MOVEMENT_THRESHOLD):
                direction_changes += 1

        return direction_changes >= config.TALK_MIN_MOVEMENTS

    def cleanup_person(self, person_id):
        """Remove tracking history for a person who has disappeared."""
        self._hand_history.pop(person_id, None)
        self._nose_history.pop(person_id, None)
        self._label_history.pop(person_id, None)
        self._stable_posture.pop(person_id, None)
        self._pending_posture.pop(person_id, None)
        self._stable_actions.pop(person_id, None)
        self._pending_actions.pop(person_id, None)
