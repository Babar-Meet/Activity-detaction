"""
Rule-based action classifier using pose keypoints.
Detects: Standing, Sitting (Chair), Sitting (Ground), Waving, Using Laptop,
         Using Phone, Talking.
"""

from collections import defaultdict, deque
from utils.helpers import calculate_angle, distance
import config


class ActionClassifier:
    """
    Classifies human actions based on MediaPipe pose landmarks
    and contextual object detections.
    """

    def __init__(self):
        # Per-person histories for temporal analysis
        # key = person_id, value = deque of (x,y) positions
        self._hand_history = defaultdict(lambda: deque(maxlen=config.WAVE_HISTORY_FRAMES))
        self._nose_history = defaultdict(lambda: deque(maxlen=config.TALK_HISTORY_FRAMES))
        # Per-person label smoothing
        self._label_history = defaultdict(lambda: deque(maxlen=8))

        print("[ActionClassifier] Initialized.")

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
        posture = self._classify_posture(kp, frame_h)

        # --- Additional Actions ---
        actions = []

        if self._is_waving(person_id, kp):
            actions.append("Waving")

        if self._is_using_phone(kp):
            actions.append("Using Phone")

        if self._is_using_laptop(kp, object_detections):
            actions.append("Using Laptop")

        if self._is_talking(person_id, kp):
            actions.append("Talking")

        # Color based on posture
        if posture == "Standing":
            color = config.COLOR_STANDING
        elif posture == "Sitting (Chair)":
            color = config.COLOR_SITTING
        elif posture == "Sitting (Ground)":
            color = config.COLOR_GROUND
        else:
            color = config.COLOR_DEFAULT

        # Smooth the posture label to avoid flickering
        raw_label = posture
        self._label_history[person_id].append(raw_label)
        # Use mode of recent labels
        from collections import Counter
        counts = Counter(self._label_history[person_id])
        posture = counts.most_common(1)[0][0]

        return {
            "posture": posture,
            "actions": actions,
            "color": color,
        }

    def _classify_posture(self, kp, frame_h):
        """Determine Standing / Sitting (Chair) / Sitting (Ground)."""
        # Need hip, knee, ankle landmarks
        has_left_leg = all(k in kp for k in ["LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"])
        has_right_leg = all(k in kp for k in ["RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"])

        if not has_left_leg and not has_right_leg:
            return "Unknown"

        # Calculate knee angles for available legs
        knee_angles = []
        hip_ys = []
        knee_ys = []
        ankle_ys = []

        if has_left_leg:
            l_angle = calculate_angle(
                kp["LEFT_HIP"][:2], kp["LEFT_KNEE"][:2], kp["LEFT_ANKLE"][:2]
            )
            knee_angles.append(l_angle)
            hip_ys.append(kp["LEFT_HIP"][1])
            knee_ys.append(kp["LEFT_KNEE"][1])
            ankle_ys.append(kp["LEFT_ANKLE"][1])

        if has_right_leg:
            r_angle = calculate_angle(
                kp["RIGHT_HIP"][:2], kp["RIGHT_KNEE"][:2], kp["RIGHT_ANKLE"][:2]
            )
            knee_angles.append(r_angle)
            hip_ys.append(kp["RIGHT_HIP"][1])
            knee_ys.append(kp["RIGHT_KNEE"][1])
            ankle_ys.append(kp["RIGHT_ANKLE"][1])

        avg_knee_angle = sum(knee_angles) / len(knee_angles)
        avg_hip_y = sum(hip_ys) / len(hip_ys)
        avg_knee_y = sum(knee_ys) / len(knee_ys)

        # Hip position ratio (how low in the frame)
        hip_ratio = avg_hip_y / frame_h

        # --- Standing ---
        # Legs nearly straight (knee angle > threshold)
        # Hip above knees (hip_y < knee_y in image coords, y increases downward)
        if avg_knee_angle >= config.STANDING_KNEE_ANGLE_MIN and avg_hip_y < avg_knee_y:
            return "Standing"

        # --- Sitting (Ground) ---
        # Hip very close to bottom of frame (high ratio)
        # OR hip very close to ankles
        avg_ankle_y = sum(ankle_ys) / len(ankle_ys) if ankle_ys else frame_h
        hip_ankle_diff = abs(avg_hip_y - avg_ankle_y)
        if hip_ratio > config.SITTING_GROUND_HIP_RATIO or hip_ankle_diff < 40:
            return "Sitting (Ground)"

        # --- Sitting (Chair) ---
        # Knees bent at roughly 60-130 degrees, torso upright, hip above ground
        if (config.SITTING_CHAIR_KNEE_ANGLE_MIN <= avg_knee_angle
                <= config.SITTING_CHAIR_KNEE_ANGLE_MAX):
            return "Sitting (Chair)"

        # Default: if hip is above knees but legs not straight enough for standing
        if avg_hip_y < avg_knee_y:
            return "Standing"

        return "Sitting (Chair)"

    def _is_waving(self, person_id, kp):
        """Detect waving: hand above shoulder with movement."""
        for side in ["LEFT", "RIGHT"]:
            wrist_key = f"{side}_WRIST"
            shoulder_key = f"{side}_SHOULDER"
            if wrist_key in kp and shoulder_key in kp:
                wrist = kp[wrist_key]
                shoulder = kp[shoulder_key]

                # Hand must be above shoulder (lower y value)
                if wrist[1] < shoulder[1]:
                    # Track hand position history
                    self._hand_history[person_id].append(wrist[:2])
                    hist = self._hand_history[person_id]

                    if len(hist) >= 3:
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
        if "NOSE" not in kp:
            return False
        nose = kp["NOSE"][:2]

        for side in ["LEFT", "RIGHT"]:
            wrist_key = f"{side}_WRIST"
            if wrist_key in kp:
                wrist = kp[wrist_key][:2]
                dist = distance(nose, wrist)
                if dist < config.PHONE_HAND_FACE_DISTANCE:
                    return True
        return False

    def _is_using_laptop(self, kp, object_detections):
        """Detect using laptop: person near a detected laptop with forward hand position."""
        # Find laptop detections
        laptops = [d for d in object_detections if d["class_name"] == "laptop"]
        if not laptops:
            return False

        # Get person center from available landmarks
        pts = []
        for name in ["LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP"]:
            if name in kp:
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
                    if wrist_key in kp and shoulder_key in kp:
                        if kp[wrist_key][1] > kp[shoulder_key][1]:
                            return True
        return False

    def _is_talking(self, person_id, kp):
        """
        Approximate talking detection via repeated small head movements.
        Tracks nose position across frames and counts direction changes.
        """
        if "NOSE" not in kp:
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
