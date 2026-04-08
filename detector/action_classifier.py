"""
Rule-based action classifier using pose keypoints.
Detects: Standing, Sitting (Chair), Sitting (Ground), Walking,
         Saying Hello, V Sign.
"""

import time
from collections import Counter, defaultdict, deque
from utils.helpers import calculate_angle, distance
import config


ACTION_ORDER = [
    "Saying Hello",
    "V Sign",
    "Walking",
]


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
        self._body_center_history = defaultdict(
            lambda: deque(maxlen=config.WALK_HISTORY_FRAMES)
        )
        self._ankle_phase_history = defaultdict(
            lambda: deque(maxlen=config.WALK_HISTORY_FRAMES)
        )

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

    def _midpoint_if_visible(self, kp, left_key, right_key):
        """Return midpoint for a left/right keypoint pair when both are visible."""
        if not self._is_visible(kp, left_key):
            return None
        if not self._is_visible(kp, right_key):
            return None

        lx, ly = kp[left_key][:2]
        rx, ry = kp[right_key][:2]
        return ((lx + rx) / 2.0, (ly + ry) / 2.0)

    def _estimate_body_scale(self, kp):
        """Estimate person body scale in pixels for adaptive motion thresholds."""
        scales = []

        if self._is_visible(kp, "LEFT_SHOULDER") and self._is_visible(kp, "RIGHT_SHOULDER"):
            scales.append(distance(kp["LEFT_SHOULDER"][:2], kp["RIGHT_SHOULDER"][:2]))

        shoulder_mid = self._midpoint_if_visible(kp, "LEFT_SHOULDER", "RIGHT_SHOULDER")
        hip_mid = self._midpoint_if_visible(kp, "LEFT_HIP", "RIGHT_HIP")
        if shoulder_mid and hip_mid:
            scales.append(distance(shoulder_mid, hip_mid))

        if self._is_visible(kp, "LEFT_HIP") and self._is_visible(kp, "LEFT_ANKLE"):
            scales.append(distance(kp["LEFT_HIP"][:2], kp["LEFT_ANKLE"][:2]))
        if self._is_visible(kp, "RIGHT_HIP") and self._is_visible(kp, "RIGHT_ANKLE"):
            scales.append(distance(kp["RIGHT_HIP"][:2], kp["RIGHT_ANKLE"][:2]))

        if not scales:
            return 80.0
        return max(40.0, max(scales))

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
                'actions': list of str (gesture labels)
                'color': BGR tuple for the posture
        """
        now_ts = time.time()
        frame_h = frame_shape[0]

        # Keep all landmarks so hand-specific keys can be used when available.
        kp = {
            name: (float(val[0]), float(val[1]), float(val[2]))
            for name, val in landmarks.items()
            if isinstance(val, (tuple, list)) and len(val) >= 3
        }

        # --- Posture Classification ---
        raw_posture, posture_confidence = self._classify_posture(kp, frame_h, object_detections)

        # Smooth raw posture by short history mode before time delay gating.
        self._label_history[person_id].append(raw_posture)
        counts = Counter(self._label_history[person_id])
        smoothed_raw_posture = counts.most_common(1)[0][0]
        posture = self._stabilize_posture(person_id, smoothed_raw_posture, now_ts)

        # --- Additional Actions ---
        raw_actions = []

        if self._is_saying_hello(person_id, kp):
            raw_actions.append("Saying Hello")

        if self._is_v_sign(kp):
            raw_actions.append("V Sign")

        if self._is_walking(person_id, kp, posture):
            raw_actions.append("Walking")

        # Suppress extra noisy action categories for a clearer end-user UI.

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

    def _is_near_chair(self, kp, object_detections):
        """Check whether hip center is spatially consistent with a chair detection."""
        chairs = [
            d
            for d in object_detections
            if d.get("raw_class_name", d.get("class_name")) == "chair"
        ]
        if not chairs:
            return False

        hip_points = []
        for name in ["LEFT_HIP", "RIGHT_HIP"]:
            if self._is_visible(kp, name):
                hip_points.append(kp[name][:2])
        if not hip_points:
            return False

        hip_cx = sum(p[0] for p in hip_points) / len(hip_points)
        hip_cy = sum(p[1] for p in hip_points) / len(hip_points)

        body_scale = self._estimate_body_scale(kp)
        max_dx = max(40.0, body_scale * config.CHAIR_HIP_HORIZONTAL_RATIO)
        max_dy = max(35.0, float(config.CHAIR_HIP_VERTICAL_NEAR_PIXELS))

        for chair in chairs:
            x1, y1, x2, y2 = chair["bbox"]
            chair_cx = (x1 + x2) / 2.0

            if abs(hip_cx - chair_cx) <= max_dx and -25 <= (hip_cy - y1) <= max_dy:
                return True

            if x1 <= hip_cx <= x2 and (y1 - 30) <= hip_cy <= y2:
                return True

        return False

    def _classify_posture(self, kp, frame_h, object_detections):
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
        hip_angles = []
        hip_ys = []
        knee_ys = []
        ankle_ys = []
        torso_heights = []
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
            if self._is_visible(kp, "LEFT_SHOULDER"):
                hip_angles.append(
                    calculate_angle(
                        kp["LEFT_SHOULDER"][:2], kp["LEFT_HIP"][:2], kp["LEFT_KNEE"][:2]
                    )
                )
                torso_heights.append(abs(kp["LEFT_SHOULDER"][1] - kp["LEFT_HIP"][1]))

        if has_right_leg:
            r_angle = calculate_angle(
                kp["RIGHT_HIP"][:2], kp["RIGHT_KNEE"][:2], kp["RIGHT_ANKLE"][:2]
            )
            knee_angles.append(r_angle)
            hip_ys.append(kp["RIGHT_HIP"][1])
            knee_ys.append(kp["RIGHT_KNEE"][1])
            ankle_ys.append(kp["RIGHT_ANKLE"][1])
            leg_vis.append((kp["RIGHT_HIP"][2] + kp["RIGHT_KNEE"][2] + kp["RIGHT_ANKLE"][2]) / 3.0)
            if self._is_visible(kp, "RIGHT_SHOULDER"):
                hip_angles.append(
                    calculate_angle(
                        kp["RIGHT_SHOULDER"][:2], kp["RIGHT_HIP"][:2], kp["RIGHT_KNEE"][:2]
                    )
                )
                torso_heights.append(abs(kp["RIGHT_SHOULDER"][1] - kp["RIGHT_HIP"][1]))

        avg_knee_angle = sum(knee_angles) / len(knee_angles)
        avg_hip_angle = sum(hip_angles) / len(hip_angles) if hip_angles else None
        avg_hip_y = sum(hip_ys) / len(hip_ys)
        avg_knee_y = sum(knee_ys) / len(knee_ys)
        avg_vis = sum(leg_vis) / len(leg_vis) if leg_vis else 0.0

        # Hip position ratio (how low in the frame)
        hip_ratio = avg_hip_y / frame_h
        avg_ankle_y = sum(ankle_ys) / len(ankle_ys) if ankle_ys else frame_h
        hip_ankle_diff = abs(avg_hip_y - avg_ankle_y)
        hip_above_knee = (avg_hip_y + config.STANDING_HIP_ABOVE_KNEE_MARGIN) < avg_knee_y
        near_chair = self._is_near_chair(kp, object_detections)

        avg_torso_h = (
            sum(torso_heights) / len(torso_heights)
            if torso_heights
            else max(35.0, frame_h * 0.08)
        )
        leg_extension = max(1.0, abs(avg_ankle_y - avg_hip_y))
        leg_extension_ratio = leg_extension / max(1.0, avg_torso_h)

        # Ground sitting is the strongest check first.
        if (
            hip_ratio >= config.SITTING_GROUND_HIP_RATIO
            or hip_ankle_diff <= config.SITTING_GROUND_HIP_ANKLE_DIFF_MAX
        ):
            confidence = self._clamp01(0.65 + 0.35 * avg_vis)
            return "Sitting (Ground)", confidence

        knee_straight = avg_knee_angle >= config.STANDING_KNEE_ANGLE_MIN
        hip_open = avg_hip_angle is None or avg_hip_angle >= config.STANDING_HIP_ANGLE_MIN
        leg_extended = leg_extension_ratio >= config.STANDING_LEG_EXTENSION_RATIO_MIN

        knee_bent = (
            config.SITTING_CHAIR_KNEE_ANGLE_MIN
            <= avg_knee_angle
            <= config.SITTING_CHAIR_KNEE_ANGLE_MAX
        )
        hip_bent = (
            avg_hip_angle is not None
            and config.SITTING_CHAIR_HIP_ANGLE_MIN
            <= avg_hip_angle
            <= config.SITTING_CHAIR_HIP_ANGLE_MAX
        )
        leg_compact = leg_extension_ratio <= config.SITTING_CHAIR_LEG_EXTENSION_RATIO_MAX

        # Chair sitting: compact leg geometry with bent knee/hip profile.
        if near_chair and (knee_bent or hip_bent or leg_compact):
            confidence = self._clamp01(0.60 + 0.35 * avg_vis)
            return "Sitting (Chair)", confidence

        if (knee_bent and leg_compact and not leg_extended) or (hip_bent and leg_compact):
            knee_mid = (
                config.SITTING_CHAIR_KNEE_ANGLE_MIN + config.SITTING_CHAIR_KNEE_ANGLE_MAX
            ) / 2.0
            knee_half_range = max(
                1.0,
                (config.SITTING_CHAIR_KNEE_ANGLE_MAX - config.SITTING_CHAIR_KNEE_ANGLE_MIN) / 2.0,
            )
            knee_center_score = self._clamp01(
                1.0 - (abs(avg_knee_angle - knee_mid) / knee_half_range)
            )
            confidence = self._clamp01(0.45 * knee_center_score + 0.55 * avg_vis)
            return "Sitting (Chair)", confidence

        # Standing: straight legs and hip above knees.
        if knee_straight and hip_open and hip_above_knee and leg_extended:
            straightness = self._clamp01(
                (avg_knee_angle - config.STANDING_KNEE_ANGLE_MIN) / 20.0 + 0.5
            )
            hip_score = 1.0
            if avg_hip_angle is not None:
                hip_score = self._clamp01(
                    (avg_hip_angle - config.STANDING_HIP_ANGLE_MIN) / 25.0 + 0.5
                )
            confidence = self._clamp01(0.4 * straightness + 0.2 * hip_score + 0.4 * avg_vis)
            return "Standing", confidence

        # Soft posture fallback for bent hip with compact legs.
        if knee_bent or (hip_bent and leg_compact):
            confidence = self._clamp01(0.35 + 0.45 * avg_vis)
            return "Sitting (Chair)", confidence

        if near_chair and avg_knee_angle <= (config.STANDING_KNEE_ANGLE_MIN + 12):
            confidence = self._clamp01(0.45 + 0.45 * avg_vis)
            return "Sitting (Chair)", confidence

        # Soft fallback to standing when geometry is almost standing.
        if (
            avg_knee_angle >= (config.STANDING_KNEE_ANGLE_MIN - 12)
            and hip_above_knee
            and leg_extension_ratio >= (config.STANDING_LEG_EXTENSION_RATIO_MIN * 0.9)
        ):
            confidence = self._clamp01(0.4 + 0.4 * avg_vis)
            return "Standing", confidence

        # Final deterministic fallback for end-user clarity.
        if hip_ratio >= (config.SITTING_GROUND_HIP_RATIO * 0.94):
            return "Sitting (Ground)", self._clamp01(0.45 + 0.40 * avg_vis)
        if near_chair or avg_knee_angle < config.STANDING_KNEE_ANGLE_MIN:
            return "Sitting (Chair)", self._clamp01(0.40 + 0.45 * avg_vis)
        return "Standing", self._clamp01(0.35 + 0.45 * avg_vis)

    def _is_saying_hello(self, person_id, kp):
        """Detect hello gesture: raised hand with repeated side-to-side movement."""
        body_scale = self._estimate_body_scale(kp)
        move_threshold = max(
            float(config.WAVE_MOVEMENT_THRESHOLD),
            body_scale * config.HELLO_MOVEMENT_RATIO,
        )
        min_sway = max(move_threshold * 1.4, body_scale * config.HELLO_MIN_SWAY_RATIO)
        shoulder_margin = max(4.0, body_scale * config.HELLO_HAND_MARGIN_RATIO)

        for side in ["LEFT", "RIGHT"]:
            shoulder_key = f"{side}_SHOULDER"
            hist = self._hand_history[person_id][side]

            hand_point = self._first_visible_point(
                kp,
                [
                    f"{side}_INDEX_FINGER_TIP",
                    f"{side}_MIDDLE_FINGER_TIP",
                    f"{side}_WRIST",
                    f"{side}_INDEX",
                ],
                threshold=0.30,
            )
            shoulder = self._first_visible_point(kp, [shoulder_key], threshold=0.30)

            if hand_point is None or shoulder is None:
                hist.clear()
                continue

            # Hand must be above shoulder (lower y value)
            if config.WAVE_HAND_ABOVE_SHOULDER and hand_point[1] > (shoulder[1] + shoulder_margin):
                hist.clear()
                continue

            hist.append(hand_point)

            if len(hist) >= 4:
                x_positions = [p[0] for p in hist]
                sway = max(x_positions) - min(x_positions)
                movements = 0
                direction_changes = 0
                prev_dx = None

                for i in range(1, len(x_positions)):
                    dx = x_positions[i] - x_positions[i - 1]
                    if abs(dx) > move_threshold:
                        movements += 1
                    if prev_dx is not None:
                        if abs(dx) > (move_threshold * 0.35) and abs(prev_dx) > (move_threshold * 0.35):
                            if dx * prev_dx < 0:
                                direction_changes += 1
                    prev_dx = dx

                if (
                    sway >= min_sway
                    and
                    movements >= config.HELLO_MIN_MOVES
                    and direction_changes >= config.HELLO_MIN_DIRECTION_CHANGES
                ):
                    return True

                # Fallback: strong horizontal sway with at least one clear movement.
                if sway >= (min_sway * 1.35) and movements >= config.HELLO_MIN_MOVES:
                    return True

        return False

    def _first_visible_point(self, kp, names, threshold=0.35):
        """Return the first visible keypoint among candidate names."""
        for name in names:
            if self._is_visible(kp, name, threshold=threshold):
                return kp[name][:2]
        return None

    def _finger_extended(self, kp, tip_name, pip_name, mcp_name=None, min_delta=4.0):
        """Return True when a finger is extended upward, False when bent, None if unknown."""
        tip = self._first_visible_point(kp, [tip_name], threshold=0.30)
        pip = self._first_visible_point(kp, [pip_name], threshold=0.30)
        if tip is None:
            return None

        if pip is not None:
            return (pip[1] - tip[1]) >= min_delta

        if mcp_name is not None:
            mcp = self._first_visible_point(kp, [mcp_name], threshold=0.30)
            if mcp is not None:
                return (mcp[1] - tip[1]) >= (min_delta * 1.2)

        return None

    def _finger_folded(self, kp, tip_name, pip_name, mcp_name=None, relax=3.0):
        """Return True when finger appears folded, False when extended, None if unknown."""
        tip = self._first_visible_point(kp, [tip_name], threshold=0.30)
        pip = self._first_visible_point(kp, [pip_name], threshold=0.30)
        if tip is None:
            return None

        if pip is not None:
            return tip[1] >= (pip[1] - relax)

        if mcp_name is not None:
            mcp = self._first_visible_point(kp, [mcp_name], threshold=0.30)
            if mcp is not None:
                return tip[1] >= (mcp[1] - (relax * 1.5))

        return None

    def _is_v_sign(self, kp):
        """Detect a V-sign using hand landmarks, with pose-based fallback."""
        body_scale = self._estimate_body_scale(kp)
        face_point = self._first_visible_point(
            kp,
            ["FACE_MOUTH_CENTER", "NOSE"],
            threshold=0.30,
        )

        min_spread = max(8.0, body_scale * config.V_SIGN_MIN_SPREAD_RATIO)
        min_lift = max(5.0, body_scale * config.V_SIGN_MIN_FINGER_LIFT_RATIO)
        fold_relax = max(2.0, body_scale * config.V_SIGN_RING_PINKY_RELAX_RATIO)

        for side in ["LEFT", "RIGHT"]:
            shoulder_key = f"{side}_SHOULDER"

            wrist = self._first_visible_point(
                kp,
                [
                    f"{side}_WRIST",
                    f"{side}_INDEX_FINGER_MCP",
                    f"{side}_MIDDLE_FINGER_MCP",
                    f"{side}_INDEX",
                ],
                threshold=0.30,
            )
            index_tip = self._first_visible_point(
                kp,
                [f"{side}_INDEX_FINGER_TIP", f"{side}_INDEX"],
                threshold=0.30,
            )
            middle_tip = self._first_visible_point(
                kp,
                [f"{side}_MIDDLE_FINGER_TIP", f"{side}_PINKY", f"{side}_RING_FINGER_TIP"],
                threshold=0.30,
            )
            thumb_tip = self._first_visible_point(
                kp,
                [f"{side}_THUMB_TIP", f"{side}_THUMB"],
                threshold=0.30,
            )

            if wrist is None or index_tip is None or middle_tip is None:
                continue

            if (
                config.V_SIGN_REQUIRE_HAND_RAISED
                and self._is_visible(kp, shoulder_key, threshold=0.35)
                and wrist[1] >= kp[shoulder_key][1]
            ):
                continue

            finger_spread = distance(index_tip, middle_tip)
            if finger_spread < min_spread:
                continue

            if (wrist[1] - index_tip[1]) < min_lift:
                continue
            if (wrist[1] - middle_tip[1]) < (min_lift * 0.8):
                continue

            index_up = self._finger_extended(
                kp,
                f"{side}_INDEX_FINGER_TIP",
                f"{side}_INDEX_FINGER_PIP",
                mcp_name=f"{side}_INDEX_FINGER_MCP",
                min_delta=min_lift,
            )
            middle_up = self._finger_extended(
                kp,
                f"{side}_MIDDLE_FINGER_TIP",
                f"{side}_MIDDLE_FINGER_PIP",
                mcp_name=f"{side}_MIDDLE_FINGER_MCP",
                min_delta=min_lift,
            )
            if index_up is False or middle_up is False:
                continue

            ring_folded = self._finger_folded(
                kp,
                f"{side}_RING_FINGER_TIP",
                f"{side}_RING_FINGER_PIP",
                mcp_name=f"{side}_RING_FINGER_MCP",
                relax=fold_relax,
            )
            pinky_folded = self._finger_folded(
                kp,
                f"{side}_PINKY_TIP",
                f"{side}_PINKY_PIP",
                mcp_name=f"{side}_PINKY_MCP",
                relax=fold_relax,
            )

            # If ring/pinky detail exists, require them to be folded for a cleaner V-sign.
            if ring_folded is False or pinky_folded is False:
                continue

            index_mcp = self._first_visible_point(kp, [f"{side}_INDEX_FINGER_MCP"], threshold=0.3)
            middle_mcp = self._first_visible_point(kp, [f"{side}_MIDDLE_FINGER_MCP"], threshold=0.3)
            if index_mcp is not None and middle_mcp is not None:
                if index_tip[1] > index_mcp[1] or middle_tip[1] > middle_mcp[1]:
                    continue

            if thumb_tip is not None:
                thumb_dist = distance(thumb_tip, wrist)
                if thumb_dist > (finger_spread * config.V_SIGN_MAX_THUMB_TO_SPREAD_RATIO):
                    continue

            if face_point is not None:
                max_face_dist = max(45.0, body_scale * config.V_SIGN_MAX_FACE_DISTANCE_RATIO)
                if distance(wrist, face_point) > max_face_dist:
                    continue

            return True

        return False

    def _is_walking(self, person_id, kp, posture):
        """Detect walking from body translation plus alternating ankle lead."""
        hip_center = self._midpoint_if_visible(kp, "LEFT_HIP", "RIGHT_HIP")
        if hip_center is None:
            return False

        self._body_center_history[person_id].append(hip_center)

        if self._is_visible(kp, "LEFT_ANKLE", threshold=0.35) and self._is_visible(kp, "RIGHT_ANKLE", threshold=0.35):
            phase_value = kp["LEFT_ANKLE"][0] - kp["RIGHT_ANKLE"][0]
            self._ankle_phase_history[person_id].append(phase_value)

        if posture != "Standing":
            return False

        center_hist = self._body_center_history[person_id]
        if len(center_hist) < max(6, config.WALK_HISTORY_FRAMES // 2):
            return False

        body_scale = self._estimate_body_scale(kp)
        net_shift = abs(center_hist[-1][0] - center_hist[0][0])
        per_frame_shifts = [
            abs(center_hist[i][0] - center_hist[i - 1][0])
            for i in range(1, len(center_hist))
        ]
        avg_shift = sum(per_frame_shifts) / len(per_frame_shifts) if per_frame_shifts else 0.0

        min_net_shift = max(10.0, body_scale * config.WALK_MIN_CENTER_SHIFT_RATIO)
        min_avg_shift = max(1.5, body_scale * config.WALK_MIN_PER_FRAME_SHIFT_RATIO)

        if net_shift < min_net_shift or avg_shift < min_avg_shift:
            return False

        phase_hist = self._ankle_phase_history[person_id]
        if len(phase_hist) < 5:
            return net_shift >= (min_net_shift * 1.7)

        swing_amplitude = (max(phase_hist) - min(phase_hist)) * 0.5
        min_swing = max(5.0, body_scale * config.WALK_MIN_ANKLE_SWING_RATIO)

        phase_threshold = max(3.0, body_scale * 0.08)
        phase_changes = 0
        prev_sign = None
        for value in phase_hist:
            if abs(value) < phase_threshold:
                continue
            sign = 1 if value > 0 else -1
            if prev_sign is not None and sign != prev_sign:
                phase_changes += 1
            prev_sign = sign

        if phase_changes >= config.WALK_MIN_PHASE_CHANGES and swing_amplitude >= min_swing:
            return True

        return net_shift >= (min_net_shift * 1.8) and swing_amplitude >= (min_swing * 0.6)

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
        self._body_center_history.pop(person_id, None)
        self._ankle_phase_history.pop(person_id, None)
        self._label_history.pop(person_id, None)
        self._stable_posture.pop(person_id, None)
        self._pending_posture.pop(person_id, None)
        self._stable_actions.pop(person_id, None)
        self._pending_actions.pop(person_id, None)
