"""Detect and display whether the visible hand is left or right.

Run: python hand_side.py --camera 0
Press q to quit. This demo does not send servo commands.
"""
from __future__ import annotations

import argparse
import shutil
import ssl
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import certifi

import cv2
import mediapipe as mp
import numpy as np

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")
WINDOW_NAME = "Bionic Arm - Hand Tracking"
PANEL_WIDTH = 390
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
    (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)
FINGER_JOINTS = {
    "Thumb": (1, 2, 3),
    "Index": (5, 6, 7),
    "Middle": (9, 10, 11),
    "Ring": (13, 14, 15),
    "Pinky": (17, 18, 19),
}
HAND_COLORS = ((60, 210, 140), (255, 190, 70))
LANDMARK_SMOOTHING = 0.4
OPEN_ANGLE = 155.0
CLOSE_ANGLE = 140.0
STATE_CONFIRM_FRAMES = 3


class TrackingStabilizer:
    """Smooth landmark movement and keep finger states from flickering."""

    def __init__(self) -> None:
        self.points: dict[tuple[str, int], tuple[float, float, float]] = {}
        self.finger_states: dict[tuple[str, str], bool] = {}
        self.candidates: dict[tuple[str, str], tuple[bool, int]] = {}
        self.angles: dict[tuple[str, str], float] = {}

    def smooth_landmarks(self, hand_key: str, landmarks: object) -> list[SimpleNamespace]:
        smoothed = []
        for index, point in enumerate(landmarks):
            key = (hand_key, index)
            current = (float(point.x), float(point.y), float(point.z))
            previous = self.points.get(key, current)
            values = tuple(
                LANDMARK_SMOOTHING * value + (1.0 - LANDMARK_SMOOTHING) * old
                for value, old in zip(current, previous)
            )
            self.points[key] = values
            smoothed.append(SimpleNamespace(x=values[0], y=values[1], z=values[2]))
        return smoothed

    def update_finger(
        self,
        hand_key: str,
        finger: str,
        value: float,
        open_threshold: float = OPEN_ANGLE,
        close_threshold: float = CLOSE_ANGLE,
    ) -> tuple[bool, float]:
        key = (hand_key, finger)
        old_value = self.angles.get(key, value)
        filtered_value = LANDMARK_SMOOTHING * value + (1.0 - LANDMARK_SMOOTHING) * old_value
        self.angles[key] = filtered_value

        state = self.finger_states.get(key)
        if state is None:
            state = filtered_value >= (open_threshold + close_threshold) / 2
            self.finger_states[key] = state
            return state, filtered_value

        desired_state = True if filtered_value >= open_threshold else False if filtered_value <= close_threshold else state
        if desired_state == state:
            self.candidates.pop(key, None)
            return state, filtered_value

        candidate_state, frames = self.candidates.get(key, (desired_state, 0))
        frames = frames + 1 if candidate_state == desired_state else 1
        self.candidates[key] = (desired_state, frames)
        if frames >= STATE_CONFIRM_FRAMES:
            state = desired_state
            self.finger_states[key] = state
            self.candidates.pop(key, None)
        return state, filtered_value


def ensure_model() -> None:
    if MODEL_PATH.exists():
        return
    print("Downloading the MediaPipe hand model (only needed once)...")
    temporary_path = MODEL_PATH.with_suffix(".task.tmp")
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    try:
        request = urllib.request.Request(MODEL_URL, headers={"User-Agent": "BionicArmCapstone/1.0"})
        with urllib.request.urlopen(request, context=ssl_context, timeout=60) as response:
            with temporary_path.open("wb") as model_file:
                shutil.copyfileobj(response, model_file)
        temporary_path.replace(MODEL_PATH)
    except Exception as exc:
        temporary_path.unlink(missing_ok=True)
        raise SystemExit(
            "Could not securely download the MediaPipe model. "
            "Check your internet connection and certificate setup, then try again. "
            f"Details: {exc}"
        ) from exc


def finger_bend_angle(landmarks: object, a: int, joint: int, c: int) -> float:
    """Return the 2D angle at a finger joint; near 180 degrees is straighter."""
    first = np.array([landmarks[a].x - landmarks[joint].x, landmarks[a].y - landmarks[joint].y])
    second = np.array([landmarks[c].x - landmarks[joint].x, landmarks[c].y - landmarks[joint].y])
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator == 0:
        return 0.0
    cosine = float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def thumb_extension_ratio(landmarks: object) -> float:
    """Measure thumb-tip distance from the palm center, normalized by palm width."""
    palm_points = np.array([[landmarks[index].x, landmarks[index].y] for index in (5, 9, 13, 17)])
    palm_center = palm_points.mean(axis=0)
    palm_width = float(np.linalg.norm(np.array([landmarks[5].x, landmarks[5].y]) - np.array([landmarks[17].x, landmarks[17].y])))
    if palm_width == 0:
        return 0.0
    thumb_tip = np.array([landmarks[4].x, landmarks[4].y])
    return float(np.linalg.norm(thumb_tip - palm_center) / palm_width)


def draw_landmark_panel(frame: np.ndarray, result: object, stabilizer: TrackingStabilizer) -> np.ndarray:
    """Draw every detected hand and show a compact finger-state summary."""
    height, width = frame.shape[:2]
    panel = np.full((height, PANEL_WIDTH, 3), (30, 34, 42), dtype=np.uint8)
    cv2.line(panel, (0, 0), (0, height), (75, 82, 96), 2)
    cv2.putText(panel, "HAND TRACKING", (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(panel, "Finger extension estimate", (24, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (170, 185, 205), 1, cv2.LINE_AA)

    hand_landmarks = result.hand_landmarks
    handedness = result.handedness
    if not hand_landmarks:
        cv2.putText(panel, "No hand detected", (24, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (90, 120, 255), 2, cv2.LINE_AA)
        cv2.putText(panel, "Show one or both hands to the camera", (24, 166), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (190, 195, 205), 1, cv2.LINE_AA)
        return np.hstack((frame, panel))

    for hand_index, landmarks in enumerate(hand_landmarks[:2]):
        color = HAND_COLORS[hand_index % len(HAND_COLORS)]
        categories = handedness[hand_index] if hand_index < len(handedness) else []
        category = categories[0] if categories else None
        raw_side = category.category_name if category else f"Hand {hand_index + 1}"
        landmarks = stabilizer.smooth_landmarks(raw_side, landmarks)
        label = ("Left hand" if raw_side == "Right" else "Right hand") if category else raw_side
        confidence = category.score if category else 0.0

        for start, end in HAND_CONNECTIONS:
            start_point = (int(landmarks[start].x * width), int(landmarks[start].y * height))
            end_point = (int(landmarks[end].x * width), int(landmarks[end].y * height))
            cv2.line(frame, start_point, end_point, color, 4, cv2.LINE_AA)
        for index, point in enumerate(landmarks):
            x, y = int(point.x * width), int(point.y * height)
            cv2.circle(frame, (x, y), 7, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), 10, (245, 245, 245), 2, cv2.LINE_AA)
            cv2.putText(frame, str(index), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        section_y = 120 + hand_index * 270
        cv2.putText(panel, f"{label}  |  {confidence:.0%}", (24, section_y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
        extension_values = []
        for row, (finger, (a, joint, c)) in enumerate(FINGER_JOINTS.items()):
            if finger == "Thumb":
                measurement = thumb_extension_ratio(landmarks)
                extended, filtered_value = stabilizer.update_finger(raw_side, finger, measurement, 0.82, 0.66)
                extension_values.append(float(np.clip((filtered_value - 0.35) / 0.7, 0.0, 1.0)))
            else:
                angle = finger_bend_angle(landmarks, a, joint, c)
                extended, filtered_value = stabilizer.update_finger(raw_side, finger, angle)
                extension_values.append(float(np.clip((filtered_value - 90.0) / 90.0, 0.0, 1.0)))
            state = "Extended" if extended else "Bent"
            state_color = (95, 225, 155) if extended else (115, 170, 245)
            cv2.putText(panel, f"{finger:<7} {state}", (32, section_y + 42 + row * 31), cv2.FONT_HERSHEY_SIMPLEX, 0.52, state_color, 1, cv2.LINE_AA)
        openness = int(round(np.mean(extension_values) * 100))
        cv2.putText(panel, f"Hand openness: {openness}%", (24, section_y + 220), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 238, 242), 1, cv2.LINE_AA)
        if hand_index < len(hand_landmarks) - 1:
            cv2.line(panel, (24, section_y + 240), (PANEL_WIDTH - 24, section_y + 240), (75, 82, 96), 1)

    cv2.putText(panel, "Estimate only; camera angle can affect it", (24, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (165, 175, 190), 1, cv2.LINE_AA)

    return np.hstack((frame, panel))


def main() -> None:
    parser = argparse.ArgumentParser(description="Track hand landmarks and display handedness.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    args = parser.parse_args()

    ensure_model()
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}. Try another index with --camera.")

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
            start = time.monotonic()
            stabilizer = TrackingStabilizer()
            while True:
                ok, frame = camera.read()
                if not ok:
                    print("Could not read a frame from the camera.")
                    break

                # Mirror the preview for a natural selfie view. Labels identify anatomical hands.
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = int((time.monotonic() - start) * 1000)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                display = draw_landmark_panel(frame, result, stabilizer)
                cv2.putText(display, "q: quit", (20, display.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow(WINDOW_NAME, display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

