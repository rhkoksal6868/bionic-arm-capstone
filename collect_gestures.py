"""Capture labeled hand landmark samples for simple gesture classification.

Show one hand, hold a gesture, and press its number key to save one sample.
"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from hand_side import HAND_CONNECTIONS, MODEL_PATH, ensure_model

GESTURES = {
    ord("1"): "open_hand",
    ord("2"): "fist",
    ord("3"): "thumbs_up",
    ord("4"): "peace",
    ord("5"): "index_point",
}
CSV_PATH = Path(__file__).with_name("gesture_data.csv")


def feature_values(landmarks: object) -> list[float]:
    """Return wrist-relative landmarks normalized by palm size."""
    wrist = landmarks[0]
    middle_mcp = landmarks[9]
    scale = float(np.linalg.norm([middle_mcp.x - wrist.x, middle_mcp.y - wrist.y, middle_mcp.z - wrist.z]))
    scale = max(scale, 1e-6)
    values = []
    for point in landmarks:
        values.extend(((point.x - wrist.x) / scale, (point.y - wrist.y) / scale, (point.z - wrist.z) / scale))
    return values


def main() -> None:
    ensure_model()
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise SystemExit("Could not open camera 0. Try changing the camera index in collect_gestures.py.")

    feature_names = [f"{axis}{index:02d}" for index in range(21) for axis in ("x", "y", "z")]
    if not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0:
        with CSV_PATH.open("w", newline="") as file:
            csv.writer(file).writerow(["label", "handedness", *feature_names])

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    counts = {name: 0 for name in GESTURES.values()}
    connections = HAND_CONNECTIONS
    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
            start_time = cv2.getTickCount() / cv2.getTickFrequency()
            while True:
                ok, frame = camera.read()
                if not ok:
                    print("Could not read a frame from the camera.")
                    break
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp = int((cv2.getTickCount() / cv2.getTickFrequency() - start_time) * 1000)
                result = landmarker.detect_for_video(image, timestamp)

                if result.hand_landmarks:
                    points = result.hand_landmarks[0]
                    height, width = frame.shape[:2]
                    for first, second in connections:
                        a = (int(points[first].x * width), int(points[first].y * height))
                        b = (int(points[second].x * width), int(points[second].y * height))
                        cv2.line(frame, a, b, (70, 210, 140), 2, cv2.LINE_AA)
                    for point in points:
                        cv2.circle(frame, (int(point.x * width), int(point.y * height)), 4, (245, 245, 245), -1, cv2.LINE_AA)
                    cv2.putText(frame, "Hand detected - press 1 to 5 to save a sample", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (60, 230, 145), 2)
                else:
                    cv2.putText(frame, "Show one hand to the camera", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 130, 255), 2)

                cv2.putText(frame, "1 Open | 2 Fist | 3 Thumbs up | 4 Peace | 5 Point", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                cv2.putText(frame, "Saved: " + "  ".join(f"{key}={value}" for key, value in counts.items()), (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow("Gesture Data Collection", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key in GESTURES and result.hand_landmarks:
                    handedness = result.handedness[0][0].category_name if result.handedness else "Unknown"
                    row = [GESTURES[key], handedness, *feature_values(result.hand_landmarks[0])]
                    with CSV_PATH.open("a", newline="") as file:
                        csv.writer(file).writerow(row)
                    counts[GESTURES[key]] += 1
                    print(f"Saved {GESTURES[key]} sample ({counts[GESTURES[key]]} this session)")
    finally:
        camera.release()
        cv2.destroyAllWindows()

    print(f"Dataset saved to {CSV_PATH}")


if __name__ == "__main__":
    main()
