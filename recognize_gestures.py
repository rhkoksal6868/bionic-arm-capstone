"""Run trained gesture recognition on one hand in the live camera feed."""
from __future__ import annotations

import argparse
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np

from collect_gestures import feature_values
from hand_side import HAND_CONNECTIONS, MODEL_PATH, ensure_model

MODEL_FILE = Path(__file__).with_name("gesture_model.joblib")
DISPLAY_NAMES = {
    "open_hand": "Open hand", "fist": "Fist", "thumbs_up": "Thumbs up",
    "peace": "Peace sign", "index_point": "Pointing",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Recognize hand gestures from the trained model.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--confidence", type=float, default=0.65, help="Minimum model score to accept a prediction")
    args = parser.parse_args()
    if not MODEL_FILE.exists():
        raise SystemExit("Trained model not found. Run train_gesture_model.py first.")

    bundle = joblib.load(MODEL_FILE)
    model = bundle["model"]
    classes = bundle["classes"]
    ensure_model()
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}.")

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    predictions: deque[str] = deque(maxlen=7)
    connections = HAND_CONNECTIONS
    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
            start = time.monotonic()
            while True:
                ok, frame = camera.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect_for_video(image, int((time.monotonic() - start) * 1000))

                label = "No hand"
                score = 0.0
                score_line = "No hand found"
                if result.hand_landmarks:
                    points = result.hand_landmarks[0]
                    height, width = frame.shape[:2]
                    for first, second in connections:
                        a = (int(points[first].x * width), int(points[first].y * height))
                        b = (int(points[second].x * width), int(points[second].y * height))
                        cv2.line(frame, a, b, (60, 210, 140), 3, cv2.LINE_AA)
                    for point in points:
                        cv2.circle(frame, (int(point.x * width), int(point.y * height)), 6, (250, 250, 250), -1, cv2.LINE_AA)

                    probabilities = model.predict_proba([feature_values(points)])[0]
                    best = int(probabilities.argmax())
                    score = float(probabilities[best])
                    current = classes[best] if score >= args.confidence else "Unknown"
                    top_two = np.argsort(probabilities)[::-1][:2]
                    score_line = " | ".join(
                        f"{DISPLAY_NAMES.get(classes[index], classes[index])}: {probabilities[index]:.0%}"
                        for index in top_two
                    )
                    predictions.append(current)
                    most_common, votes = Counter(predictions).most_common(1)[0]
                    label = most_common if votes >= 5 else "Stabilizing..."
                else:
                    predictions.clear()

                color = (70, 220, 140) if label not in ("Unknown", "No hand", "Stabilizing...") else (70, 170, 255)
                display_name = DISPLAY_NAMES.get(label, label)
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 90), (28, 32, 40), -1)
                cv2.putText(frame, f"Gesture: {display_name}", (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
                cv2.putText(frame, f"Top: {score_line} | q: quit", (24, 73), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 225, 235), 1, cv2.LINE_AA)
                cv2.imshow("Gesture Recognition", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
