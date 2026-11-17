"""Extract MediaPipe hand landmarks from class-organized gesture images."""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import cv2
import mediapipe as mp

from collect_gestures import feature_values
from hand_side import MODEL_PATH, ensure_model

OUTPUT = Path(__file__).with_name("gesture_image_data.csv")
FEATURE_NAMES = [f"{axis}{index:02d}" for index in range(21) for axis in ("x", "y", "z")]
LABELS = {
    "open_hand": "open_hand", "open_palm": "open_hand", "open_palm_hand": "open_hand", "open_palm_open_hand": "open_hand", "openpalm": "open_hand", "palm": "open_hand",
    "fist": "fist", "closed_fist": "fist",
    "thumbs_up": "thumbs_up", "thumbsup": "thumbs_up", "like": "thumbs_up",
    "peace": "peace", "peace_sign": "peace", "peacesign": "peace", "v_sign": "peace", "victory": "peace",
    "index_point": "index_point", "pointing_up": "index_point", "one": "index_point",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def class_for(path: Path, root: Path) -> str | None:
    for part in reversed(path.relative_to(root).parts[:-1]):
        normalized = "_".join("".join(c.lower() if c.isalnum() else " " for c in part).split())
        if normalized in LABELS:
            return LABELS[normalized]
    normalized_stem = "".join(c.lower() if c.isalnum() else " " for c in path.stem)
    first_token = normalized_stem.split()[0] if normalized_stem.split() else ""
    if first_token in LABELS:
        return LABELS[first_token]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert labeled gesture images to landmark features.")
    parser.add_argument("image_root", type=Path, help="Folder containing one subfolder per gesture class")
    args = parser.parse_args()
    root = args.image_root.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Image folder does not exist: {root}")

    files = [(path, class_for(path, root)) for path in root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES]
    files = [(path, label) for path, label in files if label]
    if not files:
        raise SystemExit("No supported gesture images found in recognized class folders.")

    ensure_model()
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.25,
    )
    counts: Counter[str] = Counter()
    skipped = 0
    with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker, OUTPUT.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["label", "handedness", "source", *FEATURE_NAMES])
        for number, (path, label) in enumerate(files, start=1):
            frame = cv2.imread(str(path))
            if frame is None:
                skipped += 1
                continue
            height, width = frame.shape[:2]
            scale = min(1.0, 960.0 / max(height, width))
            if scale < 1.0:
                frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
            # Public photos may already be mirrored or show either hand. Try
            # both orientations so a miss caused by handedness/orientation
            # doesn't discard an otherwise useful labeled image.
            result = None
            for candidate in (frame, cv2.flip(frame, 1)):
                rgb = cv2.cvtColor(candidate, cv2.COLOR_BGR2RGB)
                candidate_result = landmarker.detect(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                )
                if candidate_result.hand_landmarks:
                    result = candidate_result
                    break
            if result is None or not result.hand_landmarks:
                skipped += 1
                continue
            side = result.handedness[0][0].category_name if result.handedness else "Unknown"
            writer.writerow([label, side, str(path.relative_to(root)), *feature_values(result.hand_landmarks[0])])
            counts[label] += 1
            if number % 100 == 0:
                print(f"Processed {number}/{len(files)} images...")

    print(f"Saved landmark features to {OUTPUT}")
    print("Samples by class:", dict(sorted(counts.items())))
    print(f"Skipped unreadable images or images without a detected hand: {skipped}")


if __name__ == "__main__":
    main()
