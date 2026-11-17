"""Train and evaluate a compact gesture classifier from landmark CSV files."""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent
# The current project workflow trains from the downloaded, labeled image set.
# Camera-collected samples can be included explicitly with --csv if desired.
DEFAULT_FILES = (ROOT / "gesture_image_data.csv",)
MODEL_PATH = ROOT / "gesture_model.joblib"
FEATURE_NAMES = [f"{axis}{index:02d}" for index in range(21) for axis in ("x", "y", "z")]


def augment_pose(values: np.ndarray) -> list[np.ndarray]:
    """Add mirror and small in-plane camera-roll variants of one hand pose."""
    points = values.reshape(21, 3).copy()
    variants = []
    for mirrored in (False, True):
        base = points.copy()
        if mirrored:
            base[:, 0] *= -1
        for angle in (-12.0, 0.0, 12.0):
            radians = np.deg2rad(angle)
            cosine, sine = np.cos(radians), np.sin(radians)
            rotated = base.copy()
            rotated[:, 0] = cosine * base[:, 0] - sine * base[:, 1]
            rotated[:, 1] = sine * base[:, 0] + cosine * base[:, 1]
            variants.append(rotated.reshape(-1))
    return variants


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and hold out a test set for gesture recognition.")
    parser.add_argument("--csv", nargs="*", type=Path, default=list(DEFAULT_FILES), help="Landmark CSV files to combine")
    args = parser.parse_args()

    rows = []
    for csv_path in args.csv:
        if not csv_path.exists():
            continue
        with csv_path.open(newline="") as file:
            for row in csv.DictReader(file):
                if all(row.get(name) not in (None, "") for name in FEATURE_NAMES):
                    rows.append((row["label"], [float(row[name]) for name in FEATURE_NAMES]))
    if not rows:
        raise SystemExit("No landmark samples found. Prepare image features or collect camera samples first.")

    counts = Counter(label for label, _ in rows)
    print("Samples by class:", dict(sorted(counts.items())))
    too_small = {label: count for label, count in counts.items() if count < 5}
    if too_small:
        raise SystemExit(f"At least 5 samples per class are needed for a useful held-out test. Add more: {too_small}")

    labels = np.array([label for label, _ in rows])
    features = np.asarray([values for _, values in rows], dtype=np.float32)
    train_x, test_x, train_y, test_y = train_test_split(
        features, labels, test_size=0.25, random_state=42, stratify=labels
    )
    # Split original samples first, then augment only training data so
    # transformed copies cannot leak into the held-out evaluation set.
    augmented_x = []
    augmented_y = []
    for values, label in zip(train_x, train_y):
        for variant in augment_pose(values):
            augmented_x.append(variant)
            augmented_y.append(label)
    augmented_x = np.asarray(augmented_x, dtype=np.float32)
    augmented_y = np.asarray(augmented_y)
    print(f"Training samples after mirror/rotation augmentation: {len(augmented_y)}")

    model = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42),
    )
    model.fit(augmented_x, augmented_y)
    predictions = model.predict(test_x)
    class_names = sorted(counts)
    print(f"Held-out accuracy: {accuracy_score(test_y, predictions):.3f}")
    print("Held-out classification report:")
    print(classification_report(test_y, predictions, labels=class_names, zero_division=0))
    print("Confusion matrix (rows=true, columns=predicted):")
    print(class_names)
    print(confusion_matrix(test_y, predictions, labels=class_names))

    joblib.dump({"model": model, "feature_names": FEATURE_NAMES, "classes": class_names}, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")
    print("Note: random image split may overestimate performance when several images show the same people or session.")


if __name__ == "__main__":
    main()
