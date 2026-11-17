"""Local face enrollment and recognition demo for the capstone team.

Enroll one consenting person at a time with --enroll NAME, then run with --run.
Face crops and the trained recognizer remain in the local face_data directory.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

DATA_DIR = Path(__file__).with_name("face_data")
CASCADE_PATH = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
SAMPLES_PER_PERSON = 30
FACE_SIZE = (160, 160)


def create_detector() -> cv2.CascadeClassifier:
    detector = cv2.CascadeClassifier(str(CASCADE_PATH))
    if detector.empty():
        raise SystemExit("Could not load OpenCV's face detector.")
    return detector


def find_faces(frame: np.ndarray, detector: cv2.CascadeClassifier) -> list[tuple[int, int, int, int]]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))


def normalize_face(frame: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x, y, width, height = box
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = gray[y : y + height, x : x + width]
    return cv2.resize(face, FACE_SIZE)


def enroll(name: str, camera_index: int) -> None:
    safe_name = "_".join(name.strip().split())
    if not safe_name or any(char in safe_name for char in "/\\"):
        raise SystemExit("Enter a valid name without path separators.")

    print(f"This will save face images for {name} locally in: {DATA_DIR / safe_name}")
    print("Only continue if this person agrees to face enrollment.")
    if input("Type YES to continue: ").strip() != "YES":
        raise SystemExit("Enrollment cancelled.")

    person_dir = DATA_DIR / safe_name
    person_dir.mkdir(parents=True, exist_ok=True)
    detector = create_detector()
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        raise SystemExit(f"Could not open camera {camera_index}.")

    saved = 0
    last_capture = 0.0
    print("Look at the camera and move your head slightly. Press q to cancel.")
    try:
        while saved < SAMPLES_PER_PERSON:
            ok, frame = camera.read()
            if not ok:
                print("Could not read a frame from the camera.")
                break
            faces = find_faces(frame, detector)
            original_frame = frame.copy()
            for x, y, width, height in faces:
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 220, 0), 2)

            now = time.monotonic()
            if len(faces) == 1 and now - last_capture >= 0.2:
                crop = normalize_face(original_frame, tuple(int(v) for v in faces[0]))
                cv2.imwrite(str(person_dir / f"face_{saved:03d}.png"), crop)
                saved += 1
                last_capture = now

            cv2.putText(frame, f"Enrolling {name}: {saved}/{SAMPLES_PER_PERSON}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2)
            cv2.putText(frame, "Keep only one face in view | q: cancel", (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.imshow("Capstone - Face Enrollment", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()

    if saved < SAMPLES_PER_PERSON:
        for image in person_dir.glob("face_*.png"):
            image.unlink()
        person_dir.rmdir()
        raise SystemExit("Enrollment cancelled or incomplete; partial samples were removed.")
    print(f"Enrollment complete for {name}. Samples are stored locally in {person_dir}.")


def load_recognizer() -> tuple[Any, dict[int, str]]:
    if not hasattr(cv2, "face") or not hasattr(cv2.face, "LBPHFaceRecognizer_create"):
        raise SystemExit("OpenCV face module is missing. Reinstall dependencies with: python -m pip install -r requirements.txt")

    images: list[np.ndarray] = []
    labels: list[int] = []
    names: dict[int, str] = {}
    for label, person_dir in enumerate(sorted(path for path in DATA_DIR.iterdir() if path.is_dir()) if DATA_DIR.exists() else []):
        samples = sorted(person_dir.glob("face_*.png"))
        if not samples:
            continue
        names[label] = person_dir.name.replace("_", " ")
        for sample in samples:
            image = cv2.imread(str(sample), cv2.IMREAD_GRAYSCALE)
            if image is not None:
                images.append(cv2.resize(image, FACE_SIZE))
                labels.append(label)

    if not images:
        raise SystemExit("No enrolled faces found. Enroll team members first with --enroll NAME.")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(images, np.array(labels, dtype=np.int32))
    return recognizer, names


def run_recognition(camera_index: int, threshold: float) -> None:
    recognizer, names = load_recognizer()
    detector = create_detector()
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        raise SystemExit(f"Could not open camera {camera_index}.")
    print("Face recognition is running locally. Press q to quit.")
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Could not read a frame from the camera.")
                break
            for x, y, width, height in find_faces(frame, detector):
                face = normalize_face(frame, (x, y, width, height))
                label_id, distance = recognizer.predict(face)
                name = names.get(label_id, "Unknown") if distance <= threshold else "Unknown"
                color = (0, 220, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
                cv2.putText(frame, f"{name} ({distance:.0f})", (x, max(25, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, "Local face recognition | q: quit", (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.imshow("Capstone - Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Enroll and recognize consenting capstone team members locally.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--enroll", metavar="NAME", help="Enroll one person after their consent")
    mode.add_argument("--run", action="store_true", help="Recognize enrolled people")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--threshold", type=float, default=70.0, help="LBPH distance threshold; lower is stricter")
    args = parser.parse_args()
    if args.enroll:
        enroll(args.enroll, args.camera)
    else:
        run_recognition(args.camera, args.threshold)


if __name__ == "__main__":
    main()
