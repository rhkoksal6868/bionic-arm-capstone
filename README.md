# Bionic Arm Capstone Project

## Left/right hand detection

This demo uses MediaPipe Hand Landmarker to display whether the visible hand is the left or right hand and its confidence score. It does not send commands to an Arduino.

1. Open this folder in VS Code.
2. In the VS Code terminal, install the packages:
   ```bash
   python3 -m pip install -r requirements.txt
   ```
3. Run the hand-side demo:
   ```bash
   python3 hand_side.py --camera 0
   ```
4. Press `q` to quit. If the camera does not open, try `--camera 1`.

The first run downloads the MediaPipe hand model beside `hand_side.py`, so an internet connection is needed once.

## Color-marker prototype

`main.py` is the separate OpenCV color-marker prototype. Run it with `python3 main.py --camera 0`. Press `c` with the hand open and `v` with it closed to calibrate. The current marker coordinate is a first prototype signal; it is not yet the final fingertip-to-reference distance measurement.


## Local face recognition

This optional demo enrolls consenting team members and displays their names locally. It stores face images in the `face_data/` folder on this computer and does not upload them. Ask each person before enrolling them. Delete their folder from `face_data/` to remove their saved samples.

Enroll each team member separately (replace the example name):

```bash
python3 face_identity.py --enroll Hakan
python3 face_identity.py --enroll Tuana
```

Each person must type `YES` in the terminal to consent, then look at the camera until 30 samples are collected. After enrollment, start recognition:

```bash
python3 face_identity.py --run
```

Unrecognized faces are shown as `Unknown`. Face recognition can make mistakes; use this as a project demonstration, not as a security or access-control system.

## Full-screen hand tracking display

Run `python3 hand_side.py --camera 0`. The window opens full-screen, with the camera view on the left and a readable point list with normalized x/y coordinates on the right. Press `q` to quit.

MediaPipe's standard Hand Landmarker model returns 21 predefined landmarks per detected hand. This number is part of the model output and cannot be increased with a setting. More points would require a different/custom-trained landmark model or a separate tracking method, such as additional colored glove markers detected with OpenCV.

The right-side panel now shows both detected hands, their estimated side/confidence, a per-finger `Extended`/`Bent` estimate, and a rough hand-openness percentage. The 21 landmark coordinates are no longer listed because they are low-level tracking measurements; they are useful to the program for deriving finger bend and later servo commands, but not usually useful as a live user display. Finger states are a simple 2D estimate and may vary with camera angle; calibrate and validate before using them to control motors.

## Smoothing hand tracking

The live display applies temporal smoothing to landmark positions, uses separate open/close angle thresholds to prevent boundary flicker, and requires a state change to persist for three frames. This adds a small delay while reducing visual jitter. Camera angle and partial occlusion can still affect finger estimates.

Thumb extension uses a separate estimate: the thumb tip's distance from the palm center, normalized by palm width. This captures the thumb moving away from or across the palm, which a single joint angle can miss. The initial open/closed thresholds are approximate and may need tuning for the team's camera angle and pose.

## Collect gesture data

Start `python3 collect_gestures.py`. Show one hand, hold the gesture, and press its number to save one labeled sample: `1` open hand, `2` fist, `3` thumbs up, `4` peace sign, `5` index-pointing. Repeat each gesture many times with small changes in hand position and orientation. Aim for at least 50-100 samples per gesture. Press `q` to quit. Samples append to `gesture_data.csv` as wrist-relative, palm-size-normalized coordinates for the 21 landmarks; no camera images are saved. This script only collects data; model training and live gesture classification are a later step.


## Gesture recognition workflow

The pipeline has four steps: prepare labeled image folders as landmark rows, train and evaluate a classifier, then run the classifier on live camera input. Camera and image samples use the same wrist-relative, palm-size-normalized 21-point feature format.

### 1. Prepare image data

Download and extract [HandGesture-5G](https://data.mendeley.com/datasets/wj25crzdbm/1) (CC BY 4.0). It has Open Palm, Fist, Pointing Up, L-Sign, and V-Sign folders. The pipeline maps Open Palm, Fist, Pointing Up, and V-Sign to `open_hand`, `fist`, `index_point`, and `peace`; L-Sign is skipped. This dataset does not contain `thumbs_up`.

For the missing thumbs-up class, the [HandGesture-Recognizer repository](https://github.com/shimaazizi/HandGesture-Recognizer) lists a 200-image dataset with Open Palm, Fist, Thumbs Up, and Peace Sign and an MIT repository license. Its authors report 50% accuracy for their own image classifier, so use it as supplementary data and validate on your camera. Confirm the license covers the image files before redistributing them.

Put the extracted class folders under one directory, for example:

```text
gesture_images/
  HandGesture-5G/
    Open Palm/
    Fist/
    Pointing Up/
    V-Sign/
  HandGesture-Recognizer/
    dataset/
      OpenPalm/
      Fist/
      ThumbsUp/
      PeaceSign/
```

Then extract hand landmarks and save a CSV (only the first visible hand is used; images are not copied):

```bash
python3 -m pip install -r requirements.txt
python3 prepare_image_dataset.py gesture_images
```

On macOS, run these commands in the same local terminal session that can open the camera and OpenCV windows. If MediaPipe exits with `DrishtiMetalHelper` / `service_ is unavailable`, grant Terminal or VS Code camera access in **System Settings → Privacy & Security → Camera**, fully quit and reopen the app, then retry. The project virtual environment `.venv` is configured for Python 3.13; activate it with `source .venv/bin/activate` before running the steps below.

The script tries both the original and horizontally flipped image at a lower hand-detection threshold, then skips unreadable files and images where MediaPipe still finds no hand. It writes `gesture_image_data.csv`.

### 2. Train the classifier

```bash
python3 train_gesture_model.py
```

By default, it trains only on the downloaded `gesture_image_data.csv` and does not include camera-collected `gesture_data.csv`. It requires at least five examples per class, uses a stratified 75/25 train/test split, augments training samples with mirrored and slightly rotated hand poses, and prints accuracy, per-class precision/recall/F1, and a confusion matrix. Keep different images of the same people or capture session out of both split parts when possible; a random image split can overstate generalization. The trained file is `gesture_model.joblib`. To explicitly combine image and camera samples later, pass `--csv gesture_image_data.csv gesture_data.csv`.

### 3. Check held-out performance

Read the classification report and confusion matrix before using live recognition. If a gesture has low recall or is often confused with another class, gather more diverse examples for that gesture and retrain. Do not treat high test accuracy as proof that the model works with your webcam setup.

### 4. Try live recognition

```bash
python3 recognize_gestures.py --camera 0
```

The camera demo uses a 7-frame vote and shows a gesture only when at least 5 frames agree. It displays the two highest class scores to help diagnose confusion. Scores below 65% are displayed as `Unknown`. Change the cutoff with `--confidence 0.7` if it accepts incorrect gestures too often; the SVM score is an estimate, not a calibrated probability. Press `q` to quit.

This pipeline recognizes static hand poses. It does not yet recognize movement sequences over time or control motors.
