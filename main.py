"""Color-marker hand tracking prototype for the bionic arm project.

Run with: python main.py --camera 0
Press q to quit. Press c to capture open-hand calibration and v for closed-hand.
"""
from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

try:
    import serial
except ImportError:
    serial = None


@dataclass
class Settings:
    # Tune these HSV thresholds for the actual marker and room lighting.
    lower_hsv: tuple[int, int, int] = (35, 70, 60)
    upper_hsv: tuple[int, int, int] = (90, 255, 255)
    min_area: float = 120.0
    smoothing_alpha: float = 0.25


def largest_marker_center(hsv: np.ndarray, settings: Settings) -> tuple[Optional[tuple[int, int]], np.ndarray]:
    mask = cv2.inRange(hsv, np.array(settings.lower_hsv), np.array(settings.upper_hsv))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < settings.min_area:
        return None, mask
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return None, mask
    center = (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))
    return center, mask


def map_to_servo(value: float, opened: float, closed: float) -> int:
    """Map normalized image distance to a conservative 0..180 degree command."""
    if math.isclose(opened, closed):
        return 90
    ratio = float(np.clip((value - opened) / (closed - opened), 0.0, 1.0))
    return int(round(20 + ratio * 140))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--serial", help="Optional Arduino port, e.g. /dev/cu.usbmodemXXXX")
    parser.add_argument("--baud", type=int, default=115200)
    args = parser.parse_args()

    settings = Settings()
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}. Try another index with --camera.")

    link = None
    if args.serial:
        if serial is None:
            raise SystemExit("Install pyserial for serial communication: python -m pip install -r requirements.txt")
        link = serial.Serial(args.serial, args.baud, timeout=0.1, write_timeout=0.2)
        time.sleep(2)  # Arduino Uno resets when its serial port opens.

    open_distance: Optional[float] = None
    closed_distance: Optional[float] = None
    smoothed: Optional[float] = None
    last_sent: Optional[int] = None
    last_send_time = 0.0
    print("With the marker visible, hold your hand open and press 'c'; close it and press 'v'. Press 'q' to quit.")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Could not read a frame from the camera.")
                break
            frame = cv2.flip(frame, 1)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            center, mask = largest_marker_center(hsv, settings)
            command: Optional[int] = None

            if center:
                cv2.circle(frame, center, 9, (0, 255, 0), 2)
                if open_distance is not None:
                    distance = float(center[1])  # Prototype signal; replace with fingertip/reference distance.
                    smoothed = distance if smoothed is None else settings.smoothing_alpha * distance + (1 - settings.smoothing_alpha) * smoothed
                    if closed_distance is not None:
                        command = map_to_servo(smoothed, open_distance, closed_distance)
                        cv2.putText(frame, f"Servo: {command} deg", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2)

            if command is None:
                label = "Marker invalid / calibrate (c=open, v=closed)"
                color = (0, 0, 255)
            else:
                label = f"Command: {command} deg | c=open, v=closed"
                color = (0, 220, 0)
            cv2.putText(frame, label, (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            now = time.monotonic()
            if link and command is not None and (command != last_sent or now - last_send_time > 0.1):
                link.write(f"F0:{command}\n".encode("ascii"))
                last_sent, last_send_time = command, now

            cv2.imshow("Bionic Arm - Color Marker Prototype", frame)
            cv2.imshow("HSV Threshold Mask", mask)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key in (ord("c"), ord("v")) and center:
                # Calibration uses vertical marker position in this first prototype.
                if key == ord("c"):
                    open_distance = float(center[1])
                    print(f"Open-hand calibration: {open_distance:.1f}")
                else:
                    closed_distance = float(center[1])
                    print(f"Closed-hand calibration: {closed_distance:.1f}")
    finally:
        camera.release()
        if link:
            link.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
