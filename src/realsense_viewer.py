"""Minimal desktop viewer for an Intel RealSense D435 on macOS."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np
import pyrealsense2 as rs


WINDOW_TITLE = "Intel RealSense D435 — Color and Depth"
RESET_WAIT_SECONDS = 8


def add_label(image: np.ndarray, label: str) -> None:
    cv2.rectangle(image, (0, 0), (image.shape[1], 42), (20, 20, 20), -1)
    cv2.putText(
        image,
        label,
        (14, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def stream_viewer() -> int:
    try:
        context = rs.context()
        devices = list(context.query_devices())
    except RuntimeError as exc:
        print(f"Unable to enumerate the camera: {exc}", file=sys.stderr)
        return 1
    if not devices:
        print(
            "No RealSense camera is accessible. On macOS, run this viewer with sudo.",
            file=sys.stderr,
        )
        return 1

    device = devices[0]
    name = device.get_info(rs.camera_info.name)
    serial = device.get_info(rs.camera_info.serial_number)
    try:
        usb_type = device.get_info(rs.camera_info.usb_type_descriptor)
    except RuntimeError:
        usb_type = "unknown"
    fps = 15 if usb_type.startswith("2") else 30
    print(f"Opening {name} (serial {serial})")
    print(f"USB connection: {usb_type}; streaming at 640x480, {fps} FPS")
    print("Press Q or Escape in the viewer window to close it.")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, fps)
    colorizer = rs.colorizer()
    # Equal resolutions do not imply matching pixels: use sensor calibration
    # to map depth onto the color image's pixel grid.
    align = rs.align(rs.stream.color)
    started = False

    try:
        pipeline.start(config)
        started = True
        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_TITLE, 1280, 480)

        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data()).copy()
            depth_image = np.asanyarray(
                colorizer.colorize(depth_frame).get_data()
            ).copy()

            add_label(color_image, "COLOR")
            add_label(depth_image, "DEPTH ALIGNED TO COLOR")
            combined = np.hstack((color_image, depth_image))
            cv2.imshow(WINDOW_TITLE, combined)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                break
    except RuntimeError as exc:
        print(f"Unable to start the camera: {exc}", file=sys.stderr)
        return 1
    finally:
        if started:
            pipeline.stop()
        cv2.destroyAllWindows()

    return 0


def reset_camera() -> int:
    """Reset in a short-lived process so invalid USB handles cannot be reused."""
    try:
        context = rs.context()
        devices = list(context.query_devices())
    except RuntimeError as exc:
        print(f"Unable to enumerate the camera for reset: {exc}", file=sys.stderr)
        sys.stderr.flush()
        os._exit(3)
    if not devices:
        print("No RealSense camera was found for reset.", file=sys.stderr)
        sys.stderr.flush()
        os._exit(2)

    device = devices[0]
    name = device.get_info(rs.camera_info.name)
    serial = device.get_info(rs.camera_info.serial_number)
    print(f"Resetting {name} (serial {serial}) before opening it...")
    try:
        device.hardware_reset()
    except RuntimeError as exc:
        print(f"Camera reset failed: {exc}", file=sys.stderr)
        sys.stderr.flush()
        os._exit(3)

    # hardware_reset() invalidates every handle to the device. Exit immediately
    # rather than letting native SDK destructors touch those invalid handles.
    sys.stdout.flush()
    os._exit(0)


def launch_with_recovery() -> int:
    script = str(Path(__file__).resolve())
    child_command = [sys.executable, script]

    # With matched SDK and firmware, avoid resetting a healthy camera.
    viewer = subprocess.run(child_command + ["--stream-only"], check=False)
    if viewer.returncode == 0:
        return 0

    print("Camera start failed. Resetting and retrying once...")
    reset = subprocess.run(child_command + ["--reset-only"], check=False)
    if reset.returncode == 0:
        print(f"Waiting {RESET_WAIT_SECONDS} seconds for USB reconnection...")
        time.sleep(RESET_WAIT_SECONDS)
    else:
        print(
            "The automatic reset did not complete; trying one final start anyway.",
            file=sys.stderr,
        )

    viewer = subprocess.run(child_command + ["--stream-only"], check=False)
    if viewer.returncode == 0:
        return 0

    print(
        "The camera could not recover automatically. Unplug it for five seconds, "
        "reconnect it, and try again.",
        file=sys.stderr,
    )
    return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--reset-only", action="store_true")
    parser.add_argument("--stream-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.reset_only:
        return reset_camera()
    if args.stream_only:
        return stream_viewer()
    return launch_with_recovery()


if __name__ == "__main__":
    raise SystemExit(main())
