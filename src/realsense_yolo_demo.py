"""Live RealSense color/depth alignment, NUbots ONNX detection, and estimated XYZ.

Run on the computer with the camera attached. Requires numpy, onnx,
pyrealsense2, and opencv-python for GUI (headless OpenCV works with --headless).
XYZ estimates use median depth near a bounding-box center: visible surface
estimates, not object centers. Coordinates are in meters, relative to the
color camera, with X right, Y down, Z forward. Background at the box center
can produce background coordinates. Camera/world-pose calibration is separate.

One JSON record is emitted for each processed frame, to stdout or --jsonl.
Camera messages go to stderr. Press Q/Escape in the window or Ctrl+C to stop.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from depth_estimator import DepthEstimator
from yolo_onnx_detector import YoloOnnxDetector


WINDOW_TITLE = "RealSense YOLO — XYZ in meters"


def parse_args(argv=None):
    """Read and validate model, camera, depth sampling, display, and logging options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, nargs="?",
                        default=Path(__file__).resolve().parents[1] / "yolo.onnx")
    parser.add_argument("--confidence", type=float, default=0.2)
    parser.add_argument("--nms", type=float, default=0.5)
    parser.add_argument("--nms-mode", choices=["per-class", "class-agnostic"], default="per-class")
    parser.add_argument("--serial", help="Select a camera serial number; default is the first device")
    parser.add_argument("--fps", type=int, choices=[15, 30],
                        help="Default: 15 for USB 2, otherwise 30 (640x480 streams)")
    parser.add_argument("--patch-radius", type=int, default=3)
    parser.add_argument("--min-depth", type=float, default=0.1, help="Minimum accepted depth in meters")
    parser.add_argument("--max-depth", type=float, default=10.0, help="Maximum accepted depth in meters")
    parser.add_argument("--min-valid-fraction", type=float, default=0.5)
    parser.add_argument("--min-valid-pixels", type=int, default=3)
    parser.add_argument("--jsonl", type=Path, help="Write frame records to this file (overwrites existing file)")
    parser.add_argument("--headless", action="store_true", help="Disable the display; stop with Ctrl+C")
    parser.add_argument("--max-frames", type=int, help="Stop after this many processed frame pairs")
    args = parser.parse_args(argv)
    if any(not math.isfinite(value) or not 0 <= value <= 1
           for value in (args.confidence, args.nms)):
        parser.error("confidence and nms must be finite and in [0, 1]")
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("max-frames must be positive")
    try:
        DepthEstimator(args.patch_radius, args.min_depth, args.max_depth,
                       args.min_valid_fraction, args.min_valid_pixels)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def select_camera(context, serial=None):
    """Find a camera and read its identity and USB connection type."""
    import pyrealsense2 as rs

    devices = list(context.query_devices())
    if serial:
        devices = [device for device in devices
                   if device.get_info(rs.camera_info.serial_number) == serial]
    if not devices:
        raise RuntimeError("No matching RealSense camera is accessible. Connect it to this "
                           "computer and verify access with realsense_viewer.py first.")
    device = devices[0]
    try:
        usb_type = device.get_info(rs.camera_info.usb_type_descriptor)
    except RuntimeError:
        usb_type = "unknown"
    return (device.get_info(rs.camera_info.serial_number),
            device.get_info(rs.camera_info.name), usb_type)


def process_frame(frames, align, detector, estimator):
    """Align one pair, detect boxes, estimate depth, and deproject usable points.

    Returns (color_image, aligned_depth_frame, color_frame, detections), or
    None for an incomplete pair. Reads depth data/scale/intrinsics once per
    frame. Detection dictionaries are copied before adding depth and XYZ.
    """
    import pyrealsense2 as rs

    # A partial frameset cannot supply depth-to-color calibration for align.
    if not frames.get_depth_frame() or not frames.get_color_frame():
        return None
    aligned = align.process(frames)
    depth_frame, color_frame = aligned.get_depth_frame(), aligned.get_color_frame()
    if not depth_frame or not color_frame:
        return None
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())
    if color_image.shape[:2] != depth_image.shape:
        raise ValueError("Aligned depth and color dimensions do not match")
    intrinsics = depth_frame.profile.as_video_stream_profile().get_intrinsics()
    if (intrinsics.width, intrinsics.height) != (color_image.shape[1], color_image.shape[0]):
        raise ValueError("Aligned depth intrinsics do not match the image dimensions")
    # The native SDK rejects this model for deprojection; avoid its assertion.
    if intrinsics.model == rs.distortion.modified_brown_conrady:
        raise ValueError("Aligned intrinsics use a distortion model unsupported for deprojection")
    depth_scale = depth_frame.get_units()
    detections = []
    for prediction in detector.detect(color_image):
        detection = dict(prediction)
        depth = estimator.estimate(depth_image, detection["bbox_xywh"], depth_scale=depth_scale)
        detection.update(depth=depth, xyz_m=None, position_status="unavailable_depth")
        if depth["depth_m"] is not None:
            xyz = rs.rs2_deproject_pixel_to_point(
                intrinsics, depth["sample_pixel_uv"], depth["depth_m"])
            if np.isfinite(xyz).all():
                detection.update(xyz_m=[float(value) for value in xyz], position_status="valid")
            else:
                detection["position_status"] = "nonfinite_xyz"
        detections.append(detection)
    return color_image, depth_frame, color_frame, detections


def draw_preview(color_image, depth_frame, detections, detector, colorizer, forward_ms):
    """Build a side-by-side color/depth preview with labels and sampling marks."""
    annotated = detector.annotate(color_image, detections)
    depth_preview = np.asanyarray(colorizer.colorize(depth_frame).get_data()).copy()
    for detection in detections:
        xyz = detection["xyz_m"]
        text = "depth unavailable" if xyz is None else (
            f"XYZ {xyz[0]:+.2f}, {xyz[1]:+.2f}, {xyz[2]:.2f} m")
        x, y, _, h = detection["bbox_xywh"]
        origin = (x, min(annotated.shape[0] - 5, y + h + 16))
        cv2.putText(annotated, text, origin, cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(annotated, text, origin, cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA)
        pixel = detection["depth"]["sample_pixel_uv"]
        if pixel is not None:
            for panel in (annotated, depth_preview):
                cv2.circle(panel, tuple(pixel), 3, (255, 255, 255), -1)
    for panel, text in ((annotated, f"COLOR | inference {forward_ms:.1f} ms"),
                        (depth_preview, "DEPTH ALIGNED TO COLOR")):
        cv2.rectangle(panel, (0, 0), (panel.shape[1] - 1, 25), (20, 20, 20), -1)
        cv2.putText(panel, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return np.hstack((annotated, depth_preview))


def make_record(serial, color_frame, depth_frame, detections, forward_ms, processing_ms):
    """Package frame identities, camera-relative results, and timing for JSON."""
    return {
        "camera_serial": serial,
        "color_frame_number": color_frame.get_frame_number(),
        "depth_frame_number": depth_frame.get_frame_number(),
        "color_timestamp_ms": color_frame.get_timestamp(),
        "depth_timestamp_ms": depth_frame.get_timestamp(),
        "color_timestamp_domain": str(color_frame.get_frame_timestamp_domain()),
        "depth_timestamp_domain": str(depth_frame.get_frame_timestamp_domain()),
        "coordinate_frame": "color_camera", "units": "meters",
        "position_method": "bbox_center_ray_median_depth",
        "image_size_wh": [color_frame.get_width(), color_frame.get_height()],
        "forward_ms": forward_ms, "processing_ms": processing_ms,
        "detections": detections,
    }


def run_camera(args):
    """Load objects once, stream pairs, emit JSON, display results, and clean up."""
    import pyrealsense2 as rs

    detector = YoloOnnxDetector(args.model, args.confidence, args.nms, args.nms_mode)
    estimator = DepthEstimator(args.patch_radius, args.min_depth, args.max_depth,
                               args.min_valid_fraction, args.min_valid_pixels)
    context = rs.context()
    serial, name, usb_type = select_camera(context, args.serial)
    fps = args.fps or (15 if usb_type.startswith("2") else 30)
    pipeline = rs.pipeline(context)
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, fps)
    align = rs.align(rs.stream.color)
    colorizer = None if args.headless else rs.colorizer()
    started, window_open = False, False
    try:
        pipeline.start(config)
        started = True
        print(f"Opening {name} ({serial}), USB {usb_type}, 640x480 at {fps} FPS. "
              "XYZ in meters: X right, Y down, Z forward. Q/Escape or Ctrl+C to stop.",
              file=sys.stderr)
        if not args.headless:
            cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
            window_open = True
            cv2.resizeWindow(WINDOW_TITLE, 1280, 480)
        if args.jsonl:
            args.jsonl.parent.mkdir(parents=True, exist_ok=True)
        destination = args.jsonl.open("w") if args.jsonl else nullcontext(sys.stdout)
        with destination as output:
            processed = 0
            while args.max_frames is None or processed < args.max_frames:
                frames = pipeline.wait_for_frames()
                start = time.perf_counter()
                result = process_frame(frames, align, detector, estimator)
                if result is None:
                    continue
                processing_ms = (time.perf_counter() - start) * 1000
                color_image, depth_frame, color_frame, detections = result
                record = make_record(serial, color_frame, depth_frame, detections,
                                     detector.last_forward_ms, processing_ms)
                output.write(json.dumps(record, allow_nan=False) + "\n")
                output.flush()
                processed += 1
                if not args.headless:
                    preview = draw_preview(color_image, depth_frame, detections,
                                           detector, colorizer, detector.last_forward_ms)
                    cv2.imshow(WINDOW_TITLE, preview)
                    if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                        break
                    if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                        break
    finally:
        try:
            if started:
                pipeline.stop()
        finally:
            if window_open:
                cv2.destroyAllWindows()
    return 0


def main(argv=None):
    """Run the command, handle Ctrl+C, and report errors with a failure exit code."""
    args = parse_args(argv)
    try:
        return run_camera(args)
    except KeyboardInterrupt:
        return 0
    except (RuntimeError, ValueError, OSError, ImportError, cv2.error) as exc:
        print(f"RealSense YOLO demo failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
