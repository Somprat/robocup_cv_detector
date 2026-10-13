"""Run the six-class NUbots ONNX model on a BGR image using OpenCV DNN CPU.

Dependencies: numpy, onnx, opencv-python-headless.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
from yolo_onnx_detector import YoloOnnxDetector


def threshold(value):
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("threshold must be finite and in [0, 1]")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confidence", type=threshold, default=0.2)
    parser.add_argument("--nms", type=threshold, default=0.5)
    parser.add_argument("--nms-mode", choices=["per-class", "class-agnostic"], default="per-class")
    args = parser.parse_args()

    detector = YoloOnnxDetector(args.model, args.confidence, args.nms, args.nms_mode)

    source = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if source is None:
        raise ValueError(f"Cannot read image: {args.image}")
    height, width = source.shape[:2]
    detections = detector.detect(source)
    annotated = detector.annotate(source, detections)
    output = detector.last_output
    rows = output[0].T

    report = {
        "model": str(args.model.resolve()), "image": str(args.image.resolve()),
        "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "backend": "OpenCV DNN CPU", "opencv_version": cv2.__version__,
        "image_size_wh": [width, height], "input_shape": [1, 3, 640, 640],
        "preprocessing": "BGR to RGB; top-left black square padding; resize 640x640; float32 /255",
        "output_shape": list(output.shape), "output_all_finite": True,
        "box_value_range": [float(rows[:, :4].min()), float(rows[:, :4].max())],
        "class_score_range": [float(rows[:, 4:].min()), float(rows[:, 4:].max())],
        "confidence_threshold": args.confidence, "nms_threshold": args.nms,
        "nms_mode": args.nms_mode, "candidates_above_threshold": detector.last_candidate_count,
        "retained_after_nms": detector.last_retained_count, "detections": detections,
        "forward_ms_first_run": detector.last_forward_ms,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output_dir / "annotated.jpg"), annotated):
        raise RuntimeError("Failed to save annotated image")
    np.save(args.output_dir / "raw_output.npy", output)
    (args.output_dir / "detections.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
