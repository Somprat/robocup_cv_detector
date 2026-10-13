"""Reusable six-class NUbots ONNX detector using OpenCV DNN on CPU.

Input: uint8 BGR image (height, width, 3).
Output: JSON-compatible detection dictionaries with original-image pixel boxes.
Create one detector and reuse it for successive camera frames. Instances are
not thread-safe: the network and last-run statistics are mutable.
"""

import ast
import math
from pathlib import Path
import time

import cv2
import numpy as np
import onnx


class YoloOnnxDetector:
    CLASS_NAMES = {0: "ball", 1: "goalpost", 2: "robot", 3: "L-Intersection",
                   4: "T-Intersection", 5: "X-Intersection"}
    COLORS = [(0, 220, 255), (255, 180, 0), (0, 200, 0),
              (220, 0, 220), (0, 130, 255), (255, 220, 80)]

    def __init__(self, model_path, confidence=0.2, nms=0.5, nms_mode="per-class"):
        """Validate the model and options, then load the network once."""
        self.model_path = Path(model_path)
        self.confidence = self._validate_threshold(confidence)
        self.nms = self._validate_threshold(nms)
        if nms_mode not in ("per-class", "class-agnostic"):
            raise ValueError("nms_mode must be per-class or class-agnostic")
        self.nms_mode = nms_mode
        model = onnx.load(self.model_path, load_external_data=False)
        metadata = {item.key: item.value for item in model.metadata_props}
        if ast.literal_eval(metadata.get("names", "None")) != self.CLASS_NAMES:
            raise ValueError("This detector requires the six-class NUbots model")
        shape = [dim.dim_value for dim in model.graph.input[0].type.tensor_type.shape.dim]
        if shape != [1, 3, 640, 640]:
            raise ValueError(f"Unexpected model input shape: {shape}")
        self.input_name = model.graph.input[0].name
        self.names = self.CLASS_NAMES.copy()
        self.net = cv2.dnn.readNetFromONNX(str(self.model_path))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.last_output = None
        self.last_forward_ms = None
        self.last_candidate_count = 0
        self.last_retained_count = 0

    @staticmethod
    def _validate_threshold(value):
        """Convert a threshold to a finite float between zero and one."""
        number = float(value)
        if not math.isfinite(number) or not 0 <= number <= 1:
            raise ValueError("threshold must be finite and in [0, 1]")
        return number

    @staticmethod
    def preprocess(image):
        """Validate BGR input, pad bottom/right, and make a normalized RGB blob.

        Returns the blob and square side length used to undo resizing.
        """
        if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
                or image.ndim != 3 or image.shape[2] != 3
                or image.shape[0] == 0 or image.shape[1] == 0):
            raise ValueError("image must be a nonempty uint8 BGR array of shape (H, W, 3)")
        height, width = image.shape[:2]
        side = max(width, height)
        square = np.zeros((side, side, 3), dtype=np.uint8)
        square[:height, :width] = image
        blob = cv2.dnn.blobFromImage(square, 1.0 / 255.0, (640, 640),
                                     swapRB=True, crop=False)
        return blob, side

    def detect(self, image):
        """Run one frame and return detections sorted by descending confidence.

        bbox_xywh is [left, top, width, height] in original-image pixels.
        An empty list means no usable detections. Last-run output, inference
        time, and candidate counts are available on the instance.
        """
        blob, side = self.preprocess(image)
        self.net.setInput(blob, self.input_name)
        start = time.perf_counter()
        output = self.net.forward()
        elapsed_ms = (time.perf_counter() - start) * 1000
        detections = self.postprocess(output, image.shape[:2], side)
        self.last_output = output
        self.last_forward_ms = elapsed_ms
        return detections

    def postprocess(self, output, image_shape, side):
        """Decode scores/boxes, suppress duplicates, and clip to image bounds.

        image_shape is (height, width); side comes from preprocess().
        No separate objectness score is assumed in this model's output.
        """
        if output.shape != (1, 10, 8400) or not np.isfinite(output).all():
            raise ValueError(f"Invalid model output: shape={output.shape}, expected finite [1,10,8400]")
        height, width = image_shape
        rows = output[0].T
        class_ids = rows[:, 4:].argmax(axis=1)
        scores = rows[:, 4:].max(axis=1)
        valid = (scores > self.confidence) & (rows[:, 2] > 0) & (rows[:, 3] > 0)
        candidates = np.flatnonzero(valid)
        boxes = np.column_stack((rows[:, 0] - rows[:, 2] / 2,
                                 rows[:, 1] - rows[:, 3] / 2,
                                 rows[:, 2], rows[:, 3])) * (side / 640.0)
        groups = [candidates] if self.nms_mode == "class-agnostic" else [
            candidates[class_ids[candidates] == class_id] for class_id in self.names
        ]
        retained = []
        for group in groups:
            if not len(group):
                continue
            selected = cv2.dnn.NMSBoxes(boxes[group].tolist(), scores[group].tolist(),
                                        self.confidence, self.nms)
            retained.extend(int(group[int(i)]) for i in np.asarray(selected).flatten())
        self.last_candidate_count = len(candidates)
        self.last_retained_count = len(retained)
        detections = []
        for index in sorted(retained, key=lambda i: float(scores[i]), reverse=True):
            x, y, w, h = boxes[index]
            left = max(0, min(width, math.floor(x)))
            top = max(0, min(height, math.floor(y)))
            right = max(0, min(width, math.ceil(x + w)))
            bottom = max(0, min(height, math.ceil(y + h)))
            if right <= left or bottom <= top:
                continue
            class_id = int(class_ids[index])
            detections.append({"candidate_index": index, "model_class_id": class_id,
                               "model_label": self.names[class_id],
                               "confidence": float(scores[index]),
                               "bbox_xywh": [left, top, right - left, bottom - top]})
        return detections

    def annotate(self, image, detections):
        """Return a copy of the image with detection boxes and labels drawn."""
        annotated = image.copy()
        for detection in detections:
            left, top, width, height = detection["bbox_xywh"]
            color = self.COLORS[detection["model_class_id"]]
            cv2.rectangle(annotated, (left, top),
                          (left + width - 1, top + height - 1), color, 2)
            label = f'{detection["model_label"]} {detection["confidence"]:.2f}'
            origin = (left, max(18, top - 5))
            cv2.putText(annotated, label, origin, cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(annotated, label, origin, cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, color, 1, cv2.LINE_AA)
        return annotated
