"""Estimate metric depth near the center of a detected object's bounding box.

Supply numerical depth aligned to color and boxes in original color-image
pixels, such as YoloOnnxDetector's bbox_xywh. Never use a colorized depth
preview or coordinates from the resized 640x640 model input.

This estimates visible surface depth along the camera's forward axis, not
Euclidean distance or an object's physical center. The box center can land on
background, particularly for thin objects. Passing the validity checks means
there are enough usable readings; it does not prove they belong to the object.
Only NumPy is required. No camera capture or XYZ conversion happens here.
"""

from __future__ import annotations

import math
from numbers import Integral

import numpy as np


class DepthEstimator:
    def __init__(self, patch_radius=3, min_depth_m=0.1, max_depth_m=10.0,
                 min_valid_fraction=0.5, min_valid_pixels=3):
        """Configure patch size, accepted depth range, and required coverage.

        Radius 3 gives a 7x7 patch, clipped to the detection box and image.
        Defaults require at least three valid pixels and 50% valid coverage.
        Set min_valid_pixels=1 to allow estimates for single-pixel boxes.
        """
        if (isinstance(patch_radius, bool) or not isinstance(patch_radius, Integral)
                or patch_radius < 0):
            raise ValueError("patch_radius must be a nonnegative integer")
        if (isinstance(min_valid_pixels, bool) or not isinstance(min_valid_pixels, Integral)
                or min_valid_pixels < 1):
            raise ValueError("min_valid_pixels must be a positive integer")
        min_depth_m, max_depth_m = float(min_depth_m), float(max_depth_m)
        min_valid_fraction = float(min_valid_fraction)
        if (not math.isfinite(min_depth_m) or not math.isfinite(max_depth_m)
                or not 0 < min_depth_m < max_depth_m):
            raise ValueError("depth limits must be finite and 0 < min_depth_m < max_depth_m")
        if not math.isfinite(min_valid_fraction) or not 0 < min_valid_fraction <= 1:
            raise ValueError("min_valid_fraction must be finite and in (0, 1]")
        self.patch_radius = int(patch_radius)
        self.min_depth_m = min_depth_m
        self.max_depth_m = max_depth_m
        self.min_valid_fraction = min_valid_fraction
        self.min_valid_pixels = int(min_valid_pixels)

    @staticmethod
    def _validate_image(depth_image):
        """Reject empty, multichannel, or nonnumeric depth arrays."""
        if (not isinstance(depth_image, np.ndarray) or depth_image.ndim != 2
                or depth_image.size == 0 or depth_image.dtype.kind not in "uif"):
            raise ValueError("depth_image must be a nonempty 2D real numeric NumPy array")

    def _patch_bounds(self, image_shape, bbox_xywh):
        """Validate/clip a box and locate its center patch within the image.

        Returns (center_uv, patch_xywh), or None if the box misses the image.
        Fractional box edges are rounded outward; the center is computed from
        the visible clipped box. Box widths/heights must be positive.
        """
        box = np.asarray(bbox_xywh, dtype=np.float64)
        if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= 0):
            raise ValueError("bbox_xywh must contain finite x, y, positive width, positive height")
        x, y, w, h = (float(value) for value in box)
        if not math.isfinite(x + w) or not math.isfinite(y + h):
            raise ValueError("bbox edges must be finite")
        height, width = image_shape
        left, top = max(0, math.floor(x)), max(0, math.floor(y))
        right, bottom = min(width, math.ceil(x + w)), min(height, math.ceil(y + h))
        if right <= left or bottom <= top:
            return None
        u, v = (left + right) // 2, (top + bottom) // 2
        radius = self.patch_radius
        patch_left, patch_top = max(left, u - radius), max(top, v - radius)
        patch_right, patch_bottom = min(right, u + radius + 1), min(bottom, v + radius + 1)
        return [u, v], [patch_left, patch_top,
                        patch_right - patch_left, patch_bottom - patch_top]

    def estimate(self, depth_image, bbox_xywh, *, depth_scale):
        """Return a JSON-compatible depth estimate for one original-image box.

        depth_scale is mandatory: meters per input unit (use 1.0 for an array
        already in meters). Raw RealSense Z16 values need the device's actual
        scale. Zero, negative, nonfinite, and out-of-range samples are removed.
        Median depth limits the effect of isolated outliers; depth_mad_m is
        the median absolute deviation of accepted samples, not an accuracy
        guarantee. Missing data returns depth_m=None with a status reason.
        Invalid arguments raise ValueError rather than indicating missing data.
        """
        self._validate_image(depth_image)
        depth_scale = float(depth_scale)
        if not math.isfinite(depth_scale) or depth_scale <= 0:
            raise ValueError("depth_scale must be finite and positive")
        bounds = self._patch_bounds(depth_image.shape, bbox_xywh)
        result = {"status": "outside_image", "depth_m": None,
                  "sample_pixel_uv": None, "patch_xywh": None,
                  "valid_pixels": 0, "total_pixels": 0,
                  "valid_fraction": 0.0, "depth_mad_m": None}
        if bounds is None:
            return result
        center, patch_box = bounds
        left, top, width, height = patch_box
        patch = depth_image[top:top + height, left:left + width]
        with np.errstate(over="ignore", invalid="ignore"):
            metric = patch.astype(np.float64) * depth_scale
        valid = metric[np.isfinite(metric) & (metric > 0)
                       & (metric >= self.min_depth_m) & (metric <= self.max_depth_m)]
        result.update(sample_pixel_uv=center, patch_xywh=patch_box,
                      valid_pixels=int(valid.size), total_pixels=int(patch.size),
                      valid_fraction=float(valid.size / patch.size))
        required = max(self.min_valid_pixels,
                       math.ceil(patch.size * self.min_valid_fraction))
        if valid.size < required:
            result["status"] = "insufficient_valid_depth"
            return result
        median = float(np.median(valid))
        result.update(status="valid", depth_m=median,
                      depth_mad_m=float(np.median(np.abs(valid - median))))
        return result

    def estimate_frame(self, aligned_depth_frame, bbox_xywh):
        """Read an aligned RealSense frame and its scale, then estimate depth.

        The caller must align the frame to color first. This adapter accepts a
        pyrealsense2.depth_frame without importing the SDK into this module.
        """
        if aligned_depth_frame is None or not aligned_depth_frame:
            raise ValueError("aligned_depth_frame must be a valid depth frame")
        return self.estimate(np.asanyarray(aligned_depth_frame.get_data()),
                             bbox_xywh, depth_scale=aligned_depth_frame.get_units())
