"""Live demo checks using simulated capture and the real SDK's deprojection."""

from contextlib import ExitStack, redirect_stderr
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pyrealsense2 as rs

from depth_estimator import DepthEstimator
import realsense_yolo_demo as demo


class DemoTests(unittest.TestCase):
    def setUp(self):
        self.color = np.zeros((12, 12, 3), dtype=np.uint8)
        self.depth = np.full((12, 12), 2000, dtype=np.uint16)
        intrinsics = rs.intrinsics()
        intrinsics.width = intrinsics.height = 12
        intrinsics.fx = intrinsics.fy = 100
        intrinsics.ppx = intrinsics.ppy = 6
        intrinsics.model = rs.distortion.none
        intrinsics.coeffs = [0] * 5
        profile = Mock()
        profile.as_video_stream_profile.return_value.get_intrinsics.return_value = intrinsics
        self.depth_frame = Mock(profile=profile)
        self.depth_frame.get_data.return_value = self.depth
        self.depth_frame.get_units.return_value = 0.001
        self.color_frame = Mock()
        self.color_frame.get_data.return_value = self.color
        for frame in (self.depth_frame, self.color_frame):
            frame.get_frame_number.return_value = 42
            frame.get_timestamp.return_value = 1400.0
            frame.get_frame_timestamp_domain.return_value = rs.timestamp_domain.hardware_clock
            frame.get_width.return_value = frame.get_height.return_value = 12
        self.frames = Mock()
        self.frames.get_depth_frame.return_value = self.depth_frame
        self.frames.get_color_frame.return_value = self.color_frame
        self.align = Mock()
        self.align.process.return_value = self.frames
        self.prediction = {"model_class_id": 0, "model_label": "ball",
                           "confidence": 0.9, "bbox_xywh": [4, 2, 4, 4]}
        self.detector = Mock(last_forward_ms=5.0)
        self.detector.detect.return_value = [self.prediction]
        self.estimator = DepthEstimator()

    def test_deprojection_and_json(self):
        _, depth, color, detections = demo.process_frame(
            self.frames, self.align, self.detector, self.estimator)
        result = detections[0]
        self.assertEqual(result["depth"]["sample_pixel_uv"], [6, 4])
        np.testing.assert_allclose(result["xyz_m"], [0, -0.04, 2], atol=1e-6)
        self.assertEqual(result["position_status"], "valid")
        self.assertNotIn("depth", self.prediction)
        record = demo.make_record("camera", color, depth, detections, 5, 8)
        self.assertEqual(record["coordinate_frame"], "color_camera")
        json.dumps(record, allow_nan=False)

    def test_missing_depth_keeps_detection(self):
        self.depth[:] = 0
        result = demo.process_frame(self.frames, self.align, self.detector, self.estimator)[3][0]
        self.assertIsNone(result["xyz_m"])
        self.assertEqual(result["model_label"], "ball")
        self.assertEqual(result["depth"]["status"], "insufficient_valid_depth")

    def test_partial_frames_are_skipped(self):
        self.frames.get_depth_frame.return_value = None
        self.assertIsNone(demo.process_frame(self.frames, self.align, self.detector, self.estimator))
        self.align.process.assert_not_called()
        self.detector.detect.assert_not_called()

    def test_alignment_dimension_error(self):
        self.depth_frame.get_data.return_value = np.zeros((10, 12), dtype=np.uint16)
        with self.assertRaisesRegex(ValueError, "dimensions"):
            demo.process_frame(self.frames, self.align, self.detector, self.estimator)

    def test_unsupported_distortion_rejected_before_native_call(self):
        intrinsics = self.depth_frame.profile.as_video_stream_profile().get_intrinsics()
        intrinsics.model = rs.distortion.modified_brown_conrady
        with self.assertRaisesRegex(ValueError, "distortion"):
            demo.process_frame(self.frames, self.align, self.detector, self.estimator)

    def test_camera_selection_and_usb_fallback(self):
        def device(serial):
            def info(key):
                if key == rs.camera_info.usb_type_descriptor:
                    raise RuntimeError("unavailable")
                return serial if key == rs.camera_info.serial_number else "D435"
            return SimpleNamespace(get_info=info)
        context = Mock()
        context.query_devices.return_value = [device("one"), device("two")]
        self.assertEqual(demo.select_camera(context, "two"), ("two", "D435", "unknown"))
        with self.assertRaisesRegex(RuntimeError, "No matching"):
            demo.select_camera(context, "missing")

    def test_cli_rejects_invalid_settings(self):
        with redirect_stderr(io.StringIO()):
            for options in (["--confidence", "nan"], ["--max-frames", "0"],
                            ["--min-depth", "20"], ["--patch-radius", "-1"]):
                with self.subTest(options=options), self.assertRaises(SystemExit):
                    demo.parse_args(options)
        self.assertTrue(demo.parse_args([]).model.is_absolute())

    def camera_mocks(self, stack, result):
        pipeline = Mock()
        for attribute in ("context", "config", "align"):
            stack.enter_context(patch.object(rs, attribute))
        stack.enter_context(patch.object(rs, "pipeline", return_value=pipeline))
        stack.enter_context(patch.object(demo, "select_camera", return_value=("test", "D435", "3.2")))
        loader = stack.enter_context(patch.object(demo, "YoloOnnxDetector", return_value=self.detector))
        process = stack.enter_context(patch.object(demo, "process_frame", side_effect=result))
        stack.enter_context(redirect_stderr(io.StringIO()))
        return pipeline, loader, process

    def test_headless_output_limit_and_cleanup(self):
        result = demo.process_frame(self.frames, self.align, self.detector, self.estimator)
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            output = Path(directory) / "detections.jsonl"
            pipeline, loader, process = self.camera_mocks(stack, [None, result, result])
            gui = stack.enter_context(patch.object(demo.cv2, "namedWindow"))
            args = demo.parse_args(["--headless", "--max-frames", "2", "--jsonl", str(output)])
            self.assertEqual(demo.run_camera(args), 0)
            records = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["detections"][0]["xyz_m"][2], 2)
            loader.assert_called_once()
            self.assertEqual(process.call_count, 3)
            gui.assert_not_called()
            pipeline.stop.assert_called_once()

    def test_cleanup_after_processing_failure(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            pipeline, _, _ = self.camera_mocks(stack, ValueError("processing failed"))
            args = demo.parse_args(["--headless", "--jsonl", str(Path(directory) / "out.jsonl")])
            with self.assertRaisesRegex(ValueError, "processing failed"):
                demo.run_camera(args)
            pipeline.stop.assert_called_once()


if __name__ == "__main__":
    unittest.main()
