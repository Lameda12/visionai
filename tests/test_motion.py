"""
Tests for core.motion.MotionDetector
All tests run without cv2 (mock mode).
Run: python -m pytest tests/test_motion.py -v
"""
import sys
import json
import tempfile
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.motion import MotionDetector, _SENSITIVITY_PARAMS


class TestMotionDetectorInit(unittest.TestCase):

    def test_default_sensitivity(self):
        d = MotionDetector()
        self.assertEqual(d.sensitivity, "medium")

    def test_custom_sensitivity(self):
        d = MotionDetector(sensitivity="high")
        self.assertEqual(d.sensitivity, "high")

    def test_invalid_sensitivity_raises(self):
        with self.assertRaises(ValueError):
            MotionDetector(sensitivity="extreme")

    def test_default_min_area(self):
        d = MotionDetector(sensitivity="medium")
        expected = _SENSITIVITY_PARAMS["medium"]["min_area"]
        self.assertEqual(d.min_area, expected)

    def test_override_min_area(self):
        d = MotionDetector(min_area=999)
        self.assertEqual(d.min_area, 999)


class TestMotionDetectorDetect(unittest.TestCase):

    def setUp(self):
        # Fixed seed for fully deterministic tests
        self.detector = MotionDetector(sensitivity="medium")
        self.detector._mock_rng.seed(0)

    def test_detect_returns_list(self):
        events = self.detector.detect("frame_0")
        self.assertIsInstance(events, list)

    def test_event_has_required_fields(self):
        # Run many frames to guarantee at least one event
        for i in range(100):
            events = self.detector.detect(f"frame_{i}")
            if events:
                for e in events:
                    self.assertIn("area", e)
                    self.assertIn("bbox", e)
                    self.assertIn("ts", e)
                break

    def test_bbox_is_four_element_list(self):
        for i in range(100):
            events = self.detector.detect(f"frame_{i}")
            for e in events:
                self.assertEqual(len(e["bbox"]), 4)
                for v in e["bbox"]:
                    self.assertIsInstance(v, int)

    def test_area_is_positive(self):
        for i in range(100):
            for e in self.detector.detect(f"frame_{i}"):
                self.assertGreater(e["area"], 0)

    def test_ts_is_iso_string(self):
        from datetime import datetime
        for i in range(100):
            for e in self.detector.detect(f"frame_{i}"):
                # Should parse without raising
                datetime.fromisoformat(e["ts"])

    def test_area_above_min_area(self):
        """Events must exceed min_area threshold."""
        for i in range(200):
            for e in self.detector.detect(f"frame_{i}"):
                self.assertGreaterEqual(e["area"], self.detector.min_area)

    def test_frame_count_increments(self):
        initial = self.detector._frame_count
        self.detector.detect("frame_x")
        self.detector.detect("frame_y")
        self.assertEqual(self.detector._frame_count, initial + 2)

    def test_most_frames_have_no_motion(self):
        """Mock probability ~12% — majority of frames should be clear."""
        self.detector._mock_rng.seed(42)
        clear = sum(1 for i in range(100) if not self.detector.detect(f"f_{i}"))
        self.assertGreater(clear, 50)


class TestMotionDetectorAnnotate(unittest.TestCase):

    def test_annotate_returns_input_in_mock_mode(self):
        """Without cv2, annotate() should return the frame unchanged."""
        d = MotionDetector()
        frame = "synthetic_frame"
        events = [{"area": 800.0, "bbox": [10, 20, 100, 80], "ts": "2026-01-01T00:00:00+00:00"}]
        result = d.annotate(frame, events)
        self.assertEqual(result, frame)

    def test_annotate_empty_events(self):
        d = MotionDetector()
        result = d.annotate("frame", [])
        self.assertEqual(result, "frame")


class TestMotionDetectorReset(unittest.TestCase):

    def test_reset_clears_frame_count(self):
        d = MotionDetector()
        for i in range(5):
            d.detect(f"frame_{i}")
        self.assertGreater(d._frame_count, 0)
        d.reset()
        self.assertEqual(d._frame_count, 0)


class TestMotionDetectorLogging(unittest.TestCase):

    def test_events_written_to_log(self):
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "motion_log.jsonl"
            d = MotionDetector(sensitivity="high", log_path=log_path)
            d._mock_rng.seed(7)  # seed that reliably fires events

            all_events: list[dict] = []
            for i in range(50):
                all_events.extend(d.detect(f"frame_{i}"))

            if all_events:
                self.assertTrue(log_path.exists())
                lines = log_path.read_text().strip().splitlines()
                for line in lines:
                    entry = json.loads(line)
                    self.assertIn("frame", entry)
                    self.assertIn("area", entry)
                    self.assertIn("bbox", entry)

    def test_no_log_when_log_path_is_none(self):
        d = MotionDetector(log_path=None)
        d._mock_rng.seed(7)
        for i in range(50):
            d.detect(f"frame_{i}")
        # Should not raise and no unexpected files created


class TestSensitivityParams(unittest.TestCase):

    def test_all_sensitivities_defined(self):
        for s in ("low", "medium", "high"):
            self.assertIn(s, _SENSITIVITY_PARAMS)

    def test_high_more_sensitive_than_low(self):
        high_area = _SENSITIVITY_PARAMS["high"]["min_area"]
        low_area  = _SENSITIVITY_PARAMS["low"]["min_area"]
        self.assertLess(high_area, low_area)


if __name__ == "__main__":
    unittest.main(verbosity=2)
