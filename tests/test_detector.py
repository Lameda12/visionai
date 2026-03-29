"""
Tests for core.detector.ObjectDetector
All tests run without cv2 or ultralytics (mock mode).
Run: python -m pytest tests/test_detector.py -v
"""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.detector import ObjectDetector, COCO_LABELS


class TestObjectDetectorMockMode(unittest.TestCase):
    """ObjectDetector falls back to mock data when deps are unavailable."""

    def setUp(self):
        self.detector = ObjectDetector(model="yolov8n", conf=0.5)

    # ------------------------------------------------------------------
    # Result structure
    # ------------------------------------------------------------------

    def test_run_returns_dict(self):
        result = self.detector.run("fake_image.jpg")
        self.assertIsInstance(result, dict)

    def test_result_has_required_keys(self):
        result = self.detector.run("fake_image.jpg")
        for key in ("source", "model", "count", "detections", "inference_ms"):
            self.assertIn(key, result, msg=f"Missing key: {key!r}")

    def test_count_matches_detections_length(self):
        result = self.detector.run("fake_image.jpg")
        self.assertEqual(result["count"], len(result["detections"]))

    def test_detections_are_list_of_dicts(self):
        result = self.detector.run("fake_image.jpg")
        self.assertIsInstance(result["detections"], list)
        for det in result["detections"]:
            self.assertIsInstance(det, dict)

    def test_detection_has_required_fields(self):
        result = self.detector.run("fake_image.jpg")
        for det in result["detections"]:
            for field in ("label", "confidence", "box", "class_id"):
                self.assertIn(field, det)

    def test_confidence_in_range(self):
        result = self.detector.run("fake_image.jpg")
        for det in result["detections"]:
            self.assertGreaterEqual(det["confidence"], 0.0)
            self.assertLessEqual(det["confidence"], 1.0)

    def test_box_is_four_ints(self):
        result = self.detector.run("fake_image.jpg")
        for det in result["detections"]:
            box = det["box"]
            self.assertEqual(len(box), 4)
            for coord in box:
                self.assertIsInstance(coord, int)

    def test_label_is_known_coco_class(self):
        result = self.detector.run("fake_image.jpg")
        for det in result["detections"]:
            self.assertIn(det["label"], COCO_LABELS)

    def test_inference_ms_is_positive_float(self):
        result = self.detector.run("fake_image.jpg")
        self.assertGreater(result["inference_ms"], 0)
        self.assertIsInstance(result["inference_ms"], float)

    def test_model_name_preserved(self):
        result = self.detector.run("fake_image.jpg")
        self.assertEqual(result["model"], "yolov8n")

    # ------------------------------------------------------------------
    # Determinism
    # ------------------------------------------------------------------

    def test_same_source_gives_same_result(self):
        r1 = self.detector.run("sample.jpg")
        r2 = self.detector.run("sample.jpg")
        self.assertEqual(r1["count"], r2["count"])
        self.assertEqual(r1["detections"], r2["detections"])

    def test_different_source_may_give_different_result(self):
        r1 = self.detector.run("image_a.jpg")
        r2 = self.detector.run("image_b.jpg")
        # They might differ — we just check both are valid
        self.assertIsInstance(r1["count"], int)
        self.assertIsInstance(r2["count"], int)

    # ------------------------------------------------------------------
    # Confidence threshold
    # ------------------------------------------------------------------

    def test_confidence_above_threshold(self):
        threshold = 0.5
        det = ObjectDetector(conf=threshold)
        result = det.run("test.jpg")
        for d in result["detections"]:
            self.assertGreaterEqual(d["confidence"], threshold * 0.9)  # mock applies ~threshold

    # ------------------------------------------------------------------
    # save() stub
    # ------------------------------------------------------------------

    def test_save_does_not_raise_in_mock_mode(self):
        result = self.detector.run("fake.jpg")
        # save() writes JSON even in mock mode if image is None
        import tempfile, os
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out.jpg"
            self.detector.save(result, out)
            json_path = out.with_suffix(".json")
            self.assertTrue(json_path.exists())


class TestObjectDetectorInit(unittest.TestCase):

    def test_default_conf(self):
        d = ObjectDetector()
        self.assertEqual(d.conf, 0.5)

    def test_custom_model_name(self):
        d = ObjectDetector(model="yolov8m")
        result = d.run("x.jpg")
        self.assertEqual(result["model"], "yolov8m")

    def test_count_is_non_negative(self):
        d = ObjectDetector()
        result = d.run("anything.jpg")
        self.assertGreaterEqual(result["count"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
