"""
Tests for core.face_analyzer.FaceAnalyzer
All tests run without deepface or cv2 (mock mode).
Run: python -m pytest tests/test_face.py -v
"""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.face_analyzer import FaceAnalyzer, _EMOTIONS, _GENDERS, _RACES


class TestFaceAnalyzerMockMode(unittest.TestCase):

    def setUp(self):
        self.analyzer = FaceAnalyzer(backend="opencv", enforce_detection=False, analyze=True)

    # ------------------------------------------------------------------
    # run() result structure
    # ------------------------------------------------------------------

    def test_run_returns_dict(self):
        result = self.analyzer.run("portrait.jpg")
        self.assertIsInstance(result, dict)

    def test_result_keys(self):
        result = self.analyzer.run("portrait.jpg")
        for key in ("source", "count", "backend", "faces", "inference_ms"):
            self.assertIn(key, result)

    def test_count_matches_faces_length(self):
        result = self.analyzer.run("portrait.jpg")
        self.assertEqual(result["count"], len(result["faces"]))

    def test_faces_is_list(self):
        result = self.analyzer.run("portrait.jpg")
        self.assertIsInstance(result["faces"], list)

    def test_backend_preserved(self):
        result = self.analyzer.run("portrait.jpg")
        self.assertEqual(result["backend"], "opencv")

    def test_inference_ms_positive(self):
        result = self.analyzer.run("portrait.jpg")
        self.assertGreater(result["inference_ms"], 0)

    # ------------------------------------------------------------------
    # Face entry structure (with analyze=True)
    # ------------------------------------------------------------------

    def test_face_has_region(self):
        result = self.analyzer.run("portrait.jpg")
        for face in result["faces"]:
            self.assertIn("region", face)
            region = face["region"]
            for k in ("x", "y", "w", "h"):
                self.assertIn(k, region)

    def test_face_has_emotion_fields(self):
        result = self.analyzer.run("portrait.jpg")
        for face in result["faces"]:
            self.assertIn("dominant_emotion", face)
            self.assertIn("emotion", face)
            self.assertIn(face["dominant_emotion"], _EMOTIONS)

    def test_face_has_age(self):
        result = self.analyzer.run("portrait.jpg")
        for face in result["faces"]:
            self.assertIn("age", face)
            self.assertGreater(face["age"], 0)

    def test_face_has_gender_fields(self):
        result = self.analyzer.run("portrait.jpg")
        for face in result["faces"]:
            self.assertIn("dominant_gender", face)
            self.assertIn(face["dominant_gender"], _GENDERS)

    def test_face_has_race_fields(self):
        result = self.analyzer.run("portrait.jpg")
        for face in result["faces"]:
            self.assertIn("dominant_race", face)
            self.assertIn(face["dominant_race"], _RACES)

    def test_emotion_scores_dict(self):
        result = self.analyzer.run("portrait.jpg")
        for face in result["faces"]:
            scores = face.get("emotion", {})
            self.assertIsInstance(scores, dict)
            # dominant emotion should have a high score
            dom = face["dominant_emotion"]
            self.assertGreater(scores.get(dom, 0), 50)

    # ------------------------------------------------------------------
    # analyze() convenience method
    # ------------------------------------------------------------------

    def test_analyze_returns_list(self):
        faces = self.analyzer.analyze("portrait.jpg")
        self.assertIsInstance(faces, list)

    def test_analyze_consistent_with_run(self):
        faces = self.analyzer.analyze("portrait.jpg")
        result = self.analyzer.run("portrait.jpg")
        self.assertEqual(len(faces), result["count"])

    # ------------------------------------------------------------------
    # analyze=False suppresses attributes
    # ------------------------------------------------------------------

    def test_no_analyze_omits_attributes(self):
        a = FaceAnalyzer(analyze=False)
        result = a.run("portrait.jpg", analyze=False)
        for face in result["faces"]:
            self.assertNotIn("dominant_emotion", face)
            self.assertNotIn("age", face)
            self.assertNotIn("dominant_gender", face)

    # ------------------------------------------------------------------
    # Determinism
    # ------------------------------------------------------------------

    def test_same_source_deterministic(self):
        r1 = self.analyzer.run("same.jpg")
        r2 = self.analyzer.run("same.jpg")
        self.assertEqual(r1["count"], r2["count"])

    # ------------------------------------------------------------------
    # save() stub
    # ------------------------------------------------------------------

    def test_save_writes_json(self):
        import tempfile
        result = self.analyzer.run("test.jpg")
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "face_out.jpg"
            self.analyzer.save(result, out)
            json_path = out.with_suffix(".json")
            self.assertTrue(json_path.exists())

            import json
            data = json.loads(json_path.read_text())
            self.assertIn("count", data)
            self.assertIn("faces", data)


class TestFaceAnalyzerInit(unittest.TestCase):

    def test_defaults(self):
        a = FaceAnalyzer()
        self.assertEqual(a.backend, "opencv")
        self.assertFalse(a.enforce_detection)
        self.assertTrue(a.do_analyze)

    def test_custom_backend(self):
        a = FaceAnalyzer(backend="mtcnn")
        result = a.run("x.jpg")
        self.assertEqual(result["backend"], "mtcnn")


if __name__ == "__main__":
    unittest.main(verbosity=2)
