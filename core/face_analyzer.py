"""
Face detection and attribute analysis using DeepFace.
Falls back to deterministic mock data when deepface/cv2 are not installed.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

try:
    import cv2
    import numpy as np
    from deepface import DeepFace
    _DEPS = True
except ImportError:
    _DEPS = False

from utils.logger import get_logger

log = get_logger(__name__)

_EMOTIONS = ["happy", "sad", "angry", "surprise", "fear", "disgust", "neutral"]
_GENDERS = ["Man", "Woman"]
_RACES = ["asian", "white", "middle eastern", "indian", "latino hispanic", "black"]


class FaceAnalyzer:
    """DeepFace-powered face detector and attribute analyzer."""

    def __init__(
        self,
        backend: str = "opencv",
        enforce_detection: bool = False,
        analyze: bool = True,
    ) -> None:
        self.backend = backend
        self.enforce_detection = enforce_detection
        self.do_analyze = analyze
        if not _DEPS:
            log.warning("deepface/cv2 not installed — running in mock mode")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, source: str | Path) -> list[dict]:
        """
        Detect faces and return attribute list (convenience wrapper).

        Returns:
            [
                {
                    region, dominant_emotion, emotion,
                    age, dominant_gender, gender,
                    dominant_race, race,
                },
                ...
            ]
        """
        return self.run(source, analyze=self.do_analyze)["faces"]

    def run(self, source: str | Path, analyze: bool | None = None) -> dict:
        """
        Full pipeline: detect + optional attribute analysis.

        Returns:
            {
                source, count, backend,
                faces: [{region, dominant_emotion, emotion, age, ...}, ...],
                inference_ms,
                image,      # np.ndarray or None in mock mode
                annotated,  # np.ndarray or None in mock mode
            }
        """
        do_analyze = analyze if analyze is not None else self.do_analyze

        if not _DEPS:
            return self._mock_result(source, do_analyze)

        import time
        img = cv2.imread(str(source))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {source}")

        t0 = time.perf_counter()
        try:
            raw = DeepFace.analyze(
                img_path=str(source),
                actions=["emotion", "age", "gender", "race"] if do_analyze else ["emotion"],
                detector_backend=self.backend,
                enforce_detection=self.enforce_detection,
            )
        except Exception as exc:
            log.warning("DeepFace.analyze failed: %s", exc)
            raw = []

        inference_ms = (time.perf_counter() - t0) * 1000

        if isinstance(raw, dict):
            raw = [raw]

        faces = []
        annotated = img.copy()
        for face in raw:
            region = face.get("region", {})
            x, y, w, h = region.get("x", 0), region.get("y", 0), region.get("w", 50), region.get("h", 50)
            entry: dict[str, Any] = {"region": region}
            if do_analyze:
                entry.update({
                    "dominant_emotion": face.get("dominant_emotion", "neutral"),
                    "emotion": face.get("emotion", {}),
                    "age": face.get("age", 0),
                    "dominant_gender": face.get("dominant_gender", "unknown"),
                    "gender": face.get("gender", {}),
                    "dominant_race": face.get("dominant_race", "unknown"),
                    "race": face.get("race", {}),
                })
            faces.append(entry)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 229, 160), 2)
            if do_analyze:
                label = f"{entry.get('dominant_emotion','?')} ~{int(entry.get('age', 0))}"
                cv2.putText(annotated, label, (x, max(y - 8, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 229, 160), 2)

        log.info("Analyzed %d face(s) in %.1fms", len(faces), inference_ms)
        return {
            "source": str(source),
            "count": len(faces),
            "backend": self.backend,
            "faces": faces,
            "inference_ms": round(inference_ms, 1),
            "image": img,
            "annotated": annotated,
        }

    def save(self, result: dict, path: str | Path) -> Path:
        """Save annotated image and JSON metadata."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if _DEPS and result.get("annotated") is not None:
            cv2.imwrite(str(path), result["annotated"])

        meta_path = path.with_suffix(".json")
        meta = {k: v for k, v in result.items() if k not in ("image", "annotated")}
        meta_path.write_text(json.dumps(meta, indent=2))
        log.info("Saved face analysis output → %s", path)
        return path

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _mock_result(self, source: str | Path, analyze: bool) -> dict:
        rng = random.Random(hash(str(source)) & 0xFFFF)
        faces = []
        for i in range(rng.randint(1, 3)):
            x, y = rng.randint(30, 200), rng.randint(30, 150)
            w, h = rng.randint(60, 120), rng.randint(60, 120)
            entry: dict[str, Any] = {"region": {"x": x, "y": y, "w": w, "h": h}}
            if analyze:
                dom_emotion = rng.choice(_EMOTIONS)
                emotion_scores = {e: round(rng.uniform(1, 10), 1) for e in _EMOTIONS}
                emotion_scores[dom_emotion] = round(rng.uniform(60, 95), 1)
                dom_gender = rng.choice(_GENDERS)
                gender_scores = {g: round(rng.uniform(1, 10), 1) for g in _GENDERS}
                gender_scores[dom_gender] = round(rng.uniform(70, 99), 1)
                dom_race = rng.choice(_RACES)
                race_scores = {r: round(rng.uniform(1, 10), 1) for r in _RACES}
                race_scores[dom_race] = round(rng.uniform(50, 90), 1)
                entry.update({
                    "dominant_emotion": dom_emotion,
                    "emotion": emotion_scores,
                    "age": rng.randint(18, 65),
                    "dominant_gender": dom_gender,
                    "gender": gender_scores,
                    "dominant_race": dom_race,
                    "race": race_scores,
                })
            faces.append(entry)

        return {
            "source": str(source),
            "count": len(faces),
            "backend": self.backend,
            "faces": faces,
            "inference_ms": round(rng.uniform(40.0, 200.0), 1),
            "image": None,
            "annotated": None,
            "_mock": True,
        }
