"""
Motion detection using OpenCV MOG2 background subtractor.
Falls back to deterministic mock data when cv2 is not installed.
"""
from __future__ import annotations

import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import cv2
    import numpy as np
    _DEPS = True
except ImportError:
    _DEPS = False

from utils.logger import get_logger

log = get_logger(__name__)

_SENSITIVITY_PARAMS = {
    "low":    {"history": 500, "var_threshold": 32, "min_area": 1500},
    "medium": {"history": 300, "var_threshold": 16, "min_area": 500},
    "high":   {"history": 100, "var_threshold": 8,  "min_area": 150},
}


class MotionDetector:
    """MOG2-based motion detector with per-frame event logging."""

    def __init__(
        self,
        sensitivity: str = "medium",
        min_area: int | None = None,
        log_path: str | Path | None = None,
    ) -> None:
        if sensitivity not in _SENSITIVITY_PARAMS:
            raise ValueError(f"sensitivity must be one of {list(_SENSITIVITY_PARAMS)}")
        self.sensitivity = sensitivity
        params = _SENSITIVITY_PARAMS[sensitivity]
        self.min_area = min_area if min_area is not None else params["min_area"]
        self.log_path = Path(log_path) if log_path else None
        self._frame_count = 0
        self._subtractor: Any = None
        self._mock_rng = random.Random(42)

        if not _DEPS:
            log.warning("cv2 not installed — running in mock mode")
        else:
            p = params
            self._subtractor = cv2.createBackgroundSubtractorMOG2(
                history=p["history"],
                varThreshold=p["var_threshold"],
                detectShadows=True,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: Any) -> list[dict]:
        """
        Process a single frame and return motion events.

        Args:
            frame: np.ndarray (H×W×3) BGR image, or any truthy value in mock mode.

        Returns:
            [{"area": float, "bbox": [x,y,w,h], "ts": ISO-8601}, ...]
        """
        self._frame_count += 1

        if not _DEPS:
            return self._mock_detect()

        mask = self._subtractor.apply(frame)
        # Remove shadows (127) — keep only foreground (255)
        _, thresh = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.dilate(thresh, kernel, iterations=2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        events = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            event = {
                "area": round(float(area), 1),
                "bbox": [x, y, w, h],
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            events.append(event)

        if events:
            log.info("Frame %d: %d motion event(s)", self._frame_count, len(events))
            self._log_events(events)

        return events

    def annotate(self, frame: Any, events: list[dict]) -> Any:
        """Draw bounding boxes on frame. Returns annotated copy."""
        if not _DEPS:
            return frame

        annotated = frame.copy()
        for e in events:
            x, y, w, h = e["bbox"]
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 77, 109), 2)
            cv2.putText(
                annotated,
                f"MOTION {e['area']:.0f}px²",
                (x, max(y - 8, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 77, 109), 2,
            )
        return annotated

    def reset(self) -> None:
        """Reset background model."""
        self._frame_count = 0
        if _DEPS:
            p = _SENSITIVITY_PARAMS[self.sensitivity]
            self._subtractor = cv2.createBackgroundSubtractorMOG2(
                history=p["history"],
                varThreshold=p["var_threshold"],
                detectShadows=True,
            )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _mock_detect(self) -> list[dict]:
        """Return mock events — triggers on ~1 in 8 frames for realism."""
        if self._mock_rng.random() > 0.12:
            return []
        events = []
        for _ in range(self._mock_rng.randint(1, 3)):
            x = self._mock_rng.randint(50, 500)
            y = self._mock_rng.randint(50, 350)
            w = self._mock_rng.randint(40, 200)
            h = self._mock_rng.randint(40, 160)
            area = float(w * h * self._mock_rng.uniform(0.4, 0.8))
            if area < self.min_area:
                continue
            events.append({
                "area": round(area, 1),
                "bbox": [x, y, w, h],
                "ts": datetime.now(timezone.utc).isoformat(),
            })
        if events:
            self._log_events(events)
        return events

    def _log_events(self, events: list[dict]) -> None:
        if not self.log_path:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a") as fh:
            for e in events:
                fh.write(json.dumps({"frame": self._frame_count, **e}) + "\n")
