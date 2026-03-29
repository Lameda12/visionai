"""
Object detection module using YOLOv8.
Falls back to deterministic mock data when ultralytics/cv2 are not installed.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

try:
    import cv2
    import numpy as np
    from ultralytics import YOLO
    _DEPS = True
except ImportError:
    _DEPS = False

from utils.logger import get_logger

log = get_logger(__name__)

COCO_LABELS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


class ObjectDetector:
    """YOLOv8-powered object detector with mock fallback for demo mode."""

    def __init__(
        self,
        model: str = "yolov8n",
        conf: float = 0.5,
        iou: float = 0.45,
        device: str = "cpu",
    ) -> None:
        self.model_name = model
        self.conf = conf
        self.iou = iou
        self.device = device
        self._model: Any = None
        if not _DEPS:
            log.warning("ultralytics/cv2 not installed — running in mock mode")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, source: str | Path) -> dict:
        """
        Run detection on an image file or webcam index.

        Returns:
            {
                source, model, count,
                detections: [{label, confidence, box, class_id}, ...],
                inference_ms,
                image,      # np.ndarray or None in mock mode
                annotated,  # np.ndarray or None in mock mode
            }
        """
        if not _DEPS:
            return self._mock_result(source)

        self._load_model()
        img = cv2.imread(str(source))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {source}")

        import time
        t0 = time.perf_counter()
        results = self._model(img, conf=self.conf, iou=self.iou, device=self.device, verbose=False)
        inference_ms = (time.perf_counter() - t0) * 1000

        detections = []
        annotated = img.copy()
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf_score = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = COCO_LABELS[cls_id] if cls_id < len(COCO_LABELS) else str(cls_id)
                detections.append({
                    "label": label,
                    "confidence": round(conf_score, 3),
                    "box": [x1, y1, x2, y2],
                    "class_id": cls_id,
                })
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (30, 144, 255), 2)
                cv2.putText(
                    annotated,
                    f"{label} {conf_score:.2f}",
                    (x1, max(y1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 144, 255), 2,
                )

        log.info("Detected %d object(s) in %.1fms", len(detections), inference_ms)
        return {
            "source": str(source),
            "model": self.model_name,
            "detections": detections,
            "count": len(detections),
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
        log.info("Saved detection output → %s", path)
        return path

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        if self._model is None:
            log.info("Loading %s model…", self.model_name)
            self._model = YOLO(f"{self.model_name}.pt")

    def _mock_result(self, source: str | Path) -> dict:
        """Deterministic mock result for demo/testing without real deps."""
        rng = random.Random(hash(str(source)) & 0xFFFF)
        labels_pool = COCO_LABELS[:20]
        detections = []
        for _ in range(rng.randint(2, 6)):
            label = rng.choice(labels_pool)
            conf_score = round(rng.uniform(max(self.conf, 0.40), 0.98), 3)
            x1, y1 = rng.randint(10, 200), rng.randint(10, 150)
            x2 = x1 + rng.randint(60, 220)
            y2 = y1 + rng.randint(60, 180)
            detections.append({
                "label": label,
                "confidence": conf_score,
                "box": [x1, y1, x2, y2],
                "class_id": labels_pool.index(label),
            })
        return {
            "source": str(source),
            "model": self.model_name,
            "detections": detections,
            "count": len(detections),
            "inference_ms": round(rng.uniform(12.0, 80.0), 1),
            "image": None,
            "annotated": None,
            "_mock": True,
        }
