"""Annotation drawing helpers — bounding boxes, labels, overlays."""
from __future__ import annotations

from typing import Any

try:
    import cv2
    import numpy as np
    _DEPS = True
except ImportError:
    _DEPS = False

from utils.logger import get_logger

log = get_logger(__name__)

# Colour palette (BGR)
_BLUE  = (30,  144, 255)
_GREEN = (0,   229, 160)
_RED   = (109,  77, 255)
_WHITE = (240, 240, 240)


def draw_boxes(image: Any, detections: list[dict], color: tuple = _BLUE) -> Any:
    """
    Draw YOLO detection boxes on *image*.

    Args:
        image: np.ndarray (H×W×3) BGR.
        detections: list of {"label", "confidence", "box": [x1,y1,x2,y2]}.
        color: BGR tuple.

    Returns:
        Annotated copy of *image*, or *image* unchanged if cv2 unavailable.
    """
    if not _DEPS:
        log.debug("draw_boxes: cv2 unavailable, returning image unchanged")
        return image

    out = image.copy()
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        label = det.get("label", "?")
        conf = det.get("confidence", 0.0)
        text = f"{label} {conf:.2f}"

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        ty = max(y1 - 4, th + 4)
        cv2.rectangle(out, (x1, ty - th - 4), (x1 + tw + 6, ty + 2), color, -1)
        cv2.putText(out, text, (x1 + 3, ty - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, _WHITE, 1, cv2.LINE_AA)
    return out


def draw_faces(image: Any, faces: list[dict], color: tuple = _GREEN) -> Any:
    """
    Draw face region boxes and attribute labels.

    Args:
        image: np.ndarray BGR.
        faces: list of {"region": {"x","y","w","h"}, "dominant_emotion", "age"}.
        color: BGR tuple.
    """
    if not _DEPS:
        return image

    out = image.copy()
    for face in faces:
        region = face.get("region", {})
        x, y, w, h = region.get("x", 0), region.get("y", 0), region.get("w", 50), region.get("h", 50)

        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)

        parts = []
        if "dominant_emotion" in face:
            parts.append(face["dominant_emotion"])
        if "age" in face:
            parts.append(f"~{int(face['age'])}")
        if "dominant_gender" in face:
            parts.append(face["dominant_gender"])

        if parts:
            text = "  ".join(parts)
            cv2.putText(out, text, (x, max(y - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    return out


def draw_motion(image: Any, events: list[dict], color: tuple = _RED) -> Any:
    """
    Draw motion contour bounding boxes.

    Args:
        image: np.ndarray BGR.
        events: list of {"area": float, "bbox": [x,y,w,h]}.
        color: BGR tuple.
    """
    if not _DEPS:
        return image

    out = image.copy()
    for e in events:
        x, y, w, h = e["bbox"]
        area = e.get("area", 0)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            out,
            f"MOTION  {area:.0f}px²",
            (x, max(y - 8, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA,
        )
    return out
