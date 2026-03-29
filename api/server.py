"""
VisionAI REST API server — FastAPI.

Start:
    uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET  /health
    POST /detect   ?conf=0.5&annotated_image=true
    POST /face     ?analyze=true&annotated_image=true
    POST /motion   ?sensitivity=medium&annotated_image=true
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running from any working directory
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from fastapi import FastAPI, File, HTTPException, Query, UploadFile
    from fastapi.responses import JSONResponse, Response
    _FASTAPI = True
except ImportError:
    raise ImportError(
        "FastAPI is not installed.\n"
        "Run:  pip install fastapi uvicorn[standard] python-multipart\n"
        "Or:   pip install -r requirements.txt"
    )

import yaml

from core.detector import ObjectDetector
from core.face_analyzer import FaceAnalyzer
from core.motion import MotionDetector
from utils.io import bytes_to_image, image_to_bytes
from utils.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------

_cfg_path = ROOT / "config.yaml"
_cfg: dict = {}
if _cfg_path.exists():
    with _cfg_path.open() as f:
        _cfg = yaml.safe_load(f) or {}

_det_cfg  = _cfg.get("detection", {})
_face_cfg = _cfg.get("face", {})
_mot_cfg  = _cfg.get("motion", {})
_api_cfg  = _cfg.get("api", {})

MAX_MB = _api_cfg.get("max_image_size_mb", 10)
MAX_BYTES = MAX_MB * 1024 * 1024

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="VisionAI",
    description="Object detection · Face analysis · Motion detection",
    version="1.0.0",
)


def _check_size(data: bytes, label: str = "image") -> None:
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"{label} exceeds {MAX_MB} MB limit")


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "version": "1.0.0"})


# ---------------------------------------------------------------------------
# Object detection
# ---------------------------------------------------------------------------

@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    conf: float = Query(default=None),
    annotated_image: bool = Query(default=False),
) -> Response:
    """
    Run YOLOv8 object detection on an uploaded image.

    - **conf**: confidence threshold (overrides config.yaml)
    - **annotated_image**: if true, returns annotated JPEG; otherwise JSON
    """
    data = await file.read()
    _check_size(data, "image")

    confidence = conf if conf is not None else _det_cfg.get("confidence", 0.5)
    detector = ObjectDetector(
        model=_det_cfg.get("model", "yolov8n"),
        conf=confidence,
        iou=_det_cfg.get("iou_threshold", 0.45),
        device=_det_cfg.get("device", "cpu"),
    )

    img = bytes_to_image(data)
    if img is None:
        raise HTTPException(400, "Could not decode image")

    # ObjectDetector.run expects a file path — write to a temp file
    import tempfile, os
    suffix = Path(file.filename or "img.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        result = detector.run(tmp_path)
    finally:
        os.unlink(tmp_path)

    if annotated_image and result.get("annotated") is not None:
        return Response(
            content=image_to_bytes(result["annotated"]),
            media_type="image/jpeg",
        )

    return JSONResponse({
        "count": result["count"],
        "detections": result["detections"],
        "inference_ms": result["inference_ms"],
        "model": result["model"],
        "_mock": result.get("_mock", False),
    })


# ---------------------------------------------------------------------------
# Face analysis
# ---------------------------------------------------------------------------

@app.post("/face")
async def face(
    file: UploadFile = File(...),
    analyze: bool = Query(default=True),
    annotated_image: bool = Query(default=False),
) -> Response:
    """
    Run DeepFace face detection and attribute analysis on an uploaded image.

    - **analyze**: include age / emotion / gender / race attributes
    - **annotated_image**: if true, returns annotated JPEG; otherwise JSON
    """
    data = await file.read()
    _check_size(data)

    analyzer = FaceAnalyzer(
        backend=_face_cfg.get("backend", "opencv"),
        enforce_detection=_face_cfg.get("enforce_detection", False),
        analyze=analyze,
    )

    import tempfile, os
    suffix = Path(file.filename or "img.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        result = analyzer.run(tmp_path, analyze=analyze)
    finally:
        os.unlink(tmp_path)

    if annotated_image and result.get("annotated") is not None:
        return Response(
            content=image_to_bytes(result["annotated"]),
            media_type="image/jpeg",
        )

    return JSONResponse({
        "count": result["count"],
        "faces": result["faces"],
        "inference_ms": result["inference_ms"],
        "backend": result["backend"],
        "_mock": result.get("_mock", False),
    })


# ---------------------------------------------------------------------------
# Motion detection
# ---------------------------------------------------------------------------

@app.post("/motion")
async def motion(
    file1: UploadFile = File(..., description="Background / reference frame"),
    file2: UploadFile = File(..., description="Current frame to analyse"),
    sensitivity: str = Query(default=None),
    annotated_image: bool = Query(default=False),
) -> Response:
    """
    Compute motion between two uploaded frames using MOG2.

    - **file1**: background / reference frame
    - **file2**: current frame
    - **sensitivity**: low / medium / high
    - **annotated_image**: if true, returns annotated JPEG of frame2
    """
    data1 = await file1.read()
    data2 = await file2.read()
    _check_size(data1, "frame1")
    _check_size(data2, "frame2")

    sens = sensitivity or _mot_cfg.get("sensitivity", "medium")
    if sens not in ("low", "medium", "high"):
        raise HTTPException(400, "sensitivity must be low / medium / high")

    detector = MotionDetector(sensitivity=sens)

    img1 = bytes_to_image(data1)
    img2 = bytes_to_image(data2)

    # Train on background frame, then detect in current frame
    detector.detect(img1)
    events = detector.detect(img2)

    if annotated_image and img2 is not None:
        annotated = detector.annotate(img2, events)
        from utils.io import image_to_bytes as _enc
        return Response(content=_enc(annotated), media_type="image/jpeg")

    return JSONResponse({
        "motion_detected": len(events) > 0,
        "event_count": len(events),
        "events": events,
        "sensitivity": sens,
    })
