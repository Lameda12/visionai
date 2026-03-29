"""Image I/O utilities — load, save, and convert between formats."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any

try:
    import cv2
    import numpy as np
    _DEPS = True
except ImportError:
    _DEPS = False

try:
    from PIL import Image as PILImage
    _PIL = True
except ImportError:
    _PIL = False

from utils.logger import get_logger

log = get_logger(__name__)


def load_image(path: str | Path) -> Any:
    """
    Load an image from disk.

    Returns np.ndarray (BGR) if cv2 available, else PIL Image, else None.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    if _DEPS:
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"cv2 could not read: {path}")
        return img

    if _PIL:
        return PILImage.open(path).convert("RGB")

    log.warning("Neither cv2 nor Pillow installed — returning raw bytes")
    return path.read_bytes()


def save_image(image: Any, path: str | Path, quality: int = 95) -> Path:
    """Save image to disk. Accepts np.ndarray, PIL Image, or bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if _DEPS and hasattr(image, "shape"):
        params = [cv2.IMWRITE_JPEG_QUALITY, quality] if path.suffix.lower() in (".jpg", ".jpeg") else []
        cv2.imwrite(str(path), image, params)
        log.info("Saved image → %s", path)
        return path

    if _PIL and hasattr(image, "save"):
        image.save(str(path), quality=quality)
        log.info("Saved image → %s", path)
        return path

    if isinstance(image, (bytes, bytearray)):
        path.write_bytes(image)
        log.info("Saved raw bytes → %s", path)
        return path

    raise TypeError(f"Unsupported image type: {type(image)}")


def image_to_bytes(image: Any, fmt: str = "JPEG", quality: int = 90) -> bytes:
    """Encode image to bytes for API responses."""
    if _DEPS and hasattr(image, "shape"):
        ext = f".{fmt.lower()}"
        success, buf = cv2.imencode(ext, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not success:
            raise RuntimeError("cv2.imencode failed")
        return bytes(buf)

    if _PIL and hasattr(image, "tobytes"):
        buf = io.BytesIO()
        image.save(buf, format=fmt, quality=quality)
        return buf.getvalue()

    if isinstance(image, (bytes, bytearray)):
        return bytes(image)

    raise TypeError(f"Cannot encode image of type {type(image)}")


def bytes_to_image(data: bytes) -> Any:
    """Decode image bytes → np.ndarray or PIL Image."""
    if _DEPS:
        arr = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if _PIL:
        return PILImage.open(io.BytesIO(data)).convert("RGB")

    return data
