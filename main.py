#!/usr/bin/env python3
"""
VisionAI CLI
============
Unified command-line interface for object detection, face analysis,
motion detection, and the interactive demo.

Usage examples:
    python main.py detect --source image.jpg --conf 0.5 --save
    python main.py face   --source image.jpg --analyze --save
    python main.py motion --source 0 --sensitivity medium --log
    python main.py demo
    python main.py serve  --port 8000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is importable regardless of cwd
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Sub-command handlers
# ---------------------------------------------------------------------------

def cmd_detect(args: argparse.Namespace) -> None:
    from core.detector import ObjectDetector

    detector = ObjectDetector(
        model=args.model,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
    )
    result = detector.run(args.source)

    print(f"\nDetected {result['count']} object(s)  [{result['inference_ms']:.1f} ms]")
    for d in result["detections"]:
        print(f"  {d['label']:20s}  conf={d['confidence']:.3f}  box={d['box']}")

    if args.save:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "detection_result.jpg"
        detector.save(result, out_path)
        print(f"\nSaved → {out_path}")

    if args.json:
        payload = {k: v for k, v in result.items() if k not in ("image", "annotated")}
        print(json.dumps(payload, indent=2))

    if args.show:
        try:
            import cv2
            if result.get("annotated") is not None:
                cv2.imshow("VisionAI — Detection", result["annotated"])
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print("[show] cv2 available but no annotated image (mock mode?)")
        except ImportError:
            print("[show] cv2 not installed — cannot display image")


def cmd_face(args: argparse.Namespace) -> None:
    from core.face_analyzer import FaceAnalyzer

    analyzer = FaceAnalyzer(
        backend=args.backend,
        enforce_detection=args.enforce,
        analyze=args.analyze,
    )
    result = analyzer.run(args.source, analyze=args.analyze)

    print(f"\nFaces detected: {result['count']}  [{result['inference_ms']:.1f} ms]")
    for i, face in enumerate(result["faces"], 1):
        print(f"  Face {i}:")
        if "dominant_emotion" in face:
            score = face.get("emotion", {}).get(face["dominant_emotion"], 0)
            print(f"    emotion  : {face['dominant_emotion']} ({score:.1f}%)")
        if "age" in face:
            print(f"    age      : ~{int(face['age'])}")
        if "dominant_gender" in face:
            g = face["dominant_gender"]
            gs = face.get("gender", {}).get(g, 0)
            print(f"    gender   : {g} ({gs:.1f}%)")

    if args.save:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "face_result.jpg"
        analyzer.save(result, out_path)
        print(f"\nSaved → {out_path}")

    if args.json:
        payload = {k: v for k, v in result.items() if k not in ("image", "annotated")}
        print(json.dumps(payload, indent=2))


def cmd_motion(args: argparse.Namespace) -> None:
    from core.motion import MotionDetector

    log_path = (Path(args.output_dir) / "motion_log.jsonl") if args.log else None
    detector = MotionDetector(sensitivity=args.sensitivity, log_path=log_path)

    try:
        import cv2
        source = int(args.source) if args.source.isdigit() else args.source
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open source: {args.source}")

        print(f"Monitoring {args.source!r} — press Q to quit")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            events = detector.detect(frame)
            if events:
                print(f"  Motion! {len(events)} event(s)")
            if args.show:
                annotated = detector.annotate(frame, events)
                cv2.imshow("VisionAI — Motion", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        cv2.destroyAllWindows()

    except ImportError:
        # Mock mode: simulate a short clip
        print("cv2 not installed — running mock motion detection (10 synthetic frames)")
        for i in range(10):
            events = detector.detect(f"synthetic_frame_{i}")
            if events:
                print(f"  Frame {i+1}: Motion detected — {len(events)} event(s)")
            else:
                print(f"  Frame {i+1}: No motion")
        if log_path:
            print(f"\nEvent log → {log_path}")


def cmd_demo(args: argparse.Namespace) -> None:
    demo_path = ROOT / "demos" / "run_demo.py"
    if not demo_path.exists():
        print("Demo script not found:", demo_path)
        sys.exit(1)
    import runpy
    runpy.run_path(str(demo_path), run_name="__main__")


def cmd_serve(args: argparse.Namespace) -> None:
    try:
        import uvicorn
    except ImportError:
        print("uvicorn is not installed. Run: pip install uvicorn[standard]")
        sys.exit(1)
    uvicorn.run(
        "api.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="visionai",
        description="VisionAI — modular computer vision toolkit",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # ---- detect ----
    det = sub.add_parser("detect", help="Object detection (YOLOv8)")
    det.add_argument("--source", required=True, help="Image path or webcam index (0)")
    det.add_argument("--conf",   type=float, default=0.5)
    det.add_argument("--iou",    type=float, default=0.45)
    det.add_argument("--model",  default="yolov8n")
    det.add_argument("--device", default="cpu")
    det.add_argument("--save",   action="store_true")
    det.add_argument("--show",   action="store_true")
    det.add_argument("--json",   action="store_true", help="Print JSON result")
    det.add_argument("--output-dir", default="./outputs")

    # ---- face ----
    fac = sub.add_parser("face", help="Face detection + attribute analysis")
    fac.add_argument("--source",   required=True)
    fac.add_argument("--analyze",  action="store_true", default=True)
    fac.add_argument("--no-analyze", dest="analyze", action="store_false")
    fac.add_argument("--backend",  default="opencv")
    fac.add_argument("--enforce",  action="store_true")
    fac.add_argument("--save",     action="store_true")
    fac.add_argument("--json",     action="store_true")
    fac.add_argument("--output-dir", default="./outputs")

    # ---- motion ----
    mot = sub.add_parser("motion", help="Motion detection (MOG2)")
    mot.add_argument("--source",      required=True, help="Video file or webcam index (0)")
    mot.add_argument("--sensitivity", default="medium", choices=["low", "medium", "high"])
    mot.add_argument("--log",         action="store_true", help="Log events to JSONL")
    mot.add_argument("--show",        action="store_true")
    mot.add_argument("--output-dir",  default="./outputs")

    # ---- demo ----
    sub.add_parser("demo", help="Run all modules on sample data")

    # ---- serve ----
    srv = sub.add_parser("serve", help="Start FastAPI server")
    srv.add_argument("--host",   default="0.0.0.0")
    srv.add_argument("--port",   type=int, default=8000)
    srv.add_argument("--reload", action="store_true")

    return p


_HANDLERS = {
    "detect": cmd_detect,
    "face":   cmd_face,
    "motion": cmd_motion,
    "demo":   cmd_demo,
    "serve":  cmd_serve,
}


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _HANDLERS[args.command](args)


if __name__ == "__main__":
    main()
