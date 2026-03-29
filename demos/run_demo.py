#!/usr/bin/env python3
"""
VisionAI Demo — stdlib-only runner
===================================
Runs all three modules end-to-end using mock/stub data so the demo
works without installing any dependencies.

When the real deps (cv2, ultralytics, deepface) ARE installed the
core modules automatically use them; the output format is identical.

Run:
    python demos/run_demo.py
    python main.py demo
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.detector import ObjectDetector
from core.face_analyzer import FaceAnalyzer
from core.motion import MotionDetector

OUTPUT_DIR = ROOT / "outputs" / "demo"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ANSI colour helpers (no deps)
def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m"

cyan    = lambda t: _c("96", t)
green   = lambda t: _c("92", t)
yellow  = lambda t: _c("93", t)
bold    = lambda t: _c("1",  t)
dim     = lambda t: _c("2",  t)
red     = lambda t: _c("91", t)

BAR  = cyan("─" * 60)
TICK = green("✓")


def section(title: str) -> None:
    print(f"\n{BAR}")
    print(bold(f"  {title}"))
    print(BAR)


def save_json(name: str, data: dict) -> Path:
    path = OUTPUT_DIR / f"{name}.json"
    path.write_text(json.dumps(data, indent=2))
    return path


# ---------------------------------------------------------------------------
# 1. Object Detection
# ---------------------------------------------------------------------------

def demo_detection() -> None:
    section("1 / 3  |  Object Detection  (YOLOv8n)")

    detector = ObjectDetector(model="yolov8n", conf=0.4)

    samples = ["demo_street.jpg", "demo_people.jpg"]
    for sample in samples:
        t0 = time.perf_counter()
        result = detector.run(sample)
        elapsed = (time.perf_counter() - t0) * 1000

        mock_tag = dim(" [mock]") if result.get("_mock") else ""
        print(f"\n  Image : {yellow(sample)}{mock_tag}")
        print(f"  Model : {result['model']}")
        print(f"  Time  : {result['inference_ms']:.1f} ms  (wall: {elapsed:.1f} ms)")
        print(f"  Found : {cyan(str(result['count']))} object(s)\n")

        rows = [(d["label"], d["confidence"], d["box"]) for d in result["detections"]]
        col_w = max((len(r[0]) for r in rows), default=6) + 2
        print(f"  {'Label':{col_w}}  {'Conf':>6}  Box")
        print(f"  {'─'*col_w}  {'─'*6}  {'─'*20}")
        for label, conf, box in rows:
            print(f"  {label:{col_w}}  {conf:6.3f}  {box}")

        path = save_json(f"detection_{Path(sample).stem}", result | {"image": None, "annotated": None})
        print(f"\n  {TICK} JSON → {dim(str(path.relative_to(ROOT)))}")


# ---------------------------------------------------------------------------
# 2. Face Analysis
# ---------------------------------------------------------------------------

def demo_face() -> None:
    section("2 / 3  |  Face Analysis  (DeepFace + OpenCV)")

    analyzer = FaceAnalyzer(backend="opencv", enforce_detection=False, analyze=True)
    result = analyzer.run("demo_people.jpg", analyze=True)

    mock_tag = dim(" [mock]") if result.get("_mock") else ""
    print(f"\n  Faces detected : {cyan(str(result['count']))}{mock_tag}")
    print(f"  Backend        : {result['backend']}")
    print(f"  Inference      : {result['inference_ms']:.1f} ms\n")

    for idx, face in enumerate(result["faces"], 1):
        region = face.get("region", {})
        print(f"  {bold(f'Face {idx}')}  bbox=({region.get('x',0)}, {region.get('y',0)}, "
              f"{region.get('w',0)}×{region.get('h',0)})")
        if "dominant_emotion" in face:
            emo   = face["dominant_emotion"]
            score = face.get("emotion", {}).get(emo, 0)
            print(f"    emotion  : {yellow(emo)} ({score:.1f}%)")
        if "age" in face:
            print(f"    age      : ~{int(face['age'])}")
        if "dominant_gender" in face:
            g   = face["dominant_gender"]
            gs  = face.get("gender", {}).get(g, 0)
            print(f"    gender   : {g} ({gs:.1f}%)")
        if "dominant_race" in face:
            print(f"    race     : {face['dominant_race']}")
        print()

    path = save_json("face_result", result | {"image": None, "annotated": None})
    print(f"  {TICK} JSON → {dim(str(path.relative_to(ROOT)))}")


# ---------------------------------------------------------------------------
# 3. Motion Detection
# ---------------------------------------------------------------------------

def demo_motion() -> None:
    section("3 / 3  |  Motion Detection  (MOG2 background subtraction)")

    log_path = OUTPUT_DIR / "motion_log.jsonl"
    detector = MotionDetector(sensitivity="high", log_path=log_path)

    print(f"\n  Simulating 70 synthetic frames (60 background + 10 motion)…\n")

    # "Train" background model on quiet frames
    total_events = 0
    motion_frames = 0
    for i in range(60):
        _ = detector.detect(f"bg_frame_{i}")  # all quiet in mock mode (low probability)

    # Trigger motion frames — force mock RNG into a hot zone by using
    # a separate MotionDetector seeded to reliably produce events
    from core.motion import MotionDetector as _MD
    hot = _MD(sensitivity="high", log_path=log_path)
    hot._mock_rng.seed(7)  # seed that guarantees motion hits

    all_events: list[dict] = []
    for i in range(10):
        events = hot.detect(f"motion_frame_{i}")
        total_events += len(events)
        if events:
            motion_frames += 1
            for e in events:
                all_events.append({"frame": i, **e})
            areas = ", ".join(f"{e['area']:.0f}" for e in events)
            print(f"  Frame {i+1:2d}: {red('MOTION')}  {len(events)} event(s)  areas=[{areas}]px2")
        else:
            print(f"  Frame {i+1:2d}: {green('✓ clear')}")

    print(f"\n  Summary: {motion_frames}/10 frames with motion, {total_events} total event(s)")

    # Save summary JSON
    summary = {
        "sensitivity": "high",
        "frames_processed": 70,
        "motion_frame_count": motion_frames,
        "total_events": total_events,
        "events": all_events,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    path = save_json("motion_result", summary)
    print(f"  {TICK} JSON   → {dim(str(path.relative_to(ROOT)))}")
    if log_path.exists():
        print(f"  {TICK} Events → {dim(str(log_path.relative_to(ROOT)))}")


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"\n{bold('VisionAI')}  |  OpenCV · YOLOv8 · DeepFace · MOG2")
    print(dim("  Object Detection · Face Analysis · Motion Detection"))

    t0 = time.perf_counter()
    demo_detection()
    demo_face()
    demo_motion()
    elapsed = time.perf_counter() - t0

    print(f"\n{BAR}")
    print(f"  {TICK} {bold('All demos complete')}  in {yellow(f'{elapsed:.2f}s')}")
    print(f"  Outputs → {dim(str(OUTPUT_DIR.relative_to(ROOT)))}/")
    print(f"  Next steps:")
    print(dim("    uvicorn api.server:app --reload        # start REST API"))
    print(dim("    python main.py detect --source 0 --show  # webcam live detection"))
    print(BAR + "\n")


if __name__ == "__main__":
    main()
