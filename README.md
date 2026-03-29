# VisionAI

Modular computer vision toolkit — object detection, face analysis, motion detection, REST API, and a browser UI.

## Quick start

```bash
git clone https://github.com/Lameda12/visionai.git
cd visionai

make demo     # run demo instantly — zero dependencies required
make test     # run all 56 tests   — zero dependencies required
```

## Full install (enables real CV models)

```bash
make setup    # creates venv + installs everything
make serve    # start API on localhost:8000
make ui       # open browser UI
```

Or manually:

```bash
python3 -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

After installing you get:
- Real YOLOv8 object detection (ultralytics)
- Real face analysis with age/emotion/gender (deepface)
- Real background-subtraction motion detection (opencv)

Without installing, all three modules run in **mock mode** — deterministic synthetic results, same API shape, tests still pass.

## CLI usage

```bash
# Object detection on an image
python3 main.py detect --source image.jpg --conf 0.5 --save

# Live webcam detection (requires cv2)
python3 main.py detect --source 0 --show

# Face analysis
python3 main.py face --source portrait.jpg --analyze --save

# Motion detection on webcam (requires cv2)
python3 main.py motion --source 0 --sensitivity medium --log

# Run the demo
python3 main.py demo

# Start the API server (requires fastapi + uvicorn)
python3 main.py serve --port 8000
# or: uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
```

## Web UI

Open `web/index.html` directly in a browser — no build step.
Point it at a running API server and use the three tabs to run each module.

```bash
uvicorn api.server:app --reload   # start API
open web/index.html               # open UI (macOS)
```

## REST API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/health` | Health check |
| `POST` | `/detect` | Object detection on uploaded image |
| `POST` | `/face`   | Face analysis on uploaded image |
| `POST` | `/motion` | Motion diff between two frames |

Query params:
- `/detect?conf=0.5&annotated_image=true` — returns JPEG if `annotated_image=true`, JSON otherwise
- `/face?analyze=true&annotated_image=false`
- `/motion?sensitivity=medium&annotated_image=false`

## Python API

```python
from core.detector import ObjectDetector
from core.face_analyzer import FaceAnalyzer
from core.motion import MotionDetector

# Object detection
detector = ObjectDetector(model="yolov8n", conf=0.5)
result = detector.run("image.jpg")
# result["detections"] → [{label, confidence, box, class_id}, ...]
detector.save(result, "outputs/detected.jpg")

# Face analysis
analyzer = FaceAnalyzer()
result = analyzer.run("portrait.jpg", analyze=True)
# result["faces"] → [{region, dominant_emotion, age, dominant_gender, ...}, ...]

# Motion detection
motion = MotionDetector(sensitivity="medium")
events = motion.detect(frame)            # pass np.ndarray or any value in mock mode
annotated = motion.annotate(frame, events)
```

## Configuration

Edit `config.yaml` to set defaults:

```yaml
detection:
  model: yolov8n        # yolov8n / yolov8s / yolov8m / yolov8l
  confidence: 0.5
  device: cpu           # cpu / cuda / mps

face:
  backend: opencv       # opencv / ssd / dlib / mtcnn
  enforce_detection: false
  analyze: true

motion:
  sensitivity: medium   # low / medium / high
  min_area: 500

api:
  host: 0.0.0.0
  port: 8000
  max_image_size_mb: 10
```

## Project structure

```
visionai/
├── main.py              # CLI — detect / face / motion / demo / serve
├── config.yaml          # Default settings
├── requirements.txt
├── core/
│   ├── detector.py      # ObjectDetector (YOLOv8 + mock fallback)
│   ├── face_analyzer.py # FaceAnalyzer  (DeepFace + mock fallback)
│   └── motion.py        # MotionDetector (MOG2 + mock fallback)
├── api/
│   └── server.py        # FastAPI REST server
├── utils/
│   ├── draw.py          # Bounding box / annotation helpers
│   ├── io.py            # Image load / save / encode
│   └── logger.py        # Shared logging
├── demos/
│   └── run_demo.py      # Stdlib-only demo (no installs required)
├── web/
│   └── index.html       # Browser UI (open directly, no build)
├── tests/
│   ├── test_detector.py # 17 tests
│   ├── test_face.py     # 19 tests
│   └── test_motion.py   # 20 tests
└── outputs/             # Saved results (auto-created)
```

## License

MIT
