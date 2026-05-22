# Navigation Assistance System for Differently-Abled Passengers

AI-powered navigation for differently-abled passengers at railway stations using YOLOv8.
Detects passengers via camera and provides real-time audio + visual guidance.

---

## Quick Start (3 ways)

### Option A — Demo instantly (no training, no dataset)
```
python demo.py                    # live webcam
python demo.py --no-camera        # synthetic animated demo (no webcam needed)
python demo.py --source video.mp4 # video demo
```

### Option B — Web App (full UI)
```
python app.py
# Open http://127.0.0.1:5000
```

### Option C — Generate dataset & train
```
python download_dataset.py --dataset synthetic   # generates 200 train + 50 val images
python train.py                                  # trains YOLOv8s for 50 epochs
python app.py                                    # launch web app with trained model
```

---

## Setup

```bash
# Windows
setup.bat

# Linux / Mac
bash setup.sh
```

Manual:
```bash
pip install -r requirements.txt
python prepare_dataset.py    # create folder structure
python app.py                # start web app
```

---

## Project Structure

```
navigation-assistance/
│
├── app.py                    Web application (Flask) — main entry point
├── demo.py                   Standalone demo (pretrained, no training needed)
├── train.py                  CLI training script
├── detect.py                 CLI detection (image / video / live)
├── download_dataset.py       Auto-download / generate dataset
├── prepare_dataset.py        Dataset organiser & verifier
├── requirements.txt
├── setup.bat / setup.sh
│
├── config/
│   ├── settings.py           Zone layout, class colors, thresholds
│   └── dataset.yaml          YOLO dataset config — edit for your data
│
├── data/dataset/
│   ├── images/train|val/     Place your training images here
│   └── labels/train|val/     YOLO .txt label files
│
├── models/                   Trained weights (.pt) saved here
├── runs/                     Training output (auto-created)
│
├── navigation/
│   ├── navigator.py          BFS graph routing across 9 station zones
│   └── audio_guide.py        TTS audio (pyttsx3 offline + gTTS fallback)
│
├── utils/
│   ├── detector.py           YOLOv8 inference wrapper
│   └── visualization.py      Bounding boxes, zone overlay, arrows
│
├── templates/
│   ├── base.html             Sidebar layout
│   ├── index.html            Dashboard
│   ├── demo.html             Demo mode page
│   ├── train.html            Training UI
│   ├── detect.html           Image / Video / Live detection
│   └── navigation.html       Route planner + interactive map
│
└── static/
    ├── css/style.css         Dark theme UI
    └── js/app.js             Frontend utilities
```

---

## Web UI Pages

| Page | URL | What it does |
|------|-----|--------------|
| Dashboard | `/` | Load model, system overview |
| Demo Mode | `/demo` | Instant demo, dataset download |
| Train | `/train` | Upload dataset, train model |
| Detect | `/detect` | Image / Video / Live camera |
| Navigation | `/navigation` | Route planner + live guidance |

---

## Detection Classes

| ID | Class | Color |
|----|-------|-------|
| 0 | person | Blue |
| 1 | wheelchair_user | Orange |
| 2 | blind_person | Green |
| 3 | person_with_crutches | Purple |
| 4 | elderly_person | Yellow |
| 5 | person_with_luggage | Cyan |

> With pretrained YOLOv8n (demo mode), class 0 (person) works immediately.
> All 6 classes require custom training on your labeled dataset.

---

## Station Zones (9 zones)

Entrance, Main Hall, Ticket Counter, Platform 1, Platform 2,
Restroom, Elevator, Waiting Area, Exit

Navigation uses BFS graph routing between zones with audio announcements.

---

## Training Your Dataset

### Step 1 — Get a dataset

**Option A: Generate synthetic (for testing)**
```bash
python download_dataset.py --dataset synthetic --train 500 --val 100
```

**Option B: Roboflow (real annotated data)**
1. Go to https://universe.roboflow.com
2. Search: "wheelchair person station"
3. Download in YOLOv8 format
4. Upload .zip in the Train page of the web UI

**Option C: Your own images**
- Annotate with LabelImg or Roboflow
- Export in YOLO format
- Place in `data/dataset/images/` and `data/dataset/labels/`

### Step 2 — Verify
```bash
python prepare_dataset.py --verify
```

### Step 3 — Train
```bash
# CLI
python train.py --epochs 100 --model yolov8s.pt

# Web UI: open http://localhost:5000/train
```

### Step 4 — Use your model
- Model saves to `models/navigation_model_best.pt`
- Load it in the web UI or pass via `--model` flag

---

## CLI Usage

```bash
# Demo (pretrained, instant)
python demo.py
python demo.py --no-camera
python demo.py --source station.mp4

# Detection
python detect.py --mode image  --source photo.jpg  --model models/nav_best.pt
python detect.py --mode video  --source video.mp4  --save
python detect.py --mode live   --source 0          --destination platform_1

# Training
python train.py --epochs 100 --batch 8 --model yolov8s.pt
python train.py --epochs 100 --augment  # with extra augmentation

# Dataset
python download_dataset.py --dataset synthetic --train 300 --val 60
python download_dataset.py --dataset roboflow --api-key KEY --workspace WS --project PROJ
python prepare_dataset.py --source /path/to/flat_images --split 0.8
python prepare_dataset.py --verify
```

---

## VS Code — Run Configs

Open the project, press **F5** and choose:
- `Run Web App` — starts Flask on port 5000
- `Demo - Live Camera` — instant detection demo
- `Demo - Synthetic` — animated station demo, no webcam needed
- `Generate Synthetic Dataset` — creates 200 training images
- `Train Model` — trains YOLOv8s for 50 epochs
- `Detect - Live Camera` — CLI live detection
- `Verify Dataset` — checks dataset integrity

---

## API Endpoints

| Method | URL | Description |
|--------|-----|-------------|
| POST | `/api/load_model` | Load a model |
| POST | `/api/train` | Start training |
| GET | `/api/train/status` | Poll training progress |
| POST | `/api/train/stop` | Stop training |
| POST | `/api/dataset/synthetic` | Generate synthetic dataset |
| POST | `/api/dataset/roboflow` | Download Roboflow dataset |
| POST | `/api/upload_dataset` | Upload dataset .zip |
| POST | `/api/detect/image` | Detect in image |
| POST | `/api/detect/video` | Process video |
| GET | `/video_feed` | MJPEG live stream |
| POST | `/api/camera/stop` | Stop camera |
| POST | `/api/navigation/route` | BFS route between zones |
| GET | `/api/navigation/map` | Station map JSON |
| POST | `/api/speak` | Server-side TTS |
| GET | `/api/models` | List saved models |
| POST | `/api/models/upload` | Upload .pt model |

---

## Requirements

- Python 3.8+
- PyTorch (CPU works; NVIDIA GPU recommended for training)
- OpenCV, Flask, Ultralytics YOLOv8
- pyttsx3 for offline audio
- Internet for first model download (~6 MB for yolov8n.pt)

---

## Notes

- First run downloads yolov8n.pt automatically (~6 MB)
- Training on CPU is slow; use GPU if available
- Minimum recommended dataset: 100–200 images per class
- Use YOLOv8s for best accuracy/speed balance
- All processing is local — no data sent to any server
