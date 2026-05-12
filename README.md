 # Navigation-Assistance-System-for-Differently-Abled-Passengers
 🚉 Railway Navigation Assistance System
Empowering Differently-Abled Passengers Through Computer Vision

A real-time AI-powered navigation system that detects passengers, wheelchair users, station landmarks, and obstacles from CCTV feeds — and delivers turn-by-turn audio & visual guidance to help differently-abled travellers navigate railway stations independently.

</div>

📌 Overview
Navigating a busy railway station can be overwhelming — especially for passengers with mobility impairments, visual challenges, or other disabilities. This system uses YOLOv8 object detection to continuously analyse camera feeds across the station and provide:

🔊 Audio guidance — spoken turn-by-turn directions via text-to-speech
🖥️ Visual overlays — on-screen bounding boxes, HUD, and navigation arrows
⚠️ Obstacle alerts — real-time warnings for hazards in the passenger's path
♿ Wheelchair priority — automatic detection and priority routing for mobility-aid users


🎯 Detected Classes
IDClassDescription0👤 PersonAll passengers and station staff1♿ WheelchairWheelchairs, walkers, crutches, mobility scooters2🪧 Station LandmarkPlatform signs, ticket counters, restrooms, exit gates3🚧 ObstacleLuggage, wet-floor signs, barriers, construction zones

🗂️ Project Structure
railway_nav/
├── configs/
│   └── data.yaml              # Dataset config — classes & split paths
├── dataset/
│   ├── images/
│   │   ├── train/             # 70% training images
│   │   ├── val/               # 20% validation images
│   │   └── test/              # 10% test images
│   └── labels/                # Matching YOLO .txt annotation files
├── scripts/
│   ├── prepare_dataset.py     # Validate, split & visualise dataset
│   ├── train.py               # YOLOv8 training pipeline
│   └── navigate.py            # Real-time inference + audio/visual guidance
├── runs/                      # Auto-generated: weights, charts, results
├── requirements.txt
└── README.md

⚙️ Installation
bash# Clone the repository
git clone https://github.com/your-username/railway-navigation-system.git
cd railway-navigation-system

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
requirements.txt includes:

ultralytics — YOLOv8 training & inference
opencv-python — video stream processing
pyttsx3 — cross-platform text-to-speech
matplotlib, numpy — data visualisation


🚀 Quick Start
Step 1 — Annotate Your Images
Use labelImg to draw bounding boxes with YOLO format:
bashpip install labelImg
labelImg
Label using class IDs: 0=person, 1=wheelchair, 2=station_landmark, 3=obstacle

Step 2 — Prepare the Dataset
bashpython scripts/prepare_dataset.py \
    --source ./raw/images \
    --labels ./raw/labels
Outputs:

✅ Validates all label files for format errors
✅ Splits into train / val / test (70% / 20% / 10%)
✅ Generates class distribution charts
✅ Saves annotated preview images


Step 3 — Train the Model
bashpython scripts/train.py
The script auto-selects the best YOLOv8 model based on your GPU:
GPU VRAMModelSpeedAccuracy< 4 GByolov8n⚡ FastestGood4–8 GByolov8s🚀 FastBetter8–16 GByolov8m⚖️ BalancedGreat> 16 GByolov8l🎯 SlowerBest
Force a specific variant:
bashpython scripts/train.py --model yolov8m.pt
# Resume interrupted training:
python scripts/train.py --resume

Step 4 — Run the Navigation System
bash# Live webcam feed
python scripts/navigate.py \
    --weights runs/train/railway_nav_v1/weights/best.pt \
    --destination platform_2

# Pre-recorded station video
python scripts/navigate.py \
    --weights best.pt \
    --source station_cctv.mp4 \
    --destination ticket_counter \
    --save

# Single image test
python scripts/navigate.py \
    --weights best.pt \
    --source platform_photo.jpg
Available destinations:
platform_1 · platform_2 · ticket_counter · restroom · help_desk · exit

🧭 Navigation Logic
The system uses zone-based spatial reasoning:
Camera Frame
┌─────────────────────────────────┐
│   LEFT   │   CENTRE   │  RIGHT  │
│  < 33%   │  33–67%    │  > 67%  │
└─────────────────────────────────┘
DetectionAudio OutputObstacle on left"Caution! Obstacle on your left. Please proceed carefully."Landmark on right"Station landmark on your right. Turn right."Wheelchair user"Wheelchair passenger detected. Priority assistance may be needed."Dense crowd"Heavy crowd ahead. Navigating to Platform 2."
Audio cooldown: 8 seconds between repeated announcements to avoid fatigue.

📊 Performance Targets
MetricTargetNotesmAP@50> 0.80Aim for 0.85+ with 500+ labelled imagesInference FPS> 15Real-time on any NVIDIA GPUObstacle alert latency< 100msSafety-critical path

💡 Tips to Improve Accuracy

More data — Target 300–500+ images per class
Diverse lighting — Include day, night, fluorescent, shadow, rainy conditions
Multiple camera angles — CCTV overhead + ground-level + side-angle views
Class balance — Review the distribution chart; upsample under-represented classes
Longer training — Increase epochs to 150 if mAP plateaus below 0.75


🛠 Troubleshooting
ProblemFixCUDA out of memoryLower batch in train.py to 8 or 4Low mAP on wheelchairAdd more wheelchair samples; verify label accuracyTTS not workingpip install pyttsx3 + install OS TTS engineSlow inference on CPUUse yolov8n.pt and set imgsz=416No valid pairs foundEnsure .txt label filenames match image filenames exactly

🤝 Contributing
Pull requests are welcome! Please:

Fork the repository
Create a feature branch (git checkout -b feature/your-feature)
Commit your changes (git commit -m 'Add your feature')
Push and open a Pull Request


📄 License
This project is licensed under the MIT License — see LICENSE for details.

<div align="center">
Made with ❤️ to make railway travel more accessible for everyone.
⭐ Star this repo if it helps your project!
</div>
