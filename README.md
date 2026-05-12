# Navigation Assistance System for Differently-Abled Passengers at Railway Stations

🚉 **Empowering Differently-Abled Passengers Through Computer Vision**

A real-time AI-powered navigation system using YOLOv8, OpenCV, and Streamlit. It detects differently-abled passengers (wheelchair users, blind passengers, crutch users) from CCTV feeds and provides targeted navigation assistance, audio guidance, and tracking.

---

## 📌 Features

*   **Real-time Detection:** YOLOv8-based tracking of `normal_person`, `wheelchair_user`, `blind_person`, and `crutch_user`.
*   **Intelligent Tracking:** Uses ByteTrack/DeepSORT logic via Ultralytics for maintaining passenger IDs and reducing false positives (e.g., merging person+wheelchair).
*   **Navigation Assistance Logic:** Modular routing system suggesting ramps for wheelchairs and audio guidance for blind passengers.
*   **Audio Guidance:** Multilingual text-to-speech support using `pyttsx3` and `gTTS`.
*   **Web Dashboard:** Modern Streamlit app for live detection, analytics, and routing previews.
*   **Production-Ready Structure:** Clean architecture, modular files, and Docker support.

---

## 🗂️ Project Structure

```text
railway_nav/
├── dataset/            # Automatically managed dataset splits
├── models/             # Contains data.yaml for training
├── app/                # Streamlit web application
│   └── main.py
├── training/           # YOLOv8 training and dataset handling
│   ├── dataset_handler.py
│   └── train.py
├── tracking/           # OpenCV detection & ByteTrack tracking logic
│   └── detector.py
├── navigation/         # Spatial reasoning & routing logic
│   └── logic.py
├── audio/              # Text-to-speech guidance logic
│   └── guidance.py
├── utils/              # Utility scripts
├── outputs/            # Training runs, weights, charts
├── requirements.txt
├── Dockerfile
└── main.py             # Entry point
```

---

## ⚙️ Local Setup Instructions

### 1. Prerequisites

Ensure you have Python 3.9+ installed.

### 2. Installation

```bash
# Clone the repository (if applicable)
# git clone <repo-url>
# cd railway-navigation-system

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

*(Note: Depending on your OS, you may need system packages like `ffmpeg` or `libgl1-mesa-glx` for OpenCV, and `espeak` for pyttsx3 on Linux).*

---

## 🚀 Usage Guide

### 1. Dataset Preparation

If you have a custom dataset (images and YOLO txt labels), place them in a folder and run the dataset handler to split and verify:

```bash
python training/dataset_handler.py
```
*Modify paths inside `dataset_handler.py` `__main__` block if needed.*

### 2. Training the Model

To train the YOLOv8n model from scratch on your dataset:

```bash
python training/train.py
```
*This handles augmentation, logs, and automatically saves the best weights to `outputs/`.*

### 3. Run the Streamlit Dashboard

Launch the complete application UI:

```bash
streamlit run app/main.py
```
Navigate to `http://localhost:8501` to view live detection, analytics, and assistance modules.

---

## 🐳 Docker Deployment

To deploy using Docker:

```bash
# Build the image
docker build -t railway-nav-assist .

# Run the container
docker run -p 8501:8501 --device=/dev/video0 railway-nav-assist
```
*(Remove `--device=/dev/video0` if you don't need webcam access inside the container).*

---

## 🧩 Architecture

1.  **Input:** Video feed (CCTV / Webcam).
2.  **Detection Module (YOLOv8 + ByteTrack):** Identifies and tracks passengers.
3.  **Logic Controller:** Filters false positives, applies NMS, determines passenger type.
4.  **Navigation Module:** Calculates accessible route based on passenger type and destination.
5.  **Output (Streamlit & Audio):** Displays bounding boxes, tracking IDs, and plays audio TTS instructions.

---

## 💡 Notes on Accuracy & Advanced Features

*   **False Positives:** The system is designed to classify a normal person + wheelchair as a `wheelchair_user` via the detection logic. Tuning `iou_thresh` and `conf_thresh` in the app helps mitigate low-light failure.
*   **Future Scope:** Heatmap analytics, crowd density estimation, fall detection, and emergency alerts can be modularly added to the `tracking/` and `app/` pipelines.

---
*Developed for Major Projects, Hackathons, and Railway Accessibility Research.*
