# ── Base Image ───────────────────────────────────────────────
FROM python:3.10-slim

# ── System deps for OpenCV ───────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ────────────────────────────────────────
WORKDIR /app

# ── Install Python deps first (cache layer) ─────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy application code ───────────────────────────────────
COPY . .

# ── Create required directories ─────────────────────────────
RUN mkdir -p static/uploads models runs data

# ── Download YOLOv8n base model (small, ~6MB) ───────────────
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" && \
    find / -name "yolov8n.pt" -exec cp {} /app/yolov8n.pt \; 2>/dev/null || true

# ── Expose port ──────────────────────────────────────────────
EXPOSE 10000

# ── Run with Gunicorn (production WSGI server) ───────────────
# Single worker + threads to keep memory low on free tier
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:10000", "--workers", "1", "--threads", "4", "--timeout", "120"]
