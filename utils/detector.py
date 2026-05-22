"""
Detector — wraps Ultralytics YOLOv8 for inference.
"""

import cv2
import numpy as np
from config.settings import Settings


class Detector:
    def __init__(self):
        self.model      = None
        self.model_path = None
        self.conf       = Settings.DEFAULT_CONF
        self.iou        = Settings.DEFAULT_IOU
        self.class_names = []

    # ── Load ──────────────────────────────────────────────────────
    def load(self, model_path: str):
        try:
            from ultralytics import YOLO
            self.model      = YOLO(model_path)
            self.model_path = model_path
            self.class_names = list(self.model.names.values()) if self.model.names else []
            return True, f"Model loaded: {model_path}"
        except Exception as e:
            self.model = None
            return False, f"Failed to load model: {e}"

    def is_loaded(self) -> bool:
        return self.model is not None

    # ── Detect ────────────────────────────────────────────────────
    def detect(self, frame: np.ndarray):
        """Run inference on a single BGR frame. Returns raw YOLO Results."""
        if not self.is_loaded():
            return None
        results = self.model.predict(
            frame,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
        )
        return results[0] if results else None

    # ── Format ────────────────────────────────────────────────────
    def format_results(self, results) -> list:
        """Convert YOLO Results → list of dicts."""
        if results is None:
            return []
        detections = []
        if results.boxes is None:
            return []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            name   = results.names.get(cls_id, f'class_{cls_id}')
            conf   = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            detections.append({
                'class_id':   cls_id,
                'class_name': name,
                'confidence': round(conf, 3),
                'bbox':       [x1, y1, x2, y2],
                'center':     [cx, cy],
            })
        return detections

    def set_conf(self, conf: float):
        self.conf = max(0.01, min(0.99, conf))

    def set_iou(self, iou: float):
        self.iou = max(0.01, min(0.99, iou))
