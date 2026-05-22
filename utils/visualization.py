"""
Visualizer — draws bounding boxes, zone overlays, and
navigation arrows on OpenCV frames.
"""

import cv2
import numpy as np
from config.settings import Settings


ZONE_ALPHA  = 0.12
ARROW_COLOR = (0, 230, 120)
FONT        = cv2.FONT_HERSHEY_SIMPLEX


class Visualizer:

    # ── Main draw entry ───────────────────────────────────────────
    def draw(self, frame: np.ndarray, results, navigator, destination: str = None) -> np.ndarray:
        out = frame.copy()
        h, w = out.shape[:2]

        self._draw_zone_overlay(out, w, h)
        self._draw_hud(out, w, h)

        if results is None:
            return out

        detections = []
        if results.boxes:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                name   = results.names.get(cls_id, f'class_{cls_id}')
                conf   = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                color = Settings.CLASS_COLORS.get(name, Settings.DEFAULT_COLOR)
                self._draw_box(out, x1, y1, x2, y2, name, conf, color)

                if destination:
                    zone = navigator.get_zone(cx / w, cy / h)
                    arrow = navigator.get_arrow_direction(zone, destination)
                    if arrow:
                        self._draw_arrow(out, cx, cy, arrow, color)

                detections.append({'class_name': name, 'confidence': conf,
                                   'center': [cx, cy], 'bbox': [x1, y1, x2, y2]})

        self._draw_detection_count(out, w, detections)
        return out

    # ── Bounding box ─────────────────────────────────────────────
    def _draw_box(self, img, x1, y1, x2, y2, label, conf, color):
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Corner accents
        cs = 14
        cv2.line(img, (x1, y1), (x1 + cs, y1), color, 3)
        cv2.line(img, (x1, y1), (x1, y1 + cs), color, 3)
        cv2.line(img, (x2, y1), (x2 - cs, y1), color, 3)
        cv2.line(img, (x2, y1), (x2, y1 + cs), color, 3)
        cv2.line(img, (x1, y2), (x1 + cs, y2), color, 3)
        cv2.line(img, (x1, y2), (x1, y2 - cs), color, 3)
        cv2.line(img, (x2, y2), (x2 - cs, y2), color, 3)
        cv2.line(img, (x2, y2), (x2, y2 - cs), color, 3)

        text  = f'{label} {int(conf * 100)}%'
        (tw, th), _ = cv2.getTextSize(text, FONT, 0.55, 1)
        ty = max(y1 - 8, th + 4)
        cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw + 8, ty + 2), color, -1)
        cv2.putText(img, text, (x1 + 4, ty - 2), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # ── Zone overlay ─────────────────────────────────────────────
    def _draw_zone_overlay(self, img, w, h):
        overlay = img.copy()
        zone_colors = {
            'entrance':       (200, 200, 50),
            'main_hall':      (50,  200, 200),
            'ticket_counter': (200, 50,  200),
            'platform_1':     (50,  100, 255),
            'platform_2':     (255, 100, 50),
            'restroom':       (50,  200, 100),
            'elevator':       (200, 150, 50),
            'exit':           (50,  50,  200),
            'waiting_area':   (150, 50,  150),
        }
        for zone, (x1f, y1f, x2f, y2f) in Settings.ZONES.items():
            zx1, zy1 = int(x1f * w), int(y1f * h)
            zx2, zy2 = int(x2f * w), int(y2f * h)
            c = zone_colors.get(zone, (150, 150, 150))
            cv2.rectangle(overlay, (zx1, zy1), (zx2, zy2), c, -1)
            cv2.rectangle(img,     (zx1, zy1), (zx2, zy2), c,  1)
            label = zone.replace('_', ' ').title()
            cv2.putText(img, label, (zx1 + 4, zy1 + 14), FONT, 0.38, c, 1, cv2.LINE_AA)
        cv2.addWeighted(overlay, ZONE_ALPHA, img, 1 - ZONE_ALPHA, 0, img)

    # ── HUD ──────────────────────────────────────────────────────
    def _draw_hud(self, img, w, h):
        cv2.rectangle(img, (0, 0), (w, 28), (20, 20, 20), -1)
        cv2.putText(img, 'Navigation Assistance System | Differently-Abled Passenger Aid',
                    (8, 19), FONT, 0.50, (180, 230, 255), 1, cv2.LINE_AA)

    # ── Detection count ───────────────────────────────────────────
    def _draw_detection_count(self, img, w, dets):
        h = img.shape[0]
        count = len(dets)
        text  = f'Detected: {count} person(s)'
        cv2.rectangle(img, (0, h - 28), (230, h), (20, 20, 20), -1)
        cv2.putText(img, text, (8, h - 9), FONT, 0.50, (100, 255, 150), 1, cv2.LINE_AA)

    # ── Navigation arrow ─────────────────────────────────────────
    ARROW_MAP = {
        'left':       (-60, 0),
        'right':      (60,  0),
        'up':         (0,  -60),
        'down':       (0,   60),
        'up-left':    (-45, -45),
        'up-right':   (45, -45),
        'down-left':  (-45, 45),
        'down-right': (45,  45),
        'arrived':    (0,   0),
    }

    def _draw_arrow(self, img, cx, cy, direction, color):
        if direction == 'arrived':
            cv2.circle(img, (cx, cy), 20, (0, 255, 0), 3)
            cv2.putText(img, 'HERE', (cx - 22, cy + 5), FONT, 0.6, (0, 255, 0), 2)
            return
        dx, dy = self.ARROW_MAP.get(direction, (0, 0))
        ex, ey = cx + dx, cy + dy
        cv2.arrowedLine(img, (cx, cy), (ex, ey), color, 3, tipLength=0.4)

        label = direction.replace('-', ' ').upper()
        cv2.putText(img, label, (ex - 20, ey - 8), FONT, 0.45, color, 1, cv2.LINE_AA)
