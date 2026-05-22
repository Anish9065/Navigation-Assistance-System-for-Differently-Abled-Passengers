"""Application settings."""
import os

class Settings:
    SECRET_KEY   = os.environ.get('SECRET_KEY', 'nav-assist-secret-2024')
    DEFAULT_CONF = 0.40
    DEFAULT_IOU  = 0.45
    DEFAULT_MODEL = 'yolov8n.pt'

    # Station layout zones (x1,y1,x2,y2 as fractions of frame)
    ZONES = {
        'entrance':       (0.30, 0.00, 0.70, 0.20),
        'main_hall':      (0.20, 0.20, 0.80, 0.60),
        'ticket_counter': (0.70, 0.00, 1.00, 0.40),
        'platform_1':     (0.00, 0.40, 0.35, 1.00),
        'platform_2':     (0.65, 0.40, 1.00, 1.00),
        'restroom':       (0.00, 0.00, 0.30, 0.40),
        'elevator':       (0.40, 0.55, 0.60, 0.80),
        'exit':           (0.30, 0.80, 0.70, 1.00),
        'waiting_area':   (0.20, 0.55, 0.80, 0.80),
    }

    # Class colours (BGR)
    CLASS_COLORS = {
        'person':               (219, 86,  26),
        'wheelchair_user':      (0,   140, 255),
        'blind_person':         (0,   200, 100),
        'person_with_crutches': (255, 100, 0),
        'elderly_person':       (180, 0,   255),
        'person_with_luggage':  (0,   200, 200),
    }
    DEFAULT_COLOR = (100, 200, 50)
