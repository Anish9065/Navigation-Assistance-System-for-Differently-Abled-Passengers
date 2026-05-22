"""
demo.py — Full navigation demo using YOLOv8 pretrained model.
No custom training needed. Works out of the box.

Usage:
  python demo.py                    # live webcam demo
  python demo.py --source image.jpg # image demo
  python demo.py --source video.mp4 # video demo
  python demo.py --no-camera        # synthetic demo (no webcam needed)
"""

import cv2, time, argparse, random, math, os
import numpy as np

# ── Class mapping: COCO pretrained → our navigation classes ──────
# YOLOv8n pretrained detects 80 COCO classes.
# We remap relevant COCO classes to our navigation categories.
COCO_TO_NAV = {
    'person':       'person',
    'bicycle':      'person_with_luggage',   # close enough for demo
    'wheelchair':   'wheelchair_user',
    'crutch':       'person_with_crutches',
    'handbag':      'person_with_luggage',
    'backpack':     'person_with_luggage',
    'suitcase':     'person_with_luggage',
    'umbrella':     'blind_person',           # demo mapping
    'walking stick':'person_with_crutches',
}

NAV_CLASS_COLORS = {
    'person':               (219,  86,  26),
    'wheelchair_user':      (  0, 140, 255),
    'blind_person':         (  0, 200, 100),
    'person_with_crutches': (255, 100,   0),
    'elderly_person':       (180,   0, 255),
    'person_with_luggage':  (  0, 200, 200),
}
DEFAULT_COLOR = (100, 200,  50)

# ── Station Zones (x1f, y1f, x2f, y2f) ───────────────────────────
ZONES = {
    'Entrance':       (0.30, 0.00, 0.70, 0.20),
    'Main Hall':      (0.20, 0.20, 0.80, 0.60),
    'Ticket Counter': (0.70, 0.00, 1.00, 0.40),
    'Platform 1':     (0.00, 0.40, 0.35, 1.00),
    'Platform 2':     (0.65, 0.40, 1.00, 1.00),
    'Restroom':       (0.00, 0.00, 0.30, 0.40),
    'Elevator':       (0.40, 0.55, 0.60, 0.80),
    'Exit':           (0.30, 0.80, 0.70, 1.00),
    'Waiting Area':   (0.20, 0.55, 0.80, 0.80),
}

ZONE_COLORS = {
    'Entrance':       (200, 200,  50),
    'Main Hall':      ( 50, 200, 200),
    'Ticket Counter': (200,  50, 200),
    'Platform 1':     ( 50, 100, 255),
    'Platform 2':     (255, 100,  50),
    'Restroom':       ( 50, 200, 100),
    'Elevator':       (200, 150,  50),
    'Exit':           ( 50,  50, 200),
    'Waiting Area':   (150,  50, 150),
}

DESTINATIONS = [
    'Platform 1', 'Platform 2', 'Ticket Counter',
    'Restroom', 'Elevator', 'Waiting Area', 'Exit'
]

FONT = cv2.FONT_HERSHEY_SIMPLEX


def get_zone(fx, fy):
    for zone, (x1, y1, x2, y2) in ZONES.items():
        if x1 <= fx <= x2 and y1 <= fy <= y2:
            return zone
    return 'Main Hall'


def get_direction(current, destination):
    routes = {
        ('Entrance',       'Platform 1'):     ('Turn left → walk to Platform 1',  'left'),
        ('Entrance',       'Platform 2'):     ('Turn right → walk to Platform 2', 'right'),
        ('Entrance',       'Ticket Counter'): ('Turn right for Ticket Counter',   'right'),
        ('Entrance',       'Restroom'):       ('Turn left for Restroom',           'left'),
        ('Entrance',       'Elevator'):       ('Walk straight to Elevator',        'down'),
        ('Main Hall',      'Platform 1'):     ('Turn left → Platform 1',           'left'),
        ('Main Hall',      'Platform 2'):     ('Turn right → Platform 2',          'right'),
        ('Main Hall',      'Ticket Counter'): ('Move right → Ticket Counter',      'right'),
        ('Main Hall',      'Restroom'):       ('Move upper-left → Restroom',       'up'),
        ('Main Hall',      'Elevator'):       ('Walk to center → Elevator',        'down'),
        ('Main Hall',      'Exit'):           ('Walk forward → Exit',              'down'),
        ('Main Hall',      'Waiting Area'):   ('Walk forward → Waiting Area',      'down'),
        ('Waiting Area',   'Platform 1'):     ('Turn left → Platform 1',           'left'),
        ('Waiting Area',   'Platform 2'):     ('Turn right → Platform 2',          'right'),
        ('Waiting Area',   'Exit'):           ('Walk forward → Exit',              'down'),
        ('Platform 1',     'Elevator'):       ('Walk right → Elevator',            'right'),
        ('Platform 2',     'Elevator'):       ('Walk left → Elevator',             'left'),
        ('Ticket Counter', 'Platform 1'):     ('Walk left → Platform 1',           'left'),
        ('Ticket Counter', 'Platform 2'):     ('Walk left → Platform 2',           'left'),
    }
    if current == destination:
        return ('You have arrived!', 'arrived')
    key = (current, destination)
    if key in routes:
        return routes[key]
    return (f'Head towards {destination}', 'up')


ARROW_OFFSETS = {
    'left':    (-70,   0),
    'right':   ( 70,   0),
    'up':      (  0, -70),
    'down':    (  0,  70),
    'up-left': (-50, -50),
    'arrived': (  0,   0),
}


def draw_arrow(img, cx, cy, direction, color):
    if direction == 'arrived':
        cv2.circle(img, (cx, cy), 24, (0, 255, 80), 3)
        cv2.putText(img, 'HERE', (cx-28, cy+6), FONT, 0.65, (0,255,80), 2, cv2.LINE_AA)
        return
    dx, dy = ARROW_OFFSETS.get(direction, (0, -60))
    ex, ey = cx+dx, cy+dy
    cv2.arrowedLine(img, (cx, cy), (ex, ey), color, 3, tipLength=0.4)
    cv2.putText(img, direction.upper(), (ex-20, ey-10), FONT, 0.42, color, 1, cv2.LINE_AA)


def draw_box(img, x1, y1, x2, y2, label, conf, color):
    cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
    cs = 12
    for (ox,oy) in [(0,0),(x2-x1,0),(0,y2-y1),(x2-x1,y2-y1)]:
        px,py = x1+ox, y1+oy
        cv2.line(img,(px,py),(px+(cs if ox==0 else -cs),py),color,3)
        cv2.line(img,(px,py),(px,py+(cs if oy==0 else -cs)),color,3)
    text = f'{label} {int(conf*100)}%'
    (tw,th),_ = cv2.getTextSize(text, FONT, 0.52, 1)
    ty = max(y1-6, th+4)
    cv2.rectangle(img,(x1,ty-th-4),(x1+tw+8,ty+2),color,-1)
    cv2.putText(img, text, (x1+4,ty-2), FONT, 0.52,(255,255,255),1,cv2.LINE_AA)


def draw_zones(img, w, h):
    overlay = img.copy()
    for zone,(x1f,y1f,x2f,y2f) in ZONES.items():
        zx1,zy1 = int(x1f*w), int(y1f*h)
        zx2,zy2 = int(x2f*w), int(y2f*h)
        c = ZONE_COLORS.get(zone,(150,150,150))
        cv2.rectangle(overlay,(zx1,zy1),(zx2,zy2),c,-1)
        cv2.rectangle(img,(zx1,zy1),(zx2,zy2),c,1)
        cv2.putText(img, zone, (zx1+4,zy1+14), FONT, 0.35, c, 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.10, img, 0.90, 0, img)


def draw_hud(img, w, h, destination, fps, n_detected, demo_mode):
    # Top bar
    cv2.rectangle(img,(0,0),(w,30),(15,15,25),-1)
    mode_tag = '[DEMO]' if demo_mode else '[LIVE]'
    cv2.putText(img, f'Navigation Assistance System  {mode_tag}  FPS:{fps:.0f}',
                (8,20), FONT, 0.50,(160,210,255),1,cv2.LINE_AA)

    # Destination panel
    panel_w = 240
    cv2.rectangle(img,(0,30),(panel_w,80),(25,20,40),-1)
    cv2.rectangle(img,(0,30),(panel_w,80),(100,80,180),1)
    cv2.putText(img,'DESTINATION',(8,47),FONT,0.38,(180,140,255),1,cv2.LINE_AA)
    cv2.putText(img, destination,  (8,68),FONT,0.55,(255,255,255),1,cv2.LINE_AA)

    # Detection count
    cv2.rectangle(img,(0,h-28),(220,h),(15,15,25),-1)
    cv2.putText(img,f'Detected: {n_detected} passenger(s)',
                (8,h-10),FONT,0.48,(100,255,150),1,cv2.LINE_AA)

    # Controls hint
    hint = 'Q:Quit  D:Destination  S:Snapshot  Z:Zones'
    cv2.putText(img, hint,(w-360,h-10),FONT,0.38,(100,100,130),1,cv2.LINE_AA)


def draw_info_panel(img, w, directions):
    """Draw direction panel on right side."""
    pw, ph = 260, min(30 + len(directions)*52, 300)
    px = w - pw - 8
    py = 88
    cv2.rectangle(img,(px,py),(px+pw,py+ph),(20,18,32),-1)
    cv2.rectangle(img,(px,py),(px+pw,py+ph),(80,80,120),1)
    cv2.putText(img,'NAVIGATION',(px+8,py+16),FONT,0.42,(180,200,255),1,cv2.LINE_AA)
    for i, (cls, zone, instr) in enumerate(directions[:5]):
        y0 = py + 28 + i*50
        color = NAV_CLASS_COLORS.get(cls, DEFAULT_COLOR)
        cv2.rectangle(img,(px+4,y0),(px+pw-4,y0+44),(30,28,45),-1)
        cv2.rectangle(img,(px+4,y0),(px+pw-4,y0+44),color,1)
        cv2.putText(img,cls[:18],(px+8,y0+14),FONT,0.38,color,1,cv2.LINE_AA)
        cv2.putText(img,f'[{zone}]',(px+8,y0+26),FONT,0.33,(150,150,180),1,cv2.LINE_AA)
        cv2.putText(img,instr[:32],(px+8,y0+38),FONT,0.33,(200,220,200),1,cv2.LINE_AA)


def run_demo_no_camera(destination='Platform 1'):
    """Synthetic demo: creates a fake station frame with animated passengers."""
    print("[DEMO] Running synthetic demo (no camera). Press Q to quit, D to change dest.")
    W, H = 900, 560

    # Simulated passengers
    class Passenger:
        def __init__(self, cls, x, y):
            self.cls = cls
            self.x   = float(x)
            self.y   = float(y)
            self.vx  = random.uniform(-1.2, 1.2)
            self.vy  = random.uniform(-0.5, 0.5)
            self.conf = random.uniform(0.65, 0.97)

        def update(self):
            self.x += self.vx
            self.y += self.vy
            if self.x < 30 or self.x > W-80: self.vx *= -1
            if self.y < 50 or self.y > H-100: self.vy *= -1

    passengers = [
        Passenger('person',               200, 280),
        Passenger('wheelchair_user',       500, 350),
        Passenger('blind_person',          350, 200),
        Passenger('person_with_crutches',  100, 400),
        Passenger('elderly_person',        700, 300),
        Passenger('person_with_luggage',   600, 180),
    ]

    dest_idx = DESTINATIONS.index(destination) if destination in DESTINATIONS else 0
    show_zones = True
    snap_idx   = 0
    fps_timer  = time.time()
    frame_count = 0
    fps         = 25.0

    while True:
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        # Station background
        frame[:60]  = (18, 15, 28)    # ceiling
        frame[H-70:] = (40, 35, 30)   # floor
        cv2.rectangle(frame,(W//2-6,60),(W//2+6,H-70),(50,50,60),-1)  # pillar

        destination = DESTINATIONS[dest_idx]

        if show_zones:
            draw_zones(frame, W, H)

        directions = []
        for p in passengers:
            p.update()
            x1 = int(p.x)
            y1 = int(p.y)
            x2 = int(p.x + 60)
            y2 = int(p.y + 120)
            color = NAV_CLASS_COLORS.get(p.cls, DEFAULT_COLOR)

            draw_box(frame, x1, y1, x2, y2, p.cls, p.conf, color)

            cx, cy = int(p.x+30), int(p.y+60)
            fx, fy = cx/W, cy/H
            zone   = get_zone(fx, fy)
            instr, arrow = get_direction(zone, destination)
            draw_arrow(frame, cx, cy, arrow, color)
            directions.append((p.cls, zone, instr))

        # FPS calc
        frame_count += 1
        if time.time() - fps_timer >= 1.0:
            fps = frame_count / (time.time() - fps_timer)
            frame_count = 0
            fps_timer   = time.time()

        draw_hud(frame, W, H, destination, fps, len(passengers), demo_mode=True)
        draw_info_panel(frame, W, directions)

        # TTS hint overlay
        cv2.putText(frame, f'Dest: {destination}', (W//2-80, H-10),
                    FONT, 0.45, (200,180,255), 1, cv2.LINE_AA)

        cv2.imshow('Navigation Assistance — Synthetic Demo', frame)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            dest_idx = (dest_idx + 1) % len(DESTINATIONS)
            print(f"[DEMO] Destination → {DESTINATIONS[dest_idx]}")
        elif key == ord('z'):
            show_zones = not show_zones
        elif key == ord('s'):
            snap = f'demo_snapshot_{snap_idx:04d}.jpg'
            cv2.imwrite(snap, frame)
            print(f"[DEMO] Saved: {snap}")
            snap_idx += 1

    cv2.destroyAllWindows()
    print("[DEMO] Done.")


def run_live_demo(source, destination='Platform 1'):
    """Live camera / image / video demo using YOLOv8 pretrained."""
    print("[INFO] Loading YOLOv8n pretrained model...")
    try:
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')
        print("[INFO] Model loaded. ✓")
    except ImportError:
        print("[ERROR] ultralytics not installed. Run: pip install ultralytics")
        return

    # TTS
    def speak(text):
        try:
            import pyttsx3
            e = pyttsx3.init()
            e.setProperty('rate', 140)
            e.say(text); e.runAndWait()
        except Exception:
            try:
                import subprocess
                subprocess.Popen(['say', text])  # macOS
            except Exception:
                print(f"[AUDIO] {text}")

    dest_idx   = DESTINATIONS.index(destination) if destination in DESTINATIONS else 0
    show_zones = True
    snap_idx   = 0
    last_speak = 0
    fps        = 25.0
    prev_time  = time.time()

    # Open source
    is_image = isinstance(source, str) and source.lower().split('.')[-1] in \
               ['jpg','jpeg','png','bmp','webp']

    if is_image:
        frame_orig = cv2.imread(source)
        if frame_orig is None:
            print(f"[ERROR] Cannot read: {source}")
            return

    else:
        src = int(source) if str(source).isdigit() else source
        cap = cv2.VideoCapture(src)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open: {source}")
            return

    speak(f"Navigation demo started. Destination: {DESTINATIONS[dest_idx]}")
    print(f"[INFO] Controls: Q=Quit  D=Change destination  S=Snapshot  Z=Toggle zones")

    while True:
        destination = DESTINATIONS[dest_idx]

        if is_image:
            frame = frame_orig.copy()
        else:
            ret, frame = cap.read()
            if not ret:
                break

        H, W = frame.shape[:2]

        # Inference
        results  = model.predict(frame, conf=0.35, iou=0.45, verbose=False)
        result   = results[0] if results else None

        if show_zones:
            draw_zones(frame, W, H)

        directions = []
        if result and result.boxes:
            for box in result.boxes:
                cls_id  = int(box.cls[0])
                coco_nm = result.names.get(cls_id, 'object')
                nav_cls = COCO_TO_NAV.get(coco_nm, coco_nm)
                conf    = float(box.conf[0])
                x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
                cx, cy  = (x1+x2)//2, (y1+y2)//2
                color   = NAV_CLASS_COLORS.get(nav_cls, DEFAULT_COLOR)

                draw_box(frame, x1, y1, x2, y2, nav_cls, conf, color)
                zone = get_zone(cx/W, cy/H)
                instr, arrow = get_direction(zone, destination)
                draw_arrow(frame, cx, cy, arrow, color)
                directions.append((nav_cls, zone, instr))

        # FPS
        now  = time.time()
        fps  = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 0.001))
        prev_time = now

        draw_hud(frame, W, H, destination, fps, len(directions), demo_mode=False)
        draw_info_panel(frame, W, directions)

        # Audio guide (every 5s)
        if directions and now - last_speak > 5:
            speak(directions[0][2])
            last_speak = now

        cv2.imshow('Navigation Assistance — Demo', frame)
        wait = 0 if is_image else 1
        key  = cv2.waitKey(wait) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            dest_idx = (dest_idx + 1) % len(DESTINATIONS)
            d = DESTINATIONS[dest_idx]
            print(f"[INFO] Destination → {d}")
            speak(f"New destination: {d}")
        elif key == ord('z'):
            show_zones = not show_zones
        elif key == ord('s'):
            snap = f'snapshot_{snap_idx:04d}.jpg'
            cv2.imwrite(snap, frame)
            print(f"[INFO] Saved: {snap}")
            snap_idx += 1

        if is_image:
            cv2.waitKey(0)
            break

    if not is_image:
        cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Navigation Assistance Demo')
    parser.add_argument('--source',      default='0',          help='Camera index, image, or video path')
    parser.add_argument('--no-camera',   action='store_true',  help='Synthetic demo (no webcam)')
    parser.add_argument('--destination', default='Platform 1', help='Initial destination zone')
    args = parser.parse_args()

    print("=" * 55)
    print("  Navigation Assistance System — DEMO MODE")
    print("  Works with YOLOv8 pretrained, no training needed")
    print("=" * 55)

    if args.no_camera:
        run_demo_no_camera(destination=args.destination)
    else:
        run_live_demo(source=args.source, destination=args.destination)


if __name__ == '__main__':
    main()
