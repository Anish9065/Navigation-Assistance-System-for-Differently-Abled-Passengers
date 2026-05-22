"""
detect.py — Standalone detection script.

Usage:
  python detect.py --mode image  --source path/to/image.jpg
  python detect.py --mode video  --source path/to/video.mp4
  python detect.py --mode live   --source 0
  python detect.py --mode image  --source img.jpg --model models/my_model.pt
"""

import argparse
import os
import cv2
import time

from utils.detector    import Detector
from utils.visualization import Visualizer
from navigation.navigator  import Navigator
from navigation.audio_guide import AudioGuide


def parse_args():
    p = argparse.ArgumentParser(description='Navigation Assistance — Detection')
    p.add_argument('--mode',        default='live',       choices=['image', 'video', 'live'])
    p.add_argument('--source',      default='0',          help='Image/video path or camera index')
    p.add_argument('--model',       default='yolov8n.pt', help='Model weights path')
    p.add_argument('--conf',        type=float, default=0.40)
    p.add_argument('--iou',         type=float, default=0.45)
    p.add_argument('--destination', default='platform_1', help='Navigation destination zone')
    p.add_argument('--save',        action='store_true',  help='Save output to disk')
    p.add_argument('--no-audio',    action='store_true',  help='Disable audio guide')
    p.add_argument('--show-zones',  action='store_true',  help='Draw zone overlay')
    return p.parse_args()


def main():
    args = parse_args()

    detector    = Detector()
    visualizer  = Visualizer()
    navigator   = Navigator()
    audio       = AudioGuide() if not args.no_audio else None

    print(f"[INFO] Loading model: {args.model}")
    ok, msg = detector.load(args.model)
    if not ok:
        print(f"[WARN] {msg}")
    else:
        print(f"[INFO] {msg}")
    detector.set_conf(args.conf)
    detector.set_iou(args.iou)

    if args.mode == 'image':
        run_image(args, detector, visualizer, navigator, audio)
    elif args.mode == 'video':
        run_video(args, detector, visualizer, navigator, audio)
    else:
        run_live(args, detector, visualizer, navigator, audio)


# ── Image Mode ────────────────────────────────────────────────────
def run_image(args, detector, visualizer, navigator, audio):
    if not os.path.exists(args.source):
        print(f"[ERROR] File not found: {args.source}")
        return

    frame = cv2.imread(args.source)
    if frame is None:
        print(f"[ERROR] Cannot read image: {args.source}")
        return

    print(f"[INFO] Detecting in image: {args.source}")
    results   = detector.detect(frame)
    annotated = visualizer.draw(frame, results, navigator, destination=args.destination)

    detections = detector.format_results(results)
    print(f"[INFO] {len(detections)} detection(s)")
    for d in detections:
        print(f"  • {d['class_name']} ({d['confidence']:.2f})")

    directions = navigator.get_directions_from_detections(detections, frame.shape, args.destination)
    for d in directions:
        print(f"  ↪  [{d['zone']}] {d['instruction']}")
        if audio:
            audio.speak(d['instruction'])

    if args.save:
        out = 'output_' + os.path.basename(args.source)
        cv2.imwrite(out, annotated)
        print(f"[INFO] Saved: {out}")

    cv2.imshow('Navigation Assistance — Image', annotated)
    print("  [Press any key to close]")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ── Video Mode ────────────────────────────────────────────────────
def run_video(args, detector, visualizer, navigator, audio):
    if not os.path.exists(args.source):
        print(f"[ERROR] File not found: {args.source}")
        return

    cap    = cv2.VideoCapture(args.source)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.save:
        out_path = 'output_' + os.path.splitext(os.path.basename(args.source))[0] + '.mp4'
        fourcc   = cv2.VideoWriter_fourcc(*'mp4v')
        writer   = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    frame_idx = 0
    last_audio = 0
    print(f"[INFO] Processing video | frames={total} | fps={fps:.1f}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results   = detector.detect(frame)
        annotated = visualizer.draw(frame, results, navigator, destination=args.destination)

        detections = detector.format_results(results)
        if audio and detections and (time.time() - last_audio) > 4:
            dirs = navigator.get_directions_from_detections(detections, frame.shape, args.destination)
            if dirs:
                audio.speak_async(dirs[0]['instruction'])
                last_audio = time.time()

        if writer:
            writer.write(annotated)

        cv2.imshow('Navigation Assistance — Video', annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1
        if frame_idx % 30 == 0:
            pct = frame_idx / max(total, 1) * 100
            print(f"  Progress: {pct:.0f}%  frame {frame_idx}/{total}", end='\r')

    cap.release()
    if writer:
        writer.release()
        print(f"\n[INFO] Saved: {out_path}")
    cv2.destroyAllWindows()
    print("\n[INFO] Done.")


# ── Live Mode ─────────────────────────────────────────────────────
def run_live(args, detector, visualizer, navigator, audio):
    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera: {src}")
        return

    print("[INFO] Live detection started. Press 'q' to quit, 's' to save snapshot.")
    if audio:
        audio.welcome()

    last_audio = 0
    snap_idx   = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0        = time.time()
        results   = detector.detect(frame)
        annotated = visualizer.draw(frame, results, navigator, destination=args.destination)
        elapsed   = time.time() - t0
        fps_disp  = 1 / max(elapsed, 0.001)

        cv2.putText(annotated, f'FPS: {fps_disp:.1f}', (annotated.shape[1] - 110, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 120), 2)

        detections = detector.format_results(results)
        if audio and detections and (time.time() - last_audio) > 4:
            dirs = navigator.get_directions_from_detections(
                detections, frame.shape, args.destination)
            if dirs:
                audio.speak_async(dirs[0]['instruction'])
                last_audio = time.time()

        cv2.imshow('Navigation Assistance — Live', annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            snap = f'snapshot_{snap_idx:04d}.jpg'
            cv2.imwrite(snap, annotated)
            print(f"[INFO] Saved: {snap}")
            snap_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Live detection stopped.")


if __name__ == '__main__':
    main()
