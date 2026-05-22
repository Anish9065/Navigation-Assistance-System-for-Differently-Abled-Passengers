"""
train.py — Standalone training script for the Navigation Assistance model.

Usage:
  python train.py
  python train.py --epochs 100 --batch 8 --model yolov8s.pt
  python train.py --yaml config/dataset.yaml --name my_model --epochs 50
"""

import argparse
import os
import shutil


def parse_args():
    p = argparse.ArgumentParser(description='Train YOLOv8 for Navigation Assistance')
    p.add_argument('--yaml',    default='config/dataset.yaml', help='Dataset YAML path')
    p.add_argument('--model',   default='yolov8n.pt',          help='Base model (n/s/m/l/x)')
    p.add_argument('--epochs',  type=int,   default=50,        help='Training epochs')
    p.add_argument('--imgsz',   type=int,   default=640,       help='Image size')
    p.add_argument('--batch',   type=int,   default=16,        help='Batch size (-1 = auto)')
    p.add_argument('--device',  default='',                    help='Device: cpu / 0 / 0,1')
    p.add_argument('--workers', type=int,   default=4,         help='DataLoader workers')
    p.add_argument('--project', default='runs/train',          help='Output project folder')
    p.add_argument('--name',    default='navigation_model',    help='Run name')
    p.add_argument('--resume',  action='store_true',           help='Resume last training')
    p.add_argument('--augment', action='store_true',           help='Enable extra augmentation')
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  Navigation Assistance — YOLOv8 Training")
    print("=" * 60)
    print(f"  Dataset : {args.yaml}")
    print(f"  Model   : {args.model}")
    print(f"  Epochs  : {args.epochs}")
    print(f"  ImgSize : {args.imgsz}")
    print(f"  Batch   : {args.batch}")
    print(f"  Device  : {args.device or 'auto'}")
    print("=" * 60)

    if not os.path.exists(args.yaml):
        print(f"[ERROR] Dataset YAML not found: {args.yaml}")
        print("  Edit config/dataset.yaml with your dataset paths and class names.")
        return

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics not installed. Run: pip install ultralytics")
        return

    model = YOLO(args.model)

    train_kwargs = dict(
        data     = args.yaml,
        epochs   = args.epochs,
        imgsz    = args.imgsz,
        batch    = args.batch,
        project  = args.project,
        name     = args.name,
        exist_ok = True,
        verbose  = True,
        resume   = args.resume,
    )
    if args.device:
        train_kwargs['device'] = args.device
    if args.workers:
        train_kwargs['workers'] = args.workers
    if args.augment:
        train_kwargs.update(dict(
            hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
            degrees=10, translate=0.1, scale=0.5,
            flipud=0.1, fliplr=0.5, mosaic=1.0,
        ))

    print("\n[INFO] Starting training…\n")
    results = model.train(**train_kwargs)

    # Copy best weights to models/
    best_src = os.path.join(args.project, args.name, 'weights', 'best.pt')
    os.makedirs('models', exist_ok=True)
    if os.path.exists(best_src):
        dst = os.path.join('models', f'{args.name}_best.pt')
        shutil.copy(best_src, dst)
        print(f"\n✅ Best weights saved to: {dst}")

    # Validate
    print("\n[INFO] Running validation…")
    metrics = model.val()
    print(f"  mAP@0.5    : {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95: {metrics.box.map:.4f}")

    print("\n✅ Training complete.")
    print(f"   Results stored in: {args.project}/{args.name}/")


if __name__ == '__main__':
    main()
