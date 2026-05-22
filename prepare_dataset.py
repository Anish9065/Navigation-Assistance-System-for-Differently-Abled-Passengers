"""
prepare_dataset.py — Helper to organise your dataset into YOLO format.

Usage:
  python prepare_dataset.py --source /path/to/raw_dataset --split 0.8
  python prepare_dataset.py --verify
"""

import os, shutil, random, argparse
from pathlib import Path


CLASSES = [
    'person',
    'wheelchair_user',
    'blind_person',
    'person_with_crutches',
    'elderly_person',
    'person_with_luggage',
]

DATASET_ROOT = Path('data/dataset')
DIRS = [
    DATASET_ROOT / 'images' / 'train',
    DATASET_ROOT / 'images' / 'val',
    DATASET_ROOT / 'labels' / 'train',
    DATASET_ROOT / 'labels' / 'val',
]


def create_dirs():
    for d in DIRS:
        d.mkdir(parents=True, exist_ok=True)
    print("[INFO] Dataset directories created:")
    for d in DIRS:
        print(f"  {d}")


def split_dataset(source: str, split: float = 0.8):
    """
    Copy images + labels from a flat source folder into train/val splits.
    Expects source to contain paired <name>.jpg + <name>.txt files.
    """
    src = Path(source)
    images = sorted(list(src.glob('*.jpg')) + list(src.glob('*.png')) + list(src.glob('*.jpeg')))
    if not images:
        print(f"[WARN] No images found in {source}")
        return

    random.shuffle(images)
    n_train = int(len(images) * split)
    splits  = {'train': images[:n_train], 'val': images[n_train:]}

    for split_name, imgs in splits.items():
        for img_path in imgs:
            # Copy image
            dst_img = DATASET_ROOT / 'images' / split_name / img_path.name
            shutil.copy2(img_path, dst_img)
            # Copy label if exists
            lbl_path = img_path.with_suffix('.txt')
            if lbl_path.exists():
                dst_lbl = DATASET_ROOT / 'labels' / split_name / lbl_path.name
                shutil.copy2(lbl_path, dst_lbl)
        print(f"[INFO] {split_name}: {len(imgs)} images")

    print(f"\n[INFO] Dataset split complete → {DATASET_ROOT}")


def verify_dataset():
    """Check dataset integrity — every image should have a label file."""
    print("[INFO] Verifying dataset…")
    issues = []
    total  = 0
    for split in ['train', 'val']:
        img_dir = DATASET_ROOT / 'images' / split
        lbl_dir = DATASET_ROOT / 'labels' / split
        imgs = list(img_dir.glob('*'))
        for img in imgs:
            total += 1
            lbl = lbl_dir / (img.stem + '.txt')
            if not lbl.exists():
                issues.append(f"Missing label: {lbl}")
        lbls = list(lbl_dir.glob('*.txt'))
        for lbl in lbls:
            img_found = any((img_dir / (lbl.stem + ext)).exists()
                            for ext in ['.jpg', '.jpeg', '.png', '.bmp'])
            if not img_found:
                issues.append(f"Orphan label (no image): {lbl}")

    print(f"  Total images: {total}")
    if issues:
        print(f"  ⚠️  {len(issues)} issue(s) found:")
        for i in issues[:20]:
            print(f"    • {i}")
    else:
        print("  ✅ Dataset looks good!")

    # Count class distribution
    print("\n  Class distribution:")
    class_counts = {c: 0 for c in CLASSES}
    for split in ['train', 'val']:
        for lbl_file in (DATASET_ROOT / 'labels' / split).glob('*.txt'):
            with open(lbl_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        cls_id = int(parts[0])
                        if cls_id < len(CLASSES):
                            class_counts[CLASSES[cls_id]] += 1
    for cls, cnt in class_counts.items():
        bar = '█' * (cnt // max(max(class_counts.values(), default=1) // 20, 1))
        print(f"    {cls:<30} {cnt:>5}  {bar}")


def create_sample_labels():
    """Create a few synthetic sample labels for testing (without real images)."""
    print("[INFO] Creating sample label files for testing…")
    for split in ['train', 'val']:
        lbl_dir = DATASET_ROOT / 'labels' / split
        img_dir = DATASET_ROOT / 'images' / split
        for i in range(5 if split == 'train' else 2):
            name = f'sample_{split}_{i:03d}'
            # Synthetic label: 3 random objects per image
            with open(lbl_dir / f'{name}.txt', 'w') as f:
                for _ in range(random.randint(1, 4)):
                    cls = random.randint(0, len(CLASSES) - 1)
                    cx  = round(random.uniform(0.1, 0.9), 4)
                    cy  = round(random.uniform(0.1, 0.9), 4)
                    w   = round(random.uniform(0.05, 0.3), 4)
                    h   = round(random.uniform(0.1, 0.5), 4)
                    f.write(f'{cls} {cx} {cy} {w} {h}\n')
    print("[INFO] Sample labels created. Add real images with matching names to train properly.")


def main():
    parser = argparse.ArgumentParser(description='Prepare dataset for Navigation Assistance')
    parser.add_argument('--source',  default='',   help='Source folder with images+labels')
    parser.add_argument('--split',   type=float,   default=0.8, help='Train/val split ratio')
    parser.add_argument('--verify',  action='store_true', help='Verify dataset integrity')
    parser.add_argument('--sample',  action='store_true', help='Create sample label files for testing')
    args = parser.parse_args()

    create_dirs()

    if args.source:
        split_dataset(args.source, args.split)

    if args.verify:
        verify_dataset()

    if args.sample:
        create_sample_labels()

    if not args.source and not args.verify and not args.sample:
        print("\n[INFO] Directory structure created. Next steps:")
        print("  1. Place training images in:  data/dataset/images/train/")
        print("  2. Place training labels in:  data/dataset/labels/train/")
        print("  3. Place val images in:       data/dataset/images/val/")
        print("  4. Place val labels in:       data/dataset/labels/val/")
        print("  5. Edit config/dataset.yaml if needed")
        print("  6. Run: python train.py")
        print("\n  Or use --source /path/to/flat_dataset to auto-split.")
        print("  Run --verify to check dataset integrity.")


if __name__ == '__main__':
    main()
