"""
download_dataset.py
───────────────────
Auto-downloads a public wheelchair / differently-abled person dataset
from Roboflow Universe and prepares it for YOLOv8 training.

Usage:
  python download_dataset.py                    # default dataset
  python download_dataset.py --dataset wheelchair
  python download_dataset.py --dataset coco-person
  python download_dataset.py --list             # show all options
  python download_dataset.py --api-key YOUR_KEY --workspace myws --project myproject --version 1
"""

import os, sys, shutil, zipfile, argparse, urllib.request, json
from pathlib import Path

# ── Public dataset registry (no API key needed for these) ────────
PUBLIC_DATASETS = {
    "wheelchair": {
        "description": "Wheelchair & person detection (Roboflow Universe)",
        "url": "https://public.roboflow.com/ds/wheelchair-detection/1/?key=roboflow-public",
        "fallback_url": None,
        "classes": ["wheelchair", "person"],
        "note": "Requires Roboflow free account key"
    },
    "coco-person": {
        "description": "COCO person subset — no download needed, uses YOLOv8 pretrained",
        "url": None,
        "classes": ["person"],
        "note": "Uses yolov8n.pt directly, no extra download"
    },
    "open-images-wheelchair": {
        "description": "Open Images wheelchair subset",
        "url": "https://storage.googleapis.com/openimages/v6/oidv6-class-descriptions.csv",
        "classes": ["Wheelchair", "Person"],
        "note": "Uses fiftyone to download"
    },
}

DATASET_DIR = Path("data/dataset")


def print_datasets():
    print("\nAvailable datasets:")
    print("─" * 60)
    for key, info in PUBLIC_DATASETS.items():
        print(f"  {key:<28} — {info['description']}")
        print(f"  {'':28}   Classes: {', '.join(info['classes'])}")
        if info.get('note'):
            print(f"  {'':28}   Note: {info['note']}")
        print()


def create_dirs():
    for split in ['train', 'val']:
        (DATASET_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / 'labels' / split).mkdir(parents=True, exist_ok=True)
    print("[INFO] Dataset directories ready.")


def download_via_roboflow_api(api_key, workspace, project, version):
    """Download any Roboflow project dataset via official API."""
    try:
        from roboflow import Roboflow
    except ImportError:
        print("[INFO] Installing roboflow package...")
        os.system(f"{sys.executable} -m pip install roboflow -q")
        from roboflow import Roboflow

    print(f"\n[INFO] Connecting to Roboflow...")
    rf      = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project)
    dataset = project.version(version).download("yolov8", location=str(DATASET_DIR))
    print(f"[INFO] Dataset downloaded to: {DATASET_DIR}")
    return dataset.location


def download_via_url(url, dest="data/dataset.zip"):
    """Download a zip file from URL with progress."""
    print(f"[INFO] Downloading: {url}")
    def progress(count, block, total):
        pct = min(int(count * block * 100 / total), 100)
        bar = '█' * (pct // 5) + '░' * (20 - pct // 5)
        print(f"\r  [{bar}] {pct}%", end='', flush=True)
    urllib.request.urlretrieve(url, dest, reporthook=progress)
    print()
    return dest


def extract_and_organise(zip_path):
    """Extract zip and move images/labels into YOLO structure."""
    extract_dir = Path("data/_extract_tmp")
    if extract_dir.exists():
        shutil.rmtree(extract_dir)

    print(f"[INFO] Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_dir)

    # Auto-detect YOLO structure inside zip
    yaml_files  = list(extract_dir.rglob("*.yaml"))
    image_files = list(extract_dir.rglob("*.jpg")) + \
                  list(extract_dir.rglob("*.jpeg")) + \
                  list(extract_dir.rglob("*.png"))
    label_files = list(extract_dir.rglob("*.txt"))

    print(f"[INFO] Found: {len(image_files)} images, {len(label_files)} labels")

    # Detect split folders
    train_imgs = [f for f in image_files if 'train' in str(f)]
    val_imgs   = [f for f in image_files if 'val' in str(f) or 'valid' in str(f)]
    other_imgs = [f for f in image_files if f not in train_imgs and f not in val_imgs]

    # If no split folders, auto-split 80/20
    if not train_imgs and other_imgs:
        import random; random.shuffle(other_imgs)
        n = int(len(other_imgs) * 0.8)
        train_imgs, val_imgs = other_imgs[:n], other_imgs[n:]

    def copy_split(imgs, split):
        img_dst = DATASET_DIR / 'images' / split
        lbl_dst = DATASET_DIR / 'labels' / split
        copied = 0
        for img in imgs:
            shutil.copy2(img, img_dst / img.name)
            lbl = img.with_suffix('.txt')
            if not lbl.exists():
                # try labels/ sibling
                lbl = img.parent.parent / 'labels' / img.with_suffix('.txt').name
            if lbl.exists():
                shutil.copy2(lbl, lbl_dst / lbl.name)
            copied += 1
        return copied

    t = copy_split(train_imgs, 'train')
    v = copy_split(val_imgs,   'val')
    print(f"[INFO] Organised: {t} train, {v} val images")

    # Copy YAML if found
    if yaml_files:
        shutil.copy2(yaml_files[0], 'config/dataset_downloaded.yaml')
        print(f"[INFO] Dataset YAML saved → config/dataset_downloaded.yaml")

    shutil.rmtree(extract_dir)
    if os.path.exists(zip_path):
        os.remove(zip_path)


def download_fiftyone_openimages(num_samples=500):
    """Download wheelchair + person images from Open Images via fiftyone."""
    try:
        import fiftyone as fo
        import fiftyone.zoo as foz
    except ImportError:
        print("[INFO] Installing fiftyone (this may take a moment)...")
        os.system(f"{sys.executable} -m pip install fiftyone -q")
        import fiftyone as fo
        import fiftyone.zoo as foz

    print(f"[INFO] Downloading {num_samples} wheelchair images from Open Images...")
    dataset = foz.load_zoo_dataset(
        "open-images-v7",
        split="train",
        label_types=["detections"],
        classes=["Wheelchair", "Person"],
        max_samples=num_samples,
        dataset_name="wheelchair_openimages",
    )

    export_dir = str(DATASET_DIR)
    dataset.export(
        export_dir=export_dir,
        dataset_type=fo.types.YOLOv5Dataset,
        label_field="ground_truth",
        split="train",
    )
    print(f"[INFO] Exported to {export_dir}")


def download_huggingface_dataset():
    """Download a publicly available wheelchair dataset from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        os.system(f"{sys.executable} -m pip install datasets -q")
        from datasets import load_dataset

    print("[INFO] Downloading from HuggingFace datasets...")
    # This uses a generic person detection dataset as a placeholder
    # Replace 'dataset_name' with actual HF dataset ID
    print("[WARN] No public HF wheelchair dataset configured. Using synthetic data instead.")
    generate_synthetic_dataset()


def generate_synthetic_dataset(n_train=200, n_val=50):
    """
    Generate a synthetic YOLO dataset with random bounding boxes.
    Creates blank (black) images with valid label files so the
    training pipeline can be tested end-to-end without real data.
    Uses PIL to create simple placeholder images with class labels.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
        import random, math
    except ImportError:
        os.system(f"{sys.executable} -m pip install Pillow -q")
        from PIL import Image, ImageDraw, ImageFont
        import random, math

    CLASSES = [
        'person', 'wheelchair_user', 'blind_person',
        'person_with_crutches', 'elderly_person', 'person_with_luggage'
    ]
    CLASS_COLORS = [
        (74, 144, 226), (255, 140, 0), (39, 174, 96),
        (155, 89, 182), (241, 196, 15), (52, 152, 219)
    ]
    BG_COLORS = [
        (30, 30, 50), (40, 35, 25), (20, 40, 30),
        (45, 20, 35), (25, 25, 45), (35, 40, 20)
    ]

    W, H = 640, 640

    print(f"[INFO] Generating synthetic dataset: {n_train} train + {n_val} val images...")

    def make_sample(idx, split):
        img_dir = DATASET_DIR / 'images' / split
        lbl_dir = DATASET_DIR / 'labels' / split

        bg = BG_COLORS[idx % len(BG_COLORS)]
        img = Image.new('RGB', (W, H), bg)
        draw = ImageDraw.Draw(img)

        # Station background elements
        draw.rectangle([0, 0, W, 60], fill=(20, 20, 35))          # ceiling
        draw.rectangle([0, H-80, W, H], fill=(50, 45, 40))        # floor
        draw.rectangle([W//2-5, 60, W//2+5, H-80], fill=(60,60,60))  # pillar

        labels = []
        n_objs = random.randint(1, 5)
        for _ in range(n_objs):
            cls_id = random.randint(0, len(CLASSES) - 1)
            color  = CLASS_COLORS[cls_id]

            # Random position (avoid edges)
            bw = random.randint(40, 160)
            bh = random.randint(80, 280)
            bx = random.randint(10, W - bw - 10)
            by = random.randint(60, H - bh - 80)

            # Draw person silhouette (simplified)
            head_r = bw // 4
            hx, hy = bx + bw // 2, by + head_r + 5
            draw.ellipse([hx-head_r, hy-head_r, hx+head_r, hy+head_r], fill=color, outline=(255,255,255,128))
            # Body
            body_top = hy + head_r
            draw.rectangle([bx+bw//4, body_top, bx+3*bw//4, by+bh-20], fill=color)

            # Class-specific markers
            if cls_id == 1:  # wheelchair_user
                draw.ellipse([bx, by+bh-40, bx+bw, by+bh], outline=(200,200,200), width=3)
            elif cls_id == 2:  # blind_person
                draw.line([bx+bw//2, by+bh-20, bx+bw//2+20, by+bh+10], fill=(220,220,220), width=3)
            elif cls_id == 3:  # crutches
                draw.line([bx+bw//4, by+bh//2, bx, by+bh], fill=(180,140,100), width=4)
                draw.line([bx+3*bw//4, by+bh//2, bx+bw, by+bh], fill=(180,140,100), width=4)
            elif cls_id == 5:  # luggage
                draw.rectangle([bx+bw//4, by+bh-60, bx+3*bw//4, by+bh-10], fill=(100,80,60), outline=(200,180,150))

            # Class label overlay
            draw.text((bx+2, by+2), CLASSES[cls_id][:8], fill=(255,255,255))

            # YOLO format label
            cx = (bx + bw / 2) / W
            cy = (by + bh / 2) / H
            nw = bw / W
            nh = bh / H
            labels.append(f"{cls_id} {cx:.4f} {cy:.4f} {nw:.4f} {nh:.4f}")

        # Add some noise/texture
        import random
        for _ in range(200):
            px, py = random.randint(0, W-1), random.randint(0, H-1)
            draw.point([px, py], fill=(
                min(255, bg[0]+random.randint(-20,20)),
                min(255, bg[1]+random.randint(-20,20)),
                min(255, bg[2]+random.randint(-20,20))
            ))

        name = f"synthetic_{split}_{idx:04d}"
        img.save(img_dir / f"{name}.jpg", quality=85)
        with open(lbl_dir / f"{name}.txt", 'w') as f:
            f.write('\n'.join(labels))

    for i in range(n_train):
        make_sample(i, 'train')
        if (i+1) % 50 == 0:
            print(f"  Train: {i+1}/{n_train}", end='\r')
    print(f"  Train: {n_train}/{n_train} ✓")

    for i in range(n_val):
        make_sample(i, 'val')
        if (i+1) % 10 == 0:
            print(f"  Val:   {i+1}/{n_val}", end='\r')
    print(f"  Val:   {n_val}/{n_val} ✓")

    print(f"\n[INFO] Synthetic dataset ready!")
    print(f"  Train: {n_train} images → data/dataset/images/train/")
    print(f"  Val:   {n_val} images  → data/dataset/images/val/")
    print("  NOTE: Replace with real annotated images for accurate detection.")


def write_yaml(classes):
    """Write a dataset.yaml for the downloaded/generated classes."""
    yaml_content = f"""# Auto-generated dataset config
path: data/dataset
train: images/train
val:   images/val
nc: {len(classes)}
names:
"""
    for i, c in enumerate(classes):
        yaml_content += f"  {i}: {c}\n"

    with open('config/dataset.yaml', 'w') as f:
        f.write(yaml_content)
    print("[INFO] config/dataset.yaml updated.")


def main():
    parser = argparse.ArgumentParser(description='Download dataset for Navigation Assistance')
    parser.add_argument('--dataset',   default='synthetic',
                        choices=['synthetic','roboflow','openimages','wheelchair'],
                        help='Dataset source')
    parser.add_argument('--api-key',   default='', help='Roboflow API key')
    parser.add_argument('--workspace', default='', help='Roboflow workspace')
    parser.add_argument('--project',   default='', help='Roboflow project name')
    parser.add_argument('--version',   type=int, default=1, help='Roboflow dataset version')
    parser.add_argument('--samples',   type=int, default=500, help='Number of samples (Open Images)')
    parser.add_argument('--train',     type=int, default=200, help='Synthetic train count')
    parser.add_argument('--val',       type=int, default=50,  help='Synthetic val count')
    parser.add_argument('--list',      action='store_true', help='List available datasets')
    args = parser.parse_args()

    if args.list:
        print_datasets()
        return

    create_dirs()

    CLASSES = [
        'person', 'wheelchair_user', 'blind_person',
        'person_with_crutches', 'elderly_person', 'person_with_luggage'
    ]

    print(f"\n{'='*55}")
    print(f"  Dataset Downloader — Navigation Assistance System")
    print(f"{'='*55}")
    print(f"  Mode: {args.dataset}\n")

    if args.dataset == 'synthetic':
        generate_synthetic_dataset(n_train=args.train, n_val=args.val)
        write_yaml(CLASSES)

    elif args.dataset == 'roboflow':
        if not args.api_key:
            print("[ERROR] Provide --api-key for Roboflow download.")
            print("  Get a free key at: https://app.roboflow.com/")
            return
        download_via_roboflow_api(args.api_key, args.workspace, args.project, args.version)

    elif args.dataset == 'openimages':
        download_fiftyone_openimages(num_samples=args.samples)
        write_yaml(['Person', 'Wheelchair'])

    elif args.dataset == 'wheelchair':
        print("[INFO] For wheelchair dataset, please:")
        print("  1. Visit: https://universe.roboflow.com/search?q=wheelchair")
        print("  2. Download any dataset in YOLOv8 format")
        print("  3. Upload the .zip in the web UI → Train page")
        print("  4. Or use: python download_dataset.py --dataset synthetic")

    print(f"\n{'='*55}")
    print("  Next step: python train.py")
    print(f"{'='*55}\n")


if __name__ == '__main__':
    main()
