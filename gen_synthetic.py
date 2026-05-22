"""
gen_synthetic.py — Generates synthetic training dataset for Navigation Assistance.
Creates realistic-looking station images with passenger bounding boxes.
"""
import os, sys, random, json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter

CLASSES = [
    'person', 'wheelchair_user', 'blind_person',
    'person_with_crutches', 'elderly_person', 'person_with_luggage'
]

CLASS_COLORS = [
    (74,  144, 226),   # person — blue
    (255, 140,   0),   # wheelchair_user — orange
    ( 39, 174,  96),   # blind_person — green
    (155,  89, 182),   # crutches — purple
    (241, 196,  15),   # elderly — yellow
    ( 52, 152, 219),   # luggage — sky
]

BG_PALETTES = [
    [(25,22,38),(40,35,55),(55,50,70)],   # night station
    [(50,45,35),(70,60,40),(90,80,55)],   # day concrete
    [(20,30,45),(30,45,60),(45,60,75)],   # blue-grey
    [(35,25,25),(55,40,35),(70,55,45)],   # warm interior
]

W, H = 640, 640
DATASET_ROOT = Path('data/dataset')


def make_station_background(draw, palette):
    """Draw a believable station background."""
    bg, mid, light = palette
    # Floor
    draw.rectangle([0, H*2//3, W, H], fill=mid)
    # Ceiling / upper structure
    draw.rectangle([0, 0, W, H//6], fill=bg)
    # Walls
    draw.rectangle([0, H//6, W, H*2//3], fill=(bg[0]+10, bg[1]+10, bg[2]+10))
    # Platform edge line
    draw.rectangle([0, H*2//3-4, W, H*2//3+2], fill=light)
    # Pillars
    for px in [W//5, 2*W//5, 3*W//5, 4*W//5]:
        draw.rectangle([px-8, H//6, px+8, H*2//3], fill=bg)
    # Signs
    for sx in [50, 200, 380, 530]:
        draw.rectangle([sx, 15, sx+90, 45], fill=(30,80,160))
        draw.rectangle([sx+2, 17, sx+88, 43], fill=(20,60,130))
    # Floor tiles
    for tx in range(0, W, 80):
        draw.line([tx, H*2//3, tx, H], fill=(mid[0]-10, mid[1]-10, mid[2]-10), width=1)
    for ty in range(H*2//3, H, 50):
        draw.line([0, ty, W, ty], fill=(mid[0]-10, mid[1]-10, mid[2]-10), width=1)


def draw_person(draw, x, y, pw, ph, cls_id, color):
    """Draw a simplified person silhouette for the given class."""
    head_r = max(pw // 4, 8)
    hx = x + pw // 2
    hy = y + head_r + 4
    # Head
    draw.ellipse([hx-head_r, hy-head_r, hx+head_r, hy+head_r], fill=color)
    # Body
    body_top = hy + head_r
    body_bot = y + ph - 25
    draw.rectangle([x + pw//5, body_top, x + 4*pw//5, body_bot], fill=color)
    # Legs
    mid_x = x + pw // 2
    draw.rectangle([x + pw//5,  body_bot, mid_x - 2, y+ph], fill=(color[0]//2, color[1]//2, color[2]//2))
    draw.rectangle([mid_x + 2, body_bot, x + 4*pw//5, y+ph], fill=(color[0]//2, color[1]//2, color[2]//2))

    # Class-specific additions
    if cls_id == 1:  # wheelchair
        wr = pw // 2 + 5
        wx, wy = x + pw//2, y + ph - 10
        draw.ellipse([wx-wr, wy-wr//2, wx+wr, wy+wr//2], outline=(200,200,200), width=3)
        draw.ellipse([wx-wr//4, wy-wr//4, wx+wr//4, wy+wr//4], fill=(180,180,180))
        draw.line([wx-wr, wy, wx+wr, wy], fill=(150,150,150), width=2)
    elif cls_id == 2:  # blind — white cane
        draw.line([hx, y+ph-20, hx+pw//2+10, y+ph+15],
                  fill=(240,240,240), width=4)
        draw.ellipse([hx+pw//2+6, y+ph+12, hx+pw//2+16, y+ph+22], fill=(240,240,240))
    elif cls_id == 3:  # crutches
        draw.line([x+pw//5, body_top+10, x-8, y+ph], fill=(160,120,80), width=4)
        draw.line([x+4*pw//5, body_top+10, x+pw+8, y+ph], fill=(160,120,80), width=4)
    elif cls_id == 4:  # elderly — slightly hunched + cane
        draw.line([hx+5, body_top+15, hx+pw//2, y+ph],
                  fill=(180,180,140), width=3)
    elif cls_id == 5:  # luggage
        lw, lh = pw//2, ph//3
        lx, ly = x + pw + 2, y + ph - lh
        draw.rectangle([lx, ly, lx+lw, ly+lh], fill=(100,80,60), outline=(160,130,90), width=2)
        draw.line([hx, body_top+10, lx, ly+lh//2], fill=(140,140,140), width=2)


def make_sample(idx, split, n_objects_range=(1, 5)):
    img_dir = DATASET_ROOT / 'images' / split
    lbl_dir = DATASET_ROOT / 'labels' / split

    palette = BG_PALETTES[idx % len(BG_PALETTES)]
    img  = Image.new('RGB', (W, H), palette[0])
    draw = ImageDraw.Draw(img)
    make_station_background(draw, palette)

    labels    = []
    n_objects = random.randint(*n_objects_range)

    for _ in range(n_objects):
        cls_id = random.randint(0, len(CLASSES) - 1)
        color  = CLASS_COLORS[cls_id]

        pw = random.randint(35, 120)
        ph = random.randint(80, 220)
        px = random.randint(10, W - pw - 10)
        py = random.randint(H//5, H*2//3 - ph//2)

        draw_person(draw, px, py, pw, ph, cls_id, color)

        # Shadow
        shadow_y = min(py + ph + 5, H - 3)
        draw.ellipse([px + pw//4, shadow_y - 4, px + 3*pw//4, shadow_y + 4],
                     fill=(0, 0, 0, 80) if img.mode == 'RGBA' else (20, 18, 25))

        cx = (px + pw / 2) / W
        cy = (py + ph / 2) / H
        nw = pw / W
        nh = ph / H
        labels.append(f"{cls_id} {cx:.4f} {cy:.4f} {nw:.4f} {nh:.4f}")

    # Slight blur for realism
    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=0.8))

    name = f"nav_{split}_{idx:05d}"
    img.save(img_dir / f"{name}.jpg", quality=88)
    with open(lbl_dir / f"{name}.txt", 'w') as f:
        f.write('\n'.join(labels))


def generate(n_train=300, n_val=60):
    for split in ['train', 'val']:
        (DATASET_ROOT / 'images' / split).mkdir(parents=True, exist_ok=True)
        (DATASET_ROOT / 'labels' / split).mkdir(parents=True, exist_ok=True)

    print(f"Generating {n_train} train images...")
    for i in range(n_train):
        make_sample(i, 'train')
        if (i+1) % 50 == 0:
            print(f"  {i+1}/{n_train}", end='\r')
    print(f"  {n_train}/{n_train} train done ✓")

    print(f"Generating {n_val} val images...")
    for i in range(n_val):
        make_sample(i, 'val', n_objects_range=(1, 3))
        if (i+1) % 20 == 0:
            print(f"  {i+1}/{n_val}", end='\r')
    print(f"  {n_val}/{n_val} val done ✓")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--train', type=int, default=300)
    p.add_argument('--val',   type=int, default=60)
    args = p.parse_args()
    generate(args.train, args.val)
    print("Dataset ready.")
