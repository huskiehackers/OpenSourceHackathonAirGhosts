from PIL import Image, ImageDraw, ImageFilter
import os, random, math, csv, zipfile
from pathlib import Path

random.seed(42)

# === SETTINGS ===
OUT_DIR = Path("test")  # output folder
classes = {
    "UArr": "^",
    "DArr": "v",
    "Horz": "-",
    "Vert": "|"
}
N_PER_CLASS = 500  # images per class
IMG_SIZE = (100, 100)

# === CREATE FOLDERS ===
if OUT_DIR.exists():
    for child in OUT_DIR.rglob("*"):
        if child.is_file():
            child.unlink()
else:
    OUT_DIR.mkdir(parents=True)

for cls in classes:
    (OUT_DIR / cls).mkdir(parents=True, exist_ok=True)

# === DRAWING FUNCTIONS ===
def draw_shape(draw, shape, bbox, stroke_w):
    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0
    if shape == "^":
        pts = [(x0 + w * 0.5, y0), (x0, y1), (x1, y1)]
        draw.polygon(pts, fill=255)
        if stroke_w > 0:
            inset = stroke_w
            pts_in = [(x0 + w * 0.5, y0 + inset),
                      (x0 + inset, y1 - inset),
                      (x1 - inset, y1 - inset)]
            draw.polygon(pts_in, fill=0)
    elif shape == "v":
        pts = [(x0, y0), (x1, y0), (x0 + w * 0.5, y1)]
        draw.polygon(pts, fill=255)
        if stroke_w > 0:
            inset = stroke_w
            pts_in = [(x0 + inset, y0 + inset),
                      (x1 - inset, y0 + inset),
                      (x0 + w * 0.5, y1 - inset)]
            draw.polygon(pts_in, fill=0)
    elif shape == "-":
        hbar = max(2, int(h * 0.2))
        cy = (y0 + y1) / 2
        draw.rectangle([x0, cy - hbar/2, x1, cy + hbar/2], fill=255)
        if stroke_w > 0:
            draw.rectangle([x0 + stroke_w, cy - hbar/2 + stroke_w,
                            x1 - stroke_w, cy + hbar/2 - stroke_w], fill=0)
    elif shape == "|":
        wbar = max(2, int(w * 0.2))
        cx = (x0 + x1) / 2
        draw.rectangle([cx - wbar/2, y0, cx + wbar/2, y1], fill=255)
        if stroke_w > 0:
            draw.rectangle([cx - wbar/2 + stroke_w, y0 + stroke_w,
                            cx + wbar/2 - stroke_w, y1 - stroke_w], fill=0)

# === GENERATE IMAGES ===
labels = []
for cls, ch in classes.items():
    out_folder = OUT_DIR / cls
    for i in range(N_PER_CLASS):
        img = Image.new("L", IMG_SIZE, color=0)
        draw = ImageDraw.Draw(img)

        scale = random.uniform(0.45, 0.85)
        pad_w = int((1 - scale) * IMG_SIZE[0] / 2)
        pad_h = int((1 - scale) * IMG_SIZE[1] / 2)
        jitter_x = random.randint(-6, 6)
        jitter_y = random.randint(-6, 6)
        bbox = [pad_w + jitter_x, pad_h + jitter_y,
                IMG_SIZE[0] - pad_w + jitter_x, IMG_SIZE[1] - pad_h + jitter_y]
        bbox = [max(0, bbox[0]), max(0, bbox[1]),
                min(IMG_SIZE[0], bbox[2]), min(IMG_SIZE[1], bbox[3])]

        stroke_w = random.randint(0, 4)
        draw_shape(draw, ch, bbox, stroke_w)

        # rotation
        angle = random.uniform(-10, 10)
        img = img.rotate(angle, resample=Image.Resampling.BILINEAR,
                         expand=False, fillcolor=0)

        # optional blur
        if random.random() < 0.3:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 1.2)))

        # noise
        if random.random() < 0.15:
            px = img.load()
            for _ in range(random.randint(5, 40)):
                x, y = random.randrange(IMG_SIZE[0]), random.randrange(IMG_SIZE[1])
                px[x, y] = 255 if random.random() < 0.5 else 0

        # invert (black on white)
        img = Image.eval(img, lambda p: 255 - p)

        fname = f"{cls}_{i:04d}.png"
        img.save(out_folder / fname)
        labels.append([os.path.join("test", cls, fname), cls])

# === SAVE LABELS ===
with open(OUT_DIR / "labels.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filepath", "label"])
    writer.writerows(labels)

# === README ===
readme_text = f"""Dataset of 100x100 PNG images for four classes:
UArr (^), DArr (v), Horz (-), Vert (|)

Generated: {len(labels)} images total ({N_PER_CLASS} per class)
Use with TensorFlow via image_dataset_from_directory or tf.data.
"""
(OUT_DIR / "README.txt").write_text(readme_text)

# === ZIP THE DATASET ===
zip_path = Path("test_dataset.zip")
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for root, _, files in os.walk(OUT_DIR):
        for file in files:
            fullpath = os.path.join(root, file)
            arcname = os.path.relpath(fullpath, OUT_DIR.parent)
            zf.write(fullpath, arcname)

print("Done. Dataset created at:", zip_path)
print("Total images:", len(labels))
