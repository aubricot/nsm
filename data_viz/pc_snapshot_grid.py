import os
import cv2
import numpy as np
from pathlib import Path

# ── config ───────────────────────────────────────────────────────────────────
cwd = Path.cwd()
base_wd = cwd.parent 
TRAIN_DIR = base_wd / "run_v57"
os.chdir(TRAIN_DIR)
N_PCS       = 5
N_SNAPSHOTS = 5       # every other image from the 10 available
OUTPUT_FILE = "pc1_to_pc5_snapshot_grid.png"

# ── collect images ────────────────────────────────────────────────────────────
rows = []
for pc in range(1, N_PCS + 1):
    folder = f"pc{pc}_snapshots"
    # Sort so images are in order along the PC axis
    files  = sorted([
        f for f in os.listdir(folder)
        if f.endswith(".png")
    ])
    # Every other image (indices 0, 2, 4, 6, 8)
    files  = files[::2][:N_SNAPSHOTS]
    if len(files) != N_SNAPSHOTS:
        print(f"[WARN] PC{pc}: expected {N_SNAPSHOTS} images, got {len(files)}")
    imgs = [cv2.imread(os.path.join(folder, f)) for f in files]
    # Stitch this row horizontally
    row_img = np.hstack(imgs)
    rows.append(row_img)
    print(f"PC{pc}: stitched {len(imgs)} images → row shape {row_img.shape}")

# ── stitch rows vertically ────────────────────────────────────────────────────
grid = np.vstack(rows)
cv2.imwrite(OUTPUT_FILE, grid)
print(f"\nGrid saved → {OUTPUT_FILE}  (shape {grid.shape})")