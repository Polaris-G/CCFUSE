"""Step-2 check: paired RGB/IR training augmentation of LoadMultiModalImagesAndLabels.

Draws N augmented training pairs (left RGB, right IR, labels on both) and runs three numeric checks:
  1. random crop works without Mosaic: reading the same sample twice gives different images;
     with translate/scale/flip/HSV all 0 the two reads are identical (sanity);
  2. hsv_ir=none: augmented IR == IR augmented with HSV gains 0 under the same seed (only geometry differs);
  3. RGB and IR get the same geometry: the gray (114) padding masks of both streams coincide.
Run from the repo root:
    python tools/check_aug.py --n 10 --out runs/viz/aug_pairs.jpg
"""
import argparse
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from utils.datasets import LoadMultiModalImagesAndLabels  # noqa: E402

COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 255), (255, 255, 0), (0, 165, 255)]

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/multispectral/DroneVehicle.yaml")
ap.add_argument("--hyp", default="data/hyp.ccsfuse_dronevehicle.yaml")
ap.add_argument("--hyp-set", nargs="*", default=[], metavar="KEY=VALUE", help="override hyp entries, e.g. hsv_ir=same")
ap.add_argument("--img-size", type=int, default=640)
ap.add_argument("--n", type=int, default=10)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="runs/viz/aug_pairs.jpg")
a = ap.parse_args()
logging.disable(logging.INFO)

data = yaml.safe_load(open(a.data, encoding="utf-8"))
hyp = yaml.safe_load(open(a.hyp, encoding="utf-8"))
for kv in a.hyp_set:
    k, v = kv.split("=", 1)
    hyp[k] = yaml.safe_load(v)
print("aug hyp:", {k: hyp.get(k) for k in ("translate", "scale", "fliplr", "mosaic", "hsv_h", "hsv_s", "hsv_v", "hsv_ir")})
ds = LoadMultiModalImagesAndLabels(data["train_rgb"], data["train_ir"], a.img_size, 1, augment=True,
                                   hyp=hyp, rect=False, stride=32)
random.seed(a.seed)
idx = random.sample(range(len(ds)), min(a.n, len(ds)))


def get(i, seed=None, **override):
    """ds[i] with optional hyp overrides and fixed RNG seed -> (rgb HWC BGR, ir HWC BGR, labels n x 5)."""
    ds.hyp = {**hyp, **override}
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    img, labels, _, _ = ds[i]
    ds.hyp = hyp
    img = img.numpy().transpose(1, 2, 0)
    return np.ascontiguousarray(img[..., 2::-1]), np.ascontiguousarray(img[..., :2:-1]), labels[:, 1:].numpy()


def draw(img, labels, names):
    img = img.copy()
    h, w = img.shape[:2]
    for c, xc, yc, bw, bh in labels:
        p1 = (int((xc - bw / 2) * w), int((yc - bh / 2) * h))
        p2 = (int((xc + bw / 2) * w), int((yc + bh / 2) * h))
        cv2.rectangle(img, p1, p2, COLORS[int(c) % len(COLORS)], 1)
    for k, n in enumerate(names):
        cv2.putText(img, n, (5, 15 + 15 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS[k % len(COLORS)], 1)
    return img


# --- visualization
rows = []
for i in idx:
    rgb, ir, lab = get(i)
    row = np.hstack([draw(rgb, lab, data["names"]), np.full((rgb.shape[0], 8, 3), 255, np.uint8),
                     draw(ir, lab, data["names"])])
    cv2.putText(row, Path(ds.img_files_rgb[i]).stem, (row.shape[1] - 90, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2)
    rows.append(row)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(a.out, np.vstack(rows))
print(f"saved {a.out} ({len(rows)} pairs, left=RGB right=IR)")

# --- 1. random crop without Mosaic
no_aug = dict(translate=0.0, scale=0.0, fliplr=0.0, flipud=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0)
diff = [not np.array_equal(get(i)[1], get(i)[1]) for i in idx]
same = [np.array_equal(get(i, **no_aug)[1], get(i, **no_aug)[1]) for i in idx]
print(f"[1] two reads of the same IR sample differ: {sum(diff)}/{len(idx)} "
      f"(all aug off -> identical: {sum(same)}/{len(idx)})")

# --- 2. hsv_ir=none -> IR only changed geometrically
if hyp.get("hsv_ir", "independent") == "none":
    ok, rgb_changed = 0, 0
    for k, i in enumerate(idx):
        rgb1, ir1, _ = get(i, seed=1000 + k)
        rgb0, ir0, _ = get(i, seed=1000 + k, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0)
        ok += np.array_equal(ir1, ir0)
        rgb_changed += not np.array_equal(rgb1, rgb0)
    print(f"[2] hsv_ir=none: IR identical to geometry-only IR: {ok}/{len(idx)}; RGB changed by HSV: {rgb_changed}/{len(idx)}")
else:
    print(f"[2] skipped (hsv_ir={hyp.get('hsv_ir')})")

# --- 3. same geometry: RGB gray padding is also padding in IR. Measured as coverage |rgb & ir| / |rgb|, not IoU:
# gray IR content often equals 114 (inflates the IR mask), and zoomed-in samples (scale > 1) have no padding.
cov, skipped = [], 0
for k, i in enumerate(idx):
    rgb, ir, _ = get(i, seed=2000 + k, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0)
    m_rgb, m_ir = (rgb == 114).all(2), (ir == 114).all(2)
    if m_rgb.sum() < 1000:
        skipped += 1
        continue
    cov.append((m_rgb & m_ir).sum() / m_rgb.sum())
print(f"[3] RGB padding also padding in IR: min {min(cov, default=float('nan')):.4f}, "
      f"mean {np.mean(cov) if cov else float('nan'):.4f} over {len(cov)} samples ({skipped} without padding skipped)")
