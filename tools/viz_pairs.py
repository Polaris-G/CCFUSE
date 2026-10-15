"""Draw YOLO labels on RGB/IR pairs side by side to check crop and box alignment.

Usage:
    python tools/viz_pairs.py --root datasets/DroneVehicle --split train --n 10 --out viz_train.jpg
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np

COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 255), (255, 255, 0), (0, 165, 255)]


def draw(img, label_file, names):
    h, w = img.shape[:2]
    if label_file.exists():
        for line in label_file.read_text().split("\n"):
            if not line.strip():
                continue
            c, xc, yc, bw, bh = line.split()
            c = int(c)
            xc, yc, bw, bh = float(xc) * w, float(yc) * h, float(bw) * w, float(bh) * h
            p1 = (int(xc - bw / 2), int(yc - bh / 2))
            p2 = (int(xc + bw / 2), int(yc + bh / 2))
            cv2.rectangle(img, p1, p2, COLORS[c % len(COLORS)], 1)
    for i, n in enumerate(names):  # legend
        cv2.putText(img, n, (5, 15 + 15 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS[i % len(COLORS)], 1)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/DroneVehicle")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--names", nargs="+", default=["car", "truck", "freight car", "bus", "van"])
    ap.add_argument("--out", default="viz_pairs.jpg")
    a = ap.parse_args()

    root = Path(a.root)
    stems = sorted(p.stem for p in (root / "rgb" / "images" / a.split).glob("*.jpg"))
    random.seed(a.seed)
    rows = []
    for s in random.sample(stems, min(a.n, len(stems))):
        pair = []
        for mod in ("rgb", "ir"):
            img = cv2.imread(str(root / mod / "images" / a.split / f"{s}.jpg"))
            pair.append(draw(img, root / mod / "labels" / a.split / f"{s}.txt", a.names))
        row = np.hstack([pair[0], np.full((pair[0].shape[0], 8, 3), 255, np.uint8), pair[1]])
        cv2.putText(row, s, (row.shape[1] - 90, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        rows.append(row)
    cv2.imwrite(a.out, np.vstack(rows))
    print(f"saved {a.out} ({len(rows)} pairs, left=RGB right=IR)")


if __name__ == "__main__":
    main()
