"""Write a random paired subset of a split as RGB/IR image-list .txt files plus a data yaml (no images copied).

Used for the step-1 overfit check (train and val point to the same subset):
    python tools/make_subset.py --data data/multispectral/DroneVehicle.yaml --frac 0.01 \
        --out data/multispectral/DroneVehicle_1pct.yaml
"""
import argparse
import random
from pathlib import Path

import yaml

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/multispectral/DroneVehicle.yaml")
ap.add_argument("--split", default="train", help="key prefix in the data yaml: train -> train_rgb / train_ir")
ap.add_argument("--frac", type=float, default=0.01)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True, help="output data yaml")
a = ap.parse_args()

d = yaml.safe_load(open(a.data, encoding="utf-8"))
exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
rgb_dir, ir_dir = Path(d[f"{a.split}_rgb"]), Path(d[f"{a.split}_ir"])
stems_rgb = sorted(p.name for p in rgb_dir.iterdir() if p.suffix.lower() in exts)
stems_ir = sorted(p.name for p in ir_dir.iterdir() if p.suffix.lower() in exts)
assert stems_rgb == stems_ir, "RGB and IR file names differ; pairs would be misaligned"

n = max(1, round(len(stems_rgb) * a.frac))
pick = sorted(random.Random(a.seed).sample(stems_rgb, n))

out = Path(a.out)
tag = f"{out.stem}_{a.split}"
lists = {}
for mod, root in (("rgb", rgb_dir), ("ir", ir_dir)):
    # img2label_paths replaces the first images segment with labels. The loader
    # accepts both separators, so write POSIX absolute paths (no drive letters).
    f = out.parent / f"{tag}_{mod}.txt"
    f.write_text("\n".join((root / s).resolve().as_posix() for s in pick) + "\n", encoding="utf-8")
    lists[mod] = f.resolve().as_posix()

cfg = {
    "train_rgb": lists["rgb"], "val_rgb": lists["rgb"],  # same subset for train and val: overfit check
    "train_ir": lists["ir"], "val_ir": lists["ir"],
    "nc": d["nc"], "names": d["names"],
}
out.write_text(f"# {n}/{len(stems_rgb)} pairs of {a.data} [{a.split}], seed {a.seed}; train == val (overfit check)\n"
               + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
print(f"{n} pairs -> {out} ({lists['rgb']}, {lists['ir']})")
