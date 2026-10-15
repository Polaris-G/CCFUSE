"""Load a split through CFT's LoadMultiModalImagesAndLabels and check pairing / labels.

Run from the repo root:
    python tools/check_loader.py --data data/multispectral/DroneVehicle.yaml
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

from utils.datasets import LoadMultiModalImagesAndLabels  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/multispectral/DroneVehicle.yaml")
ap.add_argument("--img-size", type=int, default=640)
a = ap.parse_args()

d = yaml.safe_load(open(a.data, encoding="utf-8"))
for key in ("train", "val"):
    ds = LoadMultiModalImagesAndLabels(d[f"{key}_rgb"], d[f"{key}_ir"], img_size=a.img_size, batch_size=4,
                                       augment=False, rect=False, stride=32, prefix=f"{key}: ")
    stems_rgb = [Path(p).stem for p in ds.img_files_rgb]
    stems_ir = [Path(p).stem for p in ds.img_files_ir]
    n_obj = sum(len(x) for x in ds.labels_rgb)
    cls = sorted({int(c) for x in ds.labels_rgb for c in x[:, 0]})
    print(f"[{key}] pairs rgb={len(stems_rgb)} ir={len(stems_ir)}  same order={stems_rgb == stems_ir}  "
          f"objects={n_obj}  classes={cls}  shapes={set(map(tuple, ds.shapes_rgb.astype(int)))}")
    img, labels, path, _ = ds[0]
    print(f"[{key}] sample0: tensor {tuple(img.shape)} (6 = RGB+IR channels), labels {tuple(labels.shape)}, "
          f"{os.path.basename(path)}")
