"""Convert raw DroneVehicle to the dual-stream YOLO layout used by the CFT codebase.

Input  (official release, flattened):
    <src>/{train,val,test}/{split}img    RGB  840x712 jpg
    <src>/{train,val,test}/{split}imgr   IR   840x712 jpg
    <src>/{train,val,test}/{split}label  RGB  xml (polygon / bndbox / point)
    <src>/{train,val,test}/{split}labelr IR   xml

Output:
    <dst>/{rgb,ir}/images/{split}/xxxxx.jpg   640x512, 100 px white border removed
    <dst>/{rgb,ir}/labels/{split}/xxxxx.txt   YOLO "cls xc yc w h" (normalized), identical on both sides
    <dst>/convert_stats.json

Paper (Sec. IV-A): remove white border, resize to 640x512, 5 classes.
Everything else below is NOT given in the paper and is inferred (see CCSFuse_复现方案.md §4, §8 #10):
  - crop exactly 100 px per side (verified: border strips are pure 255 on sampled images)
  - OBB polygon -> HBB by min/max of vertices, then clip to the cropped image
  - label source defaults to IR annotations (as CFT / CALNet do); the same file is written to
    rgb/labels and ir/labels because CFT's loader trains on labels_rgb and pairs RGB/IR by sorted order
  - <point> objects (no extent) and the single '*' object are dropped and counted

Usage:
    python tools/prepare_dronevehicle.py --src datasets/DroneVehicle_raw --dst datasets/DroneVehicle
"""
import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2

SPLITS = ("train", "val", "test")
CLASSES = ["car", "truck", "freight car", "bus", "van"]  # order of paper Table I
# raw spellings found in the xml files -> canonical class
NAME_MAP = {
    "car": "car",
    "truck": "truck",
    "truvk": "truck",  # typo, 1 object in RGB train
    "feright car": "freight car",
    "feright_car": "freight car",
    "feright": "freight car",  # 1 object
    "bus": "bus",
    "van": "van",
}
BORDER = 100  # inferred: 840x712 -> 640x512
W, H = 840 - 2 * BORDER, 712 - 2 * BORDER


def parse_xml(path, min_size):
    """Return (yolo_lines, Counter of events) for one xml file."""
    ev = Counter()
    lines = []
    root = ET.parse(path).getroot()
    for obj in root.findall("object"):
        raw = (obj.findtext("name") or "").strip()
        if raw not in NAME_MAP:
            ev[f"drop_unknown_name:{raw}"] += 1
            continue
        cls = CLASSES.index(NAME_MAP[raw])

        poly, box = obj.find("polygon"), obj.find("bndbox")
        if poly is not None:
            xs = [float(poly.findtext(f"x{i}")) for i in range(1, 5) if poly.findtext(f"x{i}") is not None]
            ys = [float(poly.findtext(f"y{i}")) for i in range(1, 5) if poly.findtext(f"y{i}") is not None]
            if len(xs) < 2 or len(ys) < 2:
                ev["drop_malformed_polygon"] += 1
                continue
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            ev["src_polygon"] += 1
        elif box is not None:
            x1, y1 = float(box.findtext("xmin")), float(box.findtext("ymin"))
            x2, y2 = float(box.findtext("xmax")), float(box.findtext("ymax"))
            ev["src_bndbox"] += 1
        elif obj.find("point") is not None:
            ev["drop_point"] += 1
            continue
        else:
            ev["drop_no_geometry"] += 1
            continue

        # shift into cropped frame and clip
        x1, x2, y1, y2 = x1 - BORDER, x2 - BORDER, y1 - BORDER, y2 - BORDER
        cx1, cy1, cx2, cy2 = max(x1, 0), max(y1, 0), min(x2, W), min(y2, H)
        if cx2 - cx1 < min_size or cy2 - cy1 < min_size:
            ev["drop_outside_or_tiny"] += 1
            continue
        if (cx1, cy1, cx2, cy2) != (x1, y1, x2, y2):
            ev["clipped"] += 1
        lines.append(f"{cls} {(cx1 + cx2) / 2 / W:.6f} {(cy1 + cy2) / 2 / H:.6f} "
                     f"{(cx2 - cx1) / W:.6f} {(cy2 - cy1) / H:.6f}")
        ev[f"class:{CLASSES[cls]}"] += 1
    if not lines:
        ev["empty_label_files"] += 1
    return lines, ev


def crop_and_save(src, dst, quality):
    im = cv2.imread(str(src))
    if im is None or im.shape[:2] != (712, 840):
        raise RuntimeError(f"unexpected image {src}: {None if im is None else im.shape}")
    ok = cv2.imwrite(str(dst), im[BORDER:BORDER + H, BORDER:BORDER + W], [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError(f"failed to write {dst}")


def process(job):
    split, stem, a = job
    src, dst = Path(a["src"]), Path(a["dst"])
    lab_dir = f"{split}labelr" if a["label_source"] == "ir" else f"{split}label"
    lines, ev = parse_xml(src / split / lab_dir / f"{stem}.xml", a["min_size"])
    text = "\n".join(lines) + ("\n" if lines else "")
    for mod, img_dir in (("rgb", f"{split}img"), ("ir", f"{split}imgr")):
        crop_and_save(src / split / img_dir / f"{stem}.jpg", dst / mod / "images" / split / f"{stem}.jpg", a["quality"])
        (dst / mod / "labels" / split / f"{stem}.txt").write_text(text, encoding="utf-8")
    return split, ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="datasets/DroneVehicle_raw")
    ap.add_argument("--dst", default="datasets/DroneVehicle")
    ap.add_argument("--label-source", choices=["ir", "rgb"], default="ir",
                    help="which annotation set to use for BOTH modalities (paper does not say; inferred: ir)")
    ap.add_argument("--min-size", type=float, default=2.0,
                    help="drop boxes whose clipped width or height is below this many pixels (inferred)")
    ap.add_argument("--quality", type=int, default=95, help="JPEG quality for the re-encoded crops")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--limit", type=int, default=0, help="only convert the first N pairs per split (debug)")
    a = ap.parse_args()

    src, dst = Path(a.src).resolve(), Path(a.dst).resolve()
    if dst.exists() and any(dst.iterdir()):
        sys.exit(f"{dst} already exists and is not empty; remove it yourself or choose another --dst")

    jobs = []
    for split in SPLITS:
        stems = sorted(p.stem for p in (src / split / f"{split}img").glob("*.jpg"))
        for d in (f"{split}imgr", f"{split}label", f"{split}labelr"):
            other = sorted(p.stem for p in (src / split / d).iterdir())
            if other != stems:
                sys.exit(f"file names in {split}/{d} do not match {split}/{split}img")
        if a.limit:
            stems = stems[:a.limit]
        for mod in ("rgb", "ir"):
            for kind in ("images", "labels"):
                (dst / mod / kind / split).mkdir(parents=True, exist_ok=True)
        args = {"src": str(src), "dst": str(dst), "label_source": a.label_source,
                "min_size": a.min_size, "quality": a.quality}
        jobs += [(split, s, args) for s in stems]

    stats = {s: Counter() for s in SPLITS}
    done = 0
    with ProcessPoolExecutor(a.workers) as ex:
        for split, ev in ex.map(process, jobs, chunksize=64):
            stats[split].update(ev)
            stats[split]["pairs"] += 1
            done += 1
            if done % 2000 == 0 or done == len(jobs):
                print(f"{done}/{len(jobs)}", flush=True)

    out = {"label_source": a.label_source, "min_size": a.min_size, "quality": a.quality,
           "classes": CLASSES, "splits": {s: dict(sorted(c.items())) for s, c in stats.items()}}
    (dst / "convert_stats.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
