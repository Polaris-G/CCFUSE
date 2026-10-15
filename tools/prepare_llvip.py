"""Convert raw LLVIP to the dual-stream YOLO layout used by the CFT codebase.

Input  (official release):
    <src>/visible/{train,test}/*.jpg     1280x1024
    <src>/infrared/{train,test}/*.jpg    1280x1024
    <src>/Annotations/*.xml              VOC bndbox, single class 'person'

Output:
    <dst>/{rgb,ir}/images/{train,test}/*.jpg   byte-identical copies (no resize; the loader resizes to 640)
    <dst>/{rgb,ir}/labels/{train,test}/*.txt   YOLO "0 xc yc w h" (normalized), identical on both sides
    <dst>/convert_stats.json

Paper (Sec. IV-A): 15,488 strictly aligned pairs (12,025 train / 3,463 test), single class person,
RGB and IR share annotations. Not in the paper (inferred): boxes narrower/shorter than --min-size px are
dropped (5 zero-width boxes exist in the release); images are not resized offline.

Usage:
    python tools/prepare_llvip.py --src datasets/LLVIP_raw/LLVIP --dst datasets/LLVIP
"""
import argparse
import json
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SPLITS = ("train", "test")
CLASSES = ["person"]


def convert_xml(path, min_size):
    ev = Counter()
    root = ET.parse(path).getroot()
    s = root.find("size")
    W, H = float(s.findtext("width")), float(s.findtext("height"))
    lines = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        if name not in CLASSES:
            ev[f"drop_unknown_name:{name}"] += 1
            continue
        b = obj.find("bndbox")
        if b is None:
            ev["drop_no_bndbox"] += 1
            continue
        x1, y1, x2, y2 = (float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, W), min(y2, H)
        if x2 - x1 < min_size or y2 - y1 < min_size:
            ev["drop_tiny_or_degenerate"] += 1
            continue
        lines.append(f"{CLASSES.index(name)} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} "
                     f"{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
        ev["class:person"] += 1
    if not lines:
        ev["empty_label_files"] += 1
    return lines, ev


def process(job):
    split, stem, src, dst, min_size = job
    lines, ev = convert_xml(src / "Annotations" / f"{stem}.xml", min_size)
    text = "\n".join(lines) + ("\n" if lines else "")
    for mod, raw in (("rgb", "visible"), ("ir", "infrared")):
        shutil.copy2(src / raw / split / f"{stem}.jpg", dst / mod / "images" / split / f"{stem}.jpg")
        (dst / mod / "labels" / split / f"{stem}.txt").write_text(text, encoding="utf-8")
    return split, ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="datasets/LLVIP_raw/LLVIP")
    ap.add_argument("--dst", default="datasets/LLVIP")
    ap.add_argument("--min-size", type=float, default=1.0,
                    help="drop boxes whose width or height is below this many pixels (inferred)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="only convert the first N pairs per split (debug)")
    a = ap.parse_args()

    src, dst = Path(a.src).resolve(), Path(a.dst).resolve()
    if dst.exists() and any(dst.iterdir()):
        sys.exit(f"{dst} already exists and is not empty; remove it yourself or choose another --dst")

    jobs = []
    for split in SPLITS:
        stems = sorted(p.stem for p in (src / "visible" / split).glob("*.jpg"))
        if sorted(p.stem for p in (src / "infrared" / split).glob("*.jpg")) != stems:
            sys.exit(f"visible/{split} and infrared/{split} file names differ")
        missing = [s for s in stems if not (src / "Annotations" / f"{s}.xml").is_file()]
        if missing:
            sys.exit(f"{len(missing)} images in {split} have no xml, e.g. {missing[:3]}")
        if a.limit:
            stems = stems[:a.limit]
        for mod in ("rgb", "ir"):
            for kind in ("images", "labels"):
                (dst / mod / kind / split).mkdir(parents=True, exist_ok=True)
        jobs += [(split, s, src, dst, a.min_size) for s in stems]

    stats = {s: Counter() for s in SPLITS}
    done = 0
    with ThreadPoolExecutor(a.workers) as ex:  # I/O bound: threads are enough
        for split, ev in ex.map(process, jobs):
            stats[split].update(ev)
            stats[split]["pairs"] += 1
            done += 1
            if done % 2000 == 0 or done == len(jobs):
                print(f"{done}/{len(jobs)}", flush=True)

    out = {"min_size": a.min_size, "classes": CLASSES,
           "splits": {s: dict(sorted(c.items())) for s, c in stats.items()}}
    (dst / "convert_stats.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
