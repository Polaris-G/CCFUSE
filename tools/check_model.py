"""Step-1 check: build a dual-stream model, transfer COCO weights, run one real batch forward+backward.

Prints the per-stream COCO transfer, verifies the IR stream really holds COCO weights, the fused P3/P4/P5 and
Detect output shapes, the loss, peak GPU memory, params and GFLOPs. Run from the repo root:
    python tools/check_model.py --cfg models/ccsfuse/yolov5m_dual_add.yaml --weights weights/yolov5m.pt
"""
import argparse
import logging
import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import thop  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from models.yolo_test import Model  # noqa: E402
from utils.ccsfuse import load_coco_dual_stream  # noqa: E402
from utils.datasets import LoadMultiModalImagesAndLabels  # noqa: E402
from utils.loss import ComputeLoss  # noqa: E402
from utils.torch_utils import intersect_dicts  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cfg", default="models/ccsfuse/yolov5m_dual_add.yaml")
ap.add_argument("--weights", default="weights/yolov5m.pt")
ap.add_argument("--data", default="data/multispectral/DroneVehicle.yaml")
ap.add_argument("--hyp", default="data/hyp.ccsfuse_dronevehicle.yaml")
ap.add_argument("--batch-size", type=int, default=4)
ap.add_argument("--img-size", type=int, default=640)
ap.add_argument("--pretrain-neck", action="store_true")
ap.add_argument("--fused-types", default="Add,AFSF", help="module types whose outputs are the fused f3/f4/f5")
a = ap.parse_args()
logging.disable(logging.INFO)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = yaml.safe_load(open(a.data, encoding="utf-8"))
hyp = yaml.safe_load(open(a.hyp, encoding="utf-8"))
model = Model(a.cfg, ch=6, nc=data["nc"]).to(dev)

# --- COCO transfer: CFT's original intersect vs. the dual-stream mapping
coco = torch.load(a.weights, map_location="cpu", weights_only=False)["model"].float()
old = intersect_dicts(coco.state_dict(), model.state_dict(), exclude=["anchor"])
sd, stats = load_coco_dual_stream(model, coco, pretrain_neck=a.pretrain_neck)
model.load_state_dict(sd, strict=False)
print(f"CFT intersect_dicts would transfer {len(old)}/{len(model.state_dict())} (RGB stream only)")
print("dual-stream transfer:", ", ".join(f"{g} {x}/{y}" for g, (x, y) in stats.items()),
      f"-> Transferred {len(sd)}/{len(model.state_dict())}")
pm = model.yaml["pretrain_map"]
w_coco = coco.model[0].conv.conv.weight
for g in ("rgb", "ir"):
    w = model.model[pm[g][0]].conv.conv.weight.cpu()
    print(f"  {g:3s} Focus weight == COCO layer 0: {torch.equal(w, w_coco)}")

# --- one real batch
ds = LoadMultiModalImagesAndLabels(data["train_rgb"], data["train_ir"], a.img_size, a.batch_size, augment=True,
                                   hyp=hyp, rect=False, stride=32)
imgs, targets, _, _ = LoadMultiModalImagesAndLabels.collate_fn([ds[i] for i in range(a.batch_size)])
imgs = imgs.to(dev).float() / 255
x_rgb, x_ir = imgs[:, :3], imgs[:, 3:]
print(f"batch: {tuple(imgs.shape)} -> rgb {tuple(x_rgb.shape)}, ir {tuple(x_ir.shape)}, {len(targets)} targets")

fused_types = tuple(a.fused_types.split(","))
shapes = {}


def record_shape(mod, inp, out):  # must return None, otherwise the hook replaces the module output
    outs = out if isinstance(out, (list, tuple)) else [out]
    shapes.setdefault(mod.i, (mod.type.split(".")[-1], [tuple(t.shape) for t in outs]))


hooks = [m.register_forward_hook(record_shape) for m in model.model
         if m.type.split(".")[-1] in fused_types + ("Detect",)]

model.hyp, model.gr, model.nc = hyp, 1.0, data["nc"]
compute_loss = ComputeLoss(model)
model.train()
torch.cuda.reset_peak_memory_stats()
with torch.autocast("cuda", enabled=dev.type == "cuda"):
    pred = model(x_rgb, x_ir)
    loss, items = compute_loss(pred, targets.to(dev))
loss.backward()
for h in hooks:
    h.remove()
for i, (t, s) in sorted(shapes.items()):
    print(f"  layer {i:2d} {t:6s} -> {s}")
print(f"loss {loss.item():.3f} (box/obj/cls {items[0]:.3f}/{items[1]:.3f}/{items[2]:.3f}), finite: "
      f"{bool(torch.isfinite(loss))}; grads finite: "
      f"{all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)}")
if dev.type == "cuda":
    print(f"peak GPU memory (fwd+bwd, batch {a.batch_size}, AMP): {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")

# --- size
n_params = sum(p.numel() for p in model.parameters())
z = torch.zeros(1, 3, a.img_size, a.img_size, device=dev)
macs = thop.profile(deepcopy(model).eval(), inputs=(z, z), verbose=False)[0]
print(f"params {n_params / 1e6:.2f} M, GFLOPs @{a.img_size} {2 * macs / 1e9:.2f}  (paper CCSFuse: 33.50 M, 174.12)")
