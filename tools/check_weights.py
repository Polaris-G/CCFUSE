"""Load YOLOv5 v5.0 COCO checkpoints with the CFT codebase and print backbone output shapes.

Run from the repo root:
    python tools/check_weights.py weights/yolov5s.pt weights/yolov5m.pt
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

for w in sys.argv[1:] or ["weights/yolov5s.pt", "weights/yolov5m.pt"]:
    # v5.0 checkpoints are pickled nn.Modules, so weights_only must be False
    ckpt = torch.load(w, map_location="cpu", weights_only=False)
    model = ckpt["model"].float().eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n=== {w} ===")
    print(f"ckpt keys: {list(ckpt.keys())}, epoch={ckpt.get('epoch')}")
    print(f"params: {n_params / 1e6:.2f} M, nc={model.yaml['nc']}, "
          f"depth_multiple={model.yaml['depth_multiple']}, width_multiple={model.yaml['width_multiple']}")

    # run the backbone (layers 0-9) layer by layer
    x = torch.zeros(1, 3, 640, 640)
    with torch.no_grad():
        for i, m in enumerate(model.model[:10]):
            x = m(x)
            tag = {4: "  <- P3/8  (stage3)", 6: "  <- P4/16 (stage4)", 9: "  <- P5/32 (stage5)"}.get(i, "")
            print(f"  layer {i:2d} {type(m).__name__:6s} -> {tuple(x.shape)}{tag}")
