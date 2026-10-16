"""Step-5 unit tests for AFSF (models/ccsfuse_modules.py). Run from the repo root:
    python tools/test_afsf.py
Each check prints PASS/FAIL; the script exits non-zero if any check fails.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from models.ccsfuse_modules import AFSF, CMFC  # noqa: E402

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)
failed = []
STAGES = ((192, 80, 8), (384, 40, 4), (768, 20, 2))  # YOLOv5m P3/P4/P5 channels, size @640, CMFC sr_ratio


def check(name, ok, info=''):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{'  ' + info if info else ''}")
    if not ok:
        failed.append(name)


def P(*ms):
    return sum(p.numel() for m in ms for p in m.parameters()) / 1e6


# 1. shapes, both call styles
for c, s, _ in STAGES:
    m = AFSF(c).to(dev).eval()
    a, b = torch.randn(2, c, s, s, device=dev), torch.randn(2, c, s, s, device=dev)
    o = m(a, b)
    check(f'shape c={c} {s}x{s}', o.shape == a.shape and torch.equal(o, m([a, b])), f'{tuple(o.shape)}')

# 2. Eqs. (7)-(10) recomputed by hand with the module's own layers
m = AFSF(192).to(dev).eval()
a, b = torch.randn(2, 192, 80, 80, device=dev), torch.randn(2, 192, 80, 80, device=dev)
with torch.no_grad():
    w_rgb = torch.sigmoid(m.desc_rgb.weight[:, :, 0, 0] @ a.mean((2, 3)).T + m.desc_rgb.bias[:, None]).T  # Eq. (7)
    w_ir = torch.sigmoid(m.desc_ir.weight[:, :, 0, 0] @ b.mean((2, 3)).T + m.desc_ir.bias[:, None]).T
    w = F.linear(F.gelu(F.linear(torch.cat((w_rgb, w_ir), 1), m.fc1.weight, m.fc1.bias)), m.fc2.weight, m.fc2.bias)
    ref = m.fuse(torch.cat((a * w[:, :192, None, None], b * w[:, 192:, None, None]), 1))  # Eqs. (8)-(10)
    err = (m(a, b) - ref).abs().max().item()
check('output equals hand-computed Eqs. (7)-(10)', err < 1e-4, f'max abs err {err:.2e}')
with torch.no_grad():
    wr, wi = m.channel_weights(a, b)
check('default (no sigmoid after MLP) allows negative weights', bool((torch.cat((wr, wi)) < 0).any()),
      f'w~ range [{min(wr.min(), wi.min()).item():.3f}, {max(wr.max(), wi.max()).item():.3f}]')
ms = AFSF(192, sigmoid_after_mlp=True).to(dev).eval()
with torch.no_grad():
    wr, wi = ms.channel_weights(a, b)
check('sigmoid_after_mlp=True -> weights in (0, 1)', bool(((wr > 0) & (wr < 1)).all() and ((wi > 0) & (wi < 1)).all()))
check('reduction=4 -> hidden 2C/4', m.fc1.out_features == 2 * 192 // 4 and AFSF(192, reduction=2).fc1.out_features == 192)

# 3. gradients reach both inputs and every parameter
m = AFSF(192).to(dev).train()
a = torch.randn(2, 192, 80, 80, device=dev, requires_grad=True)
b = torch.randn(2, 192, 80, 80, device=dev, requires_grad=True)
m(a, b).pow(2).mean().backward()
no_grad = [n for n, p in m.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
check('grad reaches both inputs and all params', a.grad.abs().sum() > 0 and b.grad.abs().sum() > 0 and not no_grad,
      f'params without grad: {no_grad}' if no_grad else '')

# 4. AMP fp16 finite forward + backward
if dev.type == 'cuda':
    m = AFSF(192).to(dev).train()
    a = (torch.randn(2, 192, 80, 80, device=dev) * 30).requires_grad_()
    b = (torch.randn(2, 192, 80, 80, device=dev) * 30).requires_grad_()
    with torch.autocast('cuda', dtype=torch.float16):
        o = m(a, b)
        loss = o.float().pow(2).mean()
    loss.backward()
    fin = bool(torch.isfinite(o).all()) and all(torch.isfinite(g).all() for g in [a.grad, b.grad] +
                                                 [p.grad for p in m.parameters()])
    check('AMP fp16 finite (input x30)', fin, f'out dtype {o.dtype}')

# 5. CMFC -> AFSF chain per stage (what step 6 wires into the network) + sizes
for c, s, r in STAGES:
    cm, af = CMFC(c, sr_ratio=r).to(dev), AFSF(c).to(dev)
    a = torch.randn(2, c, s, s, device=dev, requires_grad=True)
    b = torch.randn(2, c, s, s, device=dev, requires_grad=True)
    o = af(cm(a, b))
    o.mean().backward()
    check(f'CMFC->AFSF chain c={c}', o.shape == a.shape and a.grad.abs().sum() > 0 and b.grad.abs().sum() > 0
          and cm.lambda_rgb.grad is not None)
af_m = P(*[AFSF(c) for c, _, _ in STAGES])
cm_m = P(*[CMFC(c, sr_ratio=r) for c, _, r in STAGES])
print(f'[INFO] YOLOv5m: 3x AFSF {af_m:.2f} M, 3x CMFC {cm_m:.2f} M; Add baseline 33.43 M -> '
      f'CCSFuse ~{33.43 + af_m + cm_m:.2f} M (paper 33.50 M)')

print('ALL AFSF TESTS PASSED' if not failed else f'FAILED: {failed}')
sys.exit(1 if failed else 0)
