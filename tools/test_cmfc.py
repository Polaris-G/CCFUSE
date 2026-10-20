"""Step-4 unit tests for CMFC (models/ccsfuse_modules.py). Run from the repo root:
    python tools/test_cmfc.py
Each check prints PASS/FAIL; the script exits non-zero if any check fails.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from models.ccsfuse_modules import CMFC  # noqa: E402

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)
failed = []


def check(name, ok, info=''):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{'  ' + info if info else ''}")
    if not ok:
        failed.append(name)


def reference(m, f_q, f_kv, qkv_q, qkv_kv, proj, branch):
    """Independent cross-attention for one direction: 'sic' via F.scaled_dot_product_attention, 'ris' by hand."""
    b, c, h, w = f_q.shape
    q, _, _ = qkv_q(f_q)
    _, k, v = qkv_kv(f_kv)
    q, k, v = (t.reshape(b, -1, m.h, m.d).transpose(1, 2) for t in (q, k, v))
    if branch == 'sic':
        o = F.scaled_dot_product_attention(q, k, v)
    else:
        o = F.relu(q @ k.transpose(-2, -1) * m.d ** -0.5) ** 2 @ v
    o = proj(o.transpose(1, 2).reshape(b, -1, c))
    return f_q + o.transpose(1, 2).reshape(b, c, h, w)


# 1. shapes (plan: torch.randn(2,192,80,80) x2), all three YOLOv5m stages with plan SR ratios 8/4/2
for c, s, r in ((192, 80, 8), (384, 40, 4), (768, 20, 2)):
    m = CMFC(c, sr_ratio=r).to(dev)
    x_rgb, x_ir = torch.randn(2, c, s, s, device=dev), torch.randn(2, c, s, s, device=dev)
    o_rgb, o_ir = m(x_rgb, x_ir)
    o_rgb2, _ = m([x_rgb, x_ir])  # list call used by the YOLO graph
    check(f'shape c={c} {s}x{s} r={r}', o_rgb.shape == x_rgb.shape and o_ir.shape == x_ir.shape
          and torch.equal(o_rgb, o_rgb2), f'{tuple(o_rgb.shape)}, {tuple(o_ir.shape)}')

# 2. init: w_SIC = w_RIS = 0.5, shared lambda
m = CMFC(192, sr_ratio=8).to(dev).eval()
w_rgb, w_ir = m.branch_weights()
check('init weights 0.5/0.5, shared lambda', torch.allclose(w_rgb, torch.tensor([.5, .5], device=dev))
      and m.lambda_rgb is m.lambda_ir, f'w={w_rgb.tolist()}')
check('share_lambda=False -> separate lambdas', CMFC(192, share_lambda=False).lambda_rgb is not
      CMFC(192, share_lambda=False).lambda_ir)

# 3. w_SIC = 1 -> plain cross-attention ; w_RIS = 1 -> ReLU^2 @ V   (fp32, no autocast)
x_rgb, x_ir = torch.randn(2, 192, 80, 80, device=dev), torch.randn(2, 192, 80, 80, device=dev)
with torch.no_grad():
    for branch, lam in (('sic', [50., -50.]), ('ris', [-50., 50.])):
        m.lambda_rgb.copy_(torch.tensor(lam))
        o_rgb, o_ir = m(x_rgb, x_ir)
        r_rgb = reference(m, x_rgb, x_ir, m.qkv_rgb, m.qkv_ir, m.proj_rgb, branch)
        r_ir = reference(m, x_ir, x_rgb, m.qkv_ir, m.qkv_rgb, m.proj_ir, branch)
        err = max((o_rgb - r_rgb).abs().max().item(), (o_ir - r_ir).abs().max().item())
        rel = err / max(r_rgb.abs().max().item(), 1e-12)
        check(f'w_{branch.upper()}=1 equals reference {branch}', rel < 1e-4, f'max abs err {err:.2e}, rel {rel:.2e}')
    m.lambda_rgb.fill_(1.0)
    # Table II: fixed branch weights, no lambda parameter
    mf = CMFC(192, sr_ratio=8, branch_weights=(1.0, 0.0)).to(dev).eval()
    mf.load_state_dict({k: v for k, v in m.state_dict().items() if 'lambda' not in k}, strict=False)
    o_rgb, _ = mf(x_rgb, x_ir)
    r_rgb = reference(mf, x_rgb, x_ir, mf.qkv_rgb, mf.qkv_ir, mf.proj_rgb, 'sic')
    err = (o_rgb - r_rgb).abs().max().item()
    check('branch_weights=(1,0): no lambda, equals reference sic',
          not any('lambda' in n for n, _ in mf.named_parameters()) and err / r_rgb.abs().max().item() < 1e-4,
          f'max abs err {err:.2e}')

# 4. fp16 / AMP: finite forward + backward, also with large-magnitude features (ReLU^2 is unnormalized)
if dev.type == 'cuda':
    for attn_fp32 in (True, False):
        for amp_scale in (1.0, 30.0):
            m = CMFC(192, sr_ratio=8, attn_fp32=attn_fp32).to(dev).train()
            a = (torch.randn(2, 192, 80, 80, device=dev) * amp_scale).requires_grad_()
            b = (torch.randn(2, 192, 80, 80, device=dev) * amp_scale).requires_grad_()
            with torch.autocast('cuda', dtype=torch.float16):
                o_rgb, o_ir = m(a, b)
                loss = (o_rgb.float() ** 2).mean() + (o_ir.float() ** 2).mean()
            loss.backward()
            grads = [a.grad, b.grad] + [p.grad for p in m.parameters()]
            fin = bool(torch.isfinite(o_rgb).all() and torch.isfinite(o_ir).all()) and \
                all(g is not None and torch.isfinite(g).all() for g in grads)
            check(f'AMP fp16 finite (attn_fp32={attn_fp32}, input x{amp_scale:g})', fin,
                  f'out dtype {o_rgb.dtype}, max |out| {o_rgb.abs().max().item():.3g}')
else:
    print('[SKIP] AMP checks need CUDA')

# 5. gradients reach both inputs and lambda (each direction depends on the other modality)
m = CMFC(192, sr_ratio=8).to(dev).train()
a = torch.randn(2, 192, 80, 80, device=dev, requires_grad=True)
b = torch.randn(2, 192, 80, 80, device=dev, requires_grad=True)
o_rgb, _ = m(a, b)
o_rgb.sum().backward()  # rgb output only -> IR input must still get gradient through K/V
check('grad: rgb output -> both inputs and lambda', a.grad.abs().sum() > 0 and b.grad.abs().sum() > 0
      and m.lambda_rgb.grad is not None and m.lambda_rgb.grad.abs().sum() > 0,
      f'|dL/d lambda| = {m.lambda_rgb.grad.abs().tolist()}')

# 6. proj_init='zero': identity at init; proj gets gradient now, Q/K/V after one optimizer step
m = CMFC(192, sr_ratio=8, proj_init='zero').to(dev).train()
a = torch.randn(2, 192, 80, 80, device=dev, requires_grad=True)
b = torch.randn(2, 192, 80, 80, device=dev, requires_grad=True)
o_rgb, o_ir = m(a, b)
ident = torch.equal(o_rgb, a) and torch.equal(o_ir, b)
(o_rgb.pow(2).mean() + o_ir.pow(2).mean()).backward()
g_proj = m.proj_rgb.weight.grad.abs().sum().item()
opt = torch.optim.SGD(m.parameters(), lr=0.1)
opt.step()
opt.zero_grad()
o_rgb, o_ir = m(a, b)
(o_rgb.pow(2).mean() + o_ir.pow(2).mean()).backward()
g_q = m.qkv_rgb.q.weight.grad.abs().sum().item()
check("proj_init='zero': output == input at init, proj then Q/K/V get gradients", ident and g_proj > 0 and g_q > 0,
      f'|grad proj| {g_proj:.3g}, |grad q| after 1 step {g_q:.3g}')

# 7. memory at training scale: batch 4, the three YOLOv5m stages, AMP, fwd + bwd
if dev.type == 'cuda':
    ms = [CMFC(c, sr_ratio=r).to(dev) for c, r in ((192, 8), (384, 4), (768, 2))]
    xs = [(torch.randn(4, c, s, s, device=dev, requires_grad=True),
           torch.randn(4, c, s, s, device=dev, requires_grad=True)) for c, s in ((192, 80), (384, 40), (768, 20))]
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    with torch.autocast('cuda', dtype=torch.float16):
        loss = sum((o.float() ** 2).mean() for m_, (x1, x2) in zip(ms, xs) for o in m_(x1, x2))
    loss.backward()
    peak = (torch.cuda.max_memory_allocated() - base) / 2 ** 20
    n_par = sum(p.numel() for m_ in ms for p in m_.parameters()) / 1e6
    print(f'[INFO] 3 CMFC (stage3/4/5, r=8/4/2): {n_par:.2f} M params, extra peak memory fwd+bwd batch 4: {peak:.0f} MiB')

print('ALL CMFC TESTS PASSED' if not failed else f'FAILED: {failed}')
sys.exit(1 if failed else 0)
