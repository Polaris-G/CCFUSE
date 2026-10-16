"""CCSFuse modules (Zhang et al., IEEE TGRS 2026).

CMFC: Cross-Modal Feature Compensation (paper Sec. III-C, Eqs. (1)-(6), Fig. 3).
Every setting the paper does not give is a constructor argument and marked 论文未给出，推断
(see CCSFuse_复现方案.md, Sec. 2 and Sec. 8).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class _ModalityQKV(nn.Module):
    """Eq. (1) for one modality: Q on full-resolution tokens, K/V on spatially reduced tokens (PVT SRA [49])."""

    def __init__(self, c, sr_ratio):
        super().__init__()
        self.q = nn.Linear(c, c)
        self.k = nn.Linear(c, c)
        self.v = nn.Linear(c, c)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            # 论文未给出，推断: PVT-style SR = Conv(k=r, s=r) + LayerNorm
            self.sr = nn.Conv2d(c, c, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(c)

    def forward(self, f):  # f: B,C,H,W -> q: B,N,C ; k, v: B,M,C
        q = self.q(f.flatten(2).transpose(1, 2))
        if self.sr_ratio > 1:
            f = self.sr(f)
            kv = self.norm(f.flatten(2).transpose(1, 2))
        else:
            kv = f.flatten(2).transpose(1, 2)
        return q, self.k(kv), self.v(kv)


class CMFC(nn.Module):
    """Cross-Modal Feature Compensation. forward(f_rgb, f_ir) or forward([f_rgb, f_ir]) -> (f_rgb_cmfc, f_ir_cmfc).

    RGB queries attend to IR keys/values (f_{rgb<-ir}) and vice versa. Attention = w_SIC * Softmax(QK^T/sqrt(d))
    + w_RIS * ReLU(QK^T/sqrt(d))^2 with (w_SIC, w_RIS) = softmax(lambda1, lambda2), lambdas learnable, init 1
    (Eqs. (2)-(5)); output = f + f_{rgb<-ir} (Eq. (6)). Shapes are unchanged.

    Args (all 论文未给出，推断 except c):
        c: channels of the stage feature.
        num_heads: int, or 'auto' = max(c // 64, 1).
        sr_ratio: spatial reduction of K/V (plan default 8 / 4 / 2 for stage 3 / 4 / 5).
        out_proj: Linear(c, c) after attention @ V (Transformer convention, not in the paper's equations).
        share_lambda: one (lambda1, lambda2) pair for both directions; False -> one pair per direction.
        ris_norm: divide the ReLU^2 branch by M (number of K tokens) to bound its magnitude.
        attn_fp32: compute QK^T, both branches and @V in fp32 even under AMP (ReLU^2 is unnormalized).
    """

    def __init__(self, c, num_heads=8, sr_ratio=4, out_proj=True, share_lambda=True, ris_norm=False,
                 attn_fp32=True):
        super().__init__()
        num_heads = max(c // 64, 1) if num_heads == 'auto' else int(num_heads)
        assert c % num_heads == 0, f'CMFC: channels {c} not divisible by num_heads {num_heads}'
        self.c, self.h, self.d = c, num_heads, c // num_heads
        self.scale = self.d ** -0.5
        self.ris_norm, self.attn_fp32 = ris_norm, attn_fp32
        self.qkv_rgb = _ModalityQKV(c, sr_ratio)  # paper: W_rgb != W_ir (separate projections per modality)
        self.qkv_ir = _ModalityQKV(c, sr_ratio)
        self.proj_rgb = nn.Linear(c, c) if out_proj else nn.Identity()
        self.proj_ir = nn.Linear(c, c) if out_proj else nn.Identity()
        # Eq. (4): learnable branch logits, init 1 -> w_SIC = w_RIS = 0.5.
        # NOTE: plain nn.Parameter, not collected by train.py's pg0/pg1/pg2 -> must be added to the optimizer.
        self.lambda_rgb = nn.Parameter(torch.ones(2))
        self.lambda_ir = self.lambda_rgb if share_lambda else nn.Parameter(torch.ones(2))

    def branch_weights(self):
        """(w_SIC, w_RIS) for the rgb<-ir and ir<-rgb directions."""
        return F.softmax(self.lambda_rgb, 0), F.softmax(self.lambda_ir, 0)

    def _heads(self, t):  # B,L,C -> B,h,L,d
        return t.reshape(t.shape[0], t.shape[1], self.h, self.d).transpose(1, 2)

    def _attend(self, q, k, v, lam):
        """Eqs. (2)-(5): (w_SIC * Softmax(s) + w_RIS * ReLU(s)^2) @ V with s = QK^T / sqrt(d)."""
        q, k, v = self._heads(q), self._heads(k), self._heads(v)
        with torch.autocast(device_type=q.device.type, enabled=not self.attn_fp32 and torch.is_autocast_enabled()):
            if self.attn_fp32:
                q, k, v = q.float(), k.float(), v.float()
            s = (q @ k.transpose(-2, -1)) * self.scale  # B,h,N,M
            ris = F.relu(s) ** 2
            if self.ris_norm:
                ris = ris / s.shape[-1]
            w = F.softmax(lam.float() if self.attn_fp32 else lam, 0)
            out = (w[0] * s.softmax(-1) + w[1] * ris) @ v  # B,h,N,d
        return out.transpose(1, 2).reshape(out.shape[0], -1, self.c)  # B,N,C

    def forward(self, f_rgb, f_ir=None):
        if f_ir is None:  # called from the YOLO graph with a [rgb, ir] list
            f_rgb, f_ir = f_rgb
        b, c, h, w = f_rgb.shape
        q_rgb, k_rgb, v_rgb = self.qkv_rgb(f_rgb)
        q_ir, k_ir, v_ir = self.qkv_ir(f_ir)
        rgb_from_ir = self.proj_rgb(self._attend(q_rgb, k_ir, v_ir, self.lambda_rgb).to(f_rgb.dtype))
        ir_from_rgb = self.proj_ir(self._attend(q_ir, k_rgb, v_rgb, self.lambda_ir).to(f_ir.dtype))
        to_map = lambda t: t.transpose(1, 2).reshape(b, c, h, w)
        return f_rgb + to_map(rgb_from_ir), f_ir + to_map(ir_from_rgb)  # Eq. (6)
