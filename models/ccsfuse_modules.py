"""CCSFuse modules (Zhang et al., IEEE TGRS 2026).

CMFC: Cross-Modal Feature Compensation (paper Sec. III-C, Eqs. (1)-(6), Fig. 3).
AFSF: Adaptive Feature Selective Fusion (paper Sec. III-D, Eqs. (7)-(10), Fig. 4).
Every setting the paper does not give is a constructor argument and marked 论文未给出，推断
(see CCSFuse_复现方案.md, Sec. 2, 3 and 8).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common import Conv


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


class AFSF(nn.Module):
    """Adaptive Feature Selective Fusion. forward(f_rgb, f_ir) or forward([f_rgb, f_ir]) -> fused B,C,H,W.

    Eq. (7):  w_m = Sigmoid(Conv1x1(GAP(f_m)))                for m in {rgb, ir}
    Eq. (8):  [w~_rgb, w~_ir] = FC2(GELU(FC1(concat(w_rgb, w_ir))))
    Eq. (9):  f^c_m = f_m * w~_m                              (channel recalibration)
    Eq. (10): f = Conv1x1(concat(f^c_rgb, f^c_ir))            (2C -> C)

    Args (论文未给出，推断 except c):
        c: channels of each input (= output channels).
        reduction: FC1 hidden size = 2C // reduction (plan default 4).
        sigmoid_after_mlp: apply Sigmoid to the MLP output; the paper writes none, so the default is False
            (faithful) and the recalibration weights may be negative.
    Also inferred: one independent Conv1x1 (C -> C, with bias) per modality in Eq. (7); the 2C MLP output is split
    by channel into w~_rgb (first C) and w~_ir (last C); Eq. (10) uses YOLOv5 Conv (Conv + BN + SiLU).
    """

    def __init__(self, c, reduction=4, sigmoid_after_mlp=False):
        super().__init__()
        self.c, self.sigmoid_after_mlp = c, sigmoid_after_mlp
        self.desc_rgb = nn.Conv2d(c, c, 1)  # Eq. (7)
        self.desc_ir = nn.Conv2d(c, c, 1)
        self.fc1 = nn.Linear(2 * c, max(2 * c // reduction, 1))  # Eq. (8)
        self.fc2 = nn.Linear(max(2 * c // reduction, 1), 2 * c)
        self.fuse = Conv(2 * c, c, 1)  # Eq. (10)

    def channel_weights(self, f_rgb, f_ir):
        """Eqs. (7)-(8): recalibration weights w~_rgb, w~_ir, each B,C,1,1."""
        w_rgb = torch.sigmoid(self.desc_rgb(F.adaptive_avg_pool2d(f_rgb, 1)))
        w_ir = torch.sigmoid(self.desc_ir(F.adaptive_avg_pool2d(f_ir, 1)))
        w = self.fc2(F.gelu(self.fc1(torch.cat((w_rgb, w_ir), 1).flatten(1))))  # B,2C
        if self.sigmoid_after_mlp:
            w = torch.sigmoid(w)
        w = w[..., None, None]
        return w[:, :self.c], w[:, self.c:]

    def forward(self, f_rgb, f_ir=None):
        if f_ir is None:  # called from the YOLO graph with a [rgb, ir] list
            f_rgb, f_ir = f_rgb
        w_rgb, w_ir = self.channel_weights(f_rgb, f_ir)
        return self.fuse(torch.cat((f_rgb * w_rgb, f_ir * w_ir), 1))  # Eqs. (9)-(10)