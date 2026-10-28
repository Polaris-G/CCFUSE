"""Environment check for the CCSFuse conda env.

Pass/fail does not assume a vendor, an OS, or a particular GPU. The VRAM floor is a
fraction of the card that is actually visible, and the GPU's compute capability must
appear in torch.cuda.get_arch_list() (cu124 wheels do not list sm_120).

Usage (from the repo root, after `conda activate ccsfuse`):
    python check_env.py
"""
import platform
import sys
import time

ok = True


def section(title):
    print(f"\n=== {title} ===")


def fail(msg):
    global ok
    ok = False
    print(f"[FAIL] {msg}")


# 1. versions -----------------------------------------------------------------
section("Versions")
print(f"Python      : {sys.version.split()[0]} ({platform.platform()})")
try:
    import numpy as np
    print(f"numpy       : {np.__version__}")
    if tuple(int(v) for v in np.__version__.split(".")[:2]) >= (1, 24):
        fail("numpy >= 1.24 removes np.int, which the CFT codebase uses; install numpy<1.24")
except ImportError:
    fail("numpy not installed")
try:
    import cv2
    print(f"opencv      : {cv2.__version__}")
except ImportError:
    fail("opencv-python not installed")

import torch
import torch.nn as nn
import torchvision

print(f"torch       : {torch.__version__}  (built with CUDA {torch.version.cuda})")
print(f"torchvision : {torchvision.__version__}")

# 2. CUDA ---------------------------------------------------------------------
section("CUDA")
if not torch.cuda.is_available():
    built = []
    try:
        built = list(torch.cuda.get_arch_list())
    except Exception:
        pass
    fail(f"torch.cuda.is_available() is False (CPU-only torch or driver problem). arch list: {built}")
    sys.exit(1)
dev = torch.device("cuda:0")
props = torch.cuda.get_device_properties(dev)
arch = f"sm_{props.major}{props.minor}"
built = list(torch.cuda.get_arch_list())
print(f"GPU         : {props.name}, compute capability {props.major}.{props.minor} ({arch}), {props.multi_processor_count} SMs")
print(f"cuDNN       : {torch.backends.cudnn.version()} (enabled={torch.backends.cudnn.enabled})")
print(f"bf16 support: {torch.cuda.is_bf16_supported()}")
print(f"arch list   : {', '.join(built) if built else '(empty)'}")


def _arch_built(token, archs):
    # sm_120 matches sm_120 and sm_120a. A shorter token such as sm_10 must not match sm_100.
    for item in archs:
        name = item.replace("compute_", "sm_")
        if name == token:
            return True
        if name.startswith(token) and (len(name) == len(token) or not name[len(token)].isdigit()):
            return True
    return False


if not _arch_built(arch, built):
    fail(f"{arch} is not in torch.cuda.get_arch_list()={built}; this wheel has no kernels for the current GPU")

# 3. memory -------------------------------------------------------------------
section("GPU memory")
free, total = torch.cuda.mem_get_info(dev)
total_gb = total / 2**30
free_gb = free / 2**30
print(f"Driver view : free {free_gb:.2f} GB / total {total_gb:.2f} GB "
      f"(used by other processes: {(total - free) / 2**30:.2f} GB)")

# Allocate in 256 MB chunks to see how much PyTorch can really get.
# Floor scales with this GPU: 70% of an 8 GB card is 5.6 GB, of a 16 GB card is 11.2 GB.
min_frac = 0.70
chunks, chunk_bytes = [], 256 * 2**20
try:
    while True:
        chunks.append(torch.empty(chunk_bytes, dtype=torch.uint8, device=dev))
except torch.cuda.OutOfMemoryError:
    pass
except RuntimeError as exc:
    if "out of memory" not in str(exc).lower():
        raise
allocatable = len(chunks) * chunk_bytes
alloc_gb = allocatable / 2**30
print(f"Allocatable : {alloc_gb:.2f} GB (256 MB chunks until OOM); floor {min_frac:.0%} of {total_gb:.2f} GB = {min_frac * total_gb:.2f} GB")
del chunks
torch.cuda.empty_cache()
if alloc_gb > total_gb * 1.02:
    fail(f"allocatable {alloc_gb:.2f} GB exceeds physical VRAM {total_gb:.2f} GB; "
         "allocations are spilling into system RAM instead of raising OOM")
elif alloc_gb < min_frac * total_gb:
    fail(f"allocatable {alloc_gb:.2f} GB is below {min_frac:.0%} of this GPU's {total_gb:.2f} GB")

# 4. AMP training step ----------------------------------------------------------
section("AMP (fp16 autocast + GradScaler)")
torch.manual_seed(0)
model = nn.Sequential(
    nn.Conv2d(3, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.SiLU(),
    nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.SiLU(),
    nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 10),
).to(dev)
opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scaler = torch.amp.GradScaler("cuda")
x = torch.randn(4, 3, 640, 640, device=dev)
y = torch.randint(0, 10, (4,), device=dev)
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
for step in range(5):
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16):
        out = model(x)
        loss = nn.functional.cross_entropy(out, y)
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
torch.cuda.synchronize()
grad_ok = all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
print(f"output dtype: {out.dtype} (expect torch.float16)")
print(f"loss        : {loss.item():.4f}, finite={torch.isfinite(loss).item()}, grads finite={grad_ok}")
print(f"5 steps     : {time.time() - t0:.2f}s, peak mem {torch.cuda.max_memory_allocated() / 2**20:.0f} MB")
if out.dtype != torch.float16 or not torch.isfinite(loss) or not grad_ok:
    fail("AMP step produced wrong dtype or non-finite values")

# 5. CMFC-style attention scores: fp16 vs fp32 ----------------------------------
section("CMFC attention scores (stage3: N=80*80, r=8 -> M=100, 8 heads, d=24)")
B, heads, N, M, d = 2, 8, 80 * 80, 100, 192 // 8
q = torch.randn(B, heads, N, d, device=dev) * 4  # exaggerate magnitude to stress fp16
k = torch.randn(B, heads, M, d, device=dev) * 4
v = torch.randn(B, heads, M, d, device=dev)
with torch.autocast("cuda", dtype=torch.float16):
    s16 = q @ k.transpose(-2, -1) / d ** 0.5
    ris16 = torch.relu(s16) ** 2
    with torch.autocast("cuda", enabled=False):  # the recommended fp32 island
        s32 = q.float() @ k.float().transpose(-2, -1) / d ** 0.5
        ris32 = torch.relu(s32) ** 2
        out32 = (0.5 * s32.softmax(-1) + 0.5 * ris32) @ v.float()
print(f"fp16 RIS    : max {ris16.float().max().item():.1f}, has inf/nan={not torch.isfinite(ris16).all().item()}")
print(f"fp32 RIS    : max {ris32.max().item():.1f}, output finite={torch.isfinite(out32).all().item()}")
print("(fp16 inf here only means the fp32 island is necessary; it is not an environment failure)")
if not torch.isfinite(out32).all():
    fail("fp32 attention produced non-finite values")

# 6. CFT codebase imports -------------------------------------------------------
section("CFT codebase imports")
try:
    from models.yolo import Model  # noqa: F401
    from utils.datasets import LoadMultiModalImagesAndLabels  # noqa: F401
    print("models.yolo / utils.datasets import OK")
except Exception as e:  # noqa: BLE001
    fail(f"import failed: {type(e).__name__}: {e}")

print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
sys.exit(0 if ok else 1)
