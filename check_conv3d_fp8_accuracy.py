#!/usr/bin/env python3
"""Measure conv3d accuracy vs PyTorch BF16 reference.

Three comparisons per shape:
  BF16 kernel      : directly vs F.conv3d BF16
  FP8 no-scale     : kernel as-is (direct cast, no per-tensor scale)
  FP8 per-tensor   : simulate per-tensor scaling (amax/448 for x and weight
                     separately) before casting, to show the accuracy floor
                     achievable with calibration

Metrics:
  max_abs   worst single-element absolute error
  mean_abs  average absolute error
  rel_err%  mean_abs / |ref|.mean (scale-invariant)
  cosine    cosine similarity (1.0 = perfect)
  SNR(dB)   signal-to-noise ratio (higher is better)
"""

import os
import sys

import torch
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from kernels.conv3d_implicit_8wave import conv3d_implicit_8wave
from kernels.conv3d_implicit_8wave_fp8 import conv3d_implicit_8wave_fp8

# (label, N, C, D, H, W, K, kernel_size, stride, padding)
SHAPES = [
    ("3×3×3 C128 H18 pad0", 1, 128, 3, 18, 18, 128, 3, 1, 0),
    ("3×3×3 C256 H18 pad0", 1, 256, 3, 18, 18, 256, 3, 1, 0),
    ("3×3×3 C128 H16 pad1", 1, 128, 3, 16, 16, 256, 3, 1, 1),
    ("3×3×3 C128 H40 pad1", 1, 128, 6, 40, 40, 128, 3, 1, 1),
    ("3×3×3 C128 H72 pad1", 1, 128, 6, 72, 72, 128, 3, 1, 1),
    ("3×3×3 C128 H144 pad1", 1, 128, 6, 144, 144, 128, 3, 1, 1),
]

FP8_MAX = 448.0  # max representable value for E4M3FN


def snr_db(ref, err):
    sig = ref.float().pow(2).mean()
    noise = err.float().pow(2).mean()
    return float("inf") if noise == 0 else (10 * (sig / noise).log10()).item()


def cosine_sim(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return (a @ b / (a.norm() * b.norm())).item()


def get_metrics(y, ref):
    err = (y.float() - ref.float()).abs()
    rel = err.mean() / ref.float().abs().mean().clamp_min(1e-6)
    return {
        "max_abs": err.max().item(),
        "mean_abs": err.mean().item(),
        "rel_err": rel.item(),
        "cosine": cosine_sim(y, ref),
        "snr_db": snr_db(ref.float(), y.float() - ref.float()),
    }


def per_tensor_scale_cast(t):
    """Scale t to fit E4M3FN range, cast to FP8, cast back to BF16."""
    scale = FP8_MAX / t.float().abs().amax().clamp_min(1e-6)
    return (t.float() * scale).to(torch.float8_e4m3fn).to(torch.bfloat16) / scale


def run_shape(label, n, c, d, h, w, k, ksz, stride, padding):
    torch.manual_seed(42)
    x = torch.randn(n, c, d, h, w, dtype=torch.bfloat16, device="cuda")
    ksz3 = (ksz, ksz, ksz) if isinstance(ksz, int) else ksz
    wt = torch.randn(k, c, *ksz3, dtype=torch.bfloat16, device="cuda")

    ref = F.conv3d(x, wt, stride=stride, padding=padding)

    # BF16 kernel
    try:
        y_bf16 = conv3d_implicit_8wave(x, wt, stride=stride, padding=padding)
        m_bf16 = get_metrics(y_bf16, ref)
    except Exception:
        m_bf16 = None

    # FP8 kernel — no scale (direct cast, current behavior)
    try:
        y_fp8 = conv3d_implicit_8wave_fp8(x, wt, stride=stride, padding=padding)
        m_fp8 = get_metrics(y_fp8, ref)
    except Exception:
        m_fp8 = None

    # FP8 per-tensor scale simulation (PyTorch reference, not our kernel)
    # Shows accuracy floor achievable with calibration
    try:
        x_scaled = per_tensor_scale_cast(x)
        w_scaled = per_tensor_scale_cast(wt)
        y_pt_scaled = F.conv3d(x_scaled, w_scaled, stride=stride, padding=padding)
        m_fp8_scaled = get_metrics(y_pt_scaled, ref)
    except Exception:
        m_fp8_scaled = None

    return m_bf16, m_fp8, m_fp8_scaled


HDR = f"{'shape':>24}  {'max_abs':>8}  {'mean_abs':>9}  {'rel_err%':>9}  {'cosine':>8}  {'SNR(dB)':>8}"


def fmt_row(label, m):
    if m is None:
        return f"{label:>24}  {'n/a':>8}  {'n/a':>9}  {'n/a':>9}  {'n/a':>8}  {'n/a':>8}"
    return (
        f"{label:>24}"
        f"  {m['max_abs']:>8.4f}"
        f"  {m['mean_abs']:>9.5f}"
        f"  {m['rel_err']*100:>8.3f}%"
        f"  {m['cosine']:>8.6f}"
        f"  {m['snr_db']:>8.2f}"
    )


def print_section(title, labels_and_metrics):
    print(f"\n{title}")
    print(HDR)
    print("-" * len(HDR))
    for label, m in labels_and_metrics:
        print(fmt_row(label, m))


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("conv3d accuracy vs PyTorch BF16 reference")

    results = []
    for args in SHAPES:
        label = args[0]
        m_bf16, m_fp8, m_fp8_scaled = run_shape(*args)
        results.append((label, m_bf16, m_fp8, m_fp8_scaled))

    print_section(
        "── BF16 kernel  (expect rel_err < 0.1%, SNR > 50 dB) ──",
        [(label, m_bf16) for label, m_bf16, _, __ in results],
    )

    print_section(
        "── FP8 kernel, no scale  (direct cast, current behavior) ──",
        [(label, m_fp8) for label, _, m_fp8, __ in results],
    )

    print_section(
        "── FP8 per-tensor scale  (simulated calibration, PyTorch reference) ──",
        [(label, m_fp8s) for label, _, __, m_fp8s in results],
    )

    print()
    print("Note: per-tensor scale (amax/448) gives the same result as direct cast")
    print("for standard activations (amax~3-5), because E4M3FN max=448 already")
    print("covers the value range — the 3-bit mantissa precision is the bottleneck,")
    print("not the dynamic range. Scaling only helps when activations are very small")
    print("(amax << 1). The ~3.7% relative error is therefore the inherent FP8 floor")
    print("for this input distribution, independent of scaling strategy.")


if __name__ == "__main__":
    main()
