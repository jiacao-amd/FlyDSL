#!/usr/bin/env python3
"""Benchmark conv3d_implicit_8wave (BF16) and conv3d_implicit_8wave_fp8 (FP8)
vs PyTorch (torch.nn.functional.conv3d).

Kernel: 3×3×3, stride=1, pad=1. C=K=128, D=6 (Do=6 same-size).
NPQ = 6 × H × W sweeps from ~10k to ~170k (H=W multiples of 8).

Env vars:
  BENCH_IDX=<int>       run one shape (shell runner avoids OOM)
  BENCH_HEADER_ONLY=1   print header then exit
  BENCH_ITERS=<int>     iteration count (default 100)
  BENCH_WARMUP=<int>    warmup count (default 20)
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

WARMUP = int(os.environ.get("BENCH_WARMUP", 20))
ITERS = int(os.environ.get("BENCH_ITERS", 100))

KT, KH, KW = 3, 3, 3
D = 6
STRIDE = 1
PAD = 1  # same-size: Do=D=6
N = 1
C = 128
K = 128

# H=W chosen so NPQ = 6*H*H spans ~10k to ~170k, all %128==0
HW_VALS = [40, 56, 72, 104, 144, 168]

SHAPES = [(f"NPQ={6*hw*hw//1000}k H={hw}", hw) for hw in HW_VALS]

COL = 10


def bench_us(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    start.record()
    for _ in range(ITERS):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / ITERS  # µs


def tflops(npq, us):
    ops = 2 * N * K * (C * KT * KH * KW) * npq
    return ops / (us * 1e-6) / 1e12


def run_shape(hw):
    npq = N * D * hw * hw
    x = torch.randn(N, C, D, hw, hw, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(K, C, KT, KH, KW, dtype=torch.bfloat16, device="cuda")

    def try_warmup(fn):
        try:
            fn()
            return True
        except Exception:
            return False

    bf16_ok = try_warmup(lambda: conv3d_implicit_8wave(x, w, stride=STRIDE, padding=PAD))
    fp8_ok = try_warmup(lambda: conv3d_implicit_8wave_fp8(x, w, stride=STRIDE, padding=PAD))
    torch_ok = try_warmup(lambda: F.conv3d(x, w, stride=STRIDE, padding=PAD))
    torch.cuda.synchronize()

    bf16_us = bench_us(lambda: conv3d_implicit_8wave(x, w, stride=STRIDE, padding=PAD)) if bf16_ok else None
    fp8_us = bench_us(lambda: conv3d_implicit_8wave_fp8(x, w, stride=STRIDE, padding=PAD)) if fp8_ok else None
    torch_us = bench_us(lambda: F.conv3d(x, w, stride=STRIDE, padding=PAD)) if torch_ok else None

    tf = lambda us: tflops(npq, us) if us else None
    return bf16_us, fp8_us, torch_us, tf(bf16_us), tf(fp8_us), tf(torch_us), bf16_ok, fp8_ok, torch_ok


def print_header():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Conv3D {KT}×{KH}×{KW}, stride={STRIDE}, pad={PAD}, C=K={C}, D={D} (Do=6), N={N}")
    print()
    hdr = (
        f"{'shape':>16}"
        f"  {'NPQ':>7}"
        f"  {'BF16 us':>{COL}}  {'BF16 TF':>{COL}}"
        f"  {'FP8 us':>{COL}}  {'FP8 TF':>{COL}}"
        f"  {'PyTorch us':>{COL}}  {'PyTorch TF':>{COL}}"
        f"  {'BF16/PT':>8}  {'FP8/PT':>7}"
    )
    print(hdr)
    print("-" * len(hdr))


def print_row(label, hw, bf16_us, fp8_us, torch_us, bf16_tf, fp8_tf, torch_tf, bf16_ok, fp8_ok, torch_ok):
    npq = N * D * hw * hw
    fu = lambda v, ok: "n/a" if not ok else f"{v:.1f}"
    ft = lambda v, ok: "n/a" if not ok else f"{v:.2f}"
    sp = lambda a, b: f"{b/a:.2f}x" if a and b else "-"
    print(
        f"{label:>16}"
        f"  {npq:>7}"
        f"  {fu(bf16_us, bf16_ok):>{COL}}  {ft(bf16_tf, bf16_ok):>{COL}}"
        f"  {fu(fp8_us,  fp8_ok ):>{COL}}  {ft(fp8_tf,  fp8_ok ):>{COL}}"
        f"  {fu(torch_us, torch_ok):>{COL}}  {ft(torch_tf, torch_ok):>{COL}}"
        f"  {sp(bf16_us, torch_us):>8}  {sp(fp8_us, torch_us):>7}"
    )


def main():
    if os.environ.get("BENCH_HEADER_ONLY"):
        print_header()
        return

    idx = os.environ.get("BENCH_IDX")
    if idx is not None:
        label, hw = SHAPES[int(idx)]
        print_row(label, hw, *run_shape(hw))
        return

    print_header()
    for label, hw in SHAPES:
        print_row(label, hw, *run_shape(hw))
    print()


if __name__ == "__main__":
    main()
