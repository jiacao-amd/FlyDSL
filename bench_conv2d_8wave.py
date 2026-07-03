#!/usr/bin/env python3
"""Benchmark conv2d_implicit_8wave vs MIOpen (torch.nn.functional.conv2d).

Sweep: C=K in {64,128,256,512}, H=W in {8,16,32,64,128}, kernel 3x3, stride=1, pad=1.

Env vars:
  BENCH_IDX=<int>      run one shape (used by shell runner to avoid OOM)
  BENCH_HEADER_ONLY=1  print header then exit
  BENCH_ITERS=<int>    iteration count (default 100)
  BENCH_WARMUP=<int>   warmup count (default 20)
"""

import os
import sys

import torch
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from kernels.conv2d_implicit_mfma import conv2d_implicit_8wave

WARMUP = int(os.environ.get("BENCH_WARMUP", 20))
ITERS = int(os.environ.get("BENCH_ITERS", 100))

R, S = 3, 3
STRIDE = 1
PAD = 1  # same-size: Ho=H, Wo=W
N = 1

CK = 128
HW_VALS = [16, 32, 64, 128, 256, 512, 1024]

SHAPES = [(f"C{CK} {hw}²", CK, CK, hw, hw) for hw in HW_VALS]

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


def tflops(c, k, h, w, us):
    ops = 2 * N * k * (c * R * S) * h * w
    return ops / (us * 1e-6) / 1e12


def run_shape(c, k, h, w):
    x = torch.randn(N, c, h, w, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(k, c, R, S, dtype=torch.bfloat16, device="cuda")

    def try_warmup(fn):
        try:
            fn()
            return True
        except Exception:
            return False

    new_ok = try_warmup(lambda: conv2d_implicit_8wave(x, weight, stride=STRIDE, padding=PAD))
    torch_ok = try_warmup(lambda: F.conv2d(x, weight, stride=STRIDE, padding=PAD))
    torch.cuda.synchronize()

    new_us = bench_us(lambda: conv2d_implicit_8wave(x, weight, stride=STRIDE, padding=PAD)) if new_ok else None
    torch_us = bench_us(lambda: F.conv2d(x, weight, stride=STRIDE, padding=PAD)) if torch_ok else None

    tf = lambda us: tflops(c, k, h, w, us) if us else None
    return new_us, torch_us, tf(new_us), tf(torch_us), new_ok, torch_ok


def print_header():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Conv2D 3×3, stride=1, pad=1 (same-size), N={N}")
    print()
    hdr = (
        f"{'shape':>14}"
        f"  {'8wave us':>{COL}}  {'8wave TF':>{COL}}"
        f"  {'PyTorch us':>{COL}}  {'PyTorch TF':>{COL}}"
        f"  {'8wave/PT':>10}"
    )
    print(hdr)
    print("-" * len(hdr))


def print_row(label, new_us, torch_us, new_tf, torch_tf, new_ok, torch_ok):
    fu = lambda v, ok: "n/a" if not ok else f"{v:.1f}"
    ft = lambda v, ok: "n/a" if not ok else f"{v:.2f}"
    sp = lambda a, b: f"{b/a:.2f}x" if a and b else "-"
    print(
        f"{label:>14}"
        f"  {fu(new_us,   new_ok  ):>{COL}}  {ft(new_tf,   new_ok  ):>{COL}}"
        f"  {fu(torch_us, torch_ok):>{COL}}  {ft(torch_tf, torch_ok):>{COL}}"
        f"  {sp(new_us, torch_us):>10}"
    )


def main():
    if os.environ.get("BENCH_HEADER_ONLY"):
        print_header()
        return

    idx = os.environ.get("BENCH_IDX")
    if idx is not None:
        label, c, k, h, w = SHAPES[int(idx)]
        print_row(label, *run_shape(c, k, h, w))
        return

    print_header()
    for label, c, k, h, w in SHAPES:
        print_row(label, *run_shape(c, k, h, w))
    print()


if __name__ == "__main__":
    main()
