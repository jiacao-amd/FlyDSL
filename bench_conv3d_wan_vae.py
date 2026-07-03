#!/usr/bin/env python3
"""Benchmark FlyDSL conv3d kernels on WAN VAE decode/encode shapes.

WAN VAE architecture (base_dim=96, dim_mult=(1,2,4,4)):
  - Channel stages: 96, 192, 384
  - All conv3d layers: kernel 3x3x3, stride=1, pad=1
  - D includes causal padding (+2 frames in time)

BF16 kernel constraints:
  C%8==0, CRS%32==0, K>=8
  -> All WAN dims (96, 192, 384) satisfy this.

FP8 kernel constraints (CDNA4 / gfx95x only):
  C%16==0 only (LDG vector load).
  -> All WAN dims (96, 192, 384) satisfy this.
  -> NPQ / K / CRS no longer need 128-alignment: the last partial M/N/K tile is
     masked (OOB activation loads zero, OOB stores dropped, OOB-K contribution
     zeroed) so any frame count and channel size run.
  -> The prior JIT-OOM (large tiles_per_split fully unrolled) is fixed by capping
     tiles_per_split via split-K in _resolve_splitk.

Env:
  BENCH_IDX=<int>      run one shape (used by shell runner to isolate GPU state)
  BENCH_HEADER_ONLY=1  print header then exit
  BENCH_WARMUP=<int>   default 5
  BENCH_ITERS=<int>    default 20
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

WARMUP = int(os.environ.get("BENCH_WARMUP", 5))
ITERS = int(os.environ.get("BENCH_ITERS", 20))

KT, KH, KW = 3, 3, 3
STRIDE = 1
PAD = 1
N = 1

# Each entry: (label, C_in, K_out, D, H, W). D values are the real WAN frame
# counts — FP8 no longer needs NPQ 128-alignment (last M-tile is masked).
SHAPES = [
    # ---- Encode path 720p ----
    ("enc/conv_in 720p", 3, 96, 83, 90, 160),
    ("enc/res96 720p", 96, 96, 83, 90, 160),
    ("enc/res192 720p", 192, 192, 42, 45, 80),
    ("enc/res384 720p", 384, 384, 22, 23, 40),  # real D (NPQ no longer needs align)
    # ---- Decode path 720p ----
    ("dec/conv_in 720p", 16, 384, 23, 90, 160),
    ("dec/res384 720p", 384, 384, 23, 90, 160),
    ("dec/res192 480p*", 192, 192, 33, 120, 212),  # 480p proxy
    # ---- Decode path 480p ----
    ("dec/res384 480p", 384, 384, 17, 60, 106),
    ("dec/res96 480p", 96, 96, 17, 120, 212),
]

COL = 11


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


def tflops(n, k, crs, npq, us):
    return 2 * n * k * crs * npq / (us * 1e-6) / 1e12


def try_warmup(fn):
    try:
        fn()
        torch.cuda.synchronize()
        return True
    except Exception:
        return False


def safe_bench(fn):
    try:
        return bench_us(fn)
    except Exception:
        return None


def run_shape(c_in, k_out, d, h, w):
    npq = N * d * h * w
    crs = c_in * KT * KH * KW

    x = torch.randn(N, c_in, d, h, w, dtype=torch.bfloat16, device="cuda")
    wt = torch.randn(k_out, c_in, KT, KH, KW, dtype=torch.bfloat16, device="cuda")

    # BF16 (uses original D; no padding needed)
    bf16_elig = (c_in % 8 == 0) and (crs % 32 == 0) and (k_out >= 8)
    bf16_ok = bf16_elig and try_warmup(lambda: conv3d_implicit_8wave(x, wt, stride=STRIDE, padding=PAD))

    # FP8 only needs C%16==0 (LDG vector load). Partial M/N/K tiles are masked,
    # so NPQ, K, and CRS no longer need 128-alignment.
    fp8_crs_ok = c_in % 16 == 0
    fp8_elig = fp8_crs_ok
    fp8_ok = fp8_elig and try_warmup(lambda: conv3d_implicit_8wave_fp8(x, wt, stride=STRIDE, padding=PAD))

    # PyTorch (each shape runs in its own container via the runner, so MIOpen
    # workspace memory is released between shapes — full bench loop is safe).
    torch_ok = try_warmup(lambda: F.conv3d(x, wt, stride=STRIDE, padding=PAD))

    bf16_us = safe_bench(lambda: conv3d_implicit_8wave(x, wt, stride=STRIDE, padding=PAD)) if bf16_ok else None

    fp8_us = safe_bench(lambda: conv3d_implicit_8wave_fp8(x, wt, stride=STRIDE, padding=PAD)) if fp8_ok else None

    torch_us = safe_bench(lambda: F.conv3d(x, wt, stride=STRIDE, padding=PAD)) if torch_ok else None

    tf = lambda us: tflops(N, k_out, crs, npq, us) if us else None
    return dict(
        bf16_us=bf16_us,
        fp8_us=fp8_us,
        torch_us=torch_us,
        bf16_tf=tf(bf16_us),
        fp8_tf=tf(fp8_us),
        torch_tf=tf(torch_us),
        bf16_ok=bf16_ok,
        fp8_ok=fp8_ok,
        torch_ok=torch_ok,
        bf16_elig=bf16_elig,
        fp8_elig=fp8_elig,
        npq=npq,
        fp8_crs_ok=fp8_crs_ok,
    )


def print_header():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Conv3D {KT}x{KH}x{KW}, stride={STRIDE}, pad={PAD}, N={N}")
    print(f"Warmup={WARMUP}, Iters={ITERS}")
    print()
    hdr = (
        f"{'layer':>22}  {'C':>4}  {'K':>4}  {'D':>4}  {'H':>4}  {'W':>4}"
        f"  {'NPQ':>7}"
        f"  {'BF16 us':>{COL}}  {'BF16 TF':>{COL}}"
        f"  {'FP8 us':>{COL}}  {'FP8 TF':>{COL}}"
        f"  {'PT us':>{COL}}  {'PT TF':>{COL}}"
        f"  {'BF16/PT':>8}  {'FP8/PT':>7}"
    )
    print(hdr)
    print("-" * len(hdr))


def print_row(label, c, k, d, h, w, r):
    def fu(v, ok, elig, note=""):
        if not elig:
            return note if note else "skip"
        if not ok or v is None:
            return "err"
        return f"{v:.1f}"

    def ft(v, ok, elig):
        if not elig or not ok or v is None:
            return "    "
        return f"{v:.2f}"

    def sp(a, b):
        return f"{b/a:.2f}x" if a and b else "-"

    # FP8 display: 'skip' if C%16!=0 (LDG load), else timing or 'err'.
    if not r["fp8_crs_ok"]:
        fp8_u = "skip"  # C % 16 != 0
        fp8_t = "    "
    elif not r["fp8_ok"]:
        fp8_u = "err"  # eligible but kernel raised
        fp8_t = "    "
    else:
        fp8_u = f"{r['fp8_us']:.1f}" if r["fp8_us"] else "err"
        fp8_t = f"{r['fp8_tf']:.2f}" if r["fp8_tf"] else "    "

    # PT display
    if not r["torch_ok"]:
        pt_u = "PT-err"
        pt_t = "    "
    elif r["torch_us"] is None:
        pt_u = "err"
        pt_t = "    "
    else:
        pt_u = f"{r['torch_us']:.1f}"
        pt_t = f"{r['torch_tf']:.2f}" if r["torch_tf"] else "    "

    bf16_u = fu(r["bf16_us"], r["bf16_ok"], r["bf16_elig"])
    bf16_t = ft(r["bf16_tf"], r["bf16_ok"], r["bf16_elig"])

    print(
        f"{label:>22}  {c:>4}  {k:>4}  {d:>4}  {h:>4}  {w:>4}"
        f"  {r['npq']:>7}"
        f"  {bf16_u:>{COL}}  {bf16_t:>{COL}}"
        f"  {fp8_u:>{COL}}  {fp8_t:>{COL}}"
        f"  {pt_u:>{COL}}  {pt_t:>{COL}}"
        f"  {sp(r['bf16_us'], r['torch_us']):>8}"
        f"  {sp(r['fp8_us'],  r['torch_us']):>7}"
    )


def main():
    idx_env = os.environ.get("BENCH_IDX")
    header_only = os.environ.get("BENCH_HEADER_ONLY")

    if header_only:
        print_header()
        return

    if idx_env is not None:
        i = int(idx_env)
        label, c, k, d, h, w = SHAPES[i]
        r = run_shape(c, k, d, h, w)
        print_row(label, c, k, d, h, w, r)
        return

    print_header()
    for label, c, k, d, h, w in SHAPES:
        r = run_shape(c, k, d, h, w)
        print_row(label, c, k, d, h, w, r)
    print()
    print("Legend:")
    print("  'skip'   = constraint not met (C%8/CRS%32 for BF16; C%16 for FP8)")
    print("  'err'    = eligible but kernel raised")
    print("  BF16/PT  = PT_latency / BF16_latency  (>1 means FlyDSL BF16 is faster)")


if __name__ == "__main__":
    main()
