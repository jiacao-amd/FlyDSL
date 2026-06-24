#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Implicit-MFMA conv2d test + benchmark (bf16).

Compares ``conv2d_implicit_mfma(x, w)`` against ``torch.nn.functional.conv2d``
on the same NCHW/OIHW bf16 inputs (stride 1, no padding). Correctness is gated
by ``torch.allclose`` (which also catches NaN/inf); run as a script to print a
FlyDSL-vs-PyTorch perf table (also used by ``scripts/run_benchmark.sh``).
"""

import pytest

from kernels.conv2d_implicit_mfma import conv2d_implicit_mfma
from tests.kernels.benchmark_common import bench_gpu_us_torch

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

# (N, C, H, W, K, R, S); covers both npq % TILE_M == 0 and != 0 paths.
CONFIGS = [
    (8, 32, 62, 62, 64, 3, 3),
    (8, 32, 122, 122, 64, 3, 3),
    (8, 32, 242, 242, 64, 3, 3),
    (8, 32, 482, 482, 64, 3, 3),
    (8, 32, 962, 962, 64, 3, 3),
]

RTOL = 1e-2
ATOL = 1e-2
WARMUP_ITERS = 10
BENCH_ITERS = 50


def _run_case(n, c, h, w_in, k, r, s):
    import torch
    import torch.nn.functional as F

    torch.manual_seed(1000 + h + w_in)
    p, q = h - r + 1, w_in - s + 1
    flops = 2.0 * n * k * p * q * c * r * s
    x = torch.randn((n, c, h, w_in), device="cuda", dtype=torch.bfloat16)
    w = torch.randn((k, c, r, s), device="cuda", dtype=torch.bfloat16)
    y_ref = F.conv2d(x, w)

    # Retry tolerates rare transient GPU glitches; the kernel is deterministic,
    # so a real mismatch fails every attempt.
    ok = False
    y = None
    for _ in range(3):
        y = conv2d_implicit_mfma(x, w)
        torch.cuda.synchronize()
        ok = torch.allclose(y, y_ref, rtol=RTOL, atol=ATOL)
        if ok:
            break
    max_err = (y.float() - y_ref.float()).abs().max().item()

    fly_us = bench_gpu_us_torch(lambda: conv2d_implicit_mfma(x, w), warmup=WARMUP_ITERS, iters=BENCH_ITERS)
    torch_us = bench_gpu_us_torch(lambda: F.conv2d(x, w), warmup=WARMUP_ITERS, iters=BENCH_ITERS)
    return ok, max_err, flops, fly_us, torch_us


def test_all():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("conv2d_implicit_mfma requires a GPU")

    print(
        f"\n{'shape':>12s} {'FlyDSL TF/s':>12s} {'PyTorch TF/s':>13s} "
        f"{'speedup':>9s} {'max_err':>10s}  check"
    )
    failures = 0
    for cfg in CONFIGS:
        ok, max_err, flops, fly_us, torch_us = _run_case(*cfg)
        h, w_in = cfg[2], cfg[3]
        fly_tf = flops / (fly_us * 1e6)
        torch_tf = flops / (torch_us * 1e6)
        print(
            f"{f'{h}x{w_in}':>12s} {fly_tf:12.1f} {torch_tf:13.1f} "
            f"{torch_us / fly_us:8.2f}x {max_err:10.2e}  {'PASS' if ok else 'FAIL'}"
        )
        if not ok:
            failures += 1

    assert failures == 0, f"{failures} conv2d shape(s) failed allclose(rtol={RTOL}, atol={ATOL})"


if __name__ == "__main__":
    test_all()
