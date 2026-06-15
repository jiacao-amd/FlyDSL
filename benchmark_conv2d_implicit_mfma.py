#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))


def patch_rocdl_cluster_symbols_for_local_mlir():
    import flydsl._mlir.dialects.rocdl as rocdl_ods

    def missing_cluster_op(*args, **kwargs):
        raise AttributeError("ROCDL cluster op is not available in this MLIR build")

    for name in (
        "cluster_workgroup_id_x",
        "cluster_workgroup_id_y",
        "cluster_workgroup_id_z",
        "cluster_load_async_to_lds_b8",
        "cluster_load_async_to_lds_b32",
        "cluster_load_async_to_lds_b64",
        "cluster_load_async_to_lds_b128",
    ):
        if not hasattr(rocdl_ods, name):
            setattr(rocdl_ods, name, missing_cluster_op)


patch_rocdl_cluster_symbols_for_local_mlir()

from kernels.conv2d_implicit_mfma import conv2d_implicit_mfma_


DEFAULT_CASES = (
    (8, 32, 20, 20, 64, 3, 3),
    (8, 32, 32, 32, 64, 3, 3),
    (8, 32, 62, 62, 64, 3, 3),
    (8, 32, 122, 122, 64, 3, 3),
    (8, 32, 242, 242, 64, 3, 3),
    (8, 32, 482, 482, 64, 3, 3),
    (8, 32, 962, 962, 64, 3, 3),
    (8, 32, 1922, 1922, 64, 3, 3),
)


def time_ms(fn, *, iters: int, rounds: int) -> float:
    best = float("inf")
    for _ in range(rounds):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        stop.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(stop) / iters)
    return best


def bench_case(n: int, c: int, h: int, w_in: int, k: int, r: int, s: int, *, rounds: int):
    torch.cuda.empty_cache()
    torch.manual_seed(1000 + h + w_in)

    p = h - r + 1
    q = w_in - s + 1
    npq = n * p * q
    flops = 2.0 * n * k * p * q * c * r * s

    x_nchw = torch.randn((n, c, h, w_in), device="cuda", dtype=torch.float16)
    w_kcrs = torch.randn((k, c, r, s), device="cuda", dtype=torch.float16)
    x = x_nchw.permute(0, 2, 3, 1).contiguous()
    w = w_kcrs.permute(0, 2, 3, 1).contiguous()
    y_flydsl = torch.empty((npq, k), device="cuda", dtype=torch.float16)
    stream = torch.cuda.current_stream()

    conv2d_implicit_mfma_(y_flydsl, x, w, stream)
    y_torch = F.conv2d(x_nchw, w_kcrs)
    torch.cuda.synchronize()

    y_ref = y_torch.permute(0, 2, 3, 1).reshape(npq, k).contiguous()
    err = (y_flydsl - y_ref).abs().max().item()

    iters = 200 if h <= 32 else 100 if h <= 62 else 50
    flydsl_ms = time_ms(lambda: conv2d_implicit_mfma_(y_flydsl, x, w, stream), iters=iters, rounds=rounds)
    torch_ms = time_ms(lambda: F.conv2d(x_nchw, w_kcrs), iters=iters, rounds=rounds)

    del x_nchw, w_kcrs, x, w, y_flydsl, y_torch, y_ref
    torch.cuda.empty_cache()

    return {
        "shape": f"{n}x{c}x{h}x{w_in}, K={k}, R=S={r}",
        "npq": npq,
        "flydsl_ms": flydsl_ms,
        "flydsl_tflops": flops / (flydsl_ms * 1e9),
        "torch_ms": torch_ms,
        "torch_tflops": flops / (torch_ms * 1e9),
        "speedup": torch_ms / flydsl_ms,
        "err": err,
    }


def parse_shape(text: str):
    parts = tuple(int(part) for part in text.replace("x", ",").split(","))
    if len(parts) != 7:
        raise argparse.ArgumentTypeError("shape must be N,C,H,W,K,R,S")
    return parts


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark kernels/conv2d_implicit_mfma.py")
    parser.add_argument(
        "--shape",
        action="append",
        type=parse_shape,
        help="Shape as N,C,H,W,K,R,S. May be passed multiple times.",
    )
    parser.add_argument("--rounds", type=int, default=6)
    args = parser.parse_args()

    cases = tuple(args.shape) if args.shape else DEFAULT_CASES
    rows = [bench_case(*case, rounds=args.rounds) for case in cases]

    print("FlyDSL kernel: kernels/conv2d_implicit_mfma.py")
    print("FlyDSL uses NHWC/KRSC ready tensors; PyTorch baseline is F.conv2d on NCHW.")
    print(
        "shape                         NPQ      "
        "FlyDSL ms  FlyDSL TF/s   PyTorch ms  PyTorch TF/s  speedup  max error"
    )
    for row in rows:
        print(
            f"{row['shape']:<29} {row['npq']:>7}  "
            f"{row['flydsl_ms']:>9.6f}  {row['flydsl_tflops']:>11.3f}  "
            f"{row['torch_ms']:>10.6f}  {row['torch_tflops']:>12.3f}  "
            f"{row['speedup']:>7.2f}x  {row['err']:.3e}"
        )


if __name__ == "__main__":
    main()
