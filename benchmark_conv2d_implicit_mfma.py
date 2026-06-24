#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
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

from kernels.conv2d_implicit_mfma import conv2d_implicit_mfma  # noqa: E402

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

# bf16 inputs (matching F.conv2d); torch.allclose tolerances also catch NaN/inf.
DTYPE = torch.bfloat16
RTOL = 1e-2
ATOL = 1e-2


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

    x = torch.randn((n, c, h, w_in), device="cuda", dtype=DTYPE)
    w = torch.randn((k, c, r, s), device="cuda", dtype=DTYPE)

    y_torch = F.conv2d(x, w)
    ok = False
    for _ in range(3):
        y_flydsl = conv2d_implicit_mfma(x, w)
        torch.cuda.synchronize()
        err = (y_flydsl.float() - y_torch.float()).abs().max().item()
        ok = torch.allclose(y_flydsl, y_torch, rtol=RTOL, atol=ATOL)
        if ok:
            break

    iters = 200 if h <= 32 else 100 if h <= 62 else 50
    flydsl_ms = time_ms(lambda: conv2d_implicit_mfma(x, w), iters=iters, rounds=rounds)
    torch_ms = time_ms(lambda: F.conv2d(x, w), iters=iters, rounds=rounds)

    del y_flydsl, y_torch
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
        "ok": ok,
    }


def check_case(n: int, c: int, h: int, w_in: int, k: int, r: int, s: int):
    """Strict correctness vs F.conv2d (bf16, torch.allclose). Raises on mismatch.

    Retries to tolerate rare transient GPU glitches; the kernel is deterministic,
    so a real mismatch fails every attempt.
    """
    torch.manual_seed(1000 + h + w_in)
    x = torch.randn((n, c, h, w_in), device="cuda", dtype=DTYPE)
    w = torch.randn((k, c, r, s), device="cuda", dtype=DTYPE)
    y_ref = F.conv2d(x, w)
    last_err = None
    for _ in range(3):
        y = conv2d_implicit_mfma(x, w)
        torch.cuda.synchronize()
        assert y.shape == y_ref.shape, f"shape {tuple(y.shape)} != {tuple(y_ref.shape)}"
        if torch.allclose(y, y_ref, rtol=RTOL, atol=ATOL):
            return
        last_err = (y.float() - y_ref.float()).abs().max().item()
    raise AssertionError(
        f"{n}x{c}x{h}x{w_in} K={k}: not allclose(rtol={RTOL}, atol={ATOL}) "
        f"in 3 attempts, last max_abs_err={last_err:.3e}"
    )


def run_tests(cases) -> None:
    for case in cases:
        check_case(*case)
    print("OK: conv2d_implicit_mfma matches F.conv2d (bf16, allclose)")


def test_conv2d_implicit_mfma_bf16():
    # pytest entry: covers both npq % TILE_M == 0 and != 0 paths.
    for case in (
        (8, 32, 32, 32, 64, 3, 3),
        (8, 32, 62, 62, 64, 3, 3),
        (8, 32, 242, 242, 64, 3, 3),
        (8, 32, 482, 482, 64, 3, 3),
    ):
        check_case(*case)


def parse_shape(text: str):
    parts = tuple(int(part) for part in text.replace("x", ",").split(","))
    if len(parts) != 7:
        raise argparse.ArgumentTypeError("shape must be N,C,H,W,K,R,S")
    return parts


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark conv2d/conv2d_implicit_mfma.py")
    parser.add_argument(
        "--shape",
        action="append",
        type=parse_shape,
        help="Shape as N,C,H,W,K,R,S. May be passed multiple times.",
    )
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run strict correctness check (assert torch.allclose) and exit, instead of benchmarking.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Write benchmark results to this CSV file.",
    )
    args = parser.parse_args()

    cases = tuple(args.shape) if args.shape else DEFAULT_CASES

    if args.test:
        run_tests(cases)
        return

    rows = [bench_case(*case, rounds=args.rounds) for case in cases]

    print("FlyDSL kernel: conv2d/conv2d_implicit_mfma.py")
    print(
        f"Comparison: conv2d_implicit_mfma(x, w) vs F.conv2d(x, w), "
        f"same contiguous NCHW {DTYPE} inputs (NCHW->NHWC conversion included in timing)."
    )
    print(f"Accuracy gated by torch.allclose(rtol={RTOL}, atol={ATOL}); 'max error' shown for info.")
    print(
        "shape                         NPQ      "
        "FlyDSL ms  FlyDSL TF/s   PyTorch ms  PyTorch TF/s  speedup  max error  check"
    )
    for row in rows:
        print(
            f"{row['shape']:<29} {row['npq']:>7}  "
            f"{row['flydsl_ms']:>9.6f}  {row['flydsl_tflops']:>11.3f}  "
            f"{row['torch_ms']:>10.6f}  {row['torch_tflops']:>12.3f}  "
            f"{row['speedup']:>7.2f}x  {row['err']:.3e}  {'PASS' if row['ok'] else 'FAIL'}"
        )

    if args.output_csv:
        fields = [
            "shape", "npq",
            "flydsl_ms", "flydsl_tflops",
            "torch_ms", "torch_tflops",
            "speedup", "err", "ok",
        ]
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row[k] for k in fields})
        print(f"\nWrote CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
