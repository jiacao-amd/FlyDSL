#!/usr/bin/env python3
"""Compare v6 optimization-method variants for conv3d 8-wave.

This script intentionally does not import kernels/conv3d_implicit_8wave_fused.py
because another agent may be editing it.  The old8 column is a fixed target from
the previous clean sweep.
"""

import argparse
import importlib
import statistics
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "build-fly" / "python_packages"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from kernels.conv3d_implicit_8wave_fused_v3 import conv3d_implicit_8wave_fused_v3 as v3  # noqa: E402
from kernels.conv3d_implicit_mfma import conv3d_implicit_mfma as mfma  # noqa: E402


DTYPE = torch.bfloat16

OLD8_TARGET_TF = {
    (3, 128, 128): 518.0,
    (3, 128, 256): 662.0,
    (3, 256, 128): 654.0,
    (3, 256, 256): 743.0,
    (6, 128, 128): 574.0,
    (6, 128, 256): 686.0,
    (6, 256, 128): 709.0,
    (6, 256, 256): 739.0,
}

VARIANT_MODULES = {
    "v6b": "kernels.conv3d_implicit_8wave_fused_v6b",
    "v6g": "kernels.conv3d_implicit_8wave_fused_v6g",
    "v6h": "kernels.conv3d_implicit_8wave_fused_v6h",
}


def _parse_ints(s):
    return [int(x) for x in s.split(",") if x]


def _load_variant(name):
    module = importlib.import_module(VARIANT_MODULES[name])
    return getattr(module, f"conv3d_implicit_8wave_fused_{name}")


def _time_us(fn, iters, warmup, rounds):
    best = float("inf")
    for _ in range(rounds):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end) * 1e3 / iters)
    return best


def _tf(flops, us):
    return flops / (us / 1e6) / 1e12


def _short_error(exc):
    return "".join(traceback.format_exception_only(type(exc), exc)).strip().replace("\n", " | ")


def _bench_one(t, c, hw, variants, args):
    torch.cuda.empty_cache()
    torch.manual_seed(2000 + t + c + hw)
    n = 1
    k = c
    kt = kh = kw = 3
    pad = 1

    x = torch.randn((n, c, t, hw, hw), device="cuda", dtype=DTYPE)
    w = torch.randn((k, c, kt, kh, kw), device="cuda", dtype=DTYPE)
    flops = 2.0 * n * k * t * hw * hw * c * kt * kh * kw

    ref = F.conv3d(x, w, padding=pad)
    torch.cuda.synchronize()

    funcs = {
        "v3": lambda: v3(x, w, padding=pad),
        "mfma": lambda: mfma(x, w, padding=pad),
        "miopen": lambda: F.conv3d(x, w, padding=pad),
    }
    for name in variants:
        fn = _load_variant(name)
        funcs[name] = lambda fn=fn: fn(x, w, padding=pad)

    results = {}
    for name, fn in funcs.items():
        try:
            if name != "miopen":
                y = fn()
                torch.cuda.synchronize()
                if not (y.shape == ref.shape and torch.allclose(y, ref, rtol=args.rtol, atol=args.atol)):
                    results[name] = {"err": "wrong"}
                    continue
                del y
            us = statistics.median(_time_us(fn, args.iters, args.warmup, args.rounds) for _ in range(args.reps))
            results[name] = {"tf": _tf(flops, us), "us": us}
        except Exception as exc:
            torch.cuda.synchronize()
            results[name] = {"err": _short_error(exc)}

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", default="3,6")
    parser.add_argument("--channels", default="128,256")
    parser.add_argument("--hws", default="128,256")
    parser.add_argument("--variants", default="v6b,v6g,v6h")
    parser.add_argument("--iters", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--atol", type=float, default=2e-2)
    args = parser.parse_args()

    variants = [v for v in args.variants.split(",") if v]
    print("conv3d 3x3x3 pad1 K=C bf16: v6 method variants")
    print("device:", torch.cuda.get_device_name(0))

    names = variants + ["v3", "mfma", "miopen"]
    header = f"{'T':>3} {'C':>5} {'hw':>5} {'oldTgt':>8} | " + " ".join(f"{n:>8}" for n in names)
    print(header)
    print("-" * len(header))

    geos = {name: [] for name in names}
    for t in _parse_ints(args.frames):
        for c in _parse_ints(args.channels):
            for hw in _parse_ints(args.hws):
                res = _bench_one(t, c, hw, variants, args)
                old = OLD8_TARGET_TF.get((t, c, hw))
                row = f"{t:>3} {c:>5} {hw:>5} {old if old else float('nan'):>8.0f} | "
                vals = []
                for name in names:
                    item = res.get(name, {"err": "missing"})
                    if "tf" in item:
                        vals.append(f"{item['tf']:>8.0f}")
                        if name != "miopen":
                            geos[name].append(item["tf"] / res["v3"]["tf"] if "tf" in res.get("v3", {}) else float("nan"))
                    else:
                        vals.append(f"{'ERR':>8}")
                print(row + " ".join(vals), flush=True)
                for name in names:
                    item = res.get(name, {})
                    if "err" in item:
                        print(f"    {name}: {item['err']}", flush=True)

    print("-" * len(header))
    parts = []
    for name in variants + ["mfma"]:
        valid = [x for x in geos[name] if x == x]
        if valid:
            parts.append(f"{name}/v3={statistics.geometric_mean(valid):.3f}")
    print("geomean ratios: " + ", ".join(parts))


if __name__ == "__main__":
    main()
