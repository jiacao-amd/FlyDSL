#!/usr/bin/env python3
"""End-to-end WAN VAE latency: stock PyTorch vs FlyDSL BF16 / FP8 conv3d.

Loads diffusers' AutoencoderKLWan (config-only, random weights — latency is
weight-independent), then monkeypatches WanCausalConv3d.forward to route the
convolution through a FlyDSL kernel (BF16 or FP8). Every conv runs on FlyDSL:
  C a multiple of the kernel's channel-vector (16 for FP8, 8 for BF16) runs
  directly; the C=3 input stem is zero-padded in channels (math-preserving) so
  it too runs on FlyDSL instead of falling back to MIOpen.

Measures full VAE decode AND encode latency end to end, so the reported numbers
reflect the whole model, not individual conv shapes. Decode is the
video-generation hot path; encode also runs the C=3 input stem (one fallback).

Env:
  BENCH_WARMUP=<int>   default 2
  BENCH_ITERS=<int>    default 5
  VAE_FRAMES=<int>     latent temporal length (default 6)
  VAE_H=<int>          latent height (default 60)   -> ~480p-ish decode
  VAE_W=<int>          latent width  (default 104)
"""

import os
import subprocess
import sys
import time

import torch
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from diffusers import AutoencoderKLWan  # noqa: E402
from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d  # noqa: E402

from kernels.conv3d_implicit_8wave import conv3d_implicit_8wave  # noqa: E402
from kernels.conv3d_implicit_8wave_fp8 import conv3d_implicit_8wave_fp8  # noqa: E402

WARMUP = int(os.environ.get("BENCH_WARMUP", 2))
ITERS = int(os.environ.get("BENCH_ITERS", 5))
FRAMES = int(os.environ.get("VAE_FRAMES", 6))
LAT_H = int(os.environ.get("VAE_H", 60))
LAT_W = int(os.environ.get("VAE_W", 104))

# One (stage, path) measured per subprocess. Running every stage/path in a single
# long-lived process accumulates JIT-compiled GPU code objects until the process
# is SIGKILLed (exit 137) partway through a warmup — each conv shape is fine on
# its own, but the cumulative per-process compilation footprint trips a hard
# limit. A fresh process per measurement resets that state. BENCH_STAGE and
# BENCH_PATH select the single unit of work for a worker; the disk JIT cache
# (FLYDSL_RUNTIME_CACHE_DIR) is shared so nothing is recompiled needlessly.
_WORKER_STAGE = os.environ.get("BENCH_STAGE")  # "decode" | "encode"
_WORKER_PATH = os.environ.get("BENCH_PATH")  # "pytorch" | "bf16" | "fp8"

_stats = {"fly": 0, "fallback": 0}


def _causal_pad(self, x, cache_x):
    padding = list(self._padding)
    if cache_x is not None and self._padding[4] > 0:
        cache_x = cache_x.to(x.device)
        x = torch.cat([cache_x, x], dim=2)
        padding[4] -= cache_x.shape[2]
    return F.pad(x, padding)


def _torch_conv(self, x):
    return F.conv3d(x, self.weight, self.bias, self.stride, (0, 0, 0), self.dilation, self.groups)


def _padded_weight(self, cvec, attr):
    """Zero-pad weight in-channels up to a multiple of cvec, cached on the module
    (padding is math-preserving: the extra channels are zero). Caching avoids
    re-padding — and re-packing to FP8 — on every call."""
    w = getattr(self, attr, None)
    if w is None:
        c = self.in_channels
        w = F.pad(self.weight, (0, 0, 0, 0, 0, 0, 0, cvec - c % cvec))
        setattr(self, attr, w)
    return w


def _make_forward(kernel_fn, cvec, wattr):
    """Build a WanCausalConv3d.forward routing convs through kernel_fn.

    cvec is the kernel's channel-vector requirement (16 for FP8, 8 for BF16).
    C already a multiple of cvec runs directly; C not a multiple (only the C=3
    input stem) is zero-padded in channels up to cvec, which is math-preserving
    and still beats the MIOpen fallback. bf16 is the only supported dtype.
    """

    def forward(self, x, cache_x=None):
        x = _causal_pad(self, x, cache_x)
        if x.dtype != torch.bfloat16:
            _stats["fallback"] += 1
            return _torch_conv(self, x)
        _stats["fly"] += 1
        c = self.in_channels
        if c % cvec == 0:
            return kernel_fn(x, self.weight, bias=self.bias, stride=self.stride, padding=0)
        # channel-pad path (C=3 stem)
        xp = F.pad(x, (0, 0, 0, 0, 0, 0, 0, cvec - c % cvec))
        return kernel_fn(xp, _padded_weight(self, cvec, wattr), bias=self.bias, stride=self.stride, padding=0)

    return forward


_fly_fp8_forward = _make_forward(conv3d_implicit_8wave_fp8, 16, "_fly_wpad_fp8")
_fly_bf16_forward = _make_forward(conv3d_implicit_8wave, 8, "_fly_wpad_bf16")


def build_vae():
    vae = AutoencoderKLWan()
    vae = vae.to(device="cuda", dtype=torch.bfloat16).eval()
    return vae


def bench_op(op, warmup, iters):
    with torch.no_grad():
        for _ in range(warmup):
            op()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            op()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
    return (t1 - t0) / iters * 1e3  # ms


_FORWARDS = {"bf16": _fly_bf16_forward, "fp8": _fly_fp8_forward}


def _build_ops():
    """Build the VAE and return (decode_op, encode_op). The first decode runs
    here to allocate buffers and settle MIOpen before any measurement."""
    torch.manual_seed(0)
    vae = build_vae()
    z_ch = vae.config.z_dim
    z = torch.randn(1, z_ch, FRAMES, LAT_H, LAT_W, device="cuda", dtype=torch.bfloat16)

    def decode_op():
        return vae.decode(z)

    with torch.no_grad():
        out = decode_op()
        vid = (out.sample if hasattr(out, "sample") else out).contiguous()
    torch.cuda.synchronize()

    def encode_op():
        return vae.encode(vid)

    return decode_op, encode_op


def run_worker(stage, path):
    """Measure exactly one (stage, path) in this process and print a RESULT line
    the orchestrator parses. Kept to a single unit of work so per-process JIT
    accumulation never reaches the level that gets the process SIGKILLed."""
    decode_op, encode_op = _build_ops()
    op = decode_op if stage == "decode" else encode_op

    if path == "pytorch":
        torch.cuda.synchronize()
        ms = bench_op(op, WARMUP, ITERS)
        fly = fb = 0
    else:
        _orig = WanCausalConv3d.forward
        WanCausalConv3d.forward = _FORWARDS[path]
        _stats["fly"] = 0
        _stats["fallback"] = 0
        try:
            with torch.no_grad():  # warmup: JIT-compiles this stage's conv shapes
                op()
            torch.cuda.synchronize()
            ms = bench_op(op, WARMUP, ITERS)
            fly, fb = _stats["fly"], _stats["fallback"]
        finally:
            WanCausalConv3d.forward = _orig

    calls_denom = WARMUP + ITERS + 1
    print(f"RESULT {stage} {path} {ms:.4f} {fly // calls_denom} {fb // calls_denom}", flush=True)


def _run_worker_subprocess(stage, path):
    env = dict(os.environ, BENCH_STAGE=stage, BENCH_PATH=path)
    proc = subprocess.run(
        [sys.executable, "-u", os.path.abspath(__file__)],
        env=env,
        capture_output=True,
        text=True,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            _, st, pa, ms, fly, fb = line.split()
            return {"ms": float(ms), "fly": int(fly), "fb": int(fb)}
    print(f"  [{stage}/{path}] worker failed (exit {proc.returncode}); stderr tail:", flush=True)
    for line in proc.stderr.splitlines()[-8:]:
        print(f"    {line}", flush=True)
    return None


def _print_stage_table(name, results):
    stock = results.get("pytorch")
    bf16 = results.get("bf16")
    fp8 = results.get("fp8")
    print(f"=== {name} ===")
    print(f"{'path':>18}  {'latency (ms)':>13}  {'speedup':>8}")
    print("-" * 44)

    def _row(label, r):
        if r is None:
            print(f"{label:>18}  {'CRASHED':>13}  {'-':>8}")
        elif stock is None:
            print(f"{label:>18}  {r['ms']:>13.2f}  {'-':>8}")
        else:
            print(f"{label:>18}  {r['ms']:>13.2f}  {stock['ms'] / r['ms']:>7.2f}x")

    _row("PyTorch (MIOpen)", stock)
    _row("FlyDSL BF16", bf16)
    _row("FlyDSL FP8", fp8)
    ref = bf16 or fp8
    if ref is not None:
        print(f"conv layers per {name}: ~{ref['fly'] + ref['fb']}")
        if bf16 is not None:
            print(f"  BF16: {bf16['fly']} fly (incl. C=3 stem via channel-pad), {bf16['fb']} fallback")
        if fp8 is not None:
            print(f"  FP8 : {fp8['fly']} fly (incl. C=3 stem via channel-pad), {fp8['fb']} fallback")
    print(flush=True)


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"WAN VAE  warmup={WARMUP} iters={ITERS}  latent=(1,?,{FRAMES},{LAT_H},{LAT_W})", flush=True)
    print("Each (stage, path) runs in its own subprocess (avoids per-process JIT-accumulation SIGKILL).", flush=True)
    print(flush=True)
    for stage in ("decode", "encode"):
        results = {path: _run_worker_subprocess(stage, path) for path in ("pytorch", "bf16", "fp8")}
        _print_stage_table(stage, results)


if __name__ == "__main__":
    if _WORKER_STAGE and _WORKER_PATH:
        run_worker(_WORKER_STAGE, _WORKER_PATH)
    else:
        main()
