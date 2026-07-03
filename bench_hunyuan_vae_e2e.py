#!/usr/bin/env python3
"""End-to-end HunyuanVideo VAE latency: PyTorch (MIOpen) vs FlyDSL BF16 / FP8.

Loads diffusers' AutoencoderKLHunyuanVideo (config-only, random weights —
latency is weight-independent), then monkeypatches HunyuanVideoCausalConv3d.
forward to route the wrapped convolution through a FlyDSL kernel (BF16 or FP8).

Unlike WAN's WanCausalConv3d (which *is* an nn.Conv3d), HunyuanVideoCausalConv3d
*wraps* an inner nn.Conv3d at self.conv and pads with self.time_causal_padding /
self.pad_mode (mode="replicate"). We reproduce that padding exactly, then call
the FlyDSL kernel with padding=0.

Every conv runs on FlyDSL: C a multiple of the kernel's channel-vector (16 for
FP8, 8 for BF16) runs directly; C=3 / C=16 stems are zero-padded in channels
(math-preserving) so they too run on FlyDSL instead of falling back to MIOpen.

Measures full VAE decode AND encode latency end to end.

Env:
  BENCH_WARMUP=<int>   default 2
  BENCH_ITERS=<int>    default 5
  VAE_FRAMES=<int>     latent temporal length (default 5 -> 17 video frames)
  VAE_H=<int>          latent height (default 60)   -> ~480p decode
  VAE_W=<int>          latent width  (default 104)
"""

import os
import sys
import time

import torch
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from diffusers import AutoencoderKLHunyuanVideo  # noqa: E402
from diffusers.models.autoencoders.autoencoder_kl_hunyuan_video import (  # noqa: E402
    HunyuanVideoCausalConv3d,
)

from kernels.conv3d_implicit_8wave import conv3d_implicit_8wave  # noqa: E402
from kernels.conv3d_implicit_8wave_fp8 import conv3d_implicit_8wave_fp8  # noqa: E402

WARMUP = int(os.environ.get("BENCH_WARMUP", 2))
ITERS = int(os.environ.get("BENCH_ITERS", 5))
FRAMES = int(os.environ.get("VAE_FRAMES", 5))
LAT_H = int(os.environ.get("VAE_H", 60))
LAT_W = int(os.environ.get("VAE_W", 104))

_stats = {"fly": 0, "fallback": 0}


def _causal_pad(self, x):
    return F.pad(x, self.time_causal_padding, mode=self.pad_mode)


def _torch_conv(self, x):
    c = self.conv
    return F.conv3d(x, c.weight, c.bias, c.stride, (0, 0, 0), c.dilation, c.groups)


def _padded_weight(self, cvec, attr):
    """Zero-pad inner-conv weight in-channels up to a multiple of cvec, cached on
    the module (padding is math-preserving: the extra channels are zero)."""
    w = getattr(self, attr, None)
    if w is None:
        c = self.conv.in_channels
        w = F.pad(self.conv.weight, (0, 0, 0, 0, 0, 0, 0, cvec - c % cvec))
        setattr(self, attr, w)
    return w


def _make_forward(kernel_fn, cvec, wattr):
    """Build a HunyuanVideoCausalConv3d.forward routing convs through kernel_fn.

    cvec is the kernel's channel-vector requirement (16 for FP8, 8 for BF16).
    C already a multiple of cvec runs directly; C not a multiple (C=3 / C=16
    stems) is zero-padded in channels up to cvec (math-preserving).
    """

    def forward(self, x):
        x = _causal_pad(self, x)
        if x.dtype != torch.bfloat16:
            _stats["fallback"] += 1
            return _torch_conv(self, x)
        _stats["fly"] += 1
        conv = self.conv
        c = conv.in_channels
        if c % cvec == 0:
            return kernel_fn(x, conv.weight, bias=conv.bias, stride=conv.stride, padding=0)
        xp = F.pad(x, (0, 0, 0, 0, 0, 0, 0, cvec - c % cvec))
        return kernel_fn(xp, _padded_weight(self, cvec, wattr), bias=conv.bias, stride=conv.stride, padding=0)

    return forward


_fly_fp8_forward = _make_forward(conv3d_implicit_8wave_fp8, 16, "_fly_wpad_fp8")
_fly_bf16_forward = _make_forward(conv3d_implicit_8wave, 8, "_fly_wpad_bf16")


def build_vae():
    vae = AutoencoderKLHunyuanVideo()
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


def run_stage(name, op, calls_denom):
    """Bench one VAE stage (encode/decode) across the 3 paths."""
    print(f"[{name}] benchmarking PyTorch (MIOpen)... (first call runs MIOpen algo search, may be slow)", flush=True)
    torch.cuda.synchronize()
    stock_ms = bench_op(op, WARMUP, ITERS)
    print(f"[{name}]   PyTorch done: {stock_ms:.2f} ms", flush=True)

    def bench_fly(forward_fn, label):
        _orig = HunyuanVideoCausalConv3d.forward
        HunyuanVideoCausalConv3d.forward = forward_fn
        _stats["fly"] = 0
        _stats["fallback"] = 0
        try:
            print(f"[{name}] benchmarking FlyDSL {label}... (warmup JIT-compiles all conv shapes)", flush=True)
            with torch.no_grad():
                op()
            torch.cuda.synchronize()
            ms = bench_op(op, WARMUP, ITERS)
            print(f"[{name}]   FlyDSL {label} done: {ms:.2f} ms", flush=True)
            return ms, _stats["fly"], _stats["fallback"]
        finally:
            HunyuanVideoCausalConv3d.forward = _orig

    bf16_ms, bf16_fly, bf16_fb = bench_fly(_fly_bf16_forward, "BF16")
    fp8_ms, fp8_fly, fp8_fb = bench_fly(_fly_fp8_forward, "FP8")

    print(f"=== {name} ===")
    print(f"{'path':>18}  {'latency (ms)':>13}  {'speedup':>8}")
    print("-" * 44)
    print(f"{'PyTorch (MIOpen)':>18}  {stock_ms:>13.2f}  {'1.00x':>8}")
    print(f"{'FlyDSL BF16':>18}  {bf16_ms:>13.2f}  {stock_ms / bf16_ms:>7.2f}x")
    print(f"{'FlyDSL FP8':>18}  {fp8_ms:>13.2f}  {stock_ms / fp8_ms:>7.2f}x")
    print(f"conv layers per {name}: ~{(bf16_fly + bf16_fb) // calls_denom}")
    print(f"  BF16: {bf16_fly // calls_denom} fly (incl. stems via channel-pad), {bf16_fb // calls_denom} fallback")
    print(f"  FP8 : {fp8_fly // calls_denom} fly (incl. stems via channel-pad), {fp8_fb // calls_denom} fallback")
    print()


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"HunyuanVideo VAE  warmup={WARMUP} iters={ITERS}  latent=(1,?,{FRAMES},{LAT_H},{LAT_W})", flush=True)
    print("building VAE...", flush=True)
    torch.manual_seed(0)
    vae = build_vae()
    z_ch = vae.config.latent_channels

    z = torch.randn(1, z_ch, FRAMES, LAT_H, LAT_W, device="cuda", dtype=torch.bfloat16)

    def decode_op():
        return vae.decode(z)

    print("running first decode (allocates + MIOpen algo search)...", flush=True)
    with torch.no_grad():
        out = decode_op()
        vid = out.sample if hasattr(out, "sample") else out
    vid = vid.contiguous()
    torch.cuda.synchronize()

    def encode_op():
        return vae.encode(vid)

    print(f"latent (1,{z_ch},{FRAMES},{LAT_H},{LAT_W})  <->  video {tuple(vid.shape)}")
    print(flush=True)

    run_stage("decode", decode_op, WARMUP + ITERS + 1)
    run_stage("encode", encode_op, WARMUP + ITERS + 1)


if __name__ == "__main__":
    main()
