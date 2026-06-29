#!/usr/bin/env python3
"""Local (untracked) e2e test: conv3d_implicit_mfma inside the Wan VAE decoder.

Monkeypatches diffusers' ``WanCausalConv3d.forward`` to dispatch supported convs
to ``causal_conv3d_implicit_mfma`` (PyTorch fallback otherwise), then runs the
real Wan VAE decode patched vs unpatched and reports correctness (PSNR/MSE),
coverage (kernel hits vs fallbacks by reason), and VAE-decode speed.

Scope is the VAE step only (no DiT, no xDiT), bf16. The dispatcher replicates
diffusers' cat + F.pad (including the temporal feat-cache) and runs the padded
tensor through the kernel as a plain pad-0 conv, so cached convs are covered too.

    HIP_VISIBLE_DEVICES=4 python3 test_wan_vae_conv3d.py
    HIP_VISIBLE_DEVICES=4 python3 test_wan_vae_conv3d.py --verify-layers
    HIP_VISIBLE_DEVICES=4 python3 test_wan_vae_conv3d.py --frames 25 --hw 256
    HIP_VISIBLE_DEVICES=4 python3 test_wan_vae_conv3d.py --repo Wan-AI/Wan2.1-T2V-1.3B-Diffusers
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, OrderedDict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "build-fly" / "python_packages"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from kernels.conv3d_implicit_mfma import conv3d_implicit_mfma  # noqa: E402

RTOL, ATOL = 2e-2, 2e-2

# Exact 3x3x3 padded signatures measured slower than MIOpen (hard-coded fallback).
# Key = (tuple(xpad.shape), in_channels, out_channels, kernel). Spatially-trivial
# convs (1x1x1 / 3x1x1) are handled structurally in Dispatcher._reason instead.
SLOW_SHAPES = {
    ((1, 192, 3, 258, 258), 192, 192, (3, 3, 3)),  # 0.89x @512
    ((1, 384, 4, 130, 130), 384, 384, (3, 3, 3)),  # 0.99x @512 (x30)
}


def find_causal_conv3d_class(vae):
    """Locate the WanCausalConv3d class (import path varies across diffusers)."""
    try:
        from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d

        return WanCausalConv3d
    except Exception:
        pass
    for m in vae.modules():
        if type(m).__name__ == "WanCausalConv3d":
            return type(m)
    raise RuntimeError("Could not locate WanCausalConv3d class")


class Dispatcher:
    """Holds the original forward + counters; produces the patched forward."""

    def __init__(self, cls, verify_layers=False):
        self.cls = cls
        self.orig = cls.forward
        self.verify_layers = verify_layers
        self.hits = 0
        self.hits_cache = 0
        self.fallbacks = Counter()
        self.hit_shapes = OrderedDict()  # key -> count
        self.verify_seen = set()
        self.verify_max_err = 0.0
        self.verify_fail = []

    def _reason(self, conv, x):
        # cache_x is handled by replicating diffusers' cat+pad, so it is NOT a
        # fallback reason here.
        if x.dtype != torch.bfloat16:
            return "dtype"
        if tuple(conv.dilation) != (1, 1, 1) or conv.groups != 1:
            return "dil_or_groups"
        # Spatially-trivial convs (kh==kw==1: the 1x1x1 and 3x1x1 kernels) have no
        # spatial reuse -> they are effectively GEMMs where MIOpen wins (measured
        # 0.74-0.83x). Route them to the stock path.
        kt, kh, kw = tuple(conv.weight.shape[2:])
        if kh == 1 and kw == 1:
            return "spatial_1x1"
        return None

    def make_patched(self):
        disp = self

        def patched(conv, x, cache_x=None):
            reason = disp._reason(conv, x)

            # Replicate diffusers WanCausalConv3d.forward verbatim (cat + F.pad),
            # then run the padded tensor through the kernel as a plain pad-0 conv.
            cached = cache_x is not None and conv._padding[4] > 0
            padding = list(conv._padding)
            xin = x
            if cached:
                cache_x = cache_x.to(x.device)
                xin = torch.cat([cache_x, x], dim=2)
                padding[4] -= cache_x.shape[2]
            xpad = F.pad(xin, padding)

            # Hard-coded fallback for specific 3x3x3 shapes measured slower than
            # MIOpen (can't be told apart from the winning shapes by C/K/spatial
            # alone -- they differ only in the temporal dim -- so match the exact
            # padded signature). (spatially-trivial convs are caught by _reason.)
            key = (tuple(xpad.shape), conv.in_channels, conv.out_channels, tuple(conv.weight.shape[2:]))
            if reason is None and key in SLOW_SHAPES:
                reason = "slow_shape"
            if reason is not None:
                disp.fallbacks[reason] += 1
                return disp.orig(conv, x, cache_x)

            out = conv3d_implicit_mfma(
                xpad, conv.weight, bias=conv.bias, stride=tuple(conv.stride), padding=0
            ).contiguous()

            disp.hits += 1
            if cached:
                disp.hits_cache += 1
            disp.hit_shapes[key] = disp.hit_shapes.get(key, 0) + 1

            if disp.verify_layers and key not in disp.verify_seen:
                disp.verify_seen.add(key)
                ref = disp.orig(conv, x, cache_x if cached else None)
                err = (out.float() - ref.float()).abs().max().item()
                disp.verify_max_err = max(disp.verify_max_err, err)
                if not torch.allclose(out, ref, rtol=RTOL, atol=ATOL):
                    disp.verify_fail.append((key, err))
            return out

        return patched

    def __enter__(self):
        self.cls.forward = self.make_patched()
        return self

    def __exit__(self, *a):
        self.cls.forward = self.orig


def psnr(ref, out):
    mse = (out.float() - ref.float()).pow(2).mean().item()
    if mse == 0:
        return float("inf"), 0.0
    peak = ref.float().abs().max().item() ** 2
    return 10.0 * torch.log10(torch.tensor(peak / mse)).item(), mse


def time_decode(decode_fn, *, iters, rounds):
    best = float("inf")
    for _ in range(rounds):
        for _ in range(2):
            decode_fn()
        torch.cuda.synchronize()
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(iters):
            decode_fn()
        b.record()
        torch.cuda.synchronize()
        best = min(best, a.elapsed_time(b) / iters)
    return best


def run_stage(vae, cls, stage, args):
    print(f"\n########## VAE {stage.upper()} ##########")
    torch.manual_seed(0)
    if stage == "encode":
        # encode: feed a random video (N, 3, T, H, W) in the valid pixel range
        # [-1, 1]; raw randn (±4-5) overflows the deep bf16 encoder -> nan.
        x_in = torch.randn(1, 3, args.frames, args.hw, args.hw, device="cuda", dtype=torch.bfloat16).clamp_(-1, 1)
        print(f"encode video {tuple(x_in.shape)} -> latent")

        def run():
            with torch.no_grad():
                d = vae.encode(x_in).latent_dist
                return d.mode() if hasattr(d, "mode") else d.mean

    else:
        # decode: latent shape from config (z_dim ch, spatial /8, temporal (T-1)/4+1).
        z_dim = int(getattr(vae.config, "z_dim", 16))
        t_lat = (args.frames - 1) // 4 + 1
        z = torch.randn(1, z_dim, t_lat, args.hw // 8, args.hw // 8, device="cuda", dtype=torch.bfloat16)
        print(f"decode latent {tuple(z.shape)} -> ~({args.frames},{args.hw},{args.hw})")

        def run():
            with torch.no_grad():
                return vae.decode(z).sample

    # Stock reference + patched output.
    ref = run().float()
    disp = Dispatcher(cls, verify_layers=args.verify_layers)
    with disp:
        out = run().float()
    torch.cuda.synchronize()

    p, mse = psnr(ref, out)
    maxerr = (out - ref).abs().max().item()
    print(f"\n=== correctness (patched vs stock VAE {stage}) ===")
    print(f"shape {tuple(out.shape)}  PSNR {p:.2f} dB  MSE {mse:.3e}  max_abs_err {maxerr:.3e}")

    print("\n=== coverage ===")
    total = disp.hits + sum(disp.fallbacks.values())
    print(f"kernel hits: {disp.hits}/{total} conv calls  (of which {disp.hits_cache} used feat-cache)")
    if disp.fallbacks:
        print("fallbacks by reason: " + ", ".join(f"{k}={v}" for k, v in disp.fallbacks.most_common()))
    print(f"distinct kernel-hit conv shapes: {len(disp.hit_shapes)}")
    for key, cnt in disp.hit_shapes.items():
        xs, ci, co, ksz = key
        print(f"  x={xs} C={ci} K={co} k={ksz}  x{cnt}")

    if args.verify_layers:
        print("\n=== per-layer verify ===")
        print(f"checked {len(disp.verify_seen)} distinct layers, max_abs_err {disp.verify_max_err:.3e}")
        if disp.verify_fail:
            print(f"FAILED layers ({len(disp.verify_fail)}):")
            for key, err in disp.verify_fail:
                print(f"  {key}  err={err:.3e}")
        else:
            print("all supported layers within tolerance: PASS")

    print(f"\n=== speed (VAE {stage} wall-clock) ===")
    stock_ms = time_decode(run, iters=args.iters, rounds=args.rounds)
    with disp:
        patched_ms = time_decode(run, iters=args.iters, rounds=args.rounds)
    print(f"stock   {stock_ms:8.2f} ms")
    print(f"patched {patched_ms:8.2f} ms   speedup {stock_ms / patched_ms:.2f}x")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    ap.add_argument("--frames", type=int, default=17, help="video frame count")
    ap.add_argument("--hw", type=int, default=128, help="spatial size (square)")
    ap.add_argument("--encode", action="store_true", help="benchmark vae.encode")
    ap.add_argument("--both", action="store_true", help="benchmark both encode and decode")
    ap.add_argument("--verify-layers", action="store_true")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--iters", type=int, default=3)
    args = ap.parse_args()

    from diffusers import AutoencoderKLWan

    print(f"Loading VAE from {args.repo} (subfolder=vae, bf16)...")
    vae = AutoencoderKLWan.from_pretrained(args.repo, subfolder="vae", torch_dtype=torch.bfloat16)
    vae = vae.to("cuda").eval()
    for fn in ("disable_tiling", "disable_slicing"):
        if hasattr(vae, fn):
            getattr(vae, fn)()

    cls = find_causal_conv3d_class(vae)
    print(f"Patched class: {cls.__module__}.{cls.__name__}")

    if args.both:
        stages = ["encode", "decode"]
    elif args.encode:
        stages = ["encode"]
    else:
        stages = ["decode"]
    for stage in stages:
        run_stage(vae, cls, stage, args)


if __name__ == "__main__":
    main()
