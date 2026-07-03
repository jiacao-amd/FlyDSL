#!/usr/bin/env python3
"""E2E numerical accuracy: FlyDSL BF16 / FP8 conv3d vs stock PyTorch.

For each VAE (WAN or Hunyuan), runs decode and encode with all three conv
backends on the same input, then measures output accuracy against the PyTorch
reference. Reports per-element metrics (max abs, rel err, cosine, SNR).

Usage:
  python3 check_vae_e2e_accuracy.py wan      # WAN VAE
  python3 check_vae_e2e_accuracy.py hunyuan  # HunyuanVideo VAE
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F

MODEL = sys.argv[1].lower() if len(sys.argv) > 1 else "wan"
assert MODEL in ("wan", "hunyuan"), f"usage: {sys.argv[0]} wan|hunyuan"

from kernels.conv3d_implicit_8wave import conv3d_implicit_8wave  # noqa: E402
from kernels.conv3d_implicit_8wave_fp8 import conv3d_implicit_8wave_fp8  # noqa: E402

if MODEL == "wan":
    from diffusers import AutoencoderKLWan
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d

    ConvClass = WanCausalConv3d
    FRAMES, LAT_H, LAT_W = 6, 32, 56  # small enough to be fast

    def build():
        v = AutoencoderKLWan().to("cuda", torch.bfloat16).eval()
        return v, v.config.z_dim

    def make_fly(kernel, cvec, wattr):
        def fwd(self, x, cache_x=None):
            pad = list(self._padding)
            if cache_x is not None and self._padding[4] > 0:
                cache_x = cache_x.to(x.device)
                x = torch.cat([cache_x, x], dim=2)
                pad[4] -= cache_x.shape[2]
            x = F.pad(x, pad)
            c = self.in_channels
            if c % cvec:
                x = F.pad(x, (0, 0, 0, 0, 0, 0, 0, cvec - c % cvec))
                w = F.pad(self.weight, (0, 0, 0, 0, 0, 0, 0, cvec - c % cvec))
            else:
                w = self.weight
            return kernel(x, w, bias=self.bias, stride=self.stride, padding=0)

        return fwd

else:
    from diffusers import AutoencoderKLHunyuanVideo
    from diffusers.models.autoencoders.autoencoder_kl_hunyuan_video import HunyuanVideoCausalConv3d

    ConvClass = HunyuanVideoCausalConv3d
    FRAMES, LAT_H, LAT_W = 3, 32, 56

    def build():
        v = AutoencoderKLHunyuanVideo().to("cuda", torch.bfloat16).eval()
        return v, v.config.latent_channels

    def make_fly(kernel, cvec, wattr):
        def fwd(self, x):
            x = F.pad(x, self.time_causal_padding, mode=self.pad_mode)
            c = self.conv.in_channels
            if c % cvec:
                x = F.pad(x, (0, 0, 0, 0, 0, 0, 0, cvec - c % cvec))
                w = F.pad(self.conv.weight, (0, 0, 0, 0, 0, 0, 0, cvec - c % cvec))
            else:
                w = self.conv.weight
            return kernel(x, w, bias=self.conv.bias, stride=self.conv.stride, padding=0)

        return fwd


def metrics(y, ref):
    y, ref = y.float(), ref.float()
    err = (y - ref).abs()
    rel = err.mean() / ref.abs().mean().clamp_min(1e-6)
    sig = ref.pow(2).mean()
    noise = err.pow(2).mean()
    snr = 10 * (sig / noise.clamp_min(1e-12)).log10()
    cos = (y.flatten() @ ref.flatten()) / (y.norm() * ref.norm()).clamp_min(1e-12)
    return {
        "max_abs": err.max().item(),
        "mean_abs": err.mean().item(),
        "rel%": rel.item() * 100,
        "cosine": cos.item(),
        "SNR_dB": snr.item(),
    }


def fmt(m):
    return (
        f"max={m['max_abs']:.3f}  mean={m['mean_abs']:.4f}"
        f"  rel={m['rel%']:.2f}%  cos={m['cosine']:.5f}  SNR={m['SNR_dB']:.1f}dB"
    )


def run(vae, z, op_name, orig_fwd, fly_fwd_bf16, fly_fwd_fp8):
    """Run one stage (decode/encode) under all three backends, compare to PT."""
    results = {}
    for tag, fwd in [("PyTorch", orig_fwd), ("FlyDSL_BF16", fly_fwd_bf16), ("FlyDSL_FP8", fly_fwd_fp8)]:
        ConvClass.forward = fwd
        with torch.no_grad():
            out = z() if callable(z) else vae.decode(z)
            out = out.sample if hasattr(out, "sample") else (out.latent_dist.sample() if hasattr(out, "latent_dist") else out)
        torch.cuda.synchronize()
        results[tag] = out
    ConvClass.forward = orig_fwd
    ref = results["PyTorch"]
    print(f"\n  {op_name} output shape: {tuple(ref.shape)}")
    for tag in ("FlyDSL_BF16", "FlyDSL_FP8"):
        m = metrics(results[tag], ref)
        print(f"  {tag:12s}: {fmt(m)}")
    return results["PyTorch"]


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: {MODEL.upper()} VAE")
    print(f"latent: (1,?,{FRAMES},{LAT_H},{LAT_W})")

    torch.manual_seed(42)
    vae, z_ch = build()
    z = torch.randn(1, z_ch, FRAMES, LAT_H, LAT_W, device="cuda", dtype=torch.bfloat16)

    orig = ConvClass.forward
    fly_bf16 = make_fly(conv3d_implicit_8wave, 8, "_acc_bf16")
    fly_fp8 = make_fly(conv3d_implicit_8wave_fp8, 16, "_acc_fp8")

    print("\n=== DECODE (latent -> video) ===")
    ConvClass.forward = orig
    with torch.no_grad():
        dec_ref = vae.decode(z)
        vid = (dec_ref.sample if hasattr(dec_ref, "sample") else dec_ref).contiguous()
    torch.cuda.synchronize()
    print(f"  video shape: {tuple(vid.shape)}")

    dec_results = {}
    for tag, fwd in [("PyTorch", orig), ("FlyDSL_BF16", fly_bf16), ("FlyDSL_FP8", fly_fp8)]:
        ConvClass.forward = fwd
        with torch.no_grad():
            out = vae.decode(z)
            out = out.sample if hasattr(out, "sample") else out
        torch.cuda.synchronize()
        dec_results[tag] = out.float()
    ConvClass.forward = orig

    ref = dec_results["PyTorch"]
    print(f"  reference shape: {tuple(ref.shape)}")
    for tag in ("FlyDSL_BF16", "FlyDSL_FP8"):
        m = metrics(dec_results[tag], ref)
        print(f"  {tag:12s}: {fmt(m)}")

    print("\n=== ENCODE (video -> latent) ===")
    enc_results = {}
    for tag, fwd in [("PyTorch", orig), ("FlyDSL_BF16", fly_bf16), ("FlyDSL_FP8", fly_fp8)]:
        ConvClass.forward = fwd
        with torch.no_grad():
            out = vae.encode(vid)
            # Use .mean (deterministic) not .sample() — sample adds reparameterization
            # noise so two calls are inherently different even with the same weights.
            out = out.latent_dist.mean if hasattr(out, "latent_dist") else out
        torch.cuda.synchronize()
        enc_results[tag] = out.float()
    ConvClass.forward = orig

    ref = enc_results["PyTorch"]
    print(f"  reference shape: {tuple(ref.shape)}")
    for tag in ("FlyDSL_BF16", "FlyDSL_FP8"):
        m = metrics(enc_results[tag], ref)
        print(f"  {tag:12s}: {fmt(m)}")

    print("\n=== ENCODE→DECODE roundtrip accuracy ===")
    print("  (encode FlyDSL latent, then decode with PyTorch — measures latent quality)")
    for tag, fwd in [("FlyDSL_BF16", fly_bf16), ("FlyDSL_FP8", fly_fp8)]:
        ConvClass.forward = fwd
        with torch.no_grad():
            enc_out = vae.encode(vid)
            lat = enc_out.latent_dist.mean if hasattr(enc_out, "latent_dist") else enc_out
        ConvClass.forward = orig
        with torch.no_grad():
            dec_out = vae.decode(lat)
            recon = (dec_out.sample if hasattr(dec_out, "sample") else dec_out).float()
        torch.cuda.synchronize()
        m = metrics(recon, vid.float())
        print(f"  {tag:12s} recon vs original video: {fmt(m)}")


if __name__ == "__main__":
    main()
